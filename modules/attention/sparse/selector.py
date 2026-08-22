"""Fixed-budget block selection: which KV tiles each query tile attends to."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SparseSpec:
    """How much to keep and at what granularity. Budget is a fraction of the sparsifiable candidates, pins are added on top."""
    budget: float = 0.30
    block_q: int = 128
    block_kv: int = 64
    head_shared: bool = False # score once for all heads, cheaper and coarser
    force: bool = False # skip the dense short circuit, so tests can exercise the path at budget 1.0
    score_chunk_bytes: int = 256 << 20


@dataclass(frozen=True)
class BlockSelection:
    """int8 keep flags per (query tile, kv tile); the geometry every consumer reads."""
    keep: torch.Tensor # (B, H, NQ, NK), H is the query head count or 1
    block_q: int
    block_kv: int
    budget: float
    seq_q: int
    seq_kv: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.keep.shape)

    def density(self) -> float:
        """Fraction of tiles kept. Reads back from the accelerator, so this is for reporting and tests, never the hot path."""
        return float(self.keep.sum().item()) / max(self.keep.numel(), 1)


def block_count(length: int, block: int) -> int:
    return (length + block - 1) // block


def pool_blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    """Mean over each block of tokens, fp32, without materializing a padded copy."""
    seq = x.shape[-2]
    whole = (seq // block) * block
    parts = []
    if whole:
        head = x[..., :whole, :]
        parts.append(head.unflatten(-2, (whole // block, block)).mean(dim=-2, dtype=torch.float32))
    if whole < seq:
        parts.append(x[..., whole:, :].mean(dim=-2, dtype=torch.float32, keepdim=True))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)


def diagonal_blocks(nq: int, nk: int, block_q: int, block_kv: int, device) -> torch.Tensor:
    """Tiles whose query and key token ranges overlap; keeping them removes the empty-row case."""
    q_index = torch.arange(nq, device=device).unsqueeze(-1)
    k_index = torch.arange(nk, device=device).unsqueeze(0)
    return (q_index * block_q < (k_index + 1) * block_kv) & (k_index * block_kv < (q_index + 1) * block_q)


def score_blocks(query: torch.Tensor, key: torch.Tensor, spec: SparseSpec) -> torch.Tensor:
    """Mean-pooled query-key affinity per tile pair. No scale and no softmax: top-k is invariant under both."""
    pooled_q = pool_blocks(query, spec.block_q) # (B, Hq, NQ, D)
    pooled_k = pool_blocks(key, spec.block_kv) # (B, Hkv, NK, D)
    heads_q, heads_kv = pooled_q.shape[1], pooled_k.shape[1]
    if spec.head_shared:
        pooled_q = pooled_q.mean(dim=1, keepdim=True)
        pooled_k = pooled_k.mean(dim=1, keepdim=True)
    elif heads_kv != heads_q: # gqa: score on query heads, the geometry both consumers expect
        pooled_k = pooled_k.repeat_interleave(heads_q // heads_kv, dim=1)
    heads = pooled_q.shape[1]
    per_head = pooled_q.shape[2] * pooled_k.shape[2] * 4
    chunk = max(1, min(heads, spec.score_chunk_bytes // max(per_head, 1)))
    if chunk >= heads:
        return pooled_q @ pooled_k.transpose(-1, -2)
    return torch.cat([pooled_q[:, i:i + chunk] @ pooled_k[:, i:i + chunk].transpose(-1, -2) for i in range(0, heads, chunk)], dim=1)


plan_cache: dict = {}


def selection_plan(spec: SparseSpec, nq: int, nk: int, pins, drops, device, cache_key=None):
    """The parts that depend only on geometry and layout, not on the tensors: what must be kept, what may be chosen, and how many."""
    key = (cache_key, nq, nk, spec.block_q, spec.block_kv, spec.budget, str(device))
    hit = plan_cache.get(key) if cache_key is not None else None
    if hit is not None:
        return hit
    must = diagonal_blocks(nq, nk, spec.block_q, spec.block_kv, device).unsqueeze(0).unsqueeze(0)
    if pins is not None:
        must = must | pins
    forbidden = drops if drops is not None else torch.zeros_like(must)
    candidates = ~must & ~forbidden
    per_row = candidates.sum(dim=-1, keepdim=True) # (.., NQ, 1)
    keep_per_row = torch.ceil(per_row * spec.budget).to(torch.int64)
    covers_everything = bool((keep_per_row >= per_row).all()) # one readback, amortized over the generation by the cache
    built = (must, forbidden, candidates, keep_per_row, covers_everything)
    if cache_key is not None:
        if len(plan_cache) > 32:
            plan_cache.clear()
        plan_cache[key] = built
    return built


def select_blocks(query: torch.Tensor, key: torch.Tensor, spec: SparseSpec, pins: torch.Tensor | None = None, drops: torch.Tensor | None = None, cache_key=None) -> BlockSelection | None:
    """Keep the highest scoring KV tiles per query tile within the budget, plus pins and the diagonal. None means attend densely."""
    seq_q, seq_kv = query.shape[-2], key.shape[-2]
    nq, nk = block_count(seq_q, spec.block_q), block_count(seq_kv, spec.block_kv)
    device = query.device
    must, forbidden, candidates, keep_per_row, covers_everything = selection_plan(spec, nq, nk, pins, drops, device, cache_key)
    if covers_everything and not spec.force:
        return None # the budget covers every candidate, so the mask would be dense

    scores = score_blocks(query, key, spec)
    scores = scores.masked_fill(~candidates.expand_as(scores), float('-inf'))
    # rank rather than topk, so the per row budget varies without a host side k
    order = scores.argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(nk, device=device).expand_as(order))
    keep = must | ((rank < keep_per_row) & candidates)
    keep &= ~forbidden
    return BlockSelection(keep=keep.to(torch.int8), block_q=spec.block_q, block_kv=spec.block_kv, budget=spec.budget, seq_q=seq_q, seq_kv=seq_kv)


def radial_blocks(seq_q: int, seq_kv: int, density: float, spec: SparseSpec, device) -> BlockSelection:
    """A band around the diagonal at the requested density: the static control the selector has to beat."""
    nq, nk = block_count(seq_q, spec.block_q), block_count(seq_kv, spec.block_kv)
    q_center = (torch.arange(nq, device=device).unsqueeze(-1) + 0.5) * spec.block_q
    k_center = (torch.arange(nk, device=device).unsqueeze(0) + 0.5) * spec.block_kv
    distance = (q_center - k_center).abs()
    low, high = 0.0, float(max(seq_q, seq_kv))
    for _ in range(40): # bisect the bandwidth, since the band width to density map has no closed form at the edges
        mid = (low + high) / 2
        if float((distance <= mid).to(torch.float32).mean().item()) < density:
            low = mid
        else:
            high = mid
    keep = (distance <= high).unsqueeze(0).unsqueeze(0).to(torch.int8)
    return BlockSelection(keep=keep, block_q=spec.block_q, block_kv=spec.block_kv, budget=density, seq_q=seq_q, seq_kv=seq_kv)


def schedule(steps: int, budget: float, bump: float = 0.0, bump_steps: int = 0) -> tuple[float, ...]:
    """Per-step budgets, precomputed. At most two distinct values, so a compiled consumer sees at most two specializations."""
    if bump <= 0 or bump_steps <= 0 or steps <= 0:
        return tuple([budget] * max(steps, 0))
    raised = min(1.0, budget + bump)
    edge = min(bump_steps, math.ceil(steps / 2))
    return tuple([raised if (i < edge or i >= steps - edge) else budget for i in range(steps)])
