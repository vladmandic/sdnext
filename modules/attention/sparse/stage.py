"""The router stage that turns settings plus a published layout into a per call block selection."""
from dataclasses import dataclass
from modules.logger import log
from modules.attention import context
from modules.attention.sparse import layout as layout_mod
from modules.attention.sparse.selector import SparseSpec, block_count, schedule, select_blocks


# measured on a 3090: below roughly this length a 30 percent budget caps under 1.25x per block,
# so the selector cannot pay for itself; see docs/sparse-attention-tracker.md
AUTO_MIN_TOKENS = 8192

# settings the stage reads, so a change to any of them rebuilds the chain
OPTION_NAMES = ('sparse_attention_enabled', 'sparse_attention_budget', 'sparse_attention_min_tokens', 'sparse_attention_schedule_steps', 'sparse_attention_schedule_bump', 'sparse_attention_head_shared')


@dataclass(frozen=True)
class StageOptions:
    enabled: bool = False
    budget: float = 0.30
    min_tokens: int = 0 # 0 selects AUTO_MIN_TOKENS
    schedule_steps: int = 0
    schedule_bump: float = 0.0
    head_shared: bool = False

    @property
    def gate(self) -> int:
        return self.min_tokens if self.min_tokens > 0 else AUTO_MIN_TOKENS


def read_options() -> StageOptions:
    from modules import shared
    opts = shared.opts
    return StageOptions(
        enabled=bool(getattr(opts, 'sparse_attention_enabled', False)),
        budget=float(getattr(opts, 'sparse_attention_budget', 30)) / 100.0,
        min_tokens=int(getattr(opts, 'sparse_attention_min_tokens', 0)),
        schedule_steps=int(getattr(opts, 'sparse_attention_schedule_steps', 0)),
        schedule_bump=float(getattr(opts, 'sparse_attention_schedule_bump', 0)) / 100.0,
        head_shared=bool(getattr(opts, 'sparse_attention_head_shared', False)),
    )


def resolve_layout(seq: int, reported: set) -> layout_mod.TokenLayout:
    """The published layout when there is one, otherwise sparsify the whole sequence and say so once."""
    published = context.current.layout
    if isinstance(published, layout_mod.TokenLayout) and published.length == seq:
        return published
    if seq not in reported:
        reported.add(seq)
        detail = 'none published' if published is None else f'published length {getattr(published, "length", None)} does not match {seq}'
        log.info(f'Sparse attention: no token layout ({detail}), sparsifying the whole sequence at tokens={seq}')
    return layout_mod.layout_from_prefix(seq, 0)


def make_stage(options: StageOptions):
    """Return the per call selector, or None when the feature is off."""
    if not options.enabled or options.budget >= 1.0:
        return None
    reported: set = set()
    inactive: set = set()
    cache: dict = {}

    def budget_for_step() -> float:
        state = context.current
        if options.schedule_steps <= 0 or options.schedule_bump <= 0 or state.steps <= 0:
            return options.budget
        key = (state.steps, options.budget, options.schedule_bump, options.schedule_steps)
        table = cache.get(key)
        if table is None:
            table = schedule(state.steps, options.budget, options.schedule_bump, options.schedule_steps)
            cache.clear()
            cache[key] = table
        return table[min(state.step, len(table) - 1)] if table else options.budget

    def decline(reason: str):
        stage.last_skip = reason
        return None

    def stage(query, key, value, attn_mask, is_causal): # pylint: disable=unused-argument
        state = context.current
        if state.role != 'transformer' or not state.active:
            return decline('not the denoiser')
        if attn_mask is not None or is_causal: # flex would need a mask_mod to combine these; the quantized kernel composes them in R2
            return decline('masked or causal')
        if query.device.type == 'cpu' or query.dim() != 4:
            return decline('unsupported tensor')
        seq_q, seq_kv = query.shape[-2], key.shape[-2]
        if seq_q != seq_kv: # cross attention is short and already cheap
            return decline('cross attention')
        if seq_q < options.gate:
            if seq_q not in inactive: # an enabled setting that cannot act says so rather than doing nothing quietly
                inactive.add(seq_q)
                log.info(f'Sparse attention: inactive at tokens={seq_q}, below the minimum sequence of {options.gate}; attention stays dense')
            return decline('below the minimum sequence')
        budget = budget_for_step()
        if budget >= 1.0:
            return decline('budget covers everything')
        spec = SparseSpec(budget=budget, head_shared=options.head_shared)
        token_layout = resolve_layout(seq_q, reported)
        nq, nk = block_count(seq_q, spec.block_q), block_count(seq_kv, spec.block_kv)
        pins, drops = layout_mod.block_pins(token_layout, seq_q, seq_kv, spec.block_q, spec.block_kv, query.device)
        if pins.shape[-2:] != (nq, nk):
            return decline('layout geometry mismatch')
        stage.last_skip = None
        return select_blocks(query, key, spec, pins=pins, drops=drops)

    stage.options = options
    stage.last_skip = None
    return stage
