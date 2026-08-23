"""What each token in a packed sequence is, so the selector knows what it may sparsify."""
from dataclasses import dataclass
import torch


# only the bulk modalities are sparsifiable; everything else is pinned dense, and an unrecognized kind pins too
SPARSIFIABLE = frozenset({'video', 'image'})
DROPPED = frozenset({'pad'})


@dataclass(frozen=True)
class Span:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenLayout:
    """Ordered spans covering one packed sequence."""
    spans: tuple[Span, ...]
    length: int
    source: str = 'unknown' # how the layout was obtained, for the log

    def key(self) -> tuple:
        return (self.length, self.source, tuple((s.kind, s.start, s.end) for s in self.spans))

    def kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.kind for s in self.spans))

    def sparsifiable_tokens(self) -> int:
        return sum(s.end - s.start for s in self.spans if s.kind in SPARSIFIABLE)

    def token_flags(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per token: may this be sparsified, and is it padding."""
        sparse = torch.zeros(self.length, dtype=torch.bool, device=device)
        pad = torch.zeros(self.length, dtype=torch.bool, device=device)
        for span in self.spans:
            if span.kind in SPARSIFIABLE:
                sparse[span.start:span.end] = True
            elif span.kind in DROPPED:
                pad[span.start:span.end] = True
        return sparse, pad


def runs(indices: torch.Tensor) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs in a sorted 1d index tensor."""
    if indices.numel() == 0:
        return []
    values = indices.detach().to('cpu', torch.int64).sort().values
    breaks = (values[1:] - values[:-1] != 1).nonzero().flatten().tolist()
    bounds = [0, *[b + 1 for b in breaks], values.numel()]
    return [(int(values[bounds[i]].item()), int(values[bounds[i + 1] - 1].item()) + 1) for i in range(len(bounds) - 1)]


def layout_from_index_kwargs(kwargs: dict, length: int | None = None) -> TokenLayout | None:
    """Read a layout off the *_indices tensors a pipeline passes its transformer by name."""
    spans: list[Span] = []
    for name, value in kwargs.items():
        if not name.endswith('_indices') or not torch.is_tensor(value) or value.dim() != 1 or value.is_floating_point():
            continue
        kind = name[:-len('_indices')].lower()
        found = runs(value)
        for position, (start, end) in enumerate(found):
            # a video run that is not the last one is keyframe conditioning, which stays dense
            resolved = 'cond' if (kind == 'video' and position < len(found) - 1) else kind
            spans.append(Span(kind=resolved, start=start, end=end))
    if not spans:
        return None
    spans.sort(key=lambda s: s.start)
    return TokenLayout(spans=tuple(spans), length=length if length is not None else spans[-1].end, source='indices')


def layout_from_segments(segments, length: int | None = None, source: str = 'segments') -> TokenLayout:
    """Build a layout from ordered (kind, count) pairs, the form a transformer knows at its packing site."""
    spans: list[Span] = []
    cursor = 0
    for kind, count in segments:
        if count <= 0:
            continue
        spans.append(Span(kind=kind, start=cursor, end=cursor + count))
        cursor += count
    return TokenLayout(spans=tuple(spans), length=length if length is not None else cursor, source=source)


def segments_from_live(live: torch.Tensor, kind: str, pad_kind: str = 'pad') -> list[tuple[str, int]]:
    """Run length encode a boolean live mask into ordered (kind, count) pairs, the dead runs labelled as padding."""
    values = live.detach().to('cpu').bool()
    if values.numel() == 0:
        return []
    changes = (values[1:] != values[:-1]).nonzero().flatten().tolist()
    bounds = [0, *[c + 1 for c in changes], values.numel()]
    return [(kind if bool(values[bounds[i]]) else pad_kind, bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]


def publish_segments(segments, length: int | None = None, source: str = 'segments') -> None:
    """Publish a layout from the site that packs the sequence, which is the only place the segment lengths are all known."""
    from modules.attention import context
    context.set_layout(layout_from_segments(segments, length=length, source=source))


def layout_from_prefix(length: int, prefix: int) -> TokenLayout:
    """Fallback when nothing published a layout: treat a leading run as conditioning and sparsify the rest."""
    return layout_from_segments([('text', prefix), ('image', length - prefix)], length=length, source='prefix')


def block_flags(flags: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per block: do all tokens carry the flag, does any token carry it."""
    seq = flags.shape[0]
    whole = (seq // block) * block
    parts_all, parts_any = [], []
    if whole:
        view = flags[:whole].view(whole // block, block)
        parts_all.append(view.all(dim=-1))
        parts_any.append(view.any(dim=-1))
    if whole < seq:
        parts_all.append(flags[whole:].all(dim=-1, keepdim=True))
        parts_any.append(flags[whole:].any(dim=-1, keepdim=True))
    def join(parts):
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return join(parts_all), join(parts_any)


pin_cache: dict = {}


def block_pins(layout: TokenLayout, seq_q: int, seq_kv: int, block_q: int, block_kv: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiles that must stay dense and tiles that can be skipped outright, as (1, 1, NQ, NK) masks."""
    cache_key = (layout.key(), seq_q, seq_kv, block_q, block_kv, str(device))
    hit = pin_cache.get(cache_key)
    if hit is not None:
        return hit
    sparse_tokens, pad_tokens = layout.token_flags(device)
    q_sparse = sparse_tokens[:seq_q] if layout.length >= seq_q else torch.nn.functional.pad(sparse_tokens, (0, seq_q - layout.length))
    kv_sparse = sparse_tokens[:seq_kv] if layout.length >= seq_kv else torch.nn.functional.pad(sparse_tokens, (0, seq_kv - layout.length))
    kv_pad = pad_tokens[:seq_kv] if layout.length >= seq_kv else torch.nn.functional.pad(pad_tokens, (0, seq_kv - layout.length))
    q_all_sparse, _ = block_flags(q_sparse, block_q)
    kv_all_sparse, _ = block_flags(kv_sparse, block_kv)
    kv_all_pad, _ = block_flags(kv_pad, block_kv)
    # a tile is pinned when its query tile or its key tile carries anything that is not sparsifiable, boundary tiles included
    pins = (~q_all_sparse).unsqueeze(-1) | (~kv_all_sparse).unsqueeze(0)
    drops = kv_all_pad.unsqueeze(0).expand_as(pins)
    pins = (pins & ~drops).unsqueeze(0).unsqueeze(0).contiguous()
    drops = drops.unsqueeze(0).unsqueeze(0).contiguous()
    if len(pin_cache) > 32:
        pin_cache.clear()
    pin_cache[cache_key] = (pins, drops)
    return pins, drops
