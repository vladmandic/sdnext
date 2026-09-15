"""Opt-in route tracing for the sdpa router, enabled by SD_ATTN_DEBUG."""
import os
import torch
from modules.logger import log
from modules.attention import context

enabled = os.environ.get('SD_ATTN_DEBUG', None) is not None
seen: set[tuple] = set()
counts: dict[tuple, int] = {}


def observe(name: str, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor | None) -> None:
    """Log each distinct route once: backend, component role, step, shapes, dtype, mask presence and whether the inputs are contiguous; count every call."""
    contiguous = query.is_contiguous() and key.is_contiguous()
    signature = (name, context.current.role, tuple(query.shape), tuple(key.shape), str(query.dtype), attn_mask is not None, contiguous)
    counts[signature] = counts.get(signature, 0) + 1
    if signature in seen:
        return
    seen.add(signature)
    log.debug(f'Attention route: backend={name} role={context.current.role} step={context.current.step} q={list(query.shape)} k={list(key.shape)} dtype={query.dtype} mask={attn_mask is not None} contiguous={contiguous}')


def summary() -> list[str]:
    """One line per route with its call count since the last generation, busiest first."""
    lines = []
    for signature, count in sorted(counts.items(), key=lambda item: -item[1]):
        name, role, q_shape, k_shape, dtype, masked, contiguous = signature
        lines.append(f'backend={name} role={role} q={list(q_shape)} k={list(k_shape)} dtype={dtype} mask={masked} contiguous={contiguous} calls={count}')
    return lines


def end_generation() -> None:
    """Log the route counts of the generation that just ended and start the next count."""
    if enabled and counts:
        for line in summary():
            log.debug(f'Attention routes: {line}')
    counts.clear()


def reset() -> None:
    seen.clear()
    counts.clear()
