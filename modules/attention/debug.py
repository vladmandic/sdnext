"""Opt-in route tracing for the sdpa router, enabled by SD_ATTN_DEBUG."""
import os
import torch
from modules.logger import log
from modules.attention import context

enabled = os.environ.get('SD_ATTN_DEBUG', None) is not None
seen: set[tuple] = set()


def observe(name: str, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor | None) -> None:
    """Log each distinct route once: backend, component role, step, shapes, dtype and mask presence."""
    signature = (name, context.current.role, tuple(query.shape), tuple(key.shape), str(query.dtype), attn_mask is not None)
    if signature in seen:
        return
    seen.add(signature)
    log.debug(f'Attention route: backend={name} role={context.current.role} step={context.current.step} q={list(query.shape)} k={list(key.shape)} dtype={query.dtype} mask={attn_mask is not None}')


def reset() -> None:
    seen.clear()
