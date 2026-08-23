"""Declarative backend registry behind the scaled_dot_product_attention router."""
from dataclasses import dataclass, field
from typing import Callable
import torch


AttentionCall = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Platform:
    """Where the router runs: the devices backend name and the selected device."""
    backend: str
    device: torch.device | None = None


@dataclass(frozen=True)
class Constraints:
    """Shape, dtype and device conditions a backend serves; a call failing any of them moves on to the next entry."""
    allow_cpu: bool = False
    allow_mask: bool = True
    allow_float32: bool = True
    same_device: bool = False
    head_dims: frozenset[int] | None = None
    max_head_dim: int | None = None
    min_tokens: int = 0 # query and key sequences both at least this long
    min_long_side: int = 0 # query or key sequence longer than this
    min_heads: int = 0
    min_ndim: int = 0

    def accepts(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None) -> bool:
        if not self.allow_cpu and query.device.type == 'cpu':
            return False
        if self.min_ndim and query.ndim < self.min_ndim:
            return False
        if not self.allow_mask and attn_mask is not None:
            return False
        if not self.allow_float32 and query.dtype == torch.float32:
            return False
        if self.same_device and (key.device != query.device or value.device != query.device):
            return False
        head_dim = query.shape[-1]
        if self.head_dims is not None and head_dim not in self.head_dims:
            return False
        if self.max_head_dim is not None and head_dim > self.max_head_dim:
            return False
        if self.min_tokens and (query.shape[-2] < self.min_tokens or key.shape[-2] < self.min_tokens):
            return False
        if self.min_long_side and query.shape[-2] <= self.min_long_side and key.shape[-2] <= self.min_long_side:
            return False
        if self.min_heads and query.shape[-3] < self.min_heads:
            return False
        return True


@dataclass(frozen=True)
class AttentionBackend:
    """One attention implementation: how to prepare it once and which calls it serves."""
    name: str
    label: str # the sdp_overrides choice that enables it
    priority: int # higher priority entries are tried first
    prepare: Callable[[Platform, AttentionCall], AttentionCall | None] # imports and configures the implementation, returns its call or None
    constraints: Constraints = field(default_factory=Constraints)
    terminal: bool = False # serves every call the entries decline, in place of the original sdpa
    platforms: frozenset[str] | None = None # devices backends the implementation exists for, None for all
    options: tuple[str, ...] = () # settings the prepared call captures; a change to one rebuilds the chain
    caps: frozenset[str] = frozenset() # what the call can consume beyond plain sdpa arguments: 'block_mask', and 'masked_block' when it composes one with a token mask

    def available_on(self, platform: Platform) -> bool:
        return self.platforms is None or platform.backend in self.platforms


class Registry:
    def __init__(self):
        self.backends: dict[str, AttentionBackend] = {}

    def register(self, backend: AttentionBackend) -> AttentionBackend:
        if backend.name in self.backends:
            raise ValueError(f'attention backend registered twice: name={backend.name}')
        if self.by_label(backend.label) is not None:
            raise ValueError(f'attention backend label registered twice: label="{backend.label}"')
        self.backends[backend.name] = backend
        return backend

    def by_label(self, label: str) -> AttentionBackend | None:
        return next((backend for backend in self.backends.values() if backend.label == label), None)

    def ordered(self) -> list[AttentionBackend]:
        """Backends by ascending priority, the order they are prepared in."""
        return sorted(self.backends.values(), key=lambda backend: backend.priority)

    def labels(self) -> list[str]:
        return [backend.label for backend in self.ordered()]

    def options(self) -> list[str]:
        return sorted({name for backend in self.backends.values() for name in backend.options})

    def with_cap(self, cap: str) -> list[AttentionBackend]:
        return [backend for backend in self.ordered() if cap in backend.caps]


registry = Registry()
