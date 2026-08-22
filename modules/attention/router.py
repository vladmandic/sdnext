"""The single scaled_dot_product_attention entry point over the prepared backends."""
from dataclasses import dataclass
from functools import wraps
import torch
from installer import torch_info
from modules.logger import log
from modules.attention.registry import AttentionBackend, AttentionCall, Platform, Registry, registry as default_registry


@dataclass(frozen=True)
class PlanEntry:
    backend: AttentionBackend
    call: AttentionCall


@dataclass(frozen=True)
class Plan:
    """The prepared chain for one set of overrides: entries by descending priority, then the terminal or the original sdpa."""
    entries: tuple[PlanEntry, ...]
    terminal: PlanEntry | None
    original: AttentionCall
    platform: Platform
    labels: tuple[str, ...]

    def chain(self) -> list[str]:
        names = [entry.backend.name for entry in self.entries]
        names.append(self.terminal.backend.name if self.terminal is not None else 'sdpa')
        return names


current_plan: Plan | None = None


def build_plan(labels, platform: Platform, original: AttentionCall, reg: Registry | None = None) -> Plan:
    reg = reg if reg is not None else default_registry
    entries: list[PlanEntry] = []
    terminal: PlanEntry | None = None
    for backend in reg.ordered(): # ascending priority: the last prepared backend is tried first and owns the torch_info record
        if backend.label not in labels:
            continue
        if not backend.available_on(platform):
            log.warning(f'Torch attention: type="{backend.label}" not available on backend={platform.backend}')
            continue
        try:
            call = backend.prepare(platform, original)
        except Exception as err:
            log.error(f'Torch attention: type="{backend.label}" {err}')
            continue
        if call is None:
            continue
        entry = PlanEntry(backend=backend, call=call)
        if backend.terminal:
            terminal = entry
        else:
            entries.append(entry)
        torch_info.set(attention=backend.name)
    entries.reverse()
    return Plan(entries=tuple(entries), terminal=terminal, original=original, platform=platform, labels=tuple(labels))


def make_router(plan: Plan) -> AttentionCall:
    entries = plan.entries
    terminal = plan.terminal.call if plan.terminal is not None else None
    original = plan.original

    @wraps(original)
    def sdpa_router(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs):
        for entry in entries:
            if entry.backend.constraints.accepts(query, key, value, attn_mask):
                return entry.call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if terminal is not None:
            return terminal(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kwargs)
        if enable_gqa: # older sdpa signatures and platform wrappers reject the keyword, so it only travels when set
            kwargs['enable_gqa'] = enable_gqa
        return original(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

    return sdpa_router


def install_router(labels, platform: Platform, original: AttentionCall, reg: Registry | None = None) -> Plan:
    """Prepare the enabled backends and install the router; an empty plan leaves the original sdpa in place."""
    global current_plan # pylint: disable=global-statement
    plan = build_plan(labels, platform, original, reg)
    torch.nn.functional.scaled_dot_product_attention = make_router(plan) if (plan.entries or plan.terminal is not None) else original
    current_plan = plan
    return plan


def get_plan() -> Plan | None:
    return current_plan
