"""The single scaled_dot_product_attention entry point over the prepared backends."""
from dataclasses import dataclass
from functools import wraps
from typing import Callable
import torch
from installer import torch_info
from modules.logger import log
from modules.attention import context, debug
from modules.attention.registry import AttentionBackend, AttentionCall, Platform, Registry, registry as default_registry


@dataclass(frozen=True)
class PlanEntry:
    backend: AttentionBackend
    call: AttentionCall
    caps: frozenset[str] = frozenset() # the backend's declared caps, narrowed to what prepare verified in the installed implementation


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
    for backend in reg.ordered(): # ascending priority: the last prepared backend is tried first
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
        entry = PlanEntry(backend=backend, call=call, caps=backend.caps & frozenset(getattr(call, 'caps', backend.caps)))
        if backend.terminal:
            terminal = entry
        else:
            entries.append(entry)
    entries.reverse()
    return Plan(entries=tuple(entries), terminal=terminal, original=original, platform=platform, labels=tuple(labels))


def make_router(plan: Plan, observer: Callable | None = None, stage: Callable | None = None) -> AttentionCall:
    entries = plan.entries
    terminal = plan.terminal.call if plan.terminal is not None else None
    terminal_name = plan.terminal.backend.name if plan.terminal is not None else 'sdpa'
    original = plan.original

    @wraps(original)
    def sdpa_router(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs):
        for entry in entries:
            if entry.backend.constraints.accepts(query, key, value, attn_mask):
                if stage is not None and 'block_mask' in entry.caps:
                    selection = stage(query, key, value, attn_mask, is_causal)
                    if selection is not None:
                        if observer is not None:
                            observer(f'{entry.backend.name}+sparse', query, key, attn_mask)
                        return entry.call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, selection=selection)
                if observer is not None:
                    observer(entry.backend.name, query, key, attn_mask)
                return entry.call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if observer is not None: # pylint: disable=duplicate-code
            observer(terminal_name, query, key, attn_mask)
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
    debug.reset()
    observer = debug.observe if debug.enabled else None
    stage = build_sparse_stage(plan)
    torch.nn.functional.scaled_dot_product_attention = make_router(plan, observer, stage) if (plan.entries or plan.terminal is not None) else original
    current_plan = plan
    torch_info.set(attention='>'.join(plan.chain()))
    log.debug(f'Torch attention: chain={">".join(plan.chain())} overrides={list(labels)} backend={platform.backend} sparse={stage is not None}')
    return plan


def build_sparse_stage(plan: Plan):
    """Sparse attention is a stage over the chain rather than a chain member, so it needs a backend in the chain that consumes a block mask."""
    from modules.attention.sparse import stage as sparse_stage
    try:
        options = sparse_stage.read_options()
    except Exception:
        return None
    if not options.enabled:
        return None
    capable = [entry.backend.name for entry in plan.entries if 'block_mask' in entry.caps]
    if not capable:
        names = [backend.label for backend in default_registry.with_cap('block_mask')]
        log.warning(f'Sparse attention: enabled but no active backend consumes a block mask, enable one of {names} in sdp overrides; attention stays dense')
        return None
    built = sparse_stage.make_stage(options)
    if built is not None:
        log.info(f'Sparse attention: backend={capable[0]} budget={options.budget:.0%} gate={options.gate} schedule={options.schedule_steps}x+{options.schedule_bump:.0%}')
    return built


def get_plan() -> Plan | None:
    return current_plan


def reapply_options(reg: Registry | None = None) -> list[str]:
    """Settings whose change rebuilds the chain: the override set, the torch kernel flags, every option a backend captures, and the sparse stage."""
    from modules.attention.sparse import stage as sparse_stage
    reg = reg if reg is not None else default_registry
    return ['sdp_options', 'sdp_overrides', *reg.options(), *sparse_stage.OPTION_NAMES]


def reapply() -> None:
    """Rebuild the chain from the current settings; a resident compiled model is reset so its graphs trace the new router."""
    from modules import devices, shared
    devices.set_sdpa_params()
    compiled = getattr(shared, 'compiled_model_state', None)
    if compiled is not None and getattr(compiled, 'is_compiled', False):
        torch._dynamo.reset() # pylint: disable=protected-access
        log.debug('Torch attention: dynamo reset, compiled model resident')


def report() -> dict:
    """The active chain, sparse stage and generation context, for the api and the debug log."""
    from modules.attention.sparse import stage as sparse_stage
    plan = current_plan
    state = context.current
    options = sparse_stage.read_options()
    layout = state.layout
    return {
        'chain': plan.chain() if plan is not None else ['sdpa'],
        'overrides': list(plan.labels) if plan is not None else [],
        'sparse': {
            'enabled': options.enabled,
            'budget': options.budget,
            'gate': options.gate,
            'capable': [entry.backend.name for entry in plan.entries if 'block_mask' in entry.caps] if plan is not None else [],
            'layout': {'source': layout.source, 'kinds': list(layout.kinds()), 'length': layout.length} if layout is not None else None,
        },
        'backend': plan.platform.backend if plan is not None else None,
        'context': {'active': state.active, 'role': state.role, 'step': state.step, 'steps': state.steps, 'model': state.model_key},
    }
