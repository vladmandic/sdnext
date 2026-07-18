"""Stack modes for combining multiple LoRA networks beyond plain summation.

Dense modes (ties, dare_ties, dare_linear, magnitude_prune) combine the
networks' dense deltas elementwise; the result rides the normal apply tail
(side-channel hosting on sub-8-bit SDNQ, requantize at int8 and above,
direct add on unquantized layers). Select modes (klora, estlora) keep both
networks' contributions separate and choose a per-layer winner, shifting
from the first loaded network (subject) toward the second (style) across
the sampling steps. Selection scores depend only on the weights, so the
shift reduces to at most one flip per layer per generation, executed from
the step callback against a schedule finalized at apply time.

TIES arXiv:2306.01708, DARE arXiv:2311.03099, K-LoRA arXiv:2502.18461,
EST-LoRA arXiv:2508.02165 (its measured style-discrepancy estimate is
exposed as an option instead of being derived from probe generations).
"""

import weakref
import hashlib

import torch

from modules import shared
from modules.logger import log


DENSE_MODES = ('ties', 'dare_ties', 'dare_linear', 'magnitude_prune')
SELECT_MODES = ('klora', 'estlora')
KLORA_BETA = 0.5 # the paper's fixed ramp offset; only the slope is user-tunable
ROW_CHUNK = 512 # fp32 interiors run in first-dim slices; also fixes the DARE draw sequence
SAMPLE_CAP = 1 << 22 # strided subsample bound for magnitude quantiles (full-size quantile exceeds torch limits)

state: dict = {'entries': {}, 'flips': {}, 'gamma': 1.0, 'total_steps': 0, 'finalized': False}
warned: set = set()


def mode():
    return getattr(shared.opts, 'lora_stack_mode', 'sum') or 'sum'


def density():
    return float(getattr(shared.opts, 'lora_stack_density', 0.5))


def ramp_alpha():
    return float(getattr(shared.opts, 'lora_stack_alpha', 1.5))


def manual_discrepancy():
    return float(getattr(shared.opts, 'lora_stack_discrepancy', 0.5))


def signature():
    m = mode()
    if m in DENSE_MODES:
        return f'{m}:{density():.2f}'
    if m in SELECT_MODES:
        return f'{m}:{ramp_alpha():.2f}:{manual_discrepancy():.2f}'
    return 'sum'


def warn_once(key, message):
    if key not in warned:
        warned.add(key)
        log.warning(message)


def select_blocked():
    return 'Model' in (getattr(shared.opts, 'cuda_compile', None) or [])


def active_dense(n_contrib):
    return mode() in DENSE_MODES and n_contrib >= 2


def active_select(n_loaded):
    m = mode()
    if m not in SELECT_MODES:
        return False
    if n_loaded != 2:
        warn_once('select-count', f'Network stack: mode={m} networks={n_loaded} requires exactly 2, using sum')
        return False
    if select_blocked():
        warn_once('select-compile', f'Network stack: mode={m} disabled with model compile, using sum')
        return False
    return True


def seed_for(layer_name, net_name):
    payload = f'{layer_name}|{net_name}|{mode()}|{round(density(), 6)}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'little')


def magnitude_threshold(delta, dens):
    flat = delta.abs().flatten()
    step = max(1, flat.numel() // SAMPLE_CAP)
    return torch.quantile(flat[::step].float(), 1.0 - dens)


def dare_generator(device, layer_name, net_name):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_for(layer_name, net_name))
    return gen


def combine(named_deltas, layer_name):
    """Combine per-network dense deltas under the active dense mode; returns a tensor in the first delta's dtype."""
    m = mode()
    dens = density()
    deltas = [d for _, d in named_deltas]
    out_dtype = deltas[0].dtype
    result = torch.zeros_like(deltas[0], dtype=torch.float32)
    thresholds = [magnitude_threshold(d, dens) for d in deltas] if m in ('ties', 'magnitude_prune') else [None] * len(deltas)
    gens = [dare_generator(deltas[0].device, layer_name, name) for name, _ in named_deltas] if m in ('dare_ties', 'dare_linear') else [None] * len(deltas)
    for start in range(0, deltas[0].shape[0], ROW_CHUNK):
        stop = min(start + ROW_CHUNK, deltas[0].shape[0])
        chunks = []
        for i, d in enumerate(deltas):
            c = d[start:stop].to(torch.float32)
            if thresholds[i] is not None:
                c = c * (c.abs() >= thresholds[i])
            if gens[i] is not None:
                keep = torch.rand(c.shape, generator=gens[i], device=c.device, dtype=torch.float32) < dens
                c = c * keep / dens
            chunks.append(c)
        if m in ('ties', 'dare_ties'):
            total = torch.stack(chunks).sum(dim=0)
            elected = torch.sign(total)
            agree = [c * ((torch.sign(c) == elected) & (c != 0)) for c in chunks]
            count = torch.stack([(a != 0).to(torch.float32) for a in agree]).sum(dim=0).clamp(min=1.0)
            result[start:stop] = torch.stack(agree).sum(dim=0) / count
        else: # dare_linear, magnitude_prune: independent per-delta edits, plain sum
            result[start:stop] = torch.stack(chunks).sum(dim=0)
    return result.to(out_dtype)


def score_topk(up, down, k):
    """K-LoRA layer score: sum of the top-K absolute delta entries (one dense materialization)."""
    d = (up.to(torch.float32) @ down.to(torch.float32)).abs().flatten()
    values = torch.topk(d, min(int(k), d.numel()), sorted=False).values
    return float(values.sum()), float(d.sum())


def score_energy(up, down):
    """EST layer score: squared Frobenius norm of up@down via the Gram identity, no materialization."""
    u = up.to(torch.float32)
    dn = down.to(torch.float32)
    return float(((u.t() @ u) * (dn @ dn.t())).sum())


def clear():
    state['entries'] = {}
    state['flips'] = {}
    state['gamma'] = 1.0
    state['total_steps'] = 0
    state['finalized'] = False


def register(layer_name, module, kind, segments, scores, factors=None):
    """Record a select-mode layer: its two segments (or bf16 factor pairs) and static scores.

    kind 'factor': segments = [(start, stop), (start, stop)] column ranges in svd_up/svd_down
    with the transposed-layout flag appended; stashes both segments' values for flips.
    kind 'weight': factors = [(up0, down0), (up1, down1)] kept for recompute-from-backup.
    """
    entry = {'module': weakref.ref(module), 'kind': kind, 'segments': segments, 'scores': scores, 'factors': factors, 'stash': None}
    if kind == 'factor':
        (s0, s1), (t0, t1), transposed = segments
        up = module.svd_up.data
        entry['stash'] = (segment_view(up, s0, s1, transposed).clone(), segment_view(up, t0, t1, transposed).clone())
    state['entries'][layer_name] = entry
    state['finalized'] = False


def segment_view(up, start, stop, transposed):
    return up[start:stop] if transposed else up[:, start:stop]


def layer_flip_step(scores, total_steps):
    """First step index at which the style side wins; total_steps when it never does, 0 when style wins from the start."""
    m = mode()
    sc, ss = scores
    for step in range(total_steps):
        t = step / max(1, total_steps - 1)
        if m == 'klora':
            ramp = state['gamma'] * (ramp_alpha() * t + KLORA_BETA)
            if ss * ramp > sc:
                return step
        else: # estlora: content keeps the layer while sc >= gamma_t * ss
            ramp = ramp_alpha() * t + (1.0 - manual_discrepancy())
            if sc < ramp * ss:
                return step
    return total_steps


def finalize(total_steps):
    """Build the inverted flip map for the pass; select-mode layers start at their step-0 winner."""
    state['total_steps'] = int(total_steps)
    state['flips'] = {}
    for layer_name, entry in state['entries'].items():
        flip_at = layer_flip_step(entry['scores'], state['total_steps'])
        initial = 1 if flip_at == 0 else 0
        apply_selection(layer_name, entry, initial)
        if 0 < flip_at < state['total_steps']:
            state['flips'].setdefault(flip_at, []).append(layer_name)
    state['finalized'] = True


def reset(total_steps):
    """Per-pass reset from set_callbacks_p: restore initial selections and reschedule for this pass's step count."""
    if mode() not in SELECT_MODES or not state['entries']:
        return
    finalize(total_steps)


def on_step(step):
    """Flip the layers whose crossover is this step; non-flip steps are a dict miss."""
    if not state['finalized']:
        return
    for layer_name in state['flips'].get(int(step), ()):
        entry = state['entries'].get(layer_name)
        if entry is not None:
            apply_selection(layer_name, entry, 1)


def apply_selection(layer_name, entry, winner):
    module = entry['module']()
    if module is None:
        state['entries'].pop(layer_name, None)
        return
    if entry['kind'] == 'factor':
        (s0, s1), (t0, t1), transposed = entry['segments']
        up = module.svd_up.data
        keep, drop = ((t0, t1), (s0, s1)) if winner == 1 else ((s0, s1), (t0, t1))
        stash = entry['stash'][winner]
        segment_view(up, keep[0], keep[1], transposed).copy_(stash.to(device=up.device, dtype=up.dtype))
        segment_view(up, drop[0], drop[1], transposed).zero_()
    else:
        weight_selection(module, entry, winner)


def weight_selection(module, entry, winner):
    backup = getattr(module, 'network_weights_backup', None)
    if not isinstance(backup, torch.Tensor): # fuse mode keeps a bool sentinel, not a pristine copy
        warn_once('select-nobackup', 'Network stack: select flip skipped, no weight backup')
        return
    up, down = entry['factors'][winner]
    weight = backup.to(device=module.weight.device, dtype=torch.float32)
    delta = up.to(device=module.weight.device, dtype=torch.float32) @ down.to(device=module.weight.device, dtype=torch.float32)
    module.weight.data.copy_((weight + delta.reshape(weight.shape)).to(module.weight.dtype))
