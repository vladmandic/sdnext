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

state: dict = {'entries': {}, 'flips': {}, 'gamma': 1.0, 'gamma_num': 0.0, 'gamma_den': 0.0, 'gamma_e': 1.0, 'gamma_e_num': 0.0, 'gamma_e_den': 0.0, 'total_steps': 0, 'finalized': False, 'reported': None}
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


def select_possible(n_loaded):
    """True when the loaded set could engage a select mode; silent, for the fuse gate."""
    return mode() in SELECT_MODES and n_loaded == 2 and not select_blocked()


def select_engaged():
    """True while selection schedules are live on model layers."""
    return bool(state['entries'])


def active_select(n_loaded):
    m = mode()
    if m not in SELECT_MODES:
        return False
    if n_loaded != 2:
        log.warning(f'Network stack: mode={m} networks={n_loaded} required=2 fallback=sum')
        return False
    if select_blocked():
        log.warning(f'Network stack: mode={m} compile=model fallback=sum')
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


def score_pair(d0, d1, rank0, rank1):
    """Selection scores for a dense delta pair: klora top-K sums (K = rank product) or est energies; plus abs-sums for the global balance."""
    abs_sums = (float(d0.abs().sum()), float(d1.abs().sum()))
    if mode() == 'klora':
        k = max(1, int(rank0) * int(rank1))
        s0 = float(torch.topk(d0.abs().flatten(), min(k, d0.numel()), sorted=False).values.sum())
        s1 = float(torch.topk(d1.abs().flatten(), min(k, d1.numel()), sorted=False).values.sum())
    else:
        s0 = float(d0.float().square().sum())
        s1 = float(d1.float().square().sum())
    return (s0, s1), abs_sums


def register_weight_pair(layer_name, module, per_net):
    """Score and register a weight-kind selection pair; True when the layer is scheduled."""
    from modules.lora import lora_common as l
    if per_net is None or len(per_net) != 2:
        return False
    ranks, names = [], []
    for net_name, d in per_net:
        if d is None:
            return False
        net = next((n for n in l.loaded_networks if n.name == net_name), None)
        net_module = net.modules.get(layer_name, None) if net is not None else None
        if net_module is None:
            return False
        names.append(net_name)
        ranks.append(int(getattr(net_module, 'dim', 0) or 0) or 64)
    scores, abs_sums = score_pair(per_net[0][1].float(), per_net[1][1].float(), ranks[0], ranks[1])
    register(layer_name, module, 'weight', scores, nets=tuple(names), abs_sums=abs_sums)
    return True


def drop(layer_name):
    """Forget a layer's selection entry (its factors were removed or restored)."""
    if layer_name is not None and state['entries'].pop(layer_name, None) is not None:
        state['finalized'] = False


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
    state['gamma_num'] = 0.0
    state['gamma_den'] = 0.0
    state['gamma_e'] = 1.0
    state['gamma_e_num'] = 0.0
    state['gamma_e_den'] = 0.0
    state['total_steps'] = 0
    state['finalized'] = False
    state['reported'] = None


def register(layer_name, module, kind, scores, segments=None, nets=None, abs_sums=None):
    """Record a select-mode layer for schedule finalization.

    kind 'factor': segments = ((s0, s1), (t0, t1), transposed) column ranges on the svd
    channel; both segments' pristine values are stashed for flips. kind 'weight': nets =
    the two network names; the winner delta is recomputed from the layer backup at
    selection time. abs_sums feeds the global magnitude balance (klora gamma).
    """
    entry = {'layer': layer_name, 'module': weakref.ref(module), 'kind': kind, 'segments': segments, 'scores': scores, 'nets': nets, 'stash': None}
    if kind == 'factor':
        (s0, s1), (t0, t1), transposed = segments
        up = module.svd_up.data
        entry['stash'] = (segment_view(up, s0, s1, transposed).clone(), segment_view(up, t0, t1, transposed).clone())
    if abs_sums is not None:
        state['gamma_num'] += abs_sums[0]
        state['gamma_den'] += abs_sums[1]
    if mode() == 'estlora': # est scores ARE the per-layer energies; their totals give the scale-invariant balance
        state['gamma_e_num'] += scores[0]
        state['gamma_e_den'] += scores[1]
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
            # est energies are ||dW||^2, so a magnitude gap enters squared; balance the style side by
            # the total-energy ratio (mirrors klora's gamma) so the louder adapter cannot win on scale alone
            ramp = ramp_alpha() * t + (1.0 - manual_discrepancy())
            if sc < ramp * ss * state['gamma_e']:
                return step
    return total_steps


def materialize_model():
    """Weight-kind selection rewrites module weights outside the activation walk; rebuild balanced-offload modules real first (mirrors network_activate)."""
    from modules import sd_models
    if getattr(shared.opts, 'diffusers_offload_mode', None) == 'balanced' and getattr(shared, 'sd_model', None) is not None:
        sd_models.apply_balanced_offload(shared.sd_model, force=True, silent=True)


def finalize(total_steps):
    """Build the inverted flip map for the pass; select-mode layers start at their step-0 winner."""
    state['total_steps'] = int(total_steps)
    state['gamma'] = (state['gamma_num'] / state['gamma_den']) if state['gamma_den'] > 0 else 1.0
    state['gamma_e'] = (state['gamma_e_num'] / state['gamma_e_den']) if state['gamma_e_den'] > 0 else 1.0
    state['flips'] = {}
    if any(e['kind'] == 'weight' for e in state['entries'].values()):
        materialize_model()
    style_first = 0
    for layer_name, entry in list(state['entries'].items()): # snapshot: apply_selection drops entries whose module died
        flip_at = layer_flip_step(entry['scores'], state['total_steps'])
        initial = 1 if flip_at == 0 else 0
        style_first += initial
        apply_selection(layer_name, entry, initial)
        if 0 < flip_at < state['total_steps']:
            state['flips'].setdefault(flip_at, []).append(layer_name)
    state['finalized'] = True
    if len(state['entries']) > 0: # only a built schedule can carry a flip count, so this is the line that shows selection is live rather than requested
        gamma = state['gamma_e'] if mode() == 'estlora' else state['gamma']
        report = (mode(), len(state['entries']), style_first, sum(len(v) for v in state['flips'].values()), state['total_steps'], round(gamma, 3))
        if report != state['reported']: # rebuilt every pass, so a batch would otherwise repeat one line per image
            state['reported'] = report
            log.info(f'Network load: type=LoRA stack={report[0]} layers={report[1]} style={report[2]} flips={report[3]} steps={report[4]} gamma={report[5]:.3f}')


def reset(total_steps):
    """Per-pass reset from set_callbacks_p: restore initial selections and reschedule for this pass's step count."""
    if mode() not in SELECT_MODES or not state['entries'] or int(total_steps) <= 0:
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
        keep_seg, drop_seg = ((t0, t1), (s0, s1)) if winner == 1 else ((s0, s1), (t0, t1))
        stash = entry['stash'][winner]
        segment_view(up, keep_seg[0], keep_seg[1], transposed).copy_(stash.to(device=up.device, dtype=up.dtype))
        segment_view(up, drop_seg[0], drop_seg[1], transposed).zero_()
    else:
        weight_selection(module, entry, winner)


def weight_selection(module, entry, winner):
    from modules.lora import lora_common as l
    from modules.lora.lora_apply import network_apply_weights
    if getattr(module, 'sdnq_dequantizer', None) is not None:
        warn_once('select-sdnq-weight', 'Network stack: flip=skipped layer=quantized') # quantized backups are packed tensors; only the segment path can flip them
        return
    backup = getattr(module, 'network_weights_backup', None)
    if not isinstance(backup, torch.Tensor): # fuse mode keeps a bool sentinel, not a pristine copy
        warn_once('select-nobackup', 'Network stack: flip=skipped backup=none')
        return
    net = next((n for n in l.loaded_networks if n.name == entry['nets'][winner]), None)
    net_module = net.modules.get(entry['layer'], None) if net is not None else None
    if net_module is None:
        return
    weight = getattr(module, 'weight', None)
    if weight is None or weight.is_meta:
        warn_once('select-offloaded', 'Network stack: flip=skipped weight=offloaded')
        return
    device = weight.device
    updown = net_module.calc_updown(backup.to(device))[0]
    network_apply_weights(module, updown, None, device=device) # recomputes from the pristine backup, requantizing where the layer needs it
