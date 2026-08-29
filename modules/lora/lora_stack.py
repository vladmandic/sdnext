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

import time
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

state: dict = {'entries': {}, 'flips': {}, 'gamma': 1.0, 'gamma_e': 1.0, 'total_steps': 0, 'finalized': False, 'reported': None}
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
    """Selection scores for a dense delta pair: klora top-K sums (K = rank product) or est energies; plus abs-sums for the global balance.

    Row-chunked fp32 interiors with fp64 accumulators and one device sync for
    all four reductions. Full-tensor staging (fp32 copy, abs copy, top-k
    workspace) peaks hundreds of MB per large layer, which collides with block
    swapping on offloaded denoisers; chunking bounds the transient to the
    chunk. The global top-K over per-chunk top-K candidates selects the same
    element set as a whole-tensor top-K.
    """
    k = max(1, int(rank0) * int(rank1)) if mode() == 'klora' else 0
    accs = []
    for d in (d0, d1):
        score = torch.zeros((), device=d.device, dtype=torch.float64)
        abs_sum = torch.zeros((), device=d.device, dtype=torch.float64)
        cands = []
        for start in range(0, d.shape[0], ROW_CHUNK):
            c = d[start:start + ROW_CHUNK].to(torch.float32).abs() # out-of-place abs: to() may alias a caller-owned fp32 tensor
            abs_sum += c.sum(dtype=torch.float64)
            if k:
                flat = c.flatten()
                cands.append(torch.topk(flat, min(k, flat.numel()), sorted=False).values)
            else:
                score += c.square().sum(dtype=torch.float64)
        if k and cands:
            allc = torch.cat(cands) if len(cands) > 1 else cands[0]
            score = torch.topk(allc, min(k, allc.numel()), sorted=False).values.sum(dtype=torch.float64)
        accs.append((score, abs_sum))
    packed = torch.stack([accs[0][0], accs[0][1], accs[1][0], accs[1][1]]).cpu()
    return (float(packed[0]), float(packed[2])), (float(packed[1]), float(packed[3]))


def register_weight_pair(layer_name, module, per_net, wanted_names=None):
    """Score and register a weight-kind selection pair; True when the layer is scheduled.

    The scores persist in the factor cache when a pass identity is given, so a
    later apply of the same configuration registers from the record alone.
    """
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
    scores, abs_sums = score_pair(per_net[0][1], per_net[1][1], ranks[0], ranks[1])
    if wanted_names is not None:
        from modules.lora import lora_factor_cache
        lora_factor_cache.begin_pass(wanted_names)
        lora_factor_cache.store_scores(layer_name, scores, abs_sums)
    register(layer_name, module, 'weight', scores, nets=tuple(names), abs_sums=abs_sums)
    return True


def register_weight_pair_cached(layer_name, module, wanted_names):
    """Register a weight-kind pair from its cached score record; True when served.

    The record was stored under the same configuration signature, which pins
    the loaded pair, multipliers and stack settings, so both networks are known
    to target the layer and the prompt-order roles are unchanged.
    """
    from modules.lora import lora_common as l
    from modules.lora import lora_factor_cache
    if len(l.loaded_networks) != 2:
        return False
    lora_factor_cache.begin_pass(wanted_names)
    rec = lora_factor_cache.lookup_scores(layer_name)
    if rec is None:
        return False
    scores, abs_sums = rec
    register(layer_name, module, 'weight', scores, nets=tuple(n.name for n in l.loaded_networks), abs_sums=abs_sums)
    return True


def drop(layer_name):
    """Forget a layer's selection entry (its factors were removed or restored)."""
    if layer_name is not None and state['entries'].pop(layer_name, None) is not None:
        state['finalized'] = False


def score_energy(up, down):
    """EST layer score: squared Frobenius norm of up@down via the Gram identity, no materialization."""
    u = up.to(torch.float32)
    dn = down.to(torch.float32)
    return float(((u.t() @ u) * (dn @ dn.t())).sum())


def clear():
    state['entries'] = {}
    state['flips'] = {}
    state['gamma'] = 1.0
    state['gamma_e'] = 1.0
    state['total_steps'] = 0
    state['finalized'] = False
    state['reported'] = None


def register(layer_name, module, kind, scores, segments: tuple[tuple[int, int], tuple[int, int], bool] | None = None, nets=None, abs_sums=None):
    """Record a select-mode layer for schedule finalization.

    kind 'factor': segments = ((s0, s1), (t0, t1), transposed) column ranges on the svd
    channel; both segments' pristine values are stashed for flips. kind 'weight': nets =
    the two network names; the winner delta is recomputed from the layer backup at
    selection time. abs_sums feeds the global magnitude balance (klora gamma).
    """
    entry = {'layer': layer_name, 'module': weakref.ref(module), 'kind': kind, 'segments': segments, 'scores': scores, 'nets': nets, 'abs_sums': abs_sums, 'stash': None}
    if kind == 'factor':
        if segments is None:
            raise ValueError("segments is required when kind='factor'")
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
    # both balances derive from the live entries every time, so drops and re-registrations stay consistent by construction
    num = sum(e['abs_sums'][0] for e in state['entries'].values() if e['abs_sums'] is not None)
    den = sum(e['abs_sums'][1] for e in state['entries'].values() if e['abs_sums'] is not None)
    state['gamma'] = (num / den) if den > 0 else 1.0
    e_num = sum(e['scores'][0] for e in state['entries'].values()) # est scores ARE the per-layer energies; their totals give the scale-invariant balance
    e_den = sum(e['scores'][1] for e in state['entries'].values())
    state['gamma_e'] = (e_num / e_den) if e_den > 0 else 1.0
    state['flips'] = {}
    stats = {'weight_n': 0, 'factor_n': 0, 'materialize': 0.0, 'select': 0.0, 'w_move': 0.0, 'w_calc': 0.0, 'w_apply': 0.0}
    state['stats'] = stats
    stats['weight_n'] = sum(1 for e in state['entries'].values() if e['kind'] == 'weight')
    stats['factor_n'] = len(state['entries']) - stats['weight_n']
    if stats['weight_n'] > 0:
        t0 = time.time()
        materialize_model()
        stats['materialize'] = time.time() - t0
    style_first = 0
    t0 = time.time()
    for layer_name, entry in list(state['entries'].items()): # snapshot: apply_selection drops entries whose module died
        flip_at = layer_flip_step(entry['scores'], state['total_steps'])
        initial = 1 if flip_at == 0 else 0
        style_first += initial
        apply_selection(layer_name, entry, initial)
        if 0 < flip_at < state['total_steps']:
            state['flips'].setdefault(flip_at - 1, []).append(layer_name) # step callbacks fire after the denoise, so the flip runs one step early to be live during the crossover step's forward
    stats['select'] = time.time() - t0
    state['finalized'] = True
    if len(state['entries']) > 0: # only a built schedule can carry a flip count, so this is the line that shows selection is live rather than requested
        gamma = state['gamma_e'] if mode() == 'estlora' else state['gamma']
        report = (mode(), len(state['entries']), style_first, sum(len(v) for v in state['flips'].values()), state['total_steps'], round(gamma, 3))
        if report != state['reported']: # rebuilt every pass, so a batch would otherwise repeat one line per image
            state['reported'] = report
            log.info(f'Network load: type=LoRA stack={report[0]} layers={report[1]} style={report[2]} flips={report[3]} steps={report[4]} gamma={report[5]:.3f}')
        # logged every pass: the reset runs outside the activate walk, so its cost is invisible to the load timers
        log.debug(f'Network select: type=LoRA reset weight={stats["weight_n"]} factor={stats["factor_n"]} time={{materialize: {stats["materialize"]:.2f}, select: {stats["select"]:.2f}, move: {stats["w_move"]:.2f}, calc: {stats["w_calc"]:.2f}, apply: {stats["w_apply"]:.2f}}}')


def reset(total_steps):
    """Per-pass reset from set_callbacks_p: restore initial selections and reschedule for this pass's step count."""
    if mode() not in SELECT_MODES or not state['entries'] or int(total_steps) <= 0:
        return
    finalize(total_steps)


def on_step(step):
    """Flip the layers whose crossover is this step; non-flip steps are a dict miss."""
    if not state['finalized']:
        return
    layers = state['flips'].get(int(step), ())
    if not layers:
        return
    t0 = time.time()
    for layer_name in layers:
        entry = state['entries'].get(layer_name)
        if entry is not None:
            apply_selection(layer_name, entry, 1)
    log.debug(f'Network select: type=LoRA flip step={int(step)} layers={len(layers)} time={time.time() - t0:.2f}')


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
    from modules import devices
    stats = state.get('stats') or {}
    device = weight.device
    t0 = time.time()
    base = backup.to(devices.device) # a swapped-out layer keeps its weight on cpu; the delta matmul belongs on the accelerator regardless
    t1 = time.time()
    updown = net_module.calc_updown(base)[0].to(device)
    t2 = time.time()
    network_apply_weights(module, updown, None, device=device) # recomputes from the pristine backup, requantizing where the layer needs it
    stats['w_move'] = stats.get('w_move', 0.0) + (t1 - t0)
    stats['w_calc'] = stats.get('w_calc', 0.0) + (t2 - t1)
    stats['w_apply'] = stats.get('w_apply', 0.0) + (time.time() - t2)
