"""Disk cache for hosted svd factors.

Hosting a non-factorable adapter set costs one truncated svd per targeted
layer (tens of ms each, seconds per file) every time the set is applied
fresh. The resulting factors are deterministic in the checkpoint, the loaded
set (files, multipliers, dyn_dim), the host rank and the calibration
statistics, so they are cached on disk keyed by exactly that identity and
replayed bit-identically on the next apply of the same configuration.

One safetensors file per configuration under ``models/lora-factor-cache``,
holding every hosted layer's post-rotation factor pair as rowwise int8
with fp32 scales (measured fidelity-free in output space, half the bytes
of bf16). Files are named by the model and network set with an
identity-hash suffix, and the exact signature is embedded in the file
metadata. Factors are quantized before first use: ``store`` returns the
dequantized round-trip for the caller to apply, so a fresh compute and a
later cache hit attach bit-identical tensors. The ``lora_sdnq_host_cache``
option is the size budget in GB (0 disables); least-recently-used entries
are evicted past the budget. Any doubt about identity (unknown checkpoint,
unreadable lora file, signature mismatch) disables caching for the pass
rather than risking a stale hit.
"""

import os
import json
import hashlib

import torch

from modules import paths, shared
from modules.lora import lora_common as l
from modules.logger import log


cache_root = os.path.join(paths.models_path, 'lora-factor-cache')
state = {'wn': None, 'sig': None, 'path': None, 'store': {}, 'dirty': False, 'hits': 0, 'misses': 0}
FMT = '5' # bump on entry-layout changes so older files recompute instead of replaying short


def budget_gb():
    try:
        return float(getattr(shared.opts, 'lora_sdnq_host_cache', 0) or 0)
    except Exception:
        return 0.0


def signature(wanted_names):
    """Content identity of a hosted-apply configuration, or None when caching is unsafe."""
    from modules.lora import lora_calib
    model_name = lora_calib.checkpoint_name(getattr(shared, 'sd_model', None))
    if model_name is None:
        return None
    calib_path = lora_calib.calib_file(model_name)
    from modules.lora import lora_stack
    parts = {
        'model': model_name,
        'rank': int(getattr(shared.opts, 'lora_sdnq_host_rank', 0) or 0),
        'calib': int(os.path.getmtime(calib_path)) if lora_calib.enabled() and os.path.isfile(calib_path) else None, # the toggle is part of the identity: factors computed under the other setting must not replay
        'stack': lora_stack.signature(),
        'nets': [],
    }
    for name, te, unet, dyn in wanted_names:
        net = next((n for n in l.loaded_networks if n.name == name), None)
        filename = getattr(getattr(net, 'network_on_disk', None), 'filename', None)
        try:
            st = os.stat(filename)
        except Exception:
            return None
        parts['nets'].append([name, repr(te), repr(unet), repr(dyn), filename, int(st.st_mtime), st.st_size])
    return parts


def label(parts):
    """Filename prefix from the model and net names, so the cache folder reads without tooling."""
    names = [parts['model'].replace('\\', '/').split('/')[-1]] + [n[0] for n in parts['nets']]
    text = '-'.join(names)
    text = ''.join(c if c.isalnum() or c in '._-' else '-' for c in text)
    return text[:96]


def begin_pass(wanted_names):
    """Bind the pass to its cache entry; identity-memoized on the wanted_names tuple."""
    if wanted_names is state['wn']:
        return
    state['wn'] = wanted_names
    state.update(sig=None, path=None, dirty=False)
    state['store'] = {}
    if budget_gb() <= 0 or wanted_names == ():
        return
    parts = signature(wanted_names)
    if parts is None:
        return
    sig = json.dumps(parts, sort_keys=True)
    key = hashlib.sha256(sig.encode()).hexdigest()[:24]
    path = os.path.join(cache_root, f'{label(parts)}-{key}.safetensors')
    entries = {}
    if os.path.isfile(path):
        try:
            from safetensors import safe_open
            with safe_open(path, framework='pt', device='cpu') as f:
                meta = f.metadata() or {}
                if meta.get('sig') == sig and meta.get('fmt') == FMT:
                    for k in f.keys():
                        entries[k] = f.get_tensor(k)
            os.utime(path, None) # freshness for LRU eviction
        except Exception as e:
            log.debug(f'Network cache: read failed path="{path}" {e}')
            entries = {}
    state.update(sig=sig, path=path)
    state['store'] = entries
    log.debug(f'Network cache: entry="{path}" keys={len(entries)}')


def quantize_rowwise(t):
    t32 = t.detach().to(torch.float32)
    scale = t32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q = (t32 / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_rowwise(q, scale):
    # int8 * fp32 with a single fp32 rounding: identical on any device, so hit and miss replay the same values
    return q.to(torch.float32) * scale


def lookup(network_layer_name):
    """Cached (up, down, energy, calibrated, rms) for a layer, or None; factors return as fp32.

    Pure lookup with no hit/miss accounting: the fast-path probe uses it so a
    layer is only counted once, by whichever caller consumes the answer.
    """
    if state['sig'] is None:
        return None
    st = state['store']
    up_q, up_s = st.get(f'{network_layer_name}.up_q'), st.get(f'{network_layer_name}.up_s')
    down_q, down_s = st.get(f'{network_layer_name}.down_q'), st.get(f'{network_layer_name}.down_s')
    energy = st.get(f'{network_layer_name}.energy')
    calib = st.get(f'{network_layer_name}.calib')
    rms = st.get(f'{network_layer_name}.rms')
    if up_q is None or up_s is None or down_q is None or down_s is None or energy is None or calib is None or rms is None:
        return None
    return dequantize_rowwise(up_q, up_s), dequantize_rowwise(down_q, down_s), float(energy), bool(calib), float(rms)


def note_hit():
    state['hits'] += 1


def fetch(network_layer_name):
    """``lookup`` with accounting: a usable entry counts a hit, anything else a miss."""
    entry = lookup(network_layer_name)
    if entry is None:
        if state['sig'] is not None:
            state['misses'] += 1
        return None
    state['hits'] += 1
    return entry


def lookup_scores(network_layer_name):
    """Cached select scores for a layer as ((s0, s1), (a0, a1)), or None.

    Score records ride the same signature-keyed entry as factors, and the
    signature already pins everything the scores depend on (pair, multipliers,
    stack mode and params). No hit/miss accounting: a record saves scoring and
    delta assembly, not a sketch.
    """
    if state['sig'] is None:
        return None
    t = state['store'].get(f'{network_layer_name}.sel')
    if t is None:
        return None
    return (float(t[0]), float(t[1])), (float(t[2]), float(t[3]))


def store_scores(network_layer_name, scores, abs_sums):
    """Persist a select-mode score record; additive to the entry, older files upgrade on their next pass."""
    if state['sig'] is None:
        return
    state['store'][f'{network_layer_name}.sel'] = torch.tensor([scores[0], scores[1], abs_sums[0], abs_sums[1]], dtype=torch.float64)
    state['dirty'] = True


def store(network_layer_name, up, down, energy, calibrated, rms):
    """Quantize-before-use: returns the dequantized round-trip the caller must apply.

    The factors quantize to rowwise int8 whether or not a cache entry can be
    written, so the factors applied now, the factors a later hit replays, and a
    cache-off apply are the same tensors (the round-trip also zeroes null-tail
    columns the attach-side trim relies on). ``rms`` is the assembled delta's
    rms, kept so replays can evaluate the requantize routing rule without
    assembling the delta.
    """
    up_q, up_s = quantize_rowwise(up)
    down_q, down_s = quantize_rowwise(down)
    if state['sig'] is not None:
        st = state['store']
        st[f'{network_layer_name}.up_q'] = up_q.to('cpu').contiguous()
        st[f'{network_layer_name}.up_s'] = up_s.to('cpu').contiguous()
        st[f'{network_layer_name}.down_q'] = down_q.to('cpu').contiguous()
        st[f'{network_layer_name}.down_s'] = down_s.to('cpu').contiguous()
        st[f'{network_layer_name}.energy'] = torch.tensor(float(energy))
        st[f'{network_layer_name}.calib'] = torch.tensor(1 if calibrated else 0, dtype=torch.uint8)
        st[f'{network_layer_name}.rms'] = torch.tensor(float(rms))
        state['dirty'] = True
    return dequantize_rowwise(up_q, up_s).to(up.dtype), dequantize_rowwise(down_q, down_s).to(down.dtype)


def evict():
    budget = budget_gb() * 2**30
    try:
        files = [os.path.join(cache_root, f) for f in os.listdir(cache_root) if f.endswith('.safetensors')]
        sizes = {p: os.path.getsize(p) for p in files}
    except Exception:
        return
    total = sum(sizes.values())
    for p in sorted(files, key=os.path.getmtime):
        if total <= budget:
            break
        if p == state['path']:
            continue # never evict the entry of the live pass
        try:
            os.remove(p)
            total -= sizes[p]
        except Exception:
            pass


def flush():
    """Persist a dirty pass store; returns (hits, misses) since the last flush."""
    hits, misses = state['hits'], state['misses']
    state['hits'] = state['misses'] = 0
    if not state['dirty'] or state['path'] is None:
        return hits, misses
    state['dirty'] = False
    try:
        from safetensors.torch import save_file
        os.makedirs(cache_root, exist_ok=True)
        tmp = state['path'] + '.tmp'
        save_file(state['store'], tmp, metadata={'sig': state['sig'], 'fmt': FMT})
        os.replace(tmp, state['path'])
        evict()
    except Exception as e:
        log.warning(f'Network cache: write failed path="{state["path"]}" {e}')
    return hits, misses
