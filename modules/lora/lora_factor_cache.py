"""Disk cache for hosted svd factors.

Hosting a non-factorable adapter set costs one truncated svd per targeted
layer (tens of ms each, seconds per file) every time the set is applied
fresh. The resulting factors are deterministic in the checkpoint, the loaded
set (files, multipliers, dyn_dim), the host rank and the calibration
statistics, so they are cached on disk keyed by exactly that identity and
replayed bit-identically on the next apply of the same configuration.

One safetensors file per configuration under ``models/lora-factor-cache``,
holding every hosted layer's post-rotation factor pair. Files are named by
the model and network set with an identity-hash suffix, and the exact
signature is embedded in the file metadata. The
``lora_sdnq_host_cache`` option is the size budget in GB (0 disables);
least-recently-used entries are evicted past the budget. Any doubt about
identity (unknown checkpoint, unreadable lora file, signature mismatch)
disables caching for the pass rather than risking a stale hit.
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
    parts = {
        'model': model_name,
        'rank': int(getattr(shared.opts, 'lora_sdnq_host_rank', 0) or 0),
        'calib': int(os.path.getmtime(calib_path)) if os.path.isfile(calib_path) else None,
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
    store = {}
    if os.path.isfile(path):
        try:
            from safetensors import safe_open
            with safe_open(path, framework='pt', device='cpu') as f:
                if (f.metadata() or {}).get('sig') == sig:
                    for k in f.keys():
                        store[k] = f.get_tensor(k)
            os.utime(path, None) # freshness for LRU eviction
        except Exception as e:
            log.debug(f'Network cache: read failed path="{path}" {e}')
            store = {}
    state.update(sig=sig, path=path)
    state['store'] = store
    log.debug(f'Network cache: entry="{path}" keys={len(store)}')


def fetch(network_layer_name):
    """Cached (up, down, energy, calibrated) for a layer, or None."""
    if state['sig'] is None:
        return None
    up = state['store'].get(f'{network_layer_name}.up')
    down = state['store'].get(f'{network_layer_name}.down')
    energy = state['store'].get(f'{network_layer_name}.energy')
    calib = state['store'].get(f'{network_layer_name}.calib')
    if up is None or down is None or energy is None or calib is None:
        state['misses'] += 1
        return None
    state['hits'] += 1
    return up, down, float(energy), bool(calib)


def put(network_layer_name, up, down, energy, calibrated):
    if state['sig'] is None:
        return
    state['store'][f'{network_layer_name}.up'] = up.detach().to('cpu').contiguous()
    state['store'][f'{network_layer_name}.down'] = down.detach().to('cpu').contiguous()
    state['store'][f'{network_layer_name}.energy'] = torch.tensor(float(energy))
    state['store'][f'{network_layer_name}.calib'] = torch.tensor(1 if calibrated else 0, dtype=torch.uint8)
    state['dirty'] = True


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
        save_file(state['store'], tmp, metadata={'sig': state['sig']})
        os.replace(tmp, state['path'])
        evict()
    except Exception as e:
        log.warning(f'Network cache: write failed path="{state["path"]}" {e}')
    return hits, misses
