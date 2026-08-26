"""Per-checkpoint activation calibration for svd hosting on quantized layers.

Plain svd truncation of a hosted delta is optimal in weight space but not in
output space: transformer activations concentrate energy in a few input
channels (per-channel RMS spreads by one to three orders of magnitude), so
the directions that matter most for the output are not the largest in
Frobenius norm. Scaling the delta by per-channel input RMS before the svd
and folding the inverse scale into the down factor spends the same rank
budget on output error instead; measured on real checkpoints this raises
output-delta retention by ~0.05 at rank 256 and ~0.09 at rank 64, most on
MLP down projections whose inputs carry the largest outlier channels.

Statistics come from the model's own forwards: when a sub-8-bit SDNQ model
loads and no calibration is cached for it, streaming sum-of-squares hooks
attach to its quantized linears, accumulate during normal generations,
persist once enough tokens are seen, and go inert. Cached statistics load
at model load and sit on each layer as ``sdnq_calib_rms``; the hosting path
reads them through ``rms_for``. Capture is skipped when the model is
compiled (hooks would break the graph) and everything is gated by the
``lora_sdnq_host_calib`` option.
"""

import os

import torch

from modules import paths, shared, script_callbacks
from modules.logger import log


TOKENS_DONE = 65536
calib_root = os.path.join(paths.models_path, 'calibration')
capture = {'model': None, 'recs': {}, 'handles': [], 'complete': False}


def enabled():
    return bool(getattr(shared.opts, 'lora_sdnq_host_calib', False))


def calib_file(model_name):
    key = model_name.replace('/', '--').replace('\\', '--').replace(':', '-')
    return os.path.join(calib_root, f'{key}.safetensors')


def checkpoint_name(sd_model):
    info = getattr(sd_model, 'sd_checkpoint_info', None)
    return getattr(info, 'name', None)


def eligible_modules(sd_model):
    """Sub-8-bit 2-D SDNQ linears of the model's transformer: the layers hosting applies to."""
    transformer = getattr(sd_model, 'transformer', None)
    if transformer is None:
        return []
    from sdnq.common import dtype_dict
    out = []
    for name, m in transformer.named_modules():
        deq = getattr(m, 'sdnq_dequantizer', None)
        if deq is None or len(deq.original_shape) != 2:
            continue
        if dtype_dict[deq.weights_dtype]['num_bits'] >= 8:
            continue
        out.append((name, m))
    return out


def detach_capture():
    for h in capture['handles']:
        h.remove()
    capture['handles'].clear()
    capture['recs'].clear()
    capture['model'] = None
    capture['complete'] = False


def hook_for(rec, in_features):
    def hook(module, hook_args): # pylint: disable=unused-argument
        if rec['done'] or capture['complete']:
            return
        x = hook_args[0] if hook_args else None
        if not torch.is_tensor(x) or x.shape[-1] != in_features:
            return
        ss = x.detach().reshape(-1, in_features).float().square().sum(dim=0)
        if rec['ss'] is None:
            rec['ss'] = ss
        else:
            if rec['ss'].device != ss.device: # offload moves blocks between devices mid-run
                rec['ss'] = rec['ss'].to(ss.device)
            rec['ss'] += ss
        rec['n'] += x.numel() // in_features
        if rec['n'] >= TOKENS_DONE:
            rec['done'] = True
            if all(r['done'] for r in capture['recs'].values()):
                persist()
    return hook


def persist():
    """Write completed statistics and stamp them onto the layers.

    Runs from the last completing hook, inside a forward; the write is a few
    MB once per checkpoint ever. Handles stay registered but inert until the
    next safe point removes them (hook removal here would mutate the hook
    dict the forward is iterating).
    """
    if capture['complete']:
        return
    capture['complete'] = True
    from safetensors.torch import save_file
    tensors, min_n = {}, None
    for name, rec in capture['recs'].items():
        rms = (rec['ss'] / max(rec['n'], 1)).sqrt().float().cpu().contiguous().clone()
        tensors[name] = rms
        rec['m'].sdnq_calib_rms = rms
        min_n = rec['n'] if min_n is None else min(min_n, rec['n'])
    path = calib_file(capture['model'])
    try:
        os.makedirs(calib_root, exist_ok=True)
        save_file(tensors, path, metadata={'version': '1', 'model': capture['model'], 'tokens': str(min_n)})
        log.info(f'Network calibration: model="{capture["model"]}" layers={len(tensors)} tokens={min_n} saved="{path}"')
    except Exception as e:
        log.warning(f'Network calibration: save failed path="{path}" {e}')


def maybe_detach():
    """Remove inert hooks once capture finished; safe only outside a model forward."""
    if capture['complete'] and capture['handles']:
        detach_capture()


def load_stats(model_name, modules_list):
    from safetensors import safe_open
    path = calib_file(model_name)
    loaded = 0
    with safe_open(path, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        for name, m in modules_list:
            if name in keys:
                m.sdnq_calib_rms = f.get_tensor(name)
                loaded += 1
    log.info(f'Network calibration: model="{model_name}" layers={loaded} loaded="{path}"')


def on_model_loaded(sd_model):
    detach_capture()
    if not enabled():
        return
    name = checkpoint_name(sd_model)
    if name is None:
        return
    modules_list = eligible_modules(sd_model)
    if len(modules_list) == 0:
        return
    if os.path.isfile(calib_file(name)):
        load_stats(name, modules_list)
        return
    if 'Model' in (getattr(shared.opts, 'cuda_compile', None) or []):
        return # hooks inside a compiled module graph-break or misbehave; skip capture entirely
    capture['model'] = name
    for mod_name, m in modules_list:
        rec = {'m': m, 'ss': None, 'n': 0, 'done': False}
        capture['recs'][mod_name] = rec
        capture['handles'].append(m.register_forward_pre_hook(hook_for(rec, int(m.sdnq_dequantizer.original_shape[-1]))))
    log.info(f'Network calibration: model="{name}" layers={len(modules_list)} collecting activation statistics')


def rms_for(layer):
    """Per-channel input RMS for a layer, or None when absent or disabled."""
    maybe_detach()
    if not enabled():
        return None
    return getattr(layer, 'sdnq_calib_rms', None)


script_callbacks.on_model_loaded(on_model_loaded)
