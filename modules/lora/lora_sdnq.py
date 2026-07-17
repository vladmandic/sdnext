"""Exact LoRA application for SDNQ-quantized layers.

Baking a LoRA into a quantized weight requantizes it: dequantize, add the
delta, re-round onto the integer grid. When the per-element delta is smaller
than half a quantization step (a rank-decomposed delta on a uint4 layer sits
at a few percent of a step), rounding erases it; what survives is the two
grid-extrema elements per quantization group (2/group_size of the signal)
plus grid-shift noise of the same norm as the delta. The optimal in-grid
representation provably retains ~0%, so no rewrite of the stored integers
can fix this.

The exact path instead rides the SDNQ svd side-channel: the dequantizer
computes ``W = dq(q) + svd_up @ svd_down`` in the rotated domain at full
precision, in every forward mode. A LoRA delta ``B @ A`` is appended as
extra columns of ``svd_up`` and rows of ``svd_down``; because the Hadamard
rotation is block-diagonal, symmetric and self-inverse, storing ``A·H`` for
the down factor makes the round trip exact: ``(B @ (A·H)) · H = B @ A``.
Quantized weights are never touched, so apply and remove are exact and no
weight backup is needed. The side-channel storage is lossless; realized
fidelity floors at the compute dtype, because the dequantizer materializes
``base + factors`` in the result dtype and a delta below its ULP of the
base rounds exactly as it would on an unquantized model of that dtype.

Only additive low-rank modules ride the channel exactly (plain LoRA: no
DoRA, no CP ``mid``, no LyCORIS dense-bias, no ``diff_b``). On sub-8-bit
formats, sets with non-factorable contributions are hosted instead: the
families' own ``calc_updown`` delta is truncated to its top singular
directions and appended the same way. Truncation keeps the dominant part
of the effect and drops an orthogonal residual, where requantize keeps
only the grid extrema and adds grid-shift noise of the delta's own
magnitude. When activation statistics for the checkpoint exist (see
``lora_calib``), the truncation is channel-weighted to minimize output
error instead of weight error. At 8 bits and above requantize retains
most of the delta, so hosting is skipped there and the requantize path
remains.
"""

import torch

from modules import devices, shared
from modules.lora import lora_calib, lora_factor_cache
from modules.lora import lora_common as l
from modules.logger import log


fallback_layers: list[str] = []
hosted_layers: list[tuple[str, float, bool]] = []


def enabled():
    """True while the exact svd-channel machinery may take quantized layers; the requantize choice routes every layer to the legacy weight-rewrite path."""
    return getattr(shared.opts, 'lora_sdnq_apply', 'exact') != 'requantize'


def signature():
    """Identity suffix for the per-module apply stamp; empty on the default exact mechanism."""
    return '' if enabled() else '|quant=requantize'


def get_module_factors(module, device, dtype, original_shape=None):
    """Return ``(up_eff, down)`` reproducing ``calc_updown`` exactly, or None.

    ``updown = up @ down * calc_scale() * multiplier()`` for a plain linear
    LoRA; the scalars fold into the up factor. ``dyn_dim`` slices ranks the
    same way ``lyco_helpers.rebuild_conventional`` does.
    """
    if module.__class__.__name__ != 'NetworkModuleLora':
        return None
    if module.dora_scale is not None or module.bias is not None or module.ex_bias is not None:
        return None
    if getattr(module, 'mid_model', None) is not None:
        return None
    up = module.up_model.weight
    down = module.down_model.weight
    if up.ndim != 2 or down.ndim != 2:
        return None
    if original_shape is not None and (up.shape[0] != original_shape[0] or down.shape[1] != original_shape[-1]):
        return None # factor_candidate skips shape checks for layers already in factor mode; recheck here so a malformed stack falls back instead of raising in cat
    dyn_dim = module.network.dyn_dim
    if dyn_dim is not None and up.shape[1] != dyn_dim:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim]
    scalar = module.calc_scale() * module.multiplier()
    up_eff = up.to(device=device, dtype=torch.float32) * scalar
    return up_eff.to(dtype=dtype), down.to(device=device, dtype=dtype)


def factor_candidate(self, network_layer_name, wanted_names, use_previous=False):
    """True when this layer should take the exact svd-append path.

    Requires an SDNQ linear layer whose active networks all contribute plain
    factorable LoRA modules for this layer. An empty ``wanted_names`` is a
    removal request and qualifies whenever factors are currently attached.
    """
    if not enabled():
        return False # declined layers with factors still attached are stripped by the activate fallthrough
    if getattr(self, 'sdnq_dequantizer', None) is None or self.__class__.__name__ != 'SDNQLinear':
        return False
    if hasattr(self, 'sdnq_lora_svd_stash'):
        return True
    if wanted_names == ():  # nothing attached, nothing to remove
        return False
    loaded = l.loaded_networks if not use_previous else l.previously_loaded_networks
    seen = False
    for net in loaded:
        module = net.modules.get(network_layer_name, None)
        if module is None:
            continue
        seen = True
        if module.__class__.__name__ != 'NetworkModuleLora':
            return False
        if module.dora_scale is not None or module.bias is not None or module.ex_bias is not None or getattr(module, 'mid_model', None) is not None:
            return False
        if module.up_model.weight.ndim != 2 or module.down_model.weight.ndim != 2:
            return False
        if module.up_model.weight.shape[0] != self.sdnq_dequantizer.original_shape[0] or module.down_model.weight.shape[1] != self.sdnq_dequantizer.original_shape[-1]:
            return False
    return seen


def remove_factors(self):
    """Restore the layer's original svd factors; True when factors were attached."""
    stash = getattr(self, 'sdnq_lora_svd_stash', None)
    if stash is None:
        return False
    svd_up, svd_down = stash
    device = self.scale.device # the stash tuple does not follow module device moves; restore onto wherever the layer lives now
    if svd_up is not None and svd_up.device != device:
        svd_up = torch.nn.Parameter(svd_up.to(device=device), requires_grad=False)
        svd_down = torch.nn.Parameter(svd_down.to(device=device), requires_grad=False)
    self.svd_up = svd_up
    self.svd_down = svd_down
    del self.sdnq_lora_svd_stash
    return True


def apply_factors(self, network_layer_name, wanted_names, use_previous=False):
    """Attach the active networks' LoRA factors to this layer's svd side-channel.

    Replaces any previously attached factors (multiplier changes re-enter
    here with a new ``wanted_names`` signature). Returns True when the layer
    changed. Falls back to the caller's requantize path by returning None
    when factor extraction fails at this stage.
    """
    from sdnq.quant_utils import rotate_hadamard

    changed = remove_factors(self)
    if wanted_names == ():
        return changed

    deq = self.sdnq_dequantizer
    dtype = deq.result_dtype
    loaded = l.loaded_networks if not use_previous else l.previously_loaded_networks
    ups, downs = [], []
    for net in loaded:
        module = net.modules.get(network_layer_name, None)
        if module is None:
            continue
        factors = get_module_factors(module, devices.device, dtype, original_shape=deq.original_shape)
        if factors is None:
            return None
        up_eff, down = factors
        if deq.use_hadamard:
            down = rotate_hadamard(down.to(dtype=torch.float32), group_size=deq.hadamard_group_size).to(dtype=dtype)
        ups.append(up_eff)
        downs.append(down)
    if not ups:
        return changed
    append_factors(self, ups, downs)
    return True


def append_factors(self, ups, downs):
    """Concatenate ``[out, r]`` / ``[r, in]`` factor pairs onto the layer's svd channel and stash the originals."""
    deq = self.sdnq_dequantizer
    device = self.scale.device
    dtype = deq.result_dtype
    orig_up, orig_down = self.svd_up, self.svd_down
    if deq.use_quantized_matmul:
        # matmul layout stores factors transposed: svd_up [r, out], svd_down [in, r]
        parts_up = ([orig_up.to(device=devices.device, dtype=dtype)] if orig_up is not None else []) + [u.t() for u in ups]
        parts_down = ([orig_down.to(device=devices.device, dtype=dtype)] if orig_down is not None else []) + [d.t() for d in downs]
        new_up = torch.cat(parts_up, dim=0).contiguous()
        new_down = torch.cat(parts_down, dim=1).contiguous()
    else:
        parts_up = ([orig_up.to(device=devices.device, dtype=dtype)] if orig_up is not None else []) + ups
        parts_down = ([orig_down.to(device=devices.device, dtype=dtype)] if orig_down is not None else []) + downs
        new_up = torch.cat(parts_up, dim=1).contiguous()
        new_down = torch.cat(parts_down, dim=0).contiguous()
    self.sdnq_lora_svd_stash = (orig_up, orig_down)
    self.svd_up = torch.nn.Parameter(new_up.to(device=device), requires_grad=False)
    self.svd_down = torch.nn.Parameter(new_down.to(device=device), requires_grad=False)


def host_candidate(self, network_layer_name, wanted_names, use_previous=False):
    """True when a non-factorable set on this layer should be hosted as a truncated svd."""
    if not enabled():
        return False
    if int(getattr(shared.opts, 'lora_sdnq_host_rank', 0) or 0) <= 0:
        return False
    if getattr(self, 'sdnq_dequantizer', None) is None or self.__class__.__name__ != 'SDNQLinear':
        return False
    if wanted_names == ():
        return False
    from sdnq.common import dtype_dict
    if dtype_dict[self.sdnq_dequantizer.weights_dtype]['num_bits'] >= 8:
        return False # requantize retains most of the delta at 8 bits and above; truncation would lose more than it saves
    loaded = l.loaded_networks if not use_previous else l.previously_loaded_networks
    return any(net.modules.get(network_layer_name, None) is not None for net in loaded)


def apply_hosted(self, network_layer_name, updown, wanted_names, use_previous=False):
    """Host a set's delta on the svd channel: exact factors for factorable
    members, the top-k singular directions of the remainder for the rest.

    The delta comes from the families' own ``calc_updown``, so every family
    and scaling quirk is included; factorable members are subtracted out and
    appended exactly so they never compete with the hosted remainder for
    rank. When per-checkpoint activation statistics exist (``lora_calib``),
    input channels are weighted by their RMS before truncation so the kept
    directions minimize output error rather than weight error. Computed
    factors are disk-cached per configuration (``lora_factor_cache``) and
    replayed bit-identically on later applies. Returns None when the delta
    cannot ride the channel (wrong shape); the caller falls back to
    requantize.
    """
    from sdnq.quant_utils import rotate_hadamard

    deq = self.sdnq_dequantizer
    changed = remove_factors(self)
    if wanted_names == ():
        return changed
    if updown is None or updown.ndim != 2 or tuple(updown.shape) != tuple(deq.original_shape):
        return None
    dtype = deq.result_dtype
    cached = None
    if not use_previous:
        lora_factor_cache.begin_pass(wanted_names)
        cached = lora_factor_cache.fetch(network_layer_name)
    D = None if cached is not None else updown.detach().to(devices.device, torch.float32)

    ups, downs = [], []
    loaded = l.loaded_networks if not use_previous else l.previously_loaded_networks
    for net in loaded:
        module = net.modules.get(network_layer_name, None)
        if module is None:
            continue
        factors = get_module_factors(module, devices.device, dtype, original_shape=deq.original_shape)
        if factors is None:
            continue
        up_eff, down = factors
        if D is not None:
            D = D.sub_(up_eff.to(torch.float32) @ down.to(torch.float32)) # factorable members ride exactly; host only the remainder
        if deq.use_hadamard:
            down = rotate_hadamard(down.to(dtype=torch.float32), group_size=deq.hadamard_group_size).to(dtype=dtype)
        ups.append(up_eff)
        downs.append(down)

    if cached is not None:
        up_h, down_h, energy, calibrated = cached
        append_factors(self, ups + [up_h.to(device=devices.device, dtype=dtype)], downs + [down_h.to(device=devices.device, dtype=dtype)])
        hosted_layers.append((network_layer_name, energy, calibrated))
        return True

    cap = int(shared.opts.lora_sdnq_host_rank)
    q = min(cap, *D.shape)
    rms = lora_calib.rms_for(self)
    if rms is not None and rms.shape[-1] == D.shape[-1]:
        # scale input channels by their activation RMS so truncation minimizes output error rather than weight error
        rms = rms.to(device=D.device, dtype=torch.float32).clamp(min=1e-8)
        D = D.mul_(rms)
    else:
        rms = None
    # svd_lowrank draws random projections; fork so user generation seeds are untouched and re-applies are deterministic
    with torch.random.fork_rng(devices=[D.device] if D.device.type == 'cuda' else []):
        torch.manual_seed(0)
        U, S, V = torch.svd_lowrank(D, q=q, niter=4)
    energy = float(S.square().sum() / D.square().sum().clamp(min=1e-30)) # captured fraction, in the weighted domain when calibrated
    up_h = (U * S).to(dtype=dtype)
    down_h = V.t()
    if rms is not None:
        down_h = down_h / rms # unscale in the original input basis, before any rotation
    if deq.use_hadamard:
        down_h = rotate_hadamard(down_h, group_size=deq.hadamard_group_size)
    down_h = down_h.to(dtype=dtype)
    up_h, down_h = lora_factor_cache.store(network_layer_name, up_h, down_h, energy, rms is not None)
    append_factors(self, ups + [up_h], downs + [down_h])
    hosted_layers.append((network_layer_name, energy, rms is not None))
    return True


def note_fallback(self, network_layer_name):
    """Record a quantized layer taking the lossy requantize path (summary-logged per pass)."""
    if getattr(self, 'sdnq_dequantizer', None) is not None:
        fallback_layers.append(network_layer_name)


def report_fallbacks():
    hits, misses = lora_factor_cache.flush()
    if hits > 0 or misses > 0:
        log.info(f'Network load: type=LoRA quant=sdnq cache hits={hits} misses={misses}')
    if len(hosted_layers) > 0:
        energies = sorted(e for _name, e, _c in hosted_layers)
        median = energies[len(energies) // 2]
        calibrated = sum(1 for _name, _e, c in hosted_layers if c)
        log.info(f'Network load: type=LoRA quant=sdnq hosted={len(hosted_layers)} rank={int(shared.opts.lora_sdnq_host_rank)}{f" calib={calibrated}" if calibrated else ""} energy={median:.2f} min={energies[0]:.2f} non-factorable networks hosted on the svd side-channel')
        if l.debug:
            log.debug(f'Network load: type=LoRA quant=sdnq hosted={[(n, round(e, 3)) for n, e, _c in hosted_layers[:8]]}{"..." if len(hosted_layers) > 8 else ""}')
    hosted_layers.clear()
    if len(fallback_layers) > 0:
        if enabled():
            log.warning(f'Network load: type=LoRA quant=sdnq layers={len(fallback_layers)} non-factorable networks requantized in place (reduced fidelity on quantized weights)')
        else:
            log.info(f'Network load: type=LoRA quant=sdnq apply=requantize layers={len(fallback_layers)} reason=setting')
        if l.debug:
            log.debug(f'Network load: type=LoRA quant=sdnq requantized={fallback_layers[:8]}{"..." if len(fallback_layers) > 8 else ""}')
    fallback_layers.clear()
