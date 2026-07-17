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
weight backup is needed.

Only additive low-rank modules qualify (plain LoRA: no DoRA, no CP ``mid``,
no LyCORIS dense-bias, no ``diff_b``). Layers with any non-factorable
contribution fall back to the dequantize-add-requantize path.
"""

import torch

from modules import devices
from modules.lora import lora_common as l
from modules.logger import log


fallback_layers: list[str] = []


def get_module_factors(module, device, dtype):
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
    device = self.scale.device
    dtype = deq.result_dtype
    loaded = l.loaded_networks if not use_previous else l.previously_loaded_networks
    ups, downs = [], []
    for net in loaded:
        module = net.modules.get(network_layer_name, None)
        if module is None:
            continue
        factors = get_module_factors(module, devices.device, dtype)
        if factors is None:
            return None
        up_eff, down = factors
        if deq.use_hadamard:
            down = rotate_hadamard(down.to(dtype=torch.float32), group_size=deq.hadamard_group_size).to(dtype=dtype)
        ups.append(up_eff)
        downs.append(down)
    if not ups:
        return changed

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
    return True


def note_fallback(self, network_layer_name):
    """Record a quantized layer taking the lossy requantize path (summary-logged per pass)."""
    if getattr(self, 'sdnq_dequantizer', None) is not None:
        fallback_layers.append(network_layer_name)


def report_fallbacks():
    if len(fallback_layers) > 0:
        log.warning(f'Network load: type=LoRA quant=sdnq layers={len(fallback_layers)} non-factorable networks requantized in place (reduced fidelity on quantized weights)')
        if l.debug:
            log.debug(f'Network load: type=LoRA quant=sdnq requantized={fallback_layers[:8]}{"..." if len(fallback_layers) > 8 else ""}')
    fallback_layers.clear()
