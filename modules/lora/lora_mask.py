"""Spatial masks for loaded networks: confine an adapter to a painted region of the image.

The native loader merges every delta into the weights, so a region cannot be gated
where the delta is applied. A forward hook on each targeted Linear cancels the merged
delta per token instead: with D = up @ down already inside W, adding (m - 1) * (x @ D^T)
turns (W + D) x into W x + m * (D x). Nothing is un-merged and the loader is untouched.

Factors are what the layer actually carries: exact members and single-owner hosted
remainders on the svd channel of a quantized layer (recorded by lora_sdnq per rank
range), the network module's own pair elsewhere. A truncated delta is therefore
cancelled as attached, never as the file describes it.

Token geometry is read from the transformer call itself. Krea 2 receives per-token
(t, h, w) position ids, so the mask is sampled per token and no packing order is
assumed. Only image tokens are scaled; text tokens, padding and 2-d modulation
outputs pass through. A network that cannot be cancelled exactly applies unmasked
with a warning.
"""

import os
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from modules import shared, devices
from modules.lora import lora_common as l
from modules.logger import log


CHANNELS = {'r': 0, 'red': 0, 'g': 1, 'green': 1, 'b': 2, 'blue': 2, 'l': 3, 'lum': 3, 'luminance': 3}
PLANE_NAMES = ('r', 'g', 'b', 'l')
INFOTEXT_KEY = 'LoRA masks'
FALLBACK_KEY = 'LoRA mask fallback'
DENOISER_ROLES = ('transformer', 'unet')
MASK_TEXT = os.environ.get('SD_LORA_MASK_TEXT', None) is not None # cancel the adapter on the text stream as well, so a black sheet removes it entirely


def parse_channel(spec):
    """Plane index for a tag value: r, g, b pick a colour plane of the sheet, l its luminance; None when unknown."""
    if spec is None:
        return None
    return CHANNELS.get(str(spec).strip().lower(), None)


def resolve_sheet(image):
    """Mask sheet as a `(4, H, W)` float tensor in [0, 1]: the three colour planes plus luminance."""
    if image is None:
        return None
    if isinstance(image, dict): # gradio sketch payloads carry the composite under image
        image = image.get('image', None) or image.get('composite', None)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        return None
    rgb = torch.from_numpy(np.array(image.convert('RGB'))).float().div_(255.0).permute(2, 0, 1)
    lum = torch.from_numpy(np.array(image.convert('L'))).float().div_(255.0)[None]
    return torch.cat([rgb, lum], dim=0)


class Geometry:
    """Token layout of one transformer call: text length, image length, each image token's (h, w) and the grid."""

    def __init__(self, txtlen: int, imglen: int, hw: torch.Tensor, grid: tuple[int, int]):
        self.txtlen = txtlen
        self.imglen = imglen
        self.hw = hw
        self.grid = grid

    @property
    def key(self):
        return (self.txtlen, self.imglen, self.grid)


def krea2_geometry(module, args, kwargs):
    """Krea 2 packs [text | image]; image rows of position_ids carry (t, h, w)."""
    bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
    hidden, text, pos = bound.get('hidden_states'), bound.get('encoder_hidden_states'), bound.get('position_ids')
    if not (torch.is_tensor(hidden) and torch.is_tensor(text) and torch.is_tensor(pos)):
        return None
    txtlen, imglen = int(text.shape[1]), int(hidden.shape[1])
    hw = pos[0, txtlen:txtlen + imglen, 1:3].round().long()
    if hw.shape[0] != imglen:
        return None
    grid = (int(hw[:, 0].max().item()) + 1, int(hw[:, 1].max().item()) + 1)
    return Geometry(txtlen, imglen, hw, grid)


GEOMETRY = { # denoiser class -> how its call exposes the token grid
    'Krea2Transformer2DModel': krea2_geometry,
}


class StaticEntry:
    """Factor pair held by the hook: the network module's own up and down with the scale folded in."""

    def __init__(self, up, down, plane):
        self.up = up
        self.down = down
        self.plane = plane

    def factors(self, device, dtype):
        return self.up.to(device=device, dtype=dtype), self.down.to(device=device, dtype=dtype)


class SegmentEntry:
    """Rank range on a quantized layer's svd channel; read live so device moves and re-attaches are followed."""

    def __init__(self, module, start, stop, transposed, plane, down=None):
        self.module = module
        self.start = start
        self.stop = stop
        self.transposed = transposed
        self.plane = plane
        self.down = down # un-rotated copy when the channel is hadamard-rotated, else read live

    def factors(self, device, dtype):
        svd_up, svd_down = self.module.svd_up, self.module.svd_down
        if self.transposed: # matmul layout: svd_up [r, out], svd_down [in, r]
            up = svd_up[self.start:self.stop, :].t()
            down = svd_down[:, self.start:self.stop].t() if self.down is None else self.down
        else:
            up = svd_up[:, self.start:self.stop]
            down = svd_down[self.start:self.stop, :] if self.down is None else self.down
        return up.to(device=device, dtype=dtype), down.to(device=device, dtype=dtype)


class MaskRun:
    """Hooks, captured geometry and cached token masks for one generation."""

    def __init__(self):
        self.handles = []
        self.sheet = None
        self.geometry = None
        self.cache = {}
        self.names = {} # net name -> plane name
        self.layers = {} # net name -> hooked layer count
        self.fallback = {} # net name -> reason
        self.warned = set()

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.sheet = None
        self.geometry = None
        self.cache.clear()
        self.names.clear()
        self.layers.clear()
        self.fallback.clear()
        self.warned.clear()

    @property
    def active(self):
        return len(self.handles) > 0

    def capture(self, module, args, kwargs, provider):
        geometry = provider(module, args, kwargs)
        if geometry is None:
            return
        if self.geometry is None or geometry.key != self.geometry.key:
            self.cache.clear() # a new grid, as in a hires pass, invalidates every token mask
        self.geometry = geometry

    def token_mask(self, plane: int, seq_len: int, device, dtype):
        """`(1, L, 1)` per-token scale for a sequence of length L, or None when it carries no image tokens."""
        geometry = self.geometry
        if geometry is None or self.sheet is None:
            return None
        key = (plane, seq_len, device, dtype)
        cached = self.cache.get(key, None)
        if cached is not None:
            return cached
        if geometry.txtlen == geometry.imglen and seq_len == geometry.imglen:
            if 'ambiguous' not in self.warned:
                self.warned.add('ambiguous')
                log.warning(f'Network mask: text and image sequences are both {seq_len} tokens, single-stream projections left unmasked')
            return None
        if seq_len == geometry.imglen:
            offset = 0
        elif seq_len >= geometry.txtlen + geometry.imglen:
            offset = geometry.txtlen
        else:
            return None # text-only projection
        resized = F.interpolate(self.sheet[plane][None, None], size=geometry.grid, mode='bilinear', align_corners=False)[0, 0]
        hw = geometry.hw.to(device=resized.device)
        per_token = resized[hw[:, 0], hw[:, 1]]
        mask = torch.full((seq_len,), 0.0 if MASK_TEXT else 1.0, dtype=torch.float32) # text and padding keep the adapter unless the text switch is on
        mask[offset:offset + geometry.imglen] = per_token
        mask = mask.view(1, seq_len, 1).to(device=device, dtype=dtype)
        self.cache[key] = mask
        return mask


run = MaskRun()


def make_hook(entries):
    def hook(module, args, output): # pylint: disable=unused-argument
        if not torch.is_tensor(output) or len(args) == 0 or not torch.is_tensor(args[0]):
            return output
        x = args[0]
        result = output
        for entry in entries:
            mask = run.token_mask(entry.plane, output.shape[1], output.device, output.dtype) if output.ndim == 3 else None
            if mask is None:
                if not MASK_TEXT:
                    continue
                mask = 0.0 # the text switch cancels the adapter wherever no image token sits: text streams, folded sequences, modulation
            up, down = entry.factors(x.device, x.dtype)
            term = torch.matmul(torch.matmul(x, down.t()), up.t())
            result = result + (mask - 1) * term
        return result
    return hook


def blocked_reason(load_method: str):
    """Why masks cannot apply at all this pass, or None. Anything listed here leaves no per-net delta to cancel."""
    from modules.lora import lora_stack, lora_sdnq, lora_overrides
    if load_method != 'native':
        return f'method={load_method}'
    if lora_stack.mode() != 'sum':
        return f'stack={lora_stack.mode()}'
    if lora_stack.select_blocked():
        return 'compile=model'
    sd_model = getattr(shared.sd_model, 'pipe', shared.sd_model)
    if not lora_sdnq.enabled() and any(lora_overrides.is_quantized(getattr(sd_model, name, None)) for name in DENOISER_ROLES):
        return 'sdnq apply=requantize'
    return None


def layer_entry(module, net, net_module, plane):
    """Entry cancelling this net's delta on this layer, or (None, reason) when the delta is not cancellable; (None, None) skips the layer."""
    from modules.lora import lora_sdnq
    deq = getattr(module, 'sdnq_dequantizer', None)
    if deq is not None:
        segment = getattr(module, 'sdnq_lora_segments', {}).get(net.name, None)
        if segment is None:
            return None, 'requantized' # the delta went to the grid, or into a host shared with another net
        start, stop, kind = segment
        transposed = bool(getattr(module, 'sdnq_lora_segments_transposed', False))
        if kind == 'exact':
            factors = lora_sdnq.get_module_factors(net_module, devices.device, deq.result_dtype, original_shape=deq.original_shape)
            if factors is None:
                return None, 'factors'
            return StaticEntry(factors[0], factors[1], plane), None
        down = None
        if deq.use_hadamard: # the channel holds down rotated; rotating again restores it since the matrix is its own inverse
            from sdnq.quant_utils import rotate_hadamard
            entry = SegmentEntry(module, start, stop, transposed, plane)
            _up, attached = entry.factors(module.svd_down.device, torch.float32)
            down = rotate_hadamard(attached.contiguous(), group_size=deq.hadamard_group_size).to(dtype=deq.result_dtype)
        return SegmentEntry(module, start, stop, transposed, plane, down=down), None
    weight = getattr(module, 'weight', None)
    if weight is None or weight.ndim != 2:
        return None, None # conv and norm layers are not sequence projections
    factors = lora_sdnq.get_module_factors(net_module, devices.device, weight.dtype)
    if factors is None:
        return None, f'family={net_module.__class__.__name__}'
    return StaticEntry(factors[0], factors[1], plane), None


def note_fallback(p, name, reason):
    run.fallback[name] = reason
    log.warning(f'Network mask: name="{name}" fallback="{reason}" apply=unmasked')
    if p is not None:
        p.extra_generation_params[FALLBACK_KEY] = ', '.join(f'{k}={v}' for k, v in run.fallback.items())


def install(p, masked: dict, load_method: str):
    """Hook every layer the masked networks touched. ``masked`` maps net name to plane index."""
    remove()
    if not masked:
        return
    reason = blocked_reason(load_method)
    sheet = resolve_sheet(getattr(p, 'lora_mask', None)) if reason is None else None
    if reason is None and sheet is None:
        reason = 'no mask sheet'
    if reason is not None:
        for name in masked:
            note_fallback(p, name, reason)
        return
    sd_model = getattr(shared.sd_model, 'pipe', shared.sd_model)
    components = [(name, getattr(sd_model, name)) for name in DENOISER_ROLES if isinstance(getattr(sd_model, name, None), torch.nn.Module)]
    if len(components) == 0:
        for name in masked:
            note_fallback(p, name, 'no denoiser')
        return
    run.sheet = sheet
    nets = {net.name: net for net in l.loaded_networks}
    for _component_name, component in components:
        provider = GEOMETRY.get(component.__class__.__name__, None)
        if provider is None:
            for name in masked:
                note_fallback(p, name, f'geometry unknown cls={component.__class__.__name__}')
            run.clear()
            return
        per_layer = {}
        for name, plane in masked.items():
            net = nets.get(name, None)
            if net is None:
                note_fallback(p, name, 'not loaded')
                continue
            entries, blocked = [], None
            for _module_name, module in component.named_modules():
                layer = getattr(module, 'network_layer_name', None)
                net_module = net.modules.get(layer, None) if layer is not None else None
                if net_module is None:
                    continue
                entry, why = layer_entry(module, net, net_module, plane)
                if why is not None:
                    blocked = f'{why} layer={layer}'
                    break
                if entry is not None:
                    entries.append((module, entry))
            if blocked is not None:
                note_fallback(p, name, blocked)
                continue
            for module, entry in entries:
                per_layer.setdefault(module, []).append(entry)
            run.names[name] = PLANE_NAMES[plane]
            run.layers[name] = len(entries)
        if len(per_layer) == 0:
            continue
        run.handles.append(component.register_forward_pre_hook(lambda module, args, kwargs, provider=provider: run.capture(module, args, kwargs, provider), with_kwargs=True))
        for module, entries in per_layer.items():
            run.handles.append(module.register_forward_hook(make_hook(entries)))
    if len(run.names) > 0:
        summary = ', '.join(f'{name}={plane}' for name, plane in run.names.items())
        log.info(f'Network mask: masks={summary} layers={run.layers} sheet={tuple(sheet.shape[1:])}')
        if p is not None:
            p.extra_generation_params[INFOTEXT_KEY] = summary
    if not run.active:
        run.clear()


def remove():
    if run.active or run.sheet is not None:
        run.clear()
