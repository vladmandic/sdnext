"""
based on article by TimothyAlexisVass
https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
"""

import os
import torch
from modules import shared, devices
from modules.vae import sd_vae_taesd


debug_enabled = os.environ.get('SD_HDR_DEBUG', None) is not None
debug = shared.log.trace if debug_enabled else lambda *args, **kwargs: None
debug('Trace: HDR')
skip_correction = False
warned = False


def warn_once(message):
    global warned # pylint: disable=global-statement
    if not warned:
        shared.log.warning(f'VAE: {message}')
        warned = True


def sharpen_tensor(tensor, ratio=0):
    if ratio == 0:
        # debug("Sharpen: Early exit")
        return tensor
    kernel = torch.ones((3, 3), dtype=tensor.dtype, device=tensor.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    kernel = kernel.expand(tensor.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    result_tmp = torch.nn.functional.conv2d(tensor, kernel, groups=tensor.shape[-3])
    result = tensor.clone()
    result[..., 1:-1, 1:-1] = result_tmp
    output = (1.0 + ratio) * tensor + (0 - ratio) * result
    return soft_clamp_tensor(output, threshold=0.95)


def soft_clamp_tensor(tensor, threshold=0.8, boundary=4):
    # shrinking towards the mean; will also remove outliers
    if max(abs(tensor.max()), abs(tensor.min())) < boundary or threshold == 0:
        return tensor
    channel_dim = 0
    threshold *= boundary
    max_vals = tensor.max(channel_dim, keepdim=True)[0]
    max_replace = ((tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
    over_mask = tensor > threshold
    min_vals = tensor.min(channel_dim, keepdim=True)[0]
    min_replace = ((tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
    under_mask = tensor < -threshold
    tensor = torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, tensor))
    # debug(f'HDR soft clamp: threshold={threshold} boundary={boundary} shape={tensor.shape}')
    return tensor


def center_tensor(tensor, channel_shift=0.0, full_shift=0.0, offset=0.0):
    if channel_shift == 0 and full_shift == 0 and offset == 0:
        return tensor
    # debug(f'HDR center: Before Adjustment: Full mean={tensor.mean().item()} Channel means={tensor.mean(dim=(-1, -2)).float().cpu().numpy()}')
    tensor -= tensor.mean(dim=(-1, -2), keepdim=True) * channel_shift
    tensor -= tensor.mean() * full_shift - offset
    # debug(f'HDR center: channel-shift={channel_shift} full-shift={full_shift}')
    # debug(f'HDR center: After Adjustment: Full mean={tensor.mean().item()} Channel means={tensor.mean(dim=(-1, -2)).float().cpu().numpy()}')
    return tensor


def maximize_tensor(tensor, boundary=1.0):
    if boundary == 1.0:
        return tensor
    boundary *= 4
    min_val = tensor.min()
    max_val = tensor.max()
    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    tensor *= normalization_factor
    # debug(f'HDR maximize: boundary={boundary} min={min_val} max={max_val} factor={normalization_factor}')
    return tensor


def get_color(colorstr):
    rgb = torch.tensor(tuple(int(colorstr.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))).to(dtype=torch.float32)
    rgb = (rgb / 255).unsqueeze(-1).unsqueeze(-1).repeat(1, 64, 64).to(dtype=devices.dtype, device=devices.device)
    color = sd_vae_taesd.encode(rgb).squeeze(0)[0:3, 5, 5]
    return color


def color_adjust(tensor, colorstr, ratio):
    color = get_color(colorstr)
    # debug(f'HDR tint: str={colorstr} color={color} ratio={ratio}')
    for i in range(3):
        tensor[i] = center_tensor(tensor[i], full_shift=1, offset=color[i]*(ratio/2))
    return tensor


def correction(p, timestep, latent):
    if timestep > 950 and p.hdr_clamp:
        latent = soft_clamp_tensor(latent, threshold=p.hdr_threshold, boundary=p.hdr_boundary)
        p.extra_generation_params["HDR clamp"] = f'{p.hdr_threshold}/{p.hdr_boundary}'
    if 600 < timestep < 900 and p.hdr_color != 0:
        latent[1:] = center_tensor(latent[1:], channel_shift=p.hdr_color, full_shift=float(p.hdr_mode))  # Color
        p.extra_generation_params["HDR color"] = f'{p.hdr_color}'
    if 600 < timestep < 900 and p.hdr_tint_ratio != 0:
        latent = color_adjust(latent, p.hdr_color_picker, p.hdr_tint_ratio)
        p.extra_generation_params["HDR tint"] = f'{p.hdr_tint_ratio}'
    if timestep < 200 and (p.hdr_brightness != 0): # do it late so it doesn't change the composition
        latent[0:1] = center_tensor(latent[0:1], full_shift=float(p.hdr_mode), offset=p.hdr_brightness)  # Brightness
        p.extra_generation_params["HDR brightness"] = f'{p.hdr_brightness}'
    if timestep < 350 and p.hdr_sharpen != 0:
        per_step_ratio = 2 ** (timestep / 250) * p.hdr_sharpen / 16
        if abs(per_step_ratio) > 0.01:
            latent = sharpen_tensor(latent, ratio=per_step_ratio)
        p.extra_generation_params["HDR sharpen"] = f'{p.hdr_sharpen}'
    if 1 < timestep < 100 and p.hdr_maximize:
        latent = center_tensor(latent, channel_shift=p.hdr_max_center, full_shift=1.0)
        latent = maximize_tensor(latent, boundary=p.hdr_max_boundary)
        p.extra_generation_params["HDR max"] = f'{p.hdr_max_center}/{p.hdr_max_boundary}'
    return latent


def correction_callback(p, timestep, kwargs, initial: bool = False):
    global skip_correction # pylint: disable=global-statement
    if initial:
        if not any([p.hdr_clamp, p.hdr_mode, p.hdr_maximize, p.hdr_sharpen, p.hdr_color, p.hdr_brightness, p.hdr_tint_ratio]):
            skip_correction = True
            return kwargs
        else:
            skip_correction = False
    elif skip_correction:
        return kwargs
    latents = kwargs["latents"]
    # debug(f'HDR correction: latents={latents.shape}')
    if len(latents.shape) <= 3: # packed latent
        warn_once(f'HDR correction: shape={latents.shape} packed latent')
        return kwargs
    if len(latents.shape) == 4: # standard batched latent
        for i in range(latents.shape[0]):
            latents[i] = correction(p, timestep, latents[i])
            if debug_enabled:
                debug(f"Full Mean: {latents[i].mean().item()}")
                debug(f"Channel Means: {latents[i].mean(dim=(-1, -2), keepdim=True).flatten().float().cpu().numpy()}")
                debug(f"Channel Mins: {latents[i].min(-1, keepdim=True)[0].min(-2, keepdim=True)[0].flatten().float().cpu().numpy()}")
                debug(f"Channel Maxes: {latents[i].max(-1, keepdim=True)[0].min(-2, keepdim=True)[0].flatten().float().cpu().numpy()}")
    elif len(latents.shape) == 5 and latents.shape[0] == 1: # probably animatediff
        latents = latents.squeeze(0).permute(1, 0, 2, 3)
        for i in range(latents.shape[0]):
            latents[i] = correction(p, timestep, latents[i])
        latents = latents.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        warn_once(f'HDR correction: shape={latents.shape} unknown latent')
    kwargs["latents"] = latents
    return kwargs
