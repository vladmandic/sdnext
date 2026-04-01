"""
based on article by TimothyAlexisVass
https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
"""

import os
import torch
from modules import devices
from modules.logger import log
from modules.vae import sd_vae_taesd


debug_enabled = os.environ.get('SD_HDR_DEBUG', None) is not None
debug = log.trace if debug_enabled else lambda *args, **kwargs: None
debug('Trace: HDR')


def sharpen_tensor(tensor, ratio=0):
    if ratio == 0:
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
    return tensor


def center_tensor(tensor, channel_shift=0.0, full_shift=0.0, offset=0.0):
    if channel_shift == 0 and full_shift == 0 and offset == 0:
        return tensor
    tensor -= tensor.mean(dim=(-1, -2), keepdim=True) * channel_shift
    tensor -= tensor.mean() * full_shift - offset
    return tensor


def maximize_tensor(tensor, boundary=1.0):
    if boundary == 1.0:
        return tensor
    boundary *= 4
    min_val = tensor.min()
    max_val = tensor.max()
    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    tensor *= normalization_factor
    return tensor


def get_color(colorstr):
    if not colorstr:
        colorstr = "#000000"
    rgb = torch.tensor(tuple(int(colorstr.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))).to(dtype=torch.float32)
    rgb = (rgb / 255).unsqueeze(-1).unsqueeze(-1).repeat(1, 64, 64).to(dtype=devices.dtype, device=devices.device)
    color = sd_vae_taesd.encode(rgb).squeeze(0)[0:3, 5, 5]
    return color


def color_adjust(tensor, colorstr, ratio):
    color = get_color(colorstr)
    for i in range(3):
        tensor[i] = center_tensor(tensor[i], full_shift=1, offset=color[i]*(ratio/2))
    return tensor


def correction(p, timestep, latent, step=0):
    total = getattr(p, 'correction_total_steps', 0)
    if total > 0:
        progress = step / total  # 0.0 = first step, ~1.0 = last step
        is_early = progress < 0.05
        is_mid = 0.2 <= progress <= 0.7
        is_late = progress >= 0.8
        is_sharpen = progress >= 0.7
        is_very_late = progress >= 0.9
    else:
        # fallback to timestep-based ranges for non-flow-match schedulers
        is_early = timestep > 950
        is_mid = 600 < timestep < 900
        is_late = timestep < 200
        is_sharpen = timestep < 350
        is_very_late = 1 < timestep < 100
    if is_early and p.hdr_clamp:
        latent = soft_clamp_tensor(latent, threshold=p.hdr_threshold, boundary=p.hdr_boundary)
        p.extra_generation_params["Latent clamp"] = f'{p.hdr_threshold}/{p.hdr_boundary}'
    if is_mid and p.hdr_color != 0:
        n = getattr(p, 'correction_steps_mid', 1)
        num_channels = latent.shape[0]
        if num_channels <= 4:
            # SDXL-style: channel 0 is brightness, channels 1+ are color
            latent[1:] = center_tensor(latent[1:], channel_shift=p.hdr_color / n, full_shift=float(p.hdr_mode))
        else:
            # Multi-channel latents (Flux 2, etc.): apply to all channels
            latent = center_tensor(latent, channel_shift=p.hdr_color / n, full_shift=float(p.hdr_mode))
        p.extra_generation_params["Latent color"] = f'{p.hdr_color}'
    if is_mid and p.hdr_tint_ratio != 0:
        n = getattr(p, 'correction_steps_mid', 1)
        num_channels = latent.shape[0]
        if num_channels <= 4:
            # SDXL-style: TAESD color encoding maps to 4-channel latent space
            latent = color_adjust(latent, p.hdr_color_picker, p.hdr_tint_ratio / n)
        else:
            # Multi-channel latents: apply uniform offset to all channels based on tint ratio
            latent = center_tensor(latent, full_shift=1.0, offset=p.hdr_tint_ratio / n)
        p.extra_generation_params["Latent tint"] = f'{p.hdr_tint_ratio}'
        p.extra_generation_params["Latent tint color"] = p.hdr_color_picker
    if is_late and p.hdr_brightness != 0:
        n = getattr(p, 'correction_steps_late', 1)
        num_channels = latent.shape[0]
        if num_channels <= 4:
            # SDXL-style: brightness is in channel 0 (luminance)
            latent[0:1] = center_tensor(latent[0:1], full_shift=float(p.hdr_mode), offset=p.hdr_brightness / n)
        else:
            # Multi-channel latents (Flux 2, etc.): scale intensity to avoid color shifts
            scale = 1.0 + (p.hdr_brightness / n) * 0.25
            latent = latent * scale
        p.extra_generation_params["Latent brightness"] = f'{p.hdr_brightness}'
    if is_sharpen and p.hdr_sharpen != 0:
        progress_in_range = (step - int(total * 0.7)) / max(int(total * 0.3), 1) if total > 0 else timestep / 350
        per_step_ratio = 2 ** (progress_in_range * 1.4) * p.hdr_sharpen / 16
        if abs(per_step_ratio) > 0.01:
            latent = sharpen_tensor(latent, ratio=per_step_ratio)
        p.extra_generation_params["Latent sharpen"] = f'{p.hdr_sharpen}'
    if is_very_late and p.hdr_maximize:
        latent = center_tensor(latent, channel_shift=p.hdr_max_center, full_shift=1.0)
        latent = maximize_tensor(latent, boundary=p.hdr_max_boundary)
        p.extra_generation_params["Latent max"] = f'{p.hdr_max_center}/{p.hdr_max_boundary}'
    return latent


def _unpack_latents(latents, pipe, p):
    """Unpack packed latents to standard [B, C, H, W] format for correction."""
    vae_scale = getattr(pipe, 'vae_scale_factor', 8)
    if p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5) and p.is_hr_pass:
        width = max(getattr(p, 'width', 0), getattr(p, 'hr_upscale_to_x', 0))
        height = max(getattr(p, 'height', 0), getattr(p, 'hr_upscale_to_y', 0))
    else:
        width = getattr(p, 'width', 1024)
        height = getattr(p, 'height', 1024)
    if hasattr(pipe, '_unpack_latents') and hasattr(pipe, 'vae_scale_factor'):
        # Flux 1 / Bria: use pipeline's own unpack method
        unpacked = pipe._unpack_latents(latents, height, width, vae_scale)  # pylint: disable=protected-access
        return unpacked, 'flux1'
    if hasattr(pipe, '_unpatchify_latents'):
        # Flux 2: manual reshape [B, seq_len, patch_channels] -> [B, C, H, W]
        b, seq_len, patch_ch = latents.shape
        channels = patch_ch // 4
        h_patches = height // vae_scale // 2
        w_patches = width // vae_scale // 2
        if h_patches * w_patches != seq_len:
            h_patches = w_patches = int(seq_len ** 0.5)
        unpacked = latents.view(b, h_patches, w_patches, channels, 2, 2)
        unpacked = unpacked.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, h_patches * 2, w_patches * 2)
        return unpacked, 'flux2'
    return latents, 'unknown'


def _repack_latents(latents, pack_type, pipe, p):
    """Repack standard [B, C, H, W] latents back to packed format."""
    if p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5) and p.is_hr_pass:
        height = max(getattr(p, 'height', 0), getattr(p, 'hr_upscale_to_y', 0))
        width = max(getattr(p, 'width', 0), getattr(p, 'hr_upscale_to_x', 0))
    else:
        height = getattr(p, 'height', 1024)
        width = getattr(p, 'width', 1024)
    if pack_type == 'flux1':
        # Flux 1 / Bria: use pipeline's pack method
        return pipe._pack_latents(latents, latents.shape[0], latents.shape[1], height, width)  # pylint: disable=protected-access
    if pack_type == 'flux2':
        # Flux 2: manual repack [B, C, H, W] -> [B, seq_len, patch_channels]
        b, channels, h, w = latents.shape
        h_patches = h // 2
        w_patches = w // 2
        latents = latents.reshape(b, channels, h_patches, 2, w_patches, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(b, h_patches * w_patches, channels * 4)
        return latents
    return latents


def _count_steps_in_range(pipe, low, high):
    """Count scheduler timesteps that fall within (low, high) exclusive."""
    timesteps = getattr(getattr(pipe, 'scheduler', None), 'timesteps', None)
    if timesteps is None:
        return 1
    count = sum(1 for t in timesteps.tolist() if low < t < high)
    return max(count, 1)


def _count_steps_below(pipe, threshold):
    """Count scheduler timesteps below a threshold."""
    timesteps = getattr(getattr(pipe, 'scheduler', None), 'timesteps', None)
    if timesteps is None:
        return 1
    count = sum(1 for t in timesteps.tolist() if t < threshold)
    return max(count, 1)


def correction_callback(p, timestep, kwargs, pipe=None, initial: bool = False, step: int = 0):
    if initial:
        if not any([p.hdr_clamp, p.hdr_mode, p.hdr_maximize, p.hdr_sharpen, p.hdr_color, p.hdr_brightness, p.hdr_tint_ratio]):
            p.correction_skip = True
            return kwargs
        # always skip for detailer passes (already-corrected image, different resolution)
        if getattr(p, 'recursion', False):
            p.correction_skip = True
            return kwargs
        # optionally skip for hires pass
        if getattr(p, 'is_hr_pass', False) and not getattr(p, 'hdr_apply_hires', True):
            p.correction_skip = True
            return kwargs
        p.correction_skip = False
        p.correction_warned = False
        total = getattr(pipe, 'num_timesteps', 0) if pipe is not None else 0
        if total > 0:
            p.correction_total_steps = total
            p.correction_steps_mid = max(int(total * 0.5), 1)  # 20%-70% range
            p.correction_steps_late = max(int(total * 0.2), 1)  # last 20%
        elif pipe is not None:
            p.correction_total_steps = 0
            p.correction_steps_mid = _count_steps_in_range(pipe, 600, 900)
            p.correction_steps_late = _count_steps_below(pipe, 200)
    elif getattr(p, 'correction_skip', False):
        return kwargs
    latents = kwargs["latents"]
    if debug_enabled:
        debug(f'Correction callback: step={step} timestep={timestep} latents_shape={latents.shape} total={getattr(p, "correction_total_steps", "unset")} skip={getattr(p, "correction_skip", "unset")}')
    if len(latents.shape) <= 3:  # packed latent
        if pipe is None:
            if not getattr(p, 'correction_warned', False):
                log.warning(f'Latent correction: shape={latents.shape} packed latent but no pipe reference')
                p.correction_warned = True
            return kwargs
        unpacked, pack_type = _unpack_latents(latents, pipe, p)
        if pack_type == 'unknown':
            if not getattr(p, 'correction_warned', False):
                log.warning(f'Latent correction: shape={latents.shape} unknown packed format')
                p.correction_warned = True
            return kwargs
        for i in range(unpacked.shape[0]):
            unpacked[i] = correction(p, timestep, unpacked[i], step=step)
        kwargs["latents"] = _repack_latents(unpacked, pack_type, pipe, p)
    elif len(latents.shape) == 4:  # standard batched latent
        for i in range(latents.shape[0]):
            latents[i] = correction(p, timestep, latents[i], step=step)
            if debug_enabled:
                debug(f"Full Mean: {latents[i].mean().item()}")
                debug(f"Channel Means: {latents[i].mean(dim=(-1, -2), keepdim=True).flatten().float().cpu().numpy()}")
                debug(f"Channel Mins: {latents[i].min(-1, keepdim=True)[0].min(-2, keepdim=True)[0].flatten().float().cpu().numpy()}")
                debug(f"Channel Maxes: {latents[i].max(-1, keepdim=True)[0].min(-2, keepdim=True)[0].flatten().float().cpu().numpy()}")
        kwargs["latents"] = latents
    elif len(latents.shape) == 5 and latents.shape[0] == 1:  # probably animatediff
        latents = latents.squeeze(0).permute(1, 0, 2, 3)
        for i in range(latents.shape[0]):
            latents[i] = correction(p, timestep, latents[i], step=step)
        latents = latents.permute(1, 0, 2, 3).unsqueeze(0)
        kwargs["latents"] = latents
    else:
        if not getattr(p, 'correction_warned', False):
            log.warning(f'Latent correction: shape={latents.shape} unknown latent')
            p.correction_warned = True
    return kwargs
