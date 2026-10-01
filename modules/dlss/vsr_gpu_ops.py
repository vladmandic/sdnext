from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare_vsr_input_torch(
    rgb: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Prepare 16-channel input for VSR network from (N, 3, H, W) RGB tensor.
    Packs 2x2 spatial quads of RGBA pixels into 16 channels at (H/2, W/2).
    """
    _, _, h, w = rgb.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h > 0 or pad_w > 0:
        rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="replicate")

    rgb_dt = rgb.to(dtype)
    p00 = rgb_dt[:, :, 0::2, 0::2]
    p01 = rgb_dt[:, :, 0::2, 1::2]
    p10 = rgb_dt[:, :, 1::2, 0::2]
    p11 = rgb_dt[:, :, 1::2, 1::2]
    zero = torch.zeros_like(p00[:, :1])

    # Interleaved 2x2 quad packing: [p00_RGBA, p01_RGBA, p10_RGBA, p11_RGBA]
    feat16 = torch.cat([p00, zero, p01, zero, p10, zero, p11, zero], dim=1)
    return feat16


def process_vsr_output_torch(
    out_proj: torch.Tensor,
    orig_h: int,
    orig_w: int,
) -> torch.Tensor:
    """Reconstruct 2x high-resolution residual from 48-channel subpixel projection.
    Unpacks 16 subpixels (4x4) x 3 RGB channels to (N, 3, 2*orig_h, 2*orig_w).
    """
    n, _, h2, w2 = out_proj.shape
    t = out_proj.view(n, 4, 4, 3, h2, w2)
    # Permute to (N, 3, H2, 4, W2, 4) -> (N, 3, 2*H, 2*W)
    out_2x = t.permute(0, 3, 4, 1, 5, 2).contiguous().view(n, 3, h2 * 4, w2 * 4)
    target_h = orig_h * 2
    target_w = orig_w * 2
    return out_2x[:, :3, :target_h, :target_w]


def resample_torch(
    image: torch.Tensor,
    width: int,
    height: int,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Resize (N, C, H, W) or (C, H, W) tensor to target (height, width)."""
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True

    if image.shape[2] == height and image.shape[3] == width:
        out = image
    else:
        out = F.interpolate(
            image,
            size=(height, width),
            mode=mode,
            align_corners=False,
            antialias=True if mode in ("bilinear", "bicubic") else False,
        )

    if squeeze:
        out = out.squeeze(0)
    return out


def gaussian_blur_torch(x: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """Separable 2D Gaussian blur with edge replication on GPU."""
    if radius <= 0.0:
        return x
    sigma = max(0.1, radius)
    extent = int(max(1, round(3.0 * sigma)))
    k = torch.arange(-extent, extent + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-k * k / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    k2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
    c = x.shape[1]
    k2d = k2d.expand(c, 1, -1, -1)
    padded = F.pad(x, (extent, extent, extent, extent), mode="replicate")
    return F.conv2d(padded, k2d, groups=c)


def compose_vsr_torch(
    source_rgb: torch.Tensor,
    neural_residual_2x: torch.Tensor,
    *,
    scale: float = 2.0,
    target_width: int | None = None,
    target_height: int | None = None,
    detail_strength: float = 1.0,
    colour_strength: float = 1.0,
    detail_radius: float = 4.0,
) -> torch.Tensor:
    """Blend and scale 2x VSR output to target resolution with base bicubic interpolation and high-pass detail enhancement."""
    # 1. 2x bicubic base interpolation
    orig_h = source_rgb.shape[2]
    orig_w = source_rgb.shape[3]
    base_2x = F.interpolate(
        source_rgb,
        size=(orig_h * 2, orig_w * 2),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    # 2. Add neural residual with frequency-separated detail / colour weighting
    if detail_radius > 0.0 and (detail_strength != 1.0 or colour_strength != 1.0):
        lowpass = gaussian_blur_torch(neural_residual_2x, detail_radius)
        highpass = neural_residual_2x - lowpass
        residual_scaled = colour_strength * lowpass + detail_strength * highpass
    else:
        residual_scaled = neural_residual_2x * detail_strength

    output_2x = torch.clamp(base_2x + residual_scaled, 0.0, 1.0)

    # 3. Compute target dimensions
    if target_width is None or target_height is None:
        target_h = int(round(orig_h * scale))
        target_w = int(round(orig_w * scale))
    else:
        target_w = target_width
        target_h = target_height

    # 4. Resample to final scale if scale != 2.0
    if output_2x.shape[2] != target_h or output_2x.shape[3] != target_w:
        output = resample_torch(output_2x, target_w, target_h, mode="bicubic")
    else:
        output = output_2x

    return output.clamp(0.0, 1.0)
