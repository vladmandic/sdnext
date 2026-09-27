from __future__ import annotations

import torch
import torch.nn.functional as F


def box2_torch(x: torch.Tensor) -> torch.Tensor:
    """2x2 box mean, padded with zeros to a multiple of 16 rows and columns (the library's tile size).

    x: [N, C, H, W]
    """
    _, _, h, w = x.shape
    y = F.avg_pool2d(x[:, :, : h // 2 * 2, : w // 2 * 2], 2)
    hp = (y.shape[2] + 15) // 16 * 16
    wp = (y.shape[3] + 15) // 16 * 16
    return F.pad(y, (0, wp - y.shape[3], 0, hp - y.shape[2]))


def photometric_error_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """mean_rgb |a - b| blurred with the separable [1/8, 3/4, 1/8] kernel (border replicated).

    a, b: [N, C, H, W]
    """
    e = (a - b).abs().mean(1, keepdim=True)
    k = torch.tensor([0.125, 0.75, 0.125], device=e.device, dtype=e.dtype)
    k2 = (k[:, None] * k[None, :])[None, None]
    return F.conv2d(F.pad(e, (1, 1, 1, 1), mode="replicate"), k2)


def warp_torch(img: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """Backward bilinear warp of img [N,C,H,W] by absolute pixel offsets fx, fy [N,H,W] (border clamped)."""
    _n, _, h, w = img.shape
    ys = torch.arange(h, device=img.device, dtype=img.dtype)
    xs = torch.arange(w, device=img.device, dtype=img.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    gx = (grid_x[None] + fx) * (2.0 / max(w - 1, 1)) - 1.0
    gy = (grid_y[None] + fy) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def compose_fg_torch(a_full: torch.Tensor, b_full: torch.Tensor, export: torch.Tensor) -> torch.Tensor:
    """Full-resolution frames [N,3,H,W] from the half-res export."""
    n, _, h, w = a_full.shape
    _, _, hc, wc = export.shape
    dev, dt = a_full.device, a_full.dtype

    ys = torch.arange(h, device=dev, dtype=dt)
    xs = torch.arange(w, device=dev, dtype=dt)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    coarse = torch.cat([export[:, 0:4], torch.sigmoid(export[:, 4:5])], 1)
    u = (grid_x / 2.0).clamp(max=wc - 1.0)
    v = (grid_y / 2.0).clamp(max=hc - 1.0)
    gx = u * (2.0 / max(wc - 1, 1)) - 1.0
    gy = v * (2.0 / max(hc - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], -1)[None].expand(n, -1, -1, -1)

    up = F.grid_sample(coarse, grid, mode="bilinear", padding_mode="border", align_corners=True)
    m = up[:, 4:5]
    wa = warp_torch(a_full, 2 * up[:, 0], 2 * up[:, 1])
    wb = warp_torch(b_full, 2 * up[:, 2], 2 * up[:, 3])
    return (m * wa + (1.0 - m) * wb).clamp(0.0, 1.0)
