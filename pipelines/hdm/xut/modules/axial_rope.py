import math
from functools import lru_cache

import torch
from torch import nn

from ..utils import compile_wrapper


@compile_wrapper
def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    *shape, d, r = x.shape
    return x.view(*shape, d * r)


@compile_wrapper
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    return grid.flatten(0, 1)


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return torch.tensor([y_min, y_max, x_min, x_max])


@lru_cache(maxsize=8)
def make_axial_pos(
    h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None
):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


def make_axial_pos_no_cache(
    h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None
):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


def make_cropped_pos(crop_h, crop_w, target_h, target_w):
    pos_map = make_axial_pos_no_cache(target_h, target_w).unflatten(
        0, (target_h, target_w)
    )
    if target_h > target_w:
        pos_map = pos_map[crop_h : crop_h + target_w, :]
    elif target_h < target_w:
        pos_map = pos_map[:, crop_w : crop_w + target_h]
    return pos_map.flatten(0, 1)


def freqs_pixel(max_freq=10.0):
    def init(shape):
        freqs = torch.linspace(1.0, max_freq / 2, shape[-1]) * math.pi
        return freqs.log().expand(shape)

    return init


def freqs_pixel_log(max_freq=10.0):
    def init(shape):
        log_min = math.log(math.pi)
        log_max = math.log(max_freq * math.pi / 2)
        return torch.linspace(log_min, log_max, shape[-1]).expand(shape)

    return init


class AxialRoPE(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        pos_dim=2,
        start_index=0,
        freqs_init=freqs_pixel_log(max_freq=10.0),
    ):
        super().__init__()
        self.n_heads = n_heads
        self.start_index = start_index
        log_freqs = freqs_init((n_heads, dim // (2 * pos_dim), 1))
        self.freqs = nn.Parameter(log_freqs.clone().repeat(1, 1, pos_dim))

    def extra_repr(self):
        dim = self.freqs.shape[-1]
        return f"dim={dim}, n_heads={self.n_heads}, start_index={self.start_index}"

    def get_freqs(self, pos):
        if pos.shape[-1] != self.freqs.shape[-1]:
            raise ValueError(f"input shape must be (..., {self.freqs.shape[-1]})")
        freqs = pos[..., None, None, :] * self.freqs.exp()
        freqs = freqs.flatten(-2, -1).repeat_interleave(2, dim=-1)
        return freqs.transpose(-2, -3)

    @compile_wrapper
    def forward(self, x, pos):
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, self.start_index)


class AdditiveAxialRoPE(AxialRoPE):
    """
    https://arxiv.org/abs/2405.10436
    """

    def __init__(
        self,
        dim,
        n_heads,
        pos_dim=2,
        start_index=0,
        freqs_init=freqs_pixel_log(max_freq=10.0),
    ):
        super().__init__(dim, n_heads, pos_dim, start_index, freqs_init)
        self.emb = nn.Parameter(torch.randn(dim) / dim**0.5)

    def forward(self, x, pos):
        pos_emb = torch.zeros_like(x)
        pos_emb = pos_emb + self.emb
        freqs = self.get_freqs(pos)
        if x.ndim == 3:
            pos_emb = pos_emb.unsqueeze(1)
        return x + apply_rotary_emb(freqs, pos_emb, self.start_index).view(x.shape)


if __name__ == "__main__":
    x = torch.randn(2, 1, 4 * 4, 16)
    pos = torch.randn(2, 16, 1)
    model = AxialRoPE(16, 1, 1)
    print(model(x, pos).shape)
