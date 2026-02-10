"""Sharpfin functional image scaling operations.

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
Imports patched: absolute sharpfin.X → relative .X, triton import guarded.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple
import math
from contextlib import nullcontext

from .util import ResizeKernel, linear_to_srgb, srgb_to_linear

# from Pytorch >= 2.6
set_stance = getattr(torch.compiler, "set_stance", None)


def _get_resize_kernel(k: ResizeKernel):
    match k:
        case ResizeKernel.NEAREST:
            resize_kernel = nearest
            kernel_window = 0.5
        case ResizeKernel.BILINEAR:
            resize_kernel = bilinear
            kernel_window = 1.
        case ResizeKernel.MITCHELL:
            resize_kernel = mitchell # B = 1/3, C = 1/3
            kernel_window = 2.
        case ResizeKernel.CATMULL_ROM:
            resize_kernel = lambda x: mitchell(x, 0.0, 0.5)
            kernel_window = 2.
        case ResizeKernel.B_SPLINE:
            resize_kernel = lambda x: mitchell(x, 1.0, 0.0)
            kernel_window = 2.
        case ResizeKernel.LANCZOS2:
            resize_kernel = lambda x: lanczos(x, 2)
            kernel_window = 2.
        case ResizeKernel.LANCZOS3:
            resize_kernel = lambda x: lanczos(x, 3)
            kernel_window = 3.
        case ResizeKernel.MAGIC_KERNEL:
            resize_kernel = magic_kernel
            kernel_window = 1.5
        case ResizeKernel.MAGIC_KERNEL_SHARP_2013:
            resize_kernel = magic_kernel_sharp_2013
            kernel_window = 2.5
        case ResizeKernel.MAGIC_KERNEL_SHARP_2021:
            resize_kernel = magic_kernel_sharp_2021
            kernel_window = 4.5
        case _:
            raise ValueError(f"Unknown resize kernel {k}")
    return resize_kernel, kernel_window


### Resampling kernels
def nearest(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 0.5, 1., 0.)

    return weights

def bilinear(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 1.0, 1 - x, 0.)

    return weights

def mitchell(x: torch.Tensor, B: float = 1 / 3, C: float = 1 / 3) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 2, (-B - 6 * C) * x**3 + (6 * B + 30 * C) * x**2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C), 0)
    weights = torch.where(x <= 1, (12 - 9 * B - 6 * C) * x**3 + (-18 + 12 * B + 6 * C) * x**2 + (6 - 2 * B), weights)

    return weights

def magic_kernel(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 1.5, (1/2) * (x - 3/2) ** 2, 0)
    weights = torch.where(x <= 0.5, (3/4) - x ** 2, weights)

    return weights

def magic_kernel_sharp_2013(x: torch.Tensor):
    x = torch.abs(x)

    weights = torch.where(x <= 2.5, (-1/8) * (x - 5/2) ** 2, 0)
    weights = torch.where(x <= 1.5, (1 - x) * (7/4 - x), weights)
    weights = torch.where(x <= 0.5, (17/16) - (7/4) * x ** 2, weights)

    return weights

def magic_kernel_sharp_2021(x: torch.Tensor):
    x = torch.abs(x)

    weights = torch.where(x <= 4.5, (-1/288) * (x - 9/2) ** 2, 0)
    weights = torch.where(x <= 3.5, (1/36) * (x - 3) * (x - 15/4), weights)
    weights = torch.where(x <= 2.5, (1/6) * (x - 2) * (65/24 - x), weights)
    weights = torch.where(x <= 1.5, (35/36) * (x - 1) * (x - 239/140), weights)
    weights = torch.where(x <= 0.5, (577/576) - (239/144) * x ** 2, weights)

    return weights

def lanczos(x: torch.Tensor, n: int):
    return torch.where(torch.abs(x) < n, torch.sinc(x) * torch.sinc(x/n), 0)

def sharpen_conv2d(image: torch.Tensor, kernel: torch.Tensor, pad: int) -> torch.Tensor:
    image = F.pad(image, (pad,pad,pad,pad), mode='replicate')
    return F.conv2d(image, kernel, groups=image.shape[-3])

### Dithering and related functions.
def stochastic_round(
        x: torch.Tensor,
        out_dtype: torch.dtype,
        generator: torch.Generator = torch.Generator(),
    ):
    image = x * torch.iinfo(out_dtype).max
    image_quant = image.to(out_dtype)
    quant_error = image - image_quant.to(image.dtype)
    dither = torch.empty_like(image_quant, dtype=torch.bool)
    torch.bernoulli(quant_error, generator=generator, out=dither)
    return image_quant + dither

def generate_bayer_matrix(n):
    """Generate an n x n Bayer matrix where n is a power of 2."""
    assert (n & (n - 1)) == 0 and n > 0, "n must be a power of 2"

    if n == 1:
        return np.array([[0]])  # Base case

    smaller_matrix = generate_bayer_matrix(n // 2)

    return np.block([
        [4 * smaller_matrix + 0, 4 * smaller_matrix + 2],
        [4 * smaller_matrix + 3, 4 * smaller_matrix + 1]
    ])

### Scaling transforms

def _downscale_axis(
        image: torch.Tensor,
        size: int,
        resize_kernel: ResizeKernel,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel(resize_kernel)
    k = size / image.shape[-1]
    PAD = math.ceil((window - 0.5) / k)

    # Optimization note: doing torch.arange like this will compile to doing a int64 arange. Float arange
    # is much slower. So don't try to get clever and "optimize" by adding the +0.5 and *k to this.
    # Source grid is padded to allow "out of range" sampling from the source image.
    coords_source = (torch.arange(-PAD, image.shape[-1]+PAD, 1, dtype=torch.float32, device=device) + 0.5) * k
    coords_dest = (torch.arange(0, size, 1, dtype=torch.float32, device=device) + 0.5)

    # Create a grid of relative distances between each point on this axis.
    coord_grid = torch.empty((coords_source.shape[0], coords_dest.shape[0]), dtype=dtype, device=device)
    # Coord grid always constructed in torch.float32 because float16 precision breaks down for this
    # after 1024.0. This subtraction is the first opportunity we have to safely cast to float16.
    torch.sub(coords_source.unsqueeze(-1), other=coords_dest, out=coord_grid)

    weights = kernel(coord_grid)

    # Normalizing weights to sum to 1 along axis we are resizing on
    weights /= weights.sum(dim=0, keepdim=True)
    # weights /= (1/k)

    # Padded dimension is reduced by the matmul here.
    return F.pad(image, (PAD,PAD,0,0), mode='replicate') @ weights

def _upscale_axis(
        image: torch.Tensor,
        size: int,
        resize_kernel: ResizeKernel,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel(resize_kernel)
    k = size / image.shape[-1]
    PAD = math.ceil((window - 0.5) * k)

    # For upsizing, we expect out of range sampling from the destination image.
    coords_source = (torch.arange(0, image.shape[-1], 1, dtype=torch.float32, device=device) + 0.5)
    coords_dest = (torch.arange(-PAD, size+PAD, 1, dtype=torch.float32, device=device) + 0.5) / k

    coord_grid = torch.empty((coords_source.shape[0], coords_dest.shape[0]), dtype=dtype, device=device)
    torch.sub(coords_source.unsqueeze(-1), other=coords_dest, out=coord_grid)

    weights = kernel(coord_grid)

    # We need to explicitly trim padding by summing it into the real area of the destination grid.
    weights[:, PAD] += weights[:, :PAD].sum(dim=1)
    weights[:, -PAD-1] += weights[:, -PAD:].sum(dim=1)
    weights = weights[:, PAD:-PAD]

    weights /= weights.sum(dim=0, keepdim=True)

    return image @ weights

@torch.compile
def _downscale(
        image: torch.Tensor,
        out_res: tuple[int, int],
        resize_kernel: ResizeKernel,
        device: torch.device,
        dtype: torch.dtype,
        do_srgb_conversion: bool,
    ):
    H, W = out_res
    image = image.to(device=device, dtype=dtype)
    if do_srgb_conversion:
        image = srgb_to_linear(image)

    image = _downscale_axis(image, W, resize_kernel, device, dtype)
    image = _downscale_axis(image.mT, H, resize_kernel, device, dtype).mT

    if do_srgb_conversion:
        image = linear_to_srgb(image)
    image = image.clamp(0,1)
    return image

@torch.compile
def _upscale(
        image: torch.Tensor,
        out_res: tuple[int, int],
        resize_kernel:  ResizeKernel,
        device: torch.device,
        dtype: torch.dtype,
        do_srgb_conversion: bool,
    ):
    H, W = out_res
    image = image.to(device=device, dtype=dtype)
    if do_srgb_conversion:
        image = srgb_to_linear(image)

    image = _upscale_axis(image, W, resize_kernel, device, dtype)
    image = _upscale_axis(image.mT, H, resize_kernel, device, dtype).mT

    if do_srgb_conversion:
        image = linear_to_srgb(image)
    image = image.clamp(0,1)
    return image

# Triton sparse downscale - only available with Triton (CUDA)
try:
    from .triton_functional import downscale_sparse
except ImportError:
    downscale_sparse = None

def scale(
        image: torch.Tensor,
        out_res: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        do_srgb_conversion: bool = True,
        use_sparse: bool = False,
    ) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)
    if use_sparse:
        assert device.type != "cpu", "sparse implementation is only for GPU!"
        if resize_kernel != ResizeKernel.MAGIC_KERNEL_SHARP_2021:
            raise NotImplementedError
        if downscale_sparse is None:
            raise ImportError("Triton is required for sparse GPU acceleration")

    context_manager = (
        set_stance("force_eager") if set_stance and device.type == "cpu" else nullcontext()
    )
    with context_manager:
        if image.shape[-1] <= out_res[-1] and image.shape[-2] <= out_res[-2]:
            assert not use_sparse
            return _upscale(image, out_res, resize_kernel, device, dtype, do_srgb_conversion)
        elif image.shape[-1] >= out_res[-1] and image.shape[-2] >= out_res[-2]:
            if use_sparse:
                return downscale_sparse(image, out_res, resize_kernel)
            return _downscale(image, out_res, resize_kernel, device, dtype, do_srgb_conversion)
        else:
            raise ValueError("Mixed axis resizing (e.g. scaling one axis up and the other down) is not supported. File a bug report with your use case if needed.")
