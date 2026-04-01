"""Sharpfin Triton-accelerated GPU scaling functions.

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
Imports patched: absolute sharpfin.X -> relative .X
Requires: triton (only available on CUDA platforms)
"""

import torch
import math
import triton
import triton.language as tl

from .util import ResizeKernel
from typing import Tuple
import torch.nn.functional as F
from triton.language.extra import libdevice
from .util import linear_to_srgb, srgb_to_linear

# Magic Kernel Sharp with Triton optimizations. Mainly converted to polynomials so that
# FMA operators can be used.
@triton.jit
def magic_kernel_sharp_2021_triton(x: tl.tensor):
    out = tl.zeros_like(x) # inplace operation doesn't help much.
    x = tl.abs(x)

    lte_05 = x <= 0.5
    lte_15 = x <= 1.5
    lte_25 = x <= 2.5
    lte_35 = x <= 3.5
    lte_45 = x <= 4.5

    x_sq = x*x # triton would compile like this anyways but it helps readability

    out = tl.where(lte_05,                tl.fma(x_sq, -239/144, 577/576), out)
    out = tl.where(lte_15 and not lte_05, tl.fma(x_sq, 35/36,    tl.fma(x, -379/144, 239/144)), out)
    out = tl.where(lte_25 and not lte_15, tl.fma(x_sq, -1/6,     tl.fma(x, 113/144, -65/72)), out)
    out = tl.where(lte_35 and not lte_25, tl.fma(x_sq, 1/36,     tl.fma(x, -3/16, 5/16)), out)
    out = tl.where(lte_45 and not lte_35, tl.fma(x_sq, -1/288,   tl.fma(x, 1/32, -9/128)), out)

    return out

@triton.jit
def sinc_triton(x: tl.tensor):
    y = tl.fma(x, math.pi, 1e-8)
    return libdevice.fast_sinf(y) / y

@triton.jit
def lanczos_triton(x: tl.tensor, n: tl.constexpr = 3):
    return tl.where(
        tl.abs(x) < n,
        sinc_triton(x) * sinc_triton(x/n),
        0
    )

# NOTE: there is no reason to use libdevice.pow, its only differences are with subnormals
@triton.jit
def linear_to_srgb_triton(x):
    return tl.where(
        x <= 0.0031308,
        x * 12.92,
        tl.fma(1.055, libdevice.fast_powf(x, 1/2.4), -0.055)
    )

@triton.jit
def srgb_to_linear_triton(x):
    return tl.where(
        x <= 0.04045,
        x / 12.92,
        libdevice.fast_powf(tl.fma(1/1.055, x, 0.055/1.055), 2.4)
    )

from .sparse_backend import triton_dds, triton_dds_sbsc, triton_dds_zerorhs_sbsc, Matrix, SBSCMatrix

def _get_resize_kernel_triton(k: ResizeKernel):
    match k:
        case ResizeKernel.NEAREST:
            raise NotImplementedError
        case ResizeKernel.BILINEAR:
            raise NotImplementedError
        case ResizeKernel.MITCHELL:
            raise NotImplementedError
        case ResizeKernel.CATMULL_ROM:
            raise NotImplementedError
        case ResizeKernel.B_SPLINE:
            raise NotImplementedError
        case ResizeKernel.LANCZOS2:
            raise NotImplementedError
        case ResizeKernel.LANCZOS3:
            resize_kernel = lanczos_triton
            kernel_window = 3.
        case ResizeKernel.MAGIC_KERNEL:
            raise NotImplementedError
        case ResizeKernel.MAGIC_KERNEL_SHARP_2013:
            raise NotImplementedError
        case ResizeKernel.MAGIC_KERNEL_SHARP_2021:
            resize_kernel = magic_kernel_sharp_2021_triton
            kernel_window = 4.5
        case _:
            raise ValueError(f"Unknown resize kernel {k}")
    return resize_kernel, kernel_window

# Sparse Downscale and support functions.

# Amanatides, John and Woo, Andrew -- Fast Voxel Traversal
def grid_line_tiles(x0, y0, x1, y1, grid_width, grid_height):
    tiles = set()

    dx = x1 - x0
    dy = y1 - y0

    x = math.floor(x0)
    y = math.floor(y0)

    end_x = math.floor(x1)
    end_y = math.floor(y1)

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1

    t_max_x = ((x + (step_x > 0)) - x0) / dx if dx != 0 else float('inf')
    t_max_y = ((y + (step_y > 0)) - y0) / dy if dy != 0 else float('inf')

    t_delta_x = abs(1 / dx) if dx != 0 else float('inf')
    t_delta_y = abs(1 / dy) if dy != 0 else float('inf')

    while True:
        if 0 <= x < grid_width and 0 <= y < grid_height:
            tiles.add((y,x))
        if x == end_x and y == end_y:
            break
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            x += step_x
        else:
            t_max_y += t_delta_y
            y += step_y

    return tiles

def tile_mask_function(dest_size, src_size, kernel_window=4.5, tile_size=64):
    k = dest_size / src_size
    PAD = math.ceil((kernel_window-0.5) / k)

    grid_size = math.ceil((src_size + 2*PAD)/tile_size), math.ceil(dest_size/tile_size)

    line_1 = 0, 0.5/tile_size, (dest_size)/tile_size, (src_size+0.5)/tile_size
    line_2 = 0, (2*PAD - 0.5)/tile_size, (dest_size)/tile_size, (src_size + 2*PAD - 0.5)/tile_size
    lines = line_1, line_2

    mask = torch.zeros(grid_size, dtype=torch.bool)

    tiles = set()

    for (x0, y0, x1, y1) in lines:
        tiles.update(grid_line_tiles(x0, y0, x1, y1, grid_size[1], grid_size[0]))

    tiles = torch.tensor(list(tiles))

    mask[tiles[:,0], tiles[:,1]] = True

    return mask, tiles

def create_tensor_metadata(
        tile_mask: torch.Tensor,
        tiles: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        offsets_t: torch.Tensor,
    ):

    indices[:,:2] = tiles

    torch.argsort(indices[:,1], stable=True, out=indices[:,2]) # block_offsets_t
    torch.take(indices[:,0], indices[:,2], out=indices[:,3]) # col_indices_t

    # reusing the offsets buffer here helps performance
    torch.sum(tile_mask, dim=1, out=offsets[1:])
    torch.sum(tile_mask, dim=0, out=offsets_t[1:])
    torch.cumsum(offsets, dim=0, out=offsets)
    torch.cumsum(offsets_t, dim=0, out=offsets_t)

    return indices, offsets, offsets_t

# for isolating the one mandatory graph break
@torch.compiler.disable
def _get_nnz_and_buffers(tile_mask):
    num_sparse_blocks = torch.sum(tile_mask).item()

    return [
        torch.empty((4, num_sparse_blocks), dtype=torch.int64, pin_memory=True).T, # indices
        torch.zeros((tile_mask.shape[0] + 1,), dtype=torch.int32, pin_memory=True), # offsets
        torch.zeros((tile_mask.shape[1] + 1,), dtype=torch.int32, pin_memory=True) # offsets_t
    ]


def generate_sparse_matrix(dest_size, src_size, kernel_window=4.5, tile_size=64):
    tile_mask, tiles = tile_mask_function(dest_size, src_size, kernel_window, tile_size)

    buffers = _get_nnz_and_buffers(tile_mask)
    num_sparse_blocks = buffers[0].shape[0]

    indices, offsets, offsets_t = create_tensor_metadata(
        tile_mask,
        tiles,
        *buffers
    )

    indices = indices.to(device='cuda', dtype=torch.int32, non_blocking=True)

    return Matrix(
        (tile_mask.shape[0] * tile_size, tile_mask.shape[1] * tile_size),
        torch.empty(num_sparse_blocks, tile_size, tile_size, dtype=torch.float16, device='cuda'),
        row_indices=indices[:,0],
        column_indices=indices[:,1],
        offsets=offsets.to(device='cuda', non_blocking=True),
        column_indices_t=indices[:,3],
        offsets_t=offsets_t.to(device='cuda', non_blocking=True),
        block_offsets_t=indices[:,2]
    )

@triton.jit
def compute_sparse_coord_grid_kernel(
        coords_source_ptr, coords_dest_ptr, sparse_data_ptr,
        row_indices_ptr, col_indices_ptr,
        k: float, M: int, N: int, BLOCK_SIZE: tl.constexpr, SPARSE_BLOCK_SIZE: tl.constexpr
    ):
    SPARSE_BLOCK_NUMEL = SPARSE_BLOCK_SIZE * SPARSE_BLOCK_SIZE
    sparse_block = tl.program_id(0)

    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tl.load(row_indices_ptr + sparse_block) * SPARSE_BLOCK_SIZE + tile_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.load(col_indices_ptr + sparse_block) * SPARSE_BLOCK_SIZE + tile_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_row = row_offsets < M
    mask_col = col_offsets < N

    coord_source = tl.load(coords_source_ptr + row_offsets, mask=mask_row, other=0.0)
    coord_dest = tl.load(coords_dest_ptr + col_offsets, mask=mask_col, other=0.0)

    x = tl.cast(coord_source[:, None] - coord_dest[None, :], tl.float16)

    x = magic_kernel_sharp_2021_triton(x)

    x *= k

    sparse_block_ptr = sparse_data_ptr + sparse_block * SPARSE_BLOCK_NUMEL

    local_row_start = tile_row * BLOCK_SIZE
    local_col_start = tile_col * BLOCK_SIZE

    local_rows = local_row_start + tl.arange(0, BLOCK_SIZE)
    local_cols = local_col_start + tl.arange(0, BLOCK_SIZE)

    local_rows_2d = local_rows[:, None]
    local_cols_2d = local_cols[None, :]

    store_offset = local_rows_2d * SPARSE_BLOCK_SIZE + local_cols_2d

    tl.store(sparse_block_ptr + store_offset, x)

def compute_sparse_coord_grid(target_size, source_size, kernel_window, BLOCK_SIZE=32, SPARSE_BLOCK_SIZE=64):
    assert SPARSE_BLOCK_SIZE % BLOCK_SIZE == 0

    k = target_size / source_size
    PAD = math.ceil((kernel_window - 0.5) / k)

    coords_source = torch.arange((-PAD + 0.5)*k, (source_size + PAD + 0.5)*k, k, dtype=torch.float32, device='cuda')
    coords_dest = torch.arange(0.5, target_size + 0.5, 1, dtype=torch.float32, device='cuda')

    M, N = coords_source.shape[0], coords_dest.shape[0]
    x = generate_sparse_matrix(target_size, source_size, kernel_window, SPARSE_BLOCK_SIZE)

    SPARSE_NUM_BLOCKS = x.data.shape[0]

    grid = lambda meta: (SPARSE_NUM_BLOCKS, triton.cdiv(SPARSE_BLOCK_SIZE, meta['BLOCK_SIZE']), triton.cdiv(SPARSE_BLOCK_SIZE, meta['BLOCK_SIZE']))
    compute_sparse_coord_grid_kernel[grid](
        coords_source, coords_dest, x.data,
        x.row_indices, x.column_indices,
        k, M, N, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    return x

# Dense kernel for downsampling coord_grids

@triton.jit
def compute_coord_grid_kernel(
        coords_source_ptr, coords_dest_ptr, coord_grid_ptr, k,
        M, N, BLOCK_SIZE: tl.constexpr,
    ):
    row_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_row = row_offsets < M
    mask_col = col_offsets < N

    coord_source = tl.load(coords_source_ptr + row_offsets, mask=mask_row)
    coord_dest = tl.load(coords_dest_ptr + col_offsets, mask=mask_col)

    x = tl.cast(coord_source[:, None] - coord_dest[None, :], tl.float16)

    x = magic_kernel_sharp_2021_triton(x)

    x *= k

    tl.store(coord_grid_ptr + row_offsets[:, None] * N + col_offsets[None, :], x, mask=mask_row[:, None] & mask_col[None, :])

def compute_coord_grid(target_size, source_size, kernel_window=4.5, BLOCK_SIZE=32):
    k = target_size / source_size
    PAD = math.ceil((kernel_window - 0.5) / k)

    coords_source = torch.arange((-PAD + 0.5)*k, (source_size + PAD + 0.5)*k, k, dtype=torch.float32, device='cuda')
    coords_dest = torch.arange(0.5, target_size + 0.5, 1, dtype=torch.float32, device='cuda')

    M, N = coords_source.shape[0], coords_dest.shape[0]
    coord_grid = torch.empty((M, N), dtype=torch.float16, device='cuda')

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    compute_coord_grid_kernel[grid](coords_source, coords_dest, coord_grid, k, M, N, BLOCK_SIZE)
    return coord_grid

@triton.jit
def pad_replicate_kernel(
        A, B,
        M_X, N_X,
        M_Y, N_Y,
        M_PAD, N_PAD,
        stride_xc, stride_xm, stride_xn,
        stride_yc, stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        fuse_linrgb: tl.constexpr
    ):
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_cl = tl.maximum(offs_m, M_PAD) - M_PAD
    offs_m_cl = tl.minimum(offs_m_cl, M_X - 1)
    offs_n_cl = tl.maximum(offs_n, N_PAD) - N_PAD
    offs_n_cl = tl.minimum(offs_n_cl, N_X - 1)

    mask_m = offs_m < M_Y
    mask_n = offs_n < N_Y

    A_ptr = A + pid_c * stride_xc + offs_m_cl[:, None] * stride_xm + offs_n_cl[None, :] * stride_xn
    B_ptr = B + pid_c * stride_yc + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    t = tl.load(A_ptr)
    if fuse_linrgb:
        t = srgb_to_linear_triton(t)

    tl.store(B_ptr, t, mask=mask_m[:, None] & mask_n[None, :])

def pad_replicate(
        img: torch.Tensor,
        pad_h: int,
        pad_w: int,
        sparse_block_size: int = 0,
        fuse_linrgb: bool = True,
    ):
    C = img.shape[0]

    M_PAD = pad_h
    N_PAD = pad_w

    if sparse_block_size != 0:
        out_H = img.shape[-2] + M_PAD + (-(img.shape[-2] + M_PAD)) % sparse_block_size
        out_W = img.shape[-1] + N_PAD + (-(img.shape[-1] + N_PAD)) % sparse_block_size
    else:
        out_H = img.shape[-2] + M_PAD + M_PAD
        out_W = img.shape[-1] + N_PAD + N_PAD

    out = torch.empty(C, out_H, out_W, dtype=img.dtype, device=img.device)

    BLOCK_M = 1
    BLOCK_N = 512

    grid = lambda META: (
        C,
        (out.shape[1] + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (out.shape[2] + META['BLOCK_N'] - 1) // META['BLOCK_N'],
    )

    pad_replicate_kernel[grid](
        img, out,
        img.shape[1], img.shape[2],
        out.shape[1], out.shape[2],
        M_PAD, N_PAD,
        img.stride(0), img.stride(1), img.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        fuse_linrgb=fuse_linrgb,
    )
    return out

def downscale_sparse(
        image: torch.Tensor,
        target_size: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        do_gamma_handling=True,
        BLOCK_SIZE: int = 32,
        SPARSE_BLOCK_SIZE: int = 64,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    T_W = target_size[-1]
    T_H = target_size[-2]
    S_W = image.shape[-1]
    S_H = image.shape[-2]

    y_s_w = compute_sparse_coord_grid(T_W, S_W, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    y_s_h = compute_sparse_coord_grid(T_H, S_H, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)

    PAD_W = math.ceil((window - 0.5) / (T_W / S_W))
    PAD_H = math.ceil((window - 0.5) / (T_H / S_H))

    image = pad_replicate(
        image,
        PAD_H,
        PAD_W,
        SPARSE_BLOCK_SIZE,
        fuse_linrgb=do_gamma_handling
    )

    image = triton_dds(
        image,
        y_s_w,
        output_mt=True
    )

    image = triton_dds(
        image,
        y_s_h,
        fuse_srgb=do_gamma_handling,
        clamp_output=True,
        output_mt=True,
        output_slice=(T_H, T_W)
    )

    return image

def downscale_triton(
        image: torch.Tensor,
        target_size: torch.Size,
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        do_gamma_handling=True,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    y_s_w = compute_coord_grid(target_size[-1], image.shape[-1], window)
    y_s_h = compute_coord_grid(target_size[-2], image.shape[-2], window)

    PAD_W = math.ceil((window - 0.5) / (target_size[-1] / image.shape[-1]))
    PAD_H = math.ceil((window - 0.5) / (target_size[-2] / image.shape[-2]))

    image = pad_replicate(image, PAD_H, PAD_W, fuse_linrgb=do_gamma_handling)

    image = image.view(-1, image.shape[-1])
    image = image @ y_s_w
    image = image.view(3, -1, image.shape[-1])
    image = image.mT
    image = image.reshape(-1, image.shape[-1])
    image = image @ y_s_h
    image = image.view(3, -1, image.shape[-1])
    image = image.mT
    if do_gamma_handling:
        image = linear_to_srgb(image[:, :target_size[0], :target_size[1]])
    image.clamp_(0.,1.)
    return image

# Single Block Sparse Column implementations.

def evaluate_line(x, x0, y0, x1, y1):
    """Evaluate the y-coordinate at a given x along a line from (x0, y0) to (x1, y1)."""
    if x1 == x0:
        return float('inf')
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def pad_height_to_multiple(height, multiple):
    """Pad a height up to the next multiple of 'multiple'."""
    return int(math.ceil(height / multiple) * multiple)

def generate_sbsc_structure(
        dest_size,
        src_size,
        kernel_window=4.5,
        tile_size=64,
        y_tile_size=32
    ):
    k = dest_size / src_size
    PAD = math.ceil((kernel_window - 0.5) / k)

    line1 = (0, 0.5, dest_size, src_size + 0.5)
    line2 = (0, 2 * PAD - 0.5, dest_size, src_size + 2 * PAD - 0.5)

    y_mins = []
    y_maxs = []
    n_blocks = math.ceil(dest_size / tile_size)
    max_height = 0

    for i in range(n_blocks):
        x0 = i * tile_size
        x1 = min(dest_size - 1, x0 + tile_size - 1)

        yt0 = evaluate_line(x0, *line1)
        yt1 = evaluate_line(x1, *line1)
        yb0 = evaluate_line(x0, *line2)
        yb1 = evaluate_line(x1, *line2)

        y_min = min(yt0, yt1)
        y_max = max(yb0, yb1)

        height = y_max - y_min
        padded = pad_height_to_multiple(height, y_tile_size)

        y_mins.append(y_min)
        y_maxs.append(y_max)
        max_height = max(max_height, padded)

    slope_top = (line1[3] - line1[1]) / (line1[2] - line1[0])
    ideal_step = slope_top * tile_size

    lower_bounds = []
    upper_bounds = []
    for i in range(1, n_blocks):
        lower_bounds.append((y_maxs[i] - max_height) / i)
        upper_bounds.append(y_mins[i] / i)

    lower = math.ceil(max(lower_bounds)) if lower_bounds else 0
    upper = math.floor(min(upper_bounds)) if upper_bounds else int(round(ideal_step))

    fixed_offset = int(round(ideal_step))
    if fixed_offset < lower:
        fixed_offset = lower
    elif fixed_offset > upper:
        fixed_offset = upper

    return fixed_offset, max_height, n_blocks, tile_size


def generate_sbsc_matrix(dest_size, src_size, kernel_window=4.5, tile_size=64, y_tile_size=32):
    offset, block_height, num_blocks, col_width = generate_sbsc_structure(
        dest_size, src_size, kernel_window, tile_size, y_tile_size
    )

    return SBSCMatrix(
        size=((offset * (num_blocks - 1)) + block_height, dest_size),
        data=torch.empty((num_blocks, block_height, col_width), dtype=torch.float16, device='cuda'),
        offset=offset,
        block_size=y_tile_size
    )

@triton.jit
def compute_sbsc_coord_grid_kernel(
        coords_source_ptr, coords_dest_ptr,
        sparse_data_ptr, offset: tl.constexpr,
        stride_xb, stride_xw, stride_xh,
        k: float, M: int, N: int, BLOCK_SIZE: tl.constexpr, SPARSE_BLOCK_SIZE: tl.constexpr
    ):

    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)

    start_row = offset * pid_w + pid_h * BLOCK_SIZE
    start_col = pid_w * SPARSE_BLOCK_SIZE

    row_offsets = start_row + tl.arange(0, BLOCK_SIZE)
    col_offsets = start_col + tl.arange(0, SPARSE_BLOCK_SIZE)

    mask_row = row_offsets < M
    mask_col = col_offsets < N

    coord_source = tl.load(coords_source_ptr + row_offsets, mask=mask_row, other=0.0)
    coord_dest = tl.load(coords_dest_ptr + col_offsets, mask=mask_col, other=0.0)

    y = tl.cast(coord_source[:, None] - coord_dest[None, :], tl.float16)

    y = magic_kernel_sharp_2021_triton(y)

    y *= k

    sparse_block_ptr = sparse_data_ptr + pid_w * stride_xb

    local_row_start = pid_h * BLOCK_SIZE

    local_rows = local_row_start + tl.arange(0, BLOCK_SIZE)
    local_cols = tl.arange(0, SPARSE_BLOCK_SIZE)

    local_rows_2d = local_rows[:, None]
    local_cols_2d = local_cols[None, :]

    store_offset = local_rows_2d * SPARSE_BLOCK_SIZE + local_cols_2d

    tl.store(sparse_block_ptr + store_offset, y)

def compute_sbsc_coord_grid(target_size, source_size, kernel_window, BLOCK_SIZE=32, SPARSE_BLOCK_SIZE=64):
    k = target_size / source_size
    PAD = math.ceil((kernel_window - 0.5) / k)

    coords_source = torch.arange((-PAD + 0.5)*k, (source_size + PAD + 0.5)*k, k, dtype=torch.float32, device='cuda')
    coords_dest = torch.arange(0.5, target_size + 0.5, 1, dtype=torch.float32, device='cuda')

    M, N = coords_source.shape[0], coords_dest.shape[0]
    x = generate_sbsc_matrix(target_size, source_size, kernel_window, SPARSE_BLOCK_SIZE, BLOCK_SIZE)

    SPARSE_BLOCKS, BLOCK_HEIGHT, _ = x.data.shape
    stride_xb, stride_xh, stride_xw = x.data.stride()

    grid = lambda meta: (SPARSE_BLOCKS, triton.cdiv(BLOCK_HEIGHT, meta['BLOCK_SIZE']))
    compute_sbsc_coord_grid_kernel[grid](
        coords_source, coords_dest,
        x.data, x.offset,
        stride_xb, stride_xh, stride_xw,
        k, M, N, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    return x


def downscale_sbsc(
        image: torch.Tensor,
        target_size: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        do_gamma_handling: bool = True,
        BLOCK_SIZE: int = 32,
        SPARSE_BLOCK_SIZE: int = 64,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    T_W = target_size[-1]
    T_H = target_size[-2]
    S_W = image.shape[-1]
    S_H = image.shape[-2]

    y_s_w = compute_sbsc_coord_grid(T_W, S_W, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    y_s_h = compute_sbsc_coord_grid(T_H, S_H, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)

    PAD_W = math.ceil((window - 0.5) / (T_W / S_W))
    PAD_H = math.ceil((window - 0.5) / (T_H / S_H))

    image = pad_replicate(
        image,
        PAD_H,
        PAD_W,
        fuse_linrgb=do_gamma_handling,
        sparse_block_size=SPARSE_BLOCK_SIZE,
    )

    image = triton_dds_sbsc(
        image,
        y_s_w,
        output_mt=True
    )

    image = triton_dds_sbsc(
        image,
        y_s_h,
        fuse_srgb=do_gamma_handling,
        clamp_output=True,
        output_mt=True,
    )

    return image


def downscale_sbsc_zerorhs(
        image: torch.Tensor,
        target_size: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        do_gamma_handling=True,
        gamma_handling_type: str = 'fast',
        BLOCK_SIZE: int = 32,
        SPARSE_BLOCK_SIZE: int = 64,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    T_W = target_size[-1]
    T_H = target_size[-2]
    S_W = image.shape[-1]
    S_H = image.shape[-2]

    block_specs_w = generate_sbsc_structure(
        T_W, S_W, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE
    )

    block_specs_h = generate_sbsc_structure(
        T_H, S_H, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE
    )

    image = triton_dds_zerorhs_sbsc(
        image,
        T_W, S_W, window, block_specs_w,
        fuse_srgb='input' if do_gamma_handling else '',
        gamma_correction=gamma_handling_type,
        output_mt=True
    )

    image = triton_dds_zerorhs_sbsc(
        image,
        T_H, S_H, window, block_specs_h,
        fuse_srgb='output' if do_gamma_handling else '',
        gamma_correction=gamma_handling_type,
        clamp_output=True,
        output_mt=True,
    )

    return image
