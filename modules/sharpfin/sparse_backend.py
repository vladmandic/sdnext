"""Sharpfin sparse matrix backend for Triton DDS matmul.

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
Adapted from https://github.com/stanford-futuredata/stk (Apache 2.0)
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Tuple
from dataclasses import dataclass
from .triton_functional import linear_to_srgb_triton, srgb_to_linear_triton, magic_kernel_sharp_2021_triton, lanczos_triton

# Code is all adapted from https://github.com/stanford-futuredata/stk, licensed under Apache-2.0
# Very reduced set of functions for handling DDS (Dense = Dense @ Sparse) matmul only, with the
# DDS kernel modified to be more flexible on input shapes.

def _validate_matrix(shape, data, row_indices, column_indices, offsets):
    if data.dim() == 1:
        data = torch.reshape(data, [data.numel(), 1, 1])

    if data.shape[-2] != data.shape[-1]:
        raise ValueError(
            "Expected square blocking in data. "
            f"Got block shape {[data.shape[-2], data.shape[-1]]}")

    block_size = data.shape[-1]
    data = data.view([-1, block_size, block_size])

    if data.dim() != 3:
        raise ValueError(
            "Expected 3D shape for data (nnz, block, block). "
            f"Got shape {data.dim()}D shape.")

    block_size = data.shape[1]
    if shape[-2] % block_size != 0 or shape[-1] % block_size != 0:
        raise ValueError(
            "Matrix shape must be dividible by blocking. "
            f"Got shape {shape} with "
            f"{[block_size, block_size]} blocking.")

    if np.prod(shape) < data.numel():
        raise ValueError(
            "Invalid matrix. Number of nonzeros exceeds matrix capacity "
            f"({data.numel()} v. {np.prod(shape)})")

    if row_indices.dim() != 1:
        raise ValueError(
            f"Expected 1D row_indices. Got {row_indices.dim()}D row_indices.")

    if column_indices.dim() != 1:
        raise ValueError(
            f"Expected 1D column_indices. Got {column_indices.dim()}D column_indices.")

    if offsets.dim() != 1:
        raise ValueError(
            f"Expected 1D offsets. Got {offsets.dim()}D offsets.")

    if row_indices.numel() != data.shape[0]:
        raise ValueError(
            "Expected 1 index per nonzero block. "
            f"Got {row_indices.numel()} row_indices for {data.shape[0]} blocks")

    if column_indices.numel() != data.shape[0]:
        raise ValueError(
            "Expected 1 index per nonzero block. "
            f"Got {column_indices.numel()} column_indices for {data.shape[0]} blocks")

    block_rows = np.prod(shape[:-1]) / block_size
    if offsets.numel() != block_rows + 1:
        raise ValueError(
            "Expected one offset per block row plus one. "
            f"Got {offsets.numel()} offsets with {block_rows} block rows.")

    is_cuda = (data.is_cuda and
               row_indices.is_cuda and
               column_indices.is_cuda and
               offsets.is_cuda)
    is_cpu = (not data.is_cuda and
              not row_indices.is_cuda and
              not column_indices.is_cuda and
              not offsets.is_cuda)
    if not (is_cuda or is_cpu):
        raise ValueError(
            "Expected data & meta-data on common device. "
            f"Got data on {data.device}, row_indices on {row_indices.device} "
            f"column_indices on {column_indices.device} and "
            f"offsets on {offsets.device}.")

    if data.dtype != torch.float16:
        raise ValueError(
            f"Expected float16 data. Got {data.dtype} data.")
    if row_indices.dtype != torch.int16:
        raise ValueError(
            f"Expected int16 row_indices. Got {row_indices.dtype} row_indices.")
    if column_indices.dtype != torch.int16:
        raise ValueError(
            f"Expected int16 column_indices. Got {column_indices.dtype} column_indices.")
    if offsets.dtype != torch.int32:
        raise ValueError(
            f"Expected int32 offsets. Got {offsets.dtype} offsets.")
    return data

def _transpose(size, data: torch.Tensor, row_indices: torch.Tensor, column_indices: torch.Tensor, offsets):
    block_columns = size[1] // data.shape[1]

    gather_indices = column_indices.argsort()
    column_indices_t = row_indices.gather(0, gather_indices)
    block_offsets_t = gather_indices.int()

    column_indices_float = column_indices.float()

    zero = torch.zeros((1,), dtype=torch.int32, device=data.device)
    nnz_per_column = column_indices_float.histc(block_columns, 0, block_columns)
    nnz_per_column = nnz_per_column.int()
    offsets_t = torch.cat([zero, nnz_per_column.cumsum(0, dtype=torch.int32)])
    return column_indices_t, offsets_t, block_offsets_t

class SBSCMatrix(torch.nn.Module):
    """Single Block Sparse Column (SBSC) matrix format."""
    def __init__(
            self,
            size,
            data: torch.Tensor,
            offset: int,
            block_size: int
        ):
        super().__init__()
        self.data = data
        self.offset = offset
        self.size = size
        self.num_blocks = data.shape[0]
        self.col_width = data.shape[2]
        self.col_block_size = block_size

class Matrix(torch.nn.Module):
    """A matrix stored in block compressed sparse row (BCSR) format."""

    def __init__(self,
                 size,
                 data: torch.Tensor,
                 row_indices: torch.Tensor,
                 column_indices: torch.Tensor,
                 offsets: torch.Tensor,
                 column_indices_t: torch.Tensor=None,
                 offsets_t: torch.Tensor=None,
                 block_offsets_t: torch.Tensor=None):
        super().__init__()
        self._size = size
        self._data = data
        self._row_indices = row_indices
        self._column_indices = column_indices
        self._offsets = offsets

        if ((column_indices_t is None) or (offsets_t is None) or
            (block_offsets_t is None)):
            column_indices_t, offsets_t, block_offsets_t = _transpose(
                size, data, row_indices, column_indices, offsets)
        self._column_indices_t = column_indices_t
        self._offsets_t = offsets_t
        self._block_offsets_t = block_offsets_t

        self._transposed = False

        max_dim = np.iinfo(np.int16).max * self.blocking
        if column_indices.dtype == torch.int16:
            if size[0] > max_dim or size[1] > max_dim:
                raise ValueError(
                    "Sparse matrix with shape {size} exceeds representable "
                    "size with 16-bit indices.")

    def validate(self):
        _validate_matrix(self._size,
                         self._data,
                         self._row_indices,
                         self._column_indices,
                         self._offsets)

    def to(self, device):
        self._data = self._data.to(device)
        self._row_indices = self._row_indices.to(device)
        self._column_indices = self._column_indices.to(device)
        self._offsets = self._offsets.to(device)
        self._column_indices_t = self._column_indices_t.to(device)
        self._offsets_t = self._offsets_t.to(device)
        self._block_offsets_t = self._block_offsets_t.to(device)
        return self

    def cuda(self):
        return self.to(torch.cuda.current_device())

    def clone(self):
        return Matrix(
            self.size(),
            self.data.clone(),
            self.row_indices.clone(),
            self.column_indices.clone(),
            self.offsets.clone(),
            self.column_indices_t.clone(),
            self.offsets_t.clone(),
            self.block_offsets_t.clone())

    def t(self):
        if self.dim() != 2:
            raise ValueError(
                "t() expects a tensor with <= 2 dimensions, "
                f"but self is {self.dim()}D.")
        out = Matrix(self.size(),
                     self.data,
                     self.row_indices,
                     self.column_indices,
                     self.offsets,
                     self.column_indices_t,
                     self.offsets_t,
                     self.block_offsets_t)
        out._transposed = not self._transposed
        out._size = torch.Size((self._size[1], self._size[0]))
        return out

    def contiguous(self):
        raise ValueError("Not yet implemented.")

    def is_contiguous(self):
        return not self._transposed

    @property
    def is_cuda(self):
        return self._data.is_cuda

    @property
    def device(self):
        return self._data.device

    def size(self):
        return self._size

    @property
    def shape(self):
        return self.size()

    def dim(self):
        return len(self._size)

    @property
    def data(self):
        return self._data

    @property
    def row_indices(self):
        return self._row_indices

    @property
    def column_indices(self):
        return self._column_indices

    @property
    def offsets(self):
        return self._offsets

    @property
    def offsets_t(self):
        return self._offsets_t

    @property
    def column_indices_t(self):
        return self._column_indices_t

    @property
    def block_offsets_t(self):
        return self._block_offsets_t

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nnz(self):
        return self.data.numel()

    @property
    def blocking(self):
        return self.data.shape[1]

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def requires_grad_(self, x):
        self.data.requires_grad_(x)
        return self

    def view(self, *shape):
        assert self.is_contiguous()
        if shape[-1] != self.size()[-1]:
            raise ValueError(
                "Can't change view on compressed dimension. "
                f"{self.size()[-1]} v. {shape[-1]}.")
        if np.prod(shape) != np.prod(self.size()):
            raise ValueError(
                "Mismatch in numel of Matrix and new shape. "
                f"{np.prod(self.size())} v. {np.prod(shape)}")
        return Matrix(shape,
                      self.data,
                      self.row_indices,
                      self.column_indices,
                      self.offsets,
                      self.column_indices_t,
                      self.offsets_t,
                      self.block_offsets_t)

    @property
    def grad(self):
        size = self.size()
        if not self.is_contiguous():
            size = torch.Size((size[1], size[0]))
        out = Matrix(size,
                     self.data.grad,
                     self.row_indices,
                     self.column_indices,
                     self.offsets,
                     self.column_indices_t,
                     self.offsets_t,
                     self.block_offsets_t)
        return out if self.is_contiguous() else out.t()

@torch.no_grad()
def _expand_for_blocking(idxs, blocking):
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, 2]).repeat(1, blocking, 1)

    idxs[:, :, 1] *= blocking
    idxs[:, :, 1] += torch.reshape(torch.arange(blocking, device=idxs.device), [1, blocking])

    idxs = torch.reshape(idxs, [idxs.size()[0], 1, blocking, 2])
    idxs = idxs.repeat(1, blocking, 1, 1)

    idxs[:, :, :, 0] *= blocking
    idxs[:, :, :, 0] += torch.reshape(torch.arange(blocking, device=idxs.device), [1, blocking, 1])
    idxs = torch.reshape(idxs, [-1, 2])
    return idxs


@torch.no_grad()
def to_dense(x):
    assert isinstance(x, Matrix)

    shape = (np.prod(x.shape[:-1]), x.shape[-1])
    row_idxs = x.row_indices.type(torch.int32)
    col_idxs = x.column_indices.type(torch.int32)
    indices = _expand_for_blocking(torch.stack([row_idxs, col_idxs], dim=1), x.blocking)
    indices = (indices[:, 0] * shape[1] + indices[:, 1]).type(torch.int64)

    out = torch.zeros(shape[0] * shape[1], dtype=x.dtype, device=x.device)
    out.scatter_(0, indices, x.data.flatten())
    return out.reshape(x.size())


@dataclass
class TritonConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 32
    BLOCK_SIZE: int = 64
    NUM_STAGES: int = 4
    NUM_WARPS: int = 4

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dds_kernel(
        A: tl.tensor, B: tl.tensor, C: tl.tensor, M, N, K, O_M, O_N,
        stride_ac, stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cc, stride_cm, stride_cn,
        row_indices: tl.tensor, column_indices: tl.tensor,
        offsets: tl.tensor, block_offsets_t: tl.tensor,
        fuse_srgb: tl.constexpr, clamp_output: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
    ):

    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    num_pid_m = tl.num_programs(1)
    num_pid_n = tl.num_programs(2)

    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)

    offsets += pid_n

    start_inx = tl.load(offsets)
    end_inx = tl.load(offsets + 1)

    column_indices += start_inx
    block_offsets_t += start_inx

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE

    A_block_ptr = tl.make_block_ptr(
        base=A + pid_c * stride_ac, shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1)
    )

    rn = tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)

    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    bk_sub_incr = BLOCK_K * stride_bk

    for block_inx in range(end_inx - start_inx):
        a_col_idx = tl.load(column_indices + block_inx)
        ptr_A = tl.advance(A_block_ptr, (0, a_col_idx * BLOCK_SIZE))

        b_block_offset = tl.load(block_offsets_t + block_inx)
        ptr_B = B + b_block_offset * BLOCK_ELEMENTS

        for sub_block_inx in range(nsub_blocks):
            a = tl.load(ptr_A)
            b = tl.load(ptr_B)

            acc = tl.dot(a, b, acc, out_dtype=tl.float16)

            ptr_A = tl.advance(ptr_A, (0, BLOCK_K))
            ptr_B += bk_sub_incr

    if fuse_srgb:
        acc = linear_to_srgb_triton(acc)

    if clamp_output:
        acc = tl.clamp(acc, 0.0, 1.0)

    if fuse_srgb or clamp_output:
        acc = acc.to(C.dtype.element_ty)

    C_block_ptr = tl.make_block_ptr(
        base=C + pid_c * stride_cc, shape=(O_M, O_N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    tl.store(C_block_ptr, acc, boundary_check=(0, 1))


def triton_dds(
        lhs: torch.Tensor,
        rhs: Matrix,
        fuse_srgb: bool = False,
        clamp_output: bool = False,
        output_mt: bool = False,
        output_slice: None | Tuple[int,int] = None
    ):
    assert isinstance(lhs, torch.Tensor)
    assert isinstance(rhs, Matrix)
    assert lhs.ndim == 3
    CH = lhs.shape[0]
    stride_ac = lhs.stride(0)

    M, K = lhs.shape[-2:]
    N = rhs.shape[-1]

    if output_mt:
        if output_slice is not None:
            O_N, O_M = output_slice
            out = torch.empty(
                (*lhs.shape[:-2], *output_slice),
                dtype=lhs.dtype,
                device=lhs.device
            )
        else:
            O_N, O_M = N, M
            out = torch.empty(
                (*lhs.shape[:-2], rhs.shape[1], lhs.shape[-2]),
                dtype=lhs.dtype,
                device=lhs.device
            )
        stride_cm, stride_cn = out.stride(-1), out.stride(-2)
        stride_cc = out.stride(-3)
    else:
        if output_slice is not None:
            O_M, O_N = output_slice
            out = torch.empty(
                (*lhs.shape[:-2], *output_slice),
                dtype=lhs.dtype,
                device=lhs.device
            )
        else:
            O_M, O_N = M, N
            out = torch.empty(
                (*lhs.shape[:-1], rhs.shape[1]),
                dtype=lhs.dtype,
                device=lhs.device
            )
        stride_cm, stride_cn = out.stride(-2), out.stride(-1)
        stride_cc = out.stride(-3)

    trans_B = not rhs.is_contiguous()
    trans_A = (lhs.stride(-2) > 1 and lhs.stride(-1) > 1)
    assert trans_A == False, trans_B == False

    assert lhs.shape[-1] <= rhs.shape[0], "incompatible dimensions"

    stride_am, stride_ak = lhs.stride(-2), lhs.stride(-1)

    if trans_B:
        stride_bk, stride_bn = rhs.data.stride(2), rhs.data.stride(1)
        b_column_indices, b_offsets = rhs.column_indices, rhs.offsets
    else:
        stride_bk, stride_bn = rhs.data.stride(1), rhs.data.stride(2)
        b_column_indices, b_offsets = rhs.column_indices_t, rhs.offsets_t

    grid = lambda META: (CH, triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    _dds_kernel[grid](
        lhs, rhs.data, out, M, N, K, O_M, O_N,
        stride_ac, stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cc, stride_cm, stride_cn,
        rhs.row_indices, b_column_indices, b_offsets,
        rhs.block_offsets_t, fuse_srgb, clamp_output,
        GROUP_M=128, ACC_TYPE=tl.float16, BLOCK_M=min(rhs.data.shape[1], 64),
        BLOCK_N=rhs.data.shape[1], BLOCK_SIZE=rhs.data.shape[1], BLOCK_K=min(rhs.data.shape[1], 64)
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['BLOCK_SIZE', 'BLOCK_N'],
)
@triton.jit
def _dds_sbsc_kernel(
        A: tl.tensor, B: tl.tensor, C: tl.tensor, M, N, K, O_M, O_N,
        stride_ac, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cc, stride_cm, stride_cn,
        block_offset: tl.constexpr,
        fuse_srgb: tl.constexpr, clamp_output: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
    ):

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)

    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    start_row = block_offset * pid_n

    A_block_ptr = tl.make_block_ptr(
        base=A + pid_c * stride_ac, shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, start_row),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1)
    )

    B_block_ptr = tl.make_block_ptr(
        base=B + pid_n * stride_bb, shape=(BLOCK_SIZE, BLOCK_N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    for block_slice in range(nsub_blocks):
        a = tl.load(A_block_ptr, eviction_policy='evict_first', boundary_check=(0,), padding_option='zero')
        b = tl.load(B_block_ptr, eviction_policy='evict_last')

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        A_block_ptr = A_block_ptr.advance((0, BLOCK_K))
        B_block_ptr = B_block_ptr.advance((BLOCK_K, 0))

    if fuse_srgb:
        acc = linear_to_srgb_triton(acc)

    if clamp_output:
        acc = tl.clamp(acc, 0.0, 1.0)

    acc = acc.to(C.dtype.element_ty)

    C_block_ptr = tl.make_block_ptr(
        base=C + pid_c * stride_cc, shape=(O_M, O_N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    tl.store(C_block_ptr, acc, boundary_check=(0, 1), cache_modifier='.cs')

def triton_dds_sbsc(
        lhs: torch.Tensor,
        rhs: SBSCMatrix,
        fuse_srgb: bool = False,
        clamp_output: bool = False,
        output_mt: bool = False,
        output_slice: None | Tuple[int,int] = None
    ):
    assert isinstance(lhs, torch.Tensor)
    assert isinstance(rhs, SBSCMatrix)
    assert lhs.ndim == 3
    CH = lhs.shape[0]
    stride_ac = lhs.stride(0)

    M, K = lhs.shape[-2:]
    N = rhs.size[-1]

    if output_mt:
        if output_slice is not None:
            O_N, O_M = output_slice
            out = torch.empty(
                (*lhs.shape[:-2], *output_slice),
                dtype=lhs.dtype,
                device=lhs.device
            )
        else:
            O_N, O_M = N, M
            out = torch.empty(
                (*lhs.shape[:-2], rhs.size[1], lhs.shape[-2]),
                dtype=lhs.dtype,
                device=lhs.device
            )
        stride_cm, stride_cn = out.stride(-1), out.stride(-2)
        stride_cc = out.stride(-3)
    else:
        if output_slice is not None:
            O_M, O_N = output_slice
            out = torch.empty(
                (*lhs.shape[:-2], *output_slice),
                dtype=lhs.dtype,
                device=lhs.device
            )
        else:
            O_M, O_N = M, N
            out = torch.empty(
                (*lhs.shape[:-1], rhs.size[1]),
                dtype=lhs.dtype,
                device=lhs.device
            )
        stride_cm, stride_cn = out.stride(-2), out.stride(-1)
        stride_cc = out.stride(-3)

    assert lhs.shape[-1] <= rhs.size[0], f"incompatible dimensions: {lhs.shape[-1]} > {rhs.size[0]}"

    stride_am, stride_ak = lhs.stride(-2), lhs.stride(-1)

    stride_bb, stride_bk, stride_bn = rhs.data.stride(0), rhs.data.stride(1), rhs.data.stride(2)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), triton.cdiv(M, META['BLOCK_M']), CH)

    _dds_sbsc_kernel[grid](
        lhs, rhs.data, out, M, N, K, O_M, O_N,
        stride_ac, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cc, stride_cm, stride_cn,
        rhs.offset, fuse_srgb, clamp_output,
        GROUP_M=32, ACC_TYPE=tl.float16, BLOCK_M=32,
        BLOCK_N=rhs.data.shape[2], BLOCK_SIZE=rhs.data.shape[1], BLOCK_K=rhs.col_block_size
    )
    return out

from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['BLOCK_SIZE', 'BLOCK_N'],
)
@triton.jit
def _dds_sbsc_zerorhs_kernel(
        A: tl.tensor, C: tl.tensor, M, N, K, O_M, O_N,
        stride_ac, stride_am, stride_ak,
        stride_cc, stride_cm, stride_cn,
        k, PAD, block_offset: tl.constexpr,
        fuse_srgb: tl.constexpr, gamma_correction: tl.constexpr, clamp_output: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)

    nsub_blocks = triton.cdiv(BLOCK_SIZE, BLOCK_K)

    start_row = block_offset * pid_n

    offs_k = (start_row + tl.arange(0, BLOCK_K)) * stride_ak
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    A_mask = (m_range < M)[None, :].broadcast_to(BLOCK_K, BLOCK_M)

    A_M_ptr = A + pid_c * stride_ac + stride_am * m_range

    b_k = ((start_row - PAD + tl.arange(0, BLOCK_K)).to(tl.float32) + 0.5) * k
    b_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.float32) + 0.5

    b_base = (b_k[None, :] - b_n[:, None])

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float16)

    for _ in tl.range(nsub_blocks):
        A_ptr = A_M_ptr[None, :] + tl.minimum(tl.maximum(offs_k, PAD) - PAD, K - 1)[:, None]

        b = magic_kernel_sharp_2021_triton(b_base) * k

        b = b.to(tl.float16)

        a = tl.load(A_ptr, mask=A_mask)

        if fuse_srgb == 'input':
            if gamma_correction == 'fast':
                a = libdevice.fast_powf(a, 2.2).to(tl.float16)
            elif gamma_correction == 'srgb':
                a = srgb_to_linear_triton(a).to(tl.float16)

        acc = tl.dot(b, a, acc, out_dtype=tl.float16)

        offs_k += BLOCK_K * stride_ak
        b_base += BLOCK_K * k

    if fuse_srgb == 'output':
        if gamma_correction == 'fast':
            acc = libdevice.fast_powf(acc, 1.0/2.2)
        elif gamma_correction == 'srgb':
            acc = linear_to_srgb_triton(acc)

    if clamp_output:
        acc = tl.clamp(acc, 0.0, 1.0)

    if fuse_srgb == 'output' or clamp_output:
        acc = acc.to(C.dtype.element_ty)

    C_block_ptr = tl.make_block_ptr(
        base=C + pid_c * stride_cc, shape=(O_N, O_M),
        strides=(stride_cn, stride_cm),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    tl.store(C_block_ptr, acc, boundary_check=(0, 1), cache_modifier='.cs')

import math


def triton_dds_zerorhs_sbsc(
        lhs: torch.Tensor,
        target_size: int,
        source_size: int,
        kernel_window: float,
        block_specs,
        fuse_srgb: str = '',
        gamma_correction: str = 'fast',
        clamp_output: bool = False,
        output_mt: bool = False,
        output_slice: None | Tuple[int,int] = None
    ):
    assert isinstance(lhs, torch.Tensor)

    assert fuse_srgb in ['input', 'output', '']
    assert gamma_correction in ['fast', 'srgb']

    k = target_size / source_size

    PAD = math.ceil((kernel_window - 0.5) / k)

    offset, block_height, num_blocks, col_width = block_specs

    assert lhs.ndim == 3
    CH = lhs.shape[0]
    stride_ac = lhs.stride(0)

    M, K = lhs.shape[-2:]
    N = target_size

    if output_mt:
        if output_slice is not None:
            O_N, O_M = output_slice
            out = torch.empty(
                (*lhs.shape[:-2], *output_slice),
                dtype=lhs.dtype,
                device=lhs.device
            )
        else:
            O_N, O_M = N, M
            out = torch.empty(
                (*lhs.shape[:-2], N, M),
                dtype=lhs.dtype,
                device=lhs.device
            )
        stride_cm, stride_cn = out.stride(-1), out.stride(-2)
        stride_cc = out.stride(-3)
    else:
        if output_slice is not None:
            O_M, O_N = output_slice
            out = torch.empty(
                (*lhs.shape[:-2], *output_slice),
                dtype=lhs.dtype,
                device=lhs.device
            )
        else:
            O_M, O_N = M, N
            out = torch.empty(
                (*lhs.shape[:-2], M, N),
                dtype=lhs.dtype,
                device=lhs.device
            )
        stride_cm, stride_cn = out.stride(-2), out.stride(-1)
        stride_cc = out.stride(-3)

    stride_am, stride_ak = lhs.stride(-2), lhs.stride(-1)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), triton.cdiv(M, META['BLOCK_M']), CH)

    _dds_sbsc_zerorhs_kernel[grid](
        lhs, out, M, N, K, O_M, O_N,
        stride_ac, stride_am, stride_ak,
        stride_cc, stride_cm, stride_cn,
        k, PAD, offset, fuse_srgb, gamma_correction, clamp_output,
        BLOCK_M=32, BLOCK_K=16, BLOCK_N=col_width, BLOCK_SIZE=block_height,
    )
    return out
