"""
Modified from Triton MatMul example.
PyTorch torch._int_mm is broken on backward pass with Nvidia.
AMD RDNA2 doesn't support torch._int_mm, so we use int_mm via Triton.
PyTorch doesn't support FP32 output type with FP16 MM so we use Triton for it too.
"""

import torch

import triton
import triton.language as tl


def get_autotune_config():
    if triton.runtime.driver.active.get_current_target().backend == "cuda":
        return [
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  32, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N":  32, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
            triton.Config({"BLOCK_SIZE_M":  32, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
            #
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  32, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        ]
    else:
        return [
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  32, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N":  32, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_SIZE_M":  32, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K":  32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=2),
            #
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M":  64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N":  32, "BLOCK_SIZE_K":  64, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8),
            triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N":  64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4),
        ]


@triton.autotune(configs=get_autotune_config(), key=["M", "N", "K", "stride_bk"])
@triton.jit
def int_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def int_mm(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    int_mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


@triton.autotune(configs=get_autotune_config(), key=["M", "N", "K", "stride_bk"])
@triton.jit
def fp_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fp_mm(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    fp_mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
