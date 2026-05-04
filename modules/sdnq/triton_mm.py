"""
Modified from Triton MatMul example.
PyTorch torch._int_mm is broken on backward pass with Nvidia, so we use Triton on the backward pass with Nvidia.
AMD RDNA2 doesn't support torch._int_mm as it requires INT8 WMMA, so we use INT8 DP4A via Triton.
PyTorch doesn't support FP32 output type with FP16 MM, so we use Triton for FP16 MM too.
matmul_configs we use takes AMD and Intel into consideration too.
SDNQ Triton configs can outperform RocBLAS and OneDNN.
"""

import torch

import triton
import triton.language as tl


matmul_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM}, num_warps=w, num_stages=s)
    for BM in [64, 128, 256]
    for BN in [64, 128, 256]
    for BK in [64, 128]
    for GM in [2, 4, 8]
    for w in [2, 4, 8]
    for s in [2]
]


@triton.autotune(configs=matmul_configs, key=["M", "N", "K", "stride_bk", "ACCUMULATOR_DTYPE"], cache_results=True)
@triton.jit
def triton_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M: int, N: int, K: int,
    stride_am: int, stride_ak: int,
    stride_bk: int, stride_bn: int,
    stride_cm: int, stride_cn: int,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=ACCUMULATOR_DTYPE)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# Intel requires tensor descriptors to perform good
@triton.autotune(configs=matmul_configs, key=["M", "N", "K", "stride_bk", "ACCUMULATOR_DTYPE"], cache_results=True)
@triton.jit
def triton_mm_td_kernel(
    a_ptr, b_ptr, c_ptr,
    M: int, N: int, K: int,
    stride_am: int, stride_ak: int,
    stride_bk: int, stride_bn: int,
    stride_cm: int, stride_cn: int,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))

    off_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator = tl.dot(a, b, accumulator, out_dtype=ACCUMULATOR_DTYPE)
        off_k += BLOCK_SIZE_K

    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator)


def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    mm_kernel_func = triton_mm_td_kernel if b.is_contiguous() else triton_mm_kernel
    mm_kernel_func[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        tl.int32,
    )
    return c


def fp_mm(a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    mm_kernel_func = triton_mm_td_kernel if b.is_contiguous() else triton_mm_kernel
    mm_kernel_func[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        tl.float32,
    )
    return c
