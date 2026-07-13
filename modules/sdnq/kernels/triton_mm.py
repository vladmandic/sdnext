import os
import math
import torch

import triton
import triton.language as tl


min_block_size = int(os.environ.get("SDNQ_TRITON_MM_MIN_BLOCK_SIZE", "256"))
matmul_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM}, num_warps=w, num_stages=s)
    for BM in [int(BM) for BM in os.environ.get("SDNQ_TRITON_MM_BLOCK_SIZE_M_LIST", "64,128").replace(" ","").split(",")]
    for BN in [int(BN) for BN in os.environ.get("SDNQ_TRITON_MM_BLOCK_SIZE_N_LIST", "64,128,256").replace(" ","").split(",")]
    for BK in [int(BK) for BK in os.environ.get("SDNQ_TRITON_MM_BLOCK_SIZE_K_LIST", "32,64,128").replace(" ","").split(",")]
    for GM in [int(GM) for GM in os.environ.get("SDNQ_TRITON_MM_GROUP_SIZE_M_LIST", "8").replace(" ","").split(",")]
    for w in [int(w) for w in os.environ.get("SDNQ_TRITON_MM_NUM_WARPS_LIST", "16" if torch.xpu.is_available() else "4").replace(" ","").split(",")]
    for s in [int(s) for s in os.environ.get("SDNQ_TRITON_MM_NUM_STAGES_LIST", "2").replace(" ","").split(",")]
]


@triton.autotune(configs=matmul_configs, key=["b_is_contiguous", "bias_ndim", "M_AT", "N_AT", "K_AT", "a_dtype", "out_dtype"], cache_results=True)
@triton.jit
def sdnq_triton_mm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    b_is_contiguous: tl.constexpr,
    bias_ndim: tl.constexpr,
    M_AT: tl.constexpr, # pylint: disable=unused-argument
    N_AT: tl.constexpr, # pylint: disable=unused-argument
    K_AT: tl.constexpr, # pylint: disable=unused-argument
    a_dtype: tl.constexpr, # pylint: disable=unused-argument
    out_dtype: tl.constexpr, # pylint: disable=unused-argument
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(off_m >= 0)
    tl.assume(off_n >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(GROUP_SIZE_M > 0)
    tl.assume(b_is_contiguous == 0 or b_is_contiguous == 1) # pylint: disable=consider-using-in
    tl.assume(bias_ndim >= 0 and bias_ndim <= 2) # pylint: disable=consider-using-in

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(K, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    if b_is_contiguous:
        b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(N, 1), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    else:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_bn = (off_n + tl.arange(0, BLOCK_SIZE_N)) % N
        b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * K)

    off_k = 0
    accumulator_dtype: tl.constexpr = tl.int32 if a_ptr.type.element_ty == tl.int8 else tl.float32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([off_m, off_k])
        if b_is_contiguous:
            b = b_desc.load([off_k, off_n])
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - off_k, other=0.0)
            b_ptrs += BLOCK_SIZE_K
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
        off_k += BLOCK_SIZE_K

    if bias_ndim == 1:
        accumulator = accumulator.to(tl.float32)
        bias_desc = tl.make_tensor_descriptor(base=bias_ptr, shape=(N,), strides=(1,), block_shape=(BLOCK_SIZE_N,))
        bias = bias_desc.load([off_n])[None, :].to(tl.float32)
        accumulator += bias
    elif bias_ndim == 2:
        accumulator = accumulator.to(tl.float32)
        bias_desc = tl.make_tensor_descriptor(base=bias_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
        bias = bias_desc.load([off_m, off_n]).to(tl.float32)
        accumulator += bias

    accumulator = accumulator.to(c_ptr.type.element_ty)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([off_m, off_n], accumulator)


def sdnq_triton_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.FloatTensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if bias is not None:
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert bias.ndim in {1, 2}, "Bias must be 1D or 2D"
    M, K = a.shape
    K, N = b.shape
    if out_dtype is None:
        out_dtype = torch.int32 if a.dtype == torch.int8 else torch.float32
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    sdnq_triton_mm_kernel[grid](
        a, b, c, bias,
        M, N, K,
        (1 if b.is_contiguous() else 0),
        (0 if bias is None else bias.ndim),
        math.ceil(M / min_block_size),
        math.ceil(N / min_block_size),
        math.ceil(K / min_block_size),
        str(a.dtype), str(c.dtype),
    )
    return c
