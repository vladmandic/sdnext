# pylint: disable=redefined-builtin

import torch

from modules import devices
from .common import dtype_dict, compile_func, conv_types, conv_transpose_types
from .kernel_wrappers import use_contiguous_int8_mm, use_contiguous_fp16_mm, use_contiguous_fp8_mm
from .utils import is_pow2, is_pow4, next_power_of_2


@devices.inference_context()
def get_scale_asymmetric(weight: torch.FloatTensor, dim: int | list[int], weights_dtype: str) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point, scale = torch.aminmax(weight, dim=dim, keepdims=True)
    scale = scale.sub_(zero_point).div_(dtype_dict[weights_dtype]["max"] - dtype_dict[weights_dtype]["min"])
    if dtype_dict[weights_dtype]["min"] != 0:
        zero_point.sub_(scale, alpha=dtype_dict[weights_dtype]["min"])
    return scale, zero_point


@devices.inference_context()
def get_scale_symmetric(weight: torch.FloatTensor, dim: int | list[int], weights_dtype: str) -> torch.FloatTensor:
    return torch.amax(weight.abs(), dim=dim, keepdims=True).div_(dtype_dict[weights_dtype]["max"])


@devices.inference_context()
def quantize_weight(weight: torch.FloatTensor, dim: int | list[int], weights_dtype: str, dtype: torch.dtype = None, use_stochastic_rounding: bool = False) -> tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if weight.dtype != torch.float64:
        weight = weight.to(dtype=torch.float32, copy=False)

    if dtype_dict[weights_dtype]["is_unsigned"]:
        scale, zero_point = get_scale_asymmetric(weight, dim, weights_dtype)
        if dtype is not None:
            scale = scale.to(dtype=dtype)
            zero_point = zero_point.to(dtype=dtype)
        quantized_weight = torch.sub(weight, zero_point).div_(scale)
    else:
        scale = get_scale_symmetric(weight, dim, weights_dtype)
        zero_point = None
        if dtype is not None:
            scale = scale.to(dtype=dtype)
        quantized_weight = torch.div(weight, scale)

    if dtype_dict[weights_dtype]["is_integer"]:
        if use_stochastic_rounding:
            quantized_weight.add_(torch.randn_like(quantized_weight), alpha=0.1)
        quantized_weight.round_()
    else:
        if use_stochastic_rounding:
            mantissa_difference = 1 << (23 - dtype_dict[weights_dtype]["mantissa"])
            quantized_weight = quantized_weight.to(dtype=torch.float32).view(dtype=torch.int32)
            quantized_weight = quantized_weight.add_(torch.randint_like(quantized_weight, low=0, high=mantissa_difference, dtype=torch.int32)).bitwise_and_(-mantissa_difference).view(dtype=torch.float32)
        quantized_weight.nan_to_num_()
    quantized_weight = quantized_weight.clamp_(dtype_dict[weights_dtype]["min"], dtype_dict[weights_dtype]["max"]).to(dtype_dict[weights_dtype]["torch_dtype"])
    return quantized_weight, scale, zero_point


@devices.inference_context()
def fit_codebook(values: torch.FloatTensor, num_levels: int, iterations: int = 24) -> torch.FloatTensor:
    # 1D lloyd-max: initialized from the uniform min/max grid so the fitted MSE never
    # exceeds the matching affine grid's; fixed iteration count, no host syncs, meta-safe
    values = values.reshape(-1)
    w_min, w_max = values.min(), values.max()
    levels = torch.arange(num_levels, dtype=values.dtype, device=values.device).div_(num_levels - 1).mul_(w_max - w_min).add_(w_min)
    ones = torch.ones_like(values)
    for _ in range(iterations):
        midpoints = levels[1:].add(levels[:-1]).mul_(0.5)
        assignment = torch.bucketize(values, midpoints)
        sums = torch.zeros_like(levels).scatter_add_(0, assignment, values)
        counts = torch.zeros_like(levels).scatter_add_(0, assignment, ones)
        occupied = counts > 0
        levels = torch.where(occupied, sums.div_(counts.clamp_(min=1)), levels) # empty clusters keep their previous level
    return torch.sort(levels).values


@devices.inference_context()
def quantize_weight_codebook(weight: torch.FloatTensor, dim: int | list[int], weights_dtype: str, dtype: torch.dtype = None) -> tuple[torch.Tensor, torch.FloatTensor, torch.CharTensor]:
    if weight.dtype != torch.float64:
        weight = weight.to(dtype=torch.float32, copy=False)
    scale = torch.amax(weight.abs(), dim=dim, keepdims=True).clamp_(min=torch.finfo(torch.float32).tiny) # a zero row must not poison the shared codebook
    weight = torch.div(weight, scale).nan_to_num_()
    levels = fit_codebook(weight, dtype_dict[weights_dtype]["max"] + 1)
    level_absmax = levels.abs().max().clamp(min=torch.finfo(torch.float32).tiny)
    codebook = levels.div(level_absmax).mul_(127.0).round_().clamp_(-127.0, 127.0).to(dtype=torch.int8)
    scale = scale.mul_(level_absmax / 127.0)
    if dtype is not None:
        scale = scale.to(dtype=dtype)
    snapped_levels = codebook.to(dtype=weight.dtype).mul_(level_absmax / 127.0) # assign against the int8-snapped levels, not the raw fit
    midpoints = snapped_levels[1:].add(snapped_levels[:-1]).mul_(0.5)
    quantized_weight = torch.bucketize(weight, midpoints).to(dtype=dtype_dict[weights_dtype]["torch_dtype"])
    return quantized_weight, scale, codebook


@devices.inference_context()
def apply_svdquant(weight: torch.FloatTensor, rank: int = 32, niter: int = 8, dtype: torch.dtype = None) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    reshape_weight = False
    if weight.ndim > 2: # convs
        reshape_weight = True
        weight_shape = weight.shape
        weight = weight.flatten(1,-1)
    if weight.dtype != torch.float64:
        weight = weight.to(dtype=torch.float32)
    U, S, svd_down = torch.svd_lowrank(weight, q=rank, niter=niter)
    svd_up = torch.mul(U, S.unsqueeze(0))
    svd_down = svd_down.t_()
    if dtype is not None:
        svd_up = svd_up.to(dtype=dtype)
        svd_down = svd_down.to(dtype=dtype)
    weight = weight.sub(torch.mm(svd_up, svd_down))
    if reshape_weight:
        weight = weight.unflatten(-1, (*weight_shape[1:],)) # pylint: disable=possibly-used-before-assignment
    return weight, svd_up, svd_down


@devices.inference_context()
def build_hadamard_n2(n: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.FloatTensor:
    current_size = 2
    H = H_N2 = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device)
    while current_size < n:
        H = torch.kron(H, H_N2)
        current_size *= 2
    H = H.div_(n**0.5)
    H = prepare_weight_for_matmul(H, matmul_dtype="float16")
    return H


@devices.inference_context()
def build_hadamard_n4(n: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.FloatTensor:
    current_size = 4
    H = H_N4 = torch.tensor([[ 1,  1,  1, -1], [ 1,  1, -1,  1], [ 1, -1,  1,  1], [-1,  1,  1,  1]], dtype=dtype, device=device)
    while current_size < n:
        H = torch.kron(H, H_N4)
        current_size *= 4
    H = H.div_(n**0.5)
    H = prepare_weight_for_matmul(H, matmul_dtype="float16")
    return H


@devices.inference_context()
def build_hadamard(n: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.FloatTensor:
    if is_pow4(n):
        return build_hadamard_n4(n, device=device, dtype=dtype)
    elif is_pow2(n):
        return build_hadamard_n2(n, device=device, dtype=dtype)
    else:
        raise RuntimeError(f"Hadamard Group Size must be a power of 2 but got {n}.")


# 256x256 Hadamard matrix is just 256 KB at FP32
# And is the exact same matrix on all model layers
# So we can safely cache a single one
HADAMARD_MATRIX_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.FloatTensor] = {}
@devices.inference_context()
def get_hadamard(n: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.FloatTensor:
    device = devices.normalize_device(device)
    H_key = (n, device, dtype)
    H = HADAMARD_MATRIX_CACHE.get(H_key, None)
    if H is None:
        H = build_hadamard(n, dtype=dtype, device=device)
        HADAMARD_MATRIX_CACHE[H_key] = H
    return H


@devices.inference_context()
def rotate_hadamard(weight: torch.Tensor, group_size: int = 256, hadamard: torch.FloatTensor | None = None, is_conv: bool = False) -> torch.Tensor:
    if hadamard is None:
        hadamard = get_hadamard(group_size, dtype=weight.dtype, device=weight.device)
    else:
        group_size = hadamard.shape[-1]
        if hadamard.dtype != weight.dtype:
            hadamard = hadamard.to(dtype=weight.dtype)
    if is_conv:
        weight_shape = list(weight.shape)[1:]
        weight = weight.flatten(1,-1)
    weight = weight.unflatten(-1, (-1,group_size))
    result = torch.matmul(weight, hadamard).flatten(-2,-1)
    del hadamard
    if is_conv:
        result = result.unflatten(-1, weight_shape)
    return result


def get_hadamard_group_size(channel_size: int, group_size: int) -> tuple[bool, int]:
    group_size = next_power_of_2(min(channel_size, group_size))
    if channel_size % group_size != 0:
        while channel_size % group_size != 0:
            group_size = group_size // 2
    use_hadamard = group_size >= 4
    return use_hadamard, group_size


@devices.inference_context()
def apply_hadamard(weight: torch.Tensor, group_size: int = 256, hadamard: torch.FloatTensor | None = None, layer_class_name: str | None = None) -> tuple[torch.Tensor, bool, int]:
    is_conv = False
    if hadamard is not None:
        group_size = hadamard.shape[-1]
    if layer_class_name in conv_types or layer_class_name in conv_transpose_types:
        is_conv = True
        channel_size = weight.shape[1]
    else:
        channel_size = weight.shape[-1]
    use_hadamard, group_size = get_hadamard_group_size(channel_size, group_size)
    if use_hadamard:
        if hadamard is not None and group_size != hadamard.shape[-1]:
            hadamard = None
        weight = rotate_hadamard(weight, group_size=group_size, hadamard=hadamard, is_conv=is_conv)
    return weight, use_hadamard, group_size


@devices.inference_context()
def prepare_weight_for_matmul(weight: torch.Tensor, matmul_dtype: str | None = "int8") -> torch.Tensor:
    if (
        (use_contiguous_int8_mm and matmul_dtype in {"int8", "uint8"})
        or (use_contiguous_fp16_mm and matmul_dtype in {"fp16", "float16"})
        or (use_contiguous_fp8_mm and matmul_dtype in {"fp8", "float8_e4m3fn"})
    ):
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t_().contiguous().t_()
    return weight


def merge_svd_quantized(
    svd_up: torch.FloatTensor | None,
    svd_down: torch.FloatTensor | None,
    svd_up_q: torch.CharTensor | None,
    svd_up_scale: torch.FloatTensor | None,
    svd_down_q: torch.CharTensor | None,
    svd_down_scale: torch.FloatTensor | None,
    dim_up: int,
    dim_down: int,
    dtype: torch.dtype,
) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None]:
    # rowwise-int8 side-channel factors dequantize with a single fp32 rounding and
    # concatenate behind the full-precision pair, so the downstream matmul sees one
    # factor set; the quantized channel is resident storage, the merge is transient
    if svd_up_q is None:
        return svd_up, svd_down
    if svd_up is not None:
        dtype = svd_up.dtype
    up = svd_up_q.to(dtype=torch.float32).mul_(svd_up_scale).to(dtype=dtype)
    down = svd_down_q.to(dtype=torch.float32).mul_(svd_down_scale).to(dtype=dtype)
    if svd_up is None:
        return up, down
    return torch.cat([svd_up, up], dim=dim_up), torch.cat([svd_down, down], dim=dim_down)


@devices.inference_context()
def prepare_svd_for_matmul(svd_up: torch.FloatTensor, svd_down: torch.FloatTensor, use_quantized_matmul: bool) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    if svd_up is not None:
        if use_quantized_matmul:
            svd_up = prepare_weight_for_matmul(svd_up, matmul_dtype="float16")
        else:
            svd_up = svd_up.contiguous()
    if svd_down is not None:
        svd_down = prepare_weight_for_matmul(svd_down, matmul_dtype="float16")
    return svd_up, svd_down


@devices.inference_context()
def quantize_int_mm(weight: torch.FloatTensor, dim: int = -1, hadamard: torch.FloatTensor | None = None, matmul_dtype: str = "int8", use_sr: bool = False) -> tuple[torch.Tensor, torch.FloatTensor]:
    if hadamard is not None:
        weight = rotate_hadamard(weight, hadamard=hadamard)
    scale = get_scale_symmetric(weight, dim, matmul_dtype)
    weight = torch.div(weight, scale)
    if use_sr:
        weight = weight.add_(torch.randn_like(weight), alpha=0.1)
    weight = weight.round_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return weight, scale


@devices.inference_context()
def quantize_uint_mm(weight: torch.FloatTensor, dim: int = -1, hadamard: torch.FloatTensor | None = None, matmul_dtype: str = "uint8", use_sr: bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    if hadamard is not None:
        weight = rotate_hadamard(weight, hadamard=hadamard)
    matmul_dtype = matmul_dtype.removeprefix("u")
    scale, zero_point = get_scale_asymmetric(weight, dim, matmul_dtype)
    weight = torch.sub(weight, zero_point).div_(scale)
    if use_sr:
        weight = weight.add_(torch.randn_like(weight), alpha=0.1)
    weight = weight.round_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return weight, scale, zero_point


@devices.inference_context()
def quantize_fp_mm(weight: torch.FloatTensor, dim: int = -1, hadamard: torch.FloatTensor | None = None, matmul_dtype: str = "float8_e4m3fn", use_sr: bool = False) -> tuple[torch.Tensor, torch.FloatTensor]:
    if hadamard is not None:
        weight = rotate_hadamard(weight, hadamard=hadamard)
    scale = get_scale_symmetric(weight, dim, matmul_dtype)
    if use_sr:
        mantissa_difference = 1 << (23 - dtype_dict[matmul_dtype]["mantissa"])
        weight = weight.to(dtype=torch.float32).view(dtype=torch.int32)
        weight = weight.add_(torch.randint_like(weight, low=0, high=mantissa_difference, dtype=torch.int32)).bitwise_and_(-mantissa_difference).view(dtype=torch.float32)
    weight = torch.div(weight, scale).nan_to_num_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return weight, scale


rotate_hadamard_compiled = compile_func(rotate_hadamard)
