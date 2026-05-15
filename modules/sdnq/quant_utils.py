import torch

from modules import devices
from .common import dtype_dict, use_contiguous_mm


@devices.inference_context()
def get_scale_asymmetric(weight: torch.FloatTensor, reduction_axes: int | list[int], weights_dtype: str) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point = torch.amin(weight, dim=reduction_axes, keepdims=True)
    scale = torch.amax(weight, dim=reduction_axes, keepdims=True).sub_(zero_point).div_(dtype_dict[weights_dtype]["max"] - dtype_dict[weights_dtype]["min"])
    if dtype_dict[weights_dtype]["min"] != 0:
        zero_point.sub_(torch.mul(scale, dtype_dict[weights_dtype]["min"]))
    return scale, zero_point


@devices.inference_context()
def get_scale_symmetric(weight: torch.FloatTensor, reduction_axes: int | list[int], weights_dtype: str) -> torch.FloatTensor:
    return torch.amax(weight.abs(), dim=reduction_axes, keepdims=True).div_(dtype_dict[weights_dtype]["max"])


@devices.inference_context()
def quantize_weight(weight: torch.FloatTensor, reduction_axes: int | list[int], weights_dtype: str, dtype: torch.dtype = None, use_stochastic_rounding: bool = False) -> tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if weight.dtype != torch.float64:
        weight = weight.to(dtype=torch.float32)

    if dtype_dict[weights_dtype]["is_unsigned"]:
        scale, zero_point = get_scale_asymmetric(weight, reduction_axes, weights_dtype)
        if dtype is not None:
            scale = scale.to(dtype=dtype)
            zero_point = zero_point.to(dtype=dtype)
        quantized_weight = torch.sub(weight, zero_point).div_(scale)
    else:
        scale = get_scale_symmetric(weight, reduction_axes, weights_dtype)
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
def prepare_weight_for_matmul(weight: torch.Tensor) -> torch.Tensor:
    if use_contiguous_mm:
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t_().contiguous().t_()
    return weight


@devices.inference_context()
def prepare_svd_for_matmul(svd_up: torch.FloatTensor, svd_down: torch.FloatTensor, use_quantized_matmul: bool) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    if svd_up is not None:
        if use_quantized_matmul:
            svd_up = prepare_weight_for_matmul(svd_up)
        else:
            svd_up = svd_up.contiguous()
    if svd_down is not None:
        svd_down = prepare_weight_for_matmul(svd_down)
    return svd_up, svd_down
