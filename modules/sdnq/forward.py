# pylint: disable=redefined-builtin,no-member,protected-access

from typing import Callable, List, Tuple, Optional
import torch
from modules import shared

from .common import conv_types, conv_transpose_types
from .dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias
from .packed_int import unpack_int_symetric


def get_forward_func(layer_class_name: str, use_quantized_matmul: bool, is_integer: bool, use_tensorwise_fp8_matmul: bool) -> Callable: # pylint: disable=inconsistent-return-statements
    if layer_class_name in conv_types:
        if use_quantized_matmul:
            if is_integer:
                return quantized_conv_forward_int8_matmul
            else:
                if use_tensorwise_fp8_matmul:
                    return quantized_conv_forward_fp8_matmul_tensorwise
                else:
                    return quantized_conv_forward_fp8_matmul
        else:
            return quantized_conv_forward
    elif layer_class_name in conv_transpose_types:
        if layer_class_name.endswith("1d"):
            return quantized_conv_transpose_1d_forward
        elif layer_class_name.endswith("2d"):
            return quantized_conv_transpose_2d_forward
        elif layer_class_name.endswith("3d"):
            return quantized_conv_transpose_3d_forward
    else:
        if use_quantized_matmul:
            if is_integer:
                return quantized_linear_forward_int8_matmul
            else:
                if use_tensorwise_fp8_matmul:
                    return quantized_linear_forward_fp8_matmul_tensorwise
                else:
                    return quantized_linear_forward_fp8_matmul
        else:
            return quantized_linear_forward


def quantize_fp8_matmul_input(input: torch.FloatTensor) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous()
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(448)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    input_scale = input_scale.to(dtype=torch.float32)
    return input, input_scale


def quantize_fp8_matmul_input_tensorwise(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous()
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(448)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous()
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(127)
    input = torch.div(input, input_scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def fp8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    input, input_scale = quantize_fp8_matmul_input(input)
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=return_dtype).reshape(output_shape)


# sm89 doesn't support row wise scale in Windows
def fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)
    input, scale = quantize_fp8_matmul_input_tensorwise(input, scale)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, return_dtype, output_shape)


def int8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    quantized_weight_shape: torch.Size,
    weights_dtype: str,
) -> torch.FloatTensor:
    if quantized_weight_shape is not None:
        weight = unpack_int_symetric(weight, quantized_weight_shape, weights_dtype, dtype=torch.int8, transpose=True)
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    input, scale = quantize_int8_matmul_input(input, scale)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._int_mm(input, weight), scale, return_dtype, output_shape)


def process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation):
    if conv_type == 1:
        batch_size, _, L_in = input.shape
        C_out, _, K_l = result_shape
        L_out = (L_in + 2 * padding[1] - dilation[1] * (K_l - 1) - 1) // stride[1] + 1
        mm_output_shape = (batch_size, L_out, C_out)
        kernel_size = (1, K_l)
    if conv_type == 2:
        batch_size, _, H_in, W_in = input.shape
        C_out, _, K_h, K_w = result_shape
        H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1
        mm_output_shape = (batch_size, H_out, W_out, C_out)
        kernel_size = (K_h, K_w)
    else:
        batch_size, _, D_in, H_in, W_in = input.shape
        C_out, _, K_d, K_h, K_w = result_shape
        D_out = (D_in + 2 * padding[0] - dilation[0] * (K_d - 1) - 1) // stride[0] + 1
        H_out = (H_in + 2 * padding[1] - dilation[1] * (K_h - 1) - 1) // stride[1] + 1
        W_out = (W_in + 2 * padding[2] - dilation[2] * (K_w - 1) - 1) // stride[2] + 1
        mm_output_shape = (batch_size, D_out, H_out, W_out, C_out)
        kernel_size = (K_d, K_h, K_w)

    if padding_mode != "zeros":
        input = torch.nn.functional.pad(input, reversed_padding_repeated_twice, mode=padding_mode)
        padding = (0,) * (conv_type if conv_type != 1 else 2)
    elif conv_type == 3:
        input = torch.nn.functional.pad(input, reversed_padding_repeated_twice)

    if conv_type == 1:
        input = input.unsqueeze(2)

    if conv_type == 3:
        K_D_eff = K_d + (K_d - 1) * (dilation[0] - 1)
        K_H_eff = K_h + (K_h - 1) * (dilation[0] - 1)
        K_W_eff = K_w + (K_w - 1) * (dilation[0] - 1)
        input = input.unfold(2, K_D_eff, stride[0]).unfold(3, K_H_eff, stride[1]).unfold(4, K_W_eff, stride[2])
        if dilation[0] > 1:
            input = input[..., ::dilation[0], :, :]
        if dilation[1] > 1:
            input = input[..., ::dilation[1], :]
        if dilation[2] > 1:
            input = input[..., ::dilation[2]]
        input = input.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(batch_size, D_out * H_out * W_out, -1)
    else:
        input = torch.nn.functional.unfold(input, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation).transpose(1,2)
    return input, mm_output_shape


def conv_fp8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    result_shape: torch.Size,
    reversed_padding_repeated_twice: List[int],
    padding_mode: str, conv_type: int,
    groups: int, stride: List[int],
    padding: List[int], dilation: List[int],
) -> torch.FloatTensor:
    return_dtype = input.dtype
    input, mm_output_shape = process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation)
    input, input_scale = quantize_fp8_matmul_input(input)

    if groups == 1:
        result = torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=return_dtype).reshape(mm_output_shape)
    else:
        scale = scale.reshape(groups, 1, scale.shape[1] // groups)
        input_scale = input_scale.reshape(groups, input_scale.shape[0] // groups, 1)
        weight = weight.reshape(weight.shape[0], groups, weight.shape[1] // groups).transpose(0,1)
        input = input.reshape(input.shape[0], groups, input.shape[1] // groups).transpose(0,1)
        result = []
        if bias is not None:
            bias = bias.reshape(groups, bias.shape[0] // groups)
            for i in range(groups):
                result.append(torch._scaled_mm(input[i], weight[i], scale_a=input_scale[i], scale_b=scale[i], bias=bias[i], out_dtype=return_dtype))
        else:
            for i in range(groups):
                result.append(torch._scaled_mm(input[i], weight[i], scale_a=input_scale[i], scale_b=scale[i], bias=None, out_dtype=return_dtype))
        result = torch.cat(result, dim=-1).reshape(mm_output_shape)

    if conv_type == 1:
        result = result.transpose(1,2)
    elif conv_type == 2:
        result = result.permute(0,3,1,2)
    elif conv_type == 3:
        result = result.permute(0,4,1,2,3)
    return result


def conv_fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    result_shape: torch.Size,
    reversed_padding_repeated_twice: List[int],
    padding_mode: str, conv_type: int,
    groups: int, stride: List[int],
    padding: List[int], dilation: List[int],
) -> torch.FloatTensor:
    return_dtype = input.dtype
    input, mm_output_shape = process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation)
    input, scale = quantize_fp8_matmul_input_tensorwise(input, scale)
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)

    if groups == 1:
        result = torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype)
    else:
        weight = weight.reshape(weight.shape[0], groups, weight.shape[1] // groups).transpose(0,1)
        input = input.reshape(input.shape[0], groups, input.shape[1] // groups).transpose(0,1)
        result = []
        for i in range(groups):
            result.append(torch._scaled_mm(input[i], weight[i], scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype))
        result = torch.cat(result, dim=-1)
    if bias is not None:
        dequantize_symmetric_with_bias(result, scale, bias, return_dtype, mm_output_shape)
    else:
        dequantize_symmetric(result, scale, return_dtype, mm_output_shape)

    if conv_type == 1:
        result = result.transpose(1,2)
    elif conv_type == 2:
        result = result.permute(0,3,1,2)
    elif conv_type == 3:
        result = result.permute(0,4,1,2,3)
    return result


def conv_int8_matmul(
    input: torch.FloatTensor,
    weight: torch.CharTensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    result_shape: torch.Size,
    quantized_weight_shape: torch.Size,
    weights_dtype: str,
    reversed_padding_repeated_twice: List[int],
    padding_mode: str, conv_type: int,
    groups: int, stride: List[int],
    padding: List[int], dilation: List[int],
) -> torch.FloatTensor:
    return_dtype = input.dtype
    input, mm_output_shape = process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation)
    input, scale = quantize_int8_matmul_input(input, scale)
    if quantized_weight_shape is not None:
        weight = unpack_int_symetric(weight, quantized_weight_shape, weights_dtype, dtype=torch.int8, transpose=True)

    if groups == 1:
        result = torch._int_mm(input, weight)
    else:
        weight = weight.reshape(weight.shape[0], groups, weight.shape[1] // groups).transpose(0,1)
        input = input.reshape(input.shape[0], groups, input.shape[1] // groups).transpose(0,1)
        result = []
        for i in range(groups):
            result.append(torch._int_mm(input[i], weight[i]))
        result = torch.cat(result, dim=-1)
    if bias is not None:
        result = dequantize_symmetric_with_bias(result, scale, bias, return_dtype, mm_output_shape)
    else:
        result = dequantize_symmetric(result, scale, return_dtype, mm_output_shape)

    if conv_type == 1:
        result = result.transpose(1,2)
    elif conv_type == 2:
        result = result.permute(0,3,1,2)
    elif conv_type == 3:
        result = result.permute(0,4,1,2,3)
    return result


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul(input, self.weight, self.bias, self.sdnq_dequantizer.scale)


def quantized_linear_forward_fp8_matmul_tensorwise(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul_tensorwise(input, self.weight, self.bias, self.sdnq_dequantizer.scale)


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    return int8_matmul(input, self.weight, self.bias, self.sdnq_dequantizer.scale, getattr(self.sdnq_dequantizer, "quantized_weight_shape", None), self.sdnq_dequantizer.weights_dtype)


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight), self.bias)


def get_conv_args(input_ndim: int, stride, padding, dilation):
    if input_ndim == 3:
        conv_type = 1
    elif input_ndim == 4:
        conv_type = 2
    else:
        conv_type = 3
    if isinstance(stride, int):
        stride = (stride,) * conv_type
    if isinstance(padding, int):
        padding = (padding,) * conv_type
    if isinstance(dilation, int):
        dilation = (dilation,) * conv_type
    if conv_type == 1:
        stride = (1, stride[0])
        padding = (0, padding[0])
        dilation = (1, dilation[0])
    return conv_type, stride, padding, dilation


def quantized_conv_forward_fp8_matmul(self, input) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[2] < 32:
        return self._conv_forward(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    conv_type, stride, padding, dilation = get_conv_args(input.ndim, self.stride, self.padding, self.dilation)
    return conv_fp8_matmul(
        input, self.weight, self.bias,
        self.sdnq_dequantizer.scale,
        self.sdnq_dequantizer.result_shape,
        self._reversed_padding_repeated_twice,
        self.padding_mode, conv_type,
        self.groups, stride, padding, dilation,
    )


def quantized_conv_forward_fp8_matmul_tensorwise(self, input) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[2] < 32:
        return self._conv_forward(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    conv_type, stride, padding, dilation = get_conv_args(input.ndim, self.stride, self.padding, self.dilation)
    return conv_fp8_matmul_tensorwise(
        input, self.weight, self.bias,
        self.sdnq_dequantizer.scale,
        self.sdnq_dequantizer.result_shape,
        self._reversed_padding_repeated_twice,
        self.padding_mode, conv_type,
        self.groups, stride, padding, dilation,
    )


def quantized_conv_forward_int8_matmul(self, input) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[2] < 32:
        return self._conv_forward(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    conv_type, stride, padding, dilation = get_conv_args(input.ndim, self.stride, self.padding, self.dilation)
    return conv_int8_matmul(
        input, self.weight, self.bias,
        self.sdnq_dequantizer.scale,
        self.sdnq_dequantizer.result_shape,
        getattr(self.sdnq_dequantizer, "quantized_weight_shape", None),
        self.sdnq_dequantizer.weights_dtype,
        self._reversed_padding_repeated_twice,
        self.padding_mode, conv_type,
        self.groups, stride, padding, dilation,
    )


def quantized_conv_forward(self, input) -> torch.FloatTensor:
    return self._conv_forward(input, self.sdnq_dequantizer(self.weight), self.bias)


def quantized_conv_transpose_1d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation)
    return torch.nn.functional.conv_transpose1d(input, self.sdnq_dequantizer(self.weight), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


def quantized_conv_transpose_2d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation)
    return torch.nn.functional.conv_transpose2d(input, self.sdnq_dequantizer(self.weight), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


def quantized_conv_transpose_3d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation)
    return torch.nn.functional.conv_transpose3d(input, self.sdnq_dequantizer(self.weight), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


if shared.opts.sdnq_dequantize_compile:
    try:
        torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.accumulated_recompile_limit = max(8192, torch._dynamo.config.accumulated_recompile_limit)
        int8_matmul = torch.compile(int8_matmul, fullgraph=True, dynamic=False)
        fp8_matmul = torch.compile(fp8_matmul, fullgraph=True, dynamic=False)
        fp8_matmul_tensorwise = torch.compile(fp8_matmul_tensorwise, fullgraph=True, dynamic=False)
        conv_int8_matmul = torch.compile(conv_int8_matmul, fullgraph=True, dynamic=False)
        conv_fp8_matmul = torch.compile(conv_fp8_matmul, fullgraph=True, dynamic=False)
        conv_fp8_matmul_tensorwise = torch.compile(conv_fp8_matmul_tensorwise, fullgraph=True, dynamic=False)
    except Exception as e:
        shared.log.warning(f"Quantization: type=sdnq MatMul using torch.compile is not available: {e}")
