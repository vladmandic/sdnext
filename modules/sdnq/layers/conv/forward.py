# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Optional

import torch


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
