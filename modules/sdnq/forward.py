# pylint: disable=protected-access

from collections.abc import Callable

from .common import dtype_dict, conv_types, conv_transpose_types, use_tensorwise_fp8_matmul


def get_forward_func(layer_class_name: str, quantized_matmul_dtype: str, use_quantized_matmul: bool) -> Callable: # pylint: disable=inconsistent-return-statements
    if layer_class_name in conv_types:
        if use_quantized_matmul:
            if dtype_dict[quantized_matmul_dtype]["is_integer"]:
                from .layers.conv.conv_int8 import quantized_conv_forward_int8_matmul
                return quantized_conv_forward_int8_matmul
            else:
                if dtype_dict[quantized_matmul_dtype]["num_bits"] == 8:
                    if use_tensorwise_fp8_matmul:
                        from .layers.conv.conv_fp8_tensorwise import quantized_conv_forward_fp8_matmul_tensorwise
                        return quantized_conv_forward_fp8_matmul_tensorwise
                    else:
                        from .layers.conv.conv_fp8 import quantized_conv_forward_fp8_matmul
                        return quantized_conv_forward_fp8_matmul
                else:
                    from .layers.conv.conv_fp16 import quantized_conv_forward_fp16_matmul
                    return quantized_conv_forward_fp16_matmul
        else:
            from .layers.conv.forward import quantized_conv_forward
            return quantized_conv_forward
    elif layer_class_name in conv_transpose_types:
        if layer_class_name.endswith("1d"):
            from .layers.conv.forward import quantized_conv_transpose_1d_forward
            return quantized_conv_transpose_1d_forward
        elif layer_class_name.endswith("2d"):
            from .layers.conv.forward import quantized_conv_transpose_2d_forward
            return quantized_conv_transpose_2d_forward
        elif layer_class_name.endswith("3d"):
            from .layers.conv.forward import quantized_conv_transpose_3d_forward
            return quantized_conv_transpose_3d_forward
    else:
        if use_quantized_matmul:
            if dtype_dict[quantized_matmul_dtype]["is_integer"]:
                from .layers.linear.linear_int8 import quantized_linear_forward_int8_matmul
                return quantized_linear_forward_int8_matmul
            else:
                if dtype_dict[quantized_matmul_dtype]["num_bits"] == 8:
                    if use_tensorwise_fp8_matmul:
                        from .layers.linear.linear_fp8_tensorwise import quantized_linear_forward_fp8_matmul_tensorwise
                        return quantized_linear_forward_fp8_matmul_tensorwise
                    else:
                        from .layers.linear.linear_fp8 import quantized_linear_forward_fp8_matmul
                        return quantized_linear_forward_fp8_matmul
                else:
                    from .layers.linear.linear_fp16 import quantized_linear_forward_fp16_matmul
                    return quantized_linear_forward_fp16_matmul
        else:
            from .layers.linear.forward import quantized_linear_forward
            return quantized_linear_forward
