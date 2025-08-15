import functools

import torch
import torch._inductor

from torch._inductor.select_algorithm import ExternKernelChoice, ChoiceCaller, autotune_select_algorithm, extern_kernels
from torch._inductor.utils import use_aten_gemm_kernels, use_cpp_gemm_template, use_max_autotune
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
from torch._inductor.codegen.cpp_utils import create_epilogue_with_attr
from torch._inductor.lowering import register_lowering, lowerings, view
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor import ir, mkldnn_ir
from torch._inductor.ir import TensorBox
from torch._inductor.virtualized import ops, V

lowerings.pop(extern_kernels.qlinear_pointwise)
del extern_kernels.qlinear_pointwise
aten_mkldnn_qlinear_unary = ExternKernelChoice(
    torch.ops.onednn.qlinear_pointwise,
    "onednn::qlinear_pointwise",
    has_out_variant=False,
    kernel_creator=mkldnn_ir.QLinearPointwisePT2E.create,
)


@register_lowering(torch.ops.onednn.qlinear_pointwise, type_promotion_kind=None)
@register_lowering(torch.ops.onednn.qlinear_pointwise.default, type_promotion_kind=None)
def qlinear_unary(
    x: TensorBox,
    x_scale,
    x_zp,
    packed_weight: TensorBox,
    w_scale: TensorBox,
    w_zp: TensorBox,
    bias: TensorBox,
    o_scale,
    o_zero_point,
    output_dtype,
    attr,
    scalars,
    algorithm,
    layout=None,
):
    assert packed_weight.get_dtype() is torch.int8, (
        "Only int8 weights are supported by oneDNN qlinear."
    )
    x_size = x.get_size()
    if len(x_size) > 2:
        # GEMM template needs 2D input, normalize input shape here
        x = view(x, [-1, x_size[-1]])
    if not isinstance(x_scale, ir.TensorBox):
        assert isinstance(x_scale, float)
        x_scale = V.graph.add_tensor_constant(
            torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
        )
    else:
        x_scale.realize()
        if all(dim == 1 for dim in x_scale.get_size()):
            # Corner-case discovered with LLaMA series.
            # If all outer dims of x_scale are 1, make it a 0D tensor.
            # Otherwise, epilogue creator will run into indexing issues.
            x_scale = view(x_scale, [])
        assert len(x_scale.get_size()) in [0, 1], "x_scale must be 0D or 1D"

    if x_zp is None:
        # If x_zp is None, x is int8 quantized per-tensor and its scale is not reshaped,
        # then the codegened code would segfault if we don't create a tensor for x_zp.
        # It's safe to do so since x is a symmetrically quantized int8 tensor.
        # Moreover, oneDNN qlinear API doesn't accept None value for zp
        x_zp = V.graph.add_tensor_constant(
            torch.tensor(0, dtype=torch.int32), name="x_zp"
        )
    if not isinstance(x_zp, ir.TensorBox):
        assert isinstance(x_zp, int)
        x_zp = V.graph.add_tensor_constant(
            torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
        )
    else:
        x_zp.realize()

    assert x_zp.get_numel() == 1, "x_zp is incompatible with oneDNN qlinear"

    # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
    # Refer to https://github.com/pytorch/pytorch/blob
    # /f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577
    if w_zp is None:
        # If w_zp is None, then it's a dummy tensor created to denote the
        # absence of a zero point, and thus w is int8 symmetrically quantized.
        # Moreover, oneDNN qlinear API doesn't accept None value for zp
        w_zp = V.graph.add_tensor_constant(
            torch.tensor(0, dtype=torch.int32), name="w_zp"
        )
    w_scale.realize()
    w_zp.realize()
    if w_zp.get_dtype() != torch.int32 and isinstance(
        ir.InputsKernel.unwrap_storage_for_input(w_zp),
        ir.ConstantBuffer,
    ):
        # W_zp might be a ConstantBuffer with int64, convert it to int32
        w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
        w_zp = V.graph.add_tensor_constant(
            torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
        )

    bias_dtype = None if bias is None else bias.get_dtype()
    choices: list[ChoiceCaller] = []

    if use_max_autotune():
        *_, layout, x, packed_weight = mm_args(
            x, packed_weight, layout=layout, out_dtype=output_dtype
        )

        if (
            # GEMM template currently only supports symmetrically quantized weights
            isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            )
            and torch.equal(
                torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                V.graph.constants[w_zp.get_name()],
            )
        ) and use_cpp_gemm_template(layout, x, packed_weight):
            W_tensor = V.graph.constants[packed_weight.get_name()].to_dense()
            weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
            weight_compens = V.graph.add_tensor_constant(
                weight_compens_tensor,
                name=packed_weight.get_name() + "_BMatrixCompens",
            )

            def epilogue_creator(input_buffer):
                # Epilogue to convert from s32 to f32 for u8s8f32
                assert output_dtype in [
                    torch.float32,
                    torch.bfloat16,
                    torch.uint8,
                    torch.int8,
                ]
                input_loader = input_buffer.make_loader()
                weight_compens_loader = weight_compens.make_loader()
                x_scale_loader = x_scale.make_loader()
                w_scale_loader = w_scale.make_loader()
                x_zp_loader = x_zp.make_loader()
                nonlocal bias
                bias_loader = None
                if bias is not None:
                    bias_loader = bias.make_loader()

                def inner_fn(index):
                    nonlocal bias
                    input = input_loader(index)
                    # MicroKernel Output is with int32
                    # cvt to FP32 before doing compensation
                    input = ops.to_dtype(input, torch.float32)
                    weight_compens_index = (index[-1],)
                    _x_scale = x_scale_loader(())
                    _x_zp = x_zp_loader(())
                    _w_scale = w_scale_loader(weight_compens_index)
                    _weight_compo = weight_compens_loader(weight_compens_index)

                    # Step 1: Compute s8s8->s32 or u8s8->s32 GEMM & then apply compensation

                    temp = ops.mul(
                        ops.mul(
                            input,
                            _x_scale,
                        ),
                        _w_scale,
                    )
                    # NOTE: We will apply compensation even if the x_zp is 0 for int8 quantization.
                    # That's because when torch.compile is invoked for dynamic quantization,
                    # x might coincidentally have such values that x_zp might be zero despite
                    # asymmetric quantization.
                    # Besides, if x_zp is dummy for int8 x, or if x is statically quantized,
                    # we'd still perform that redundant compute to avoid making the code messy
                    # because we discovered that redundant computation of compensation did not
                    # lead to performance degradation with the input shapes tested.
                    temp = ops.sub(
                        temp,
                        ops.mul(
                            ops.mul(
                                ops.mul(
                                    _x_scale,
                                    _w_scale,
                                ),
                                _x_zp,
                            ),
                            _weight_compo,
                        ),
                    )
                    # Step 2: add Bias if applicable
                    if bias is not None:
                        _bias = bias_loader(weight_compens_index)
                        nonlocal bias_dtype
                        assert bias_dtype in [torch.float32, torch.bfloat16]
                        if bias_dtype == torch.bfloat16:
                            _bias = ops.to_dtype(_bias, torch.float32)
                        temp = ops.add(temp, _bias)

                    return temp

                output_buf = ir.Pointwise(
                    device=input_buffer.get_device(),
                    dtype=torch.float32,  # Hardcode to FP32 for u8s8f32 & s8s8f32
                    inner_fn=inner_fn,
                    ranges=input_buffer.get_size(),
                )

                # Step 3: Doing the unary post op fusion
                if attr != "none":
                    output_buf = create_epilogue_with_attr(
                        output_buf, attr, scalars=scalars, algorithm=algorithm
                    )

                # Step 4: Cast output to Target Dtype
                if output_dtype == torch.bfloat16:
                    output_cast_loader = output_buf.make_loader()

                    def inner_fn_cast_output_to_bf16(index):
                        input = output_cast_loader(index)
                        return ops.to_dtype(input, output_dtype)

                    output_buf = ir.Pointwise(
                        device=output_buf.get_device_or_error(),
                        dtype=output_dtype,
                        inner_fn=inner_fn_cast_output_to_bf16,
                        ranges=output_buf.get_size(),
                    )
                elif output_dtype in [torch.uint8, torch.int8]:
                    from .lowering import _create_constants

                    requant_input_loader = output_buf.make_loader()

                    def inner_fn_requant(index, scale, zero_point):
                        input = requant_input_loader(index)
                        inv_scale, zero_point = _create_constants(
                            1.0 / scale, zero_point, dtype=torch.float32
                        )
                        val = ops.round(input * inv_scale) + zero_point
                        if output_dtype == torch.uint8:
                            qmin, qmax = _create_constants(
                                0, 255, dtype=torch.float32
                            )
                        else:
                            qmin, qmax = _create_constants(
                                -128, 127, dtype=torch.float32
                            )
                        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                        return ops.to_dtype(clamped, output_dtype)

                    output_buf = ir.Pointwise(
                        device=output_buf.get_device_or_error(),
                        dtype=output_dtype,
                        inner_fn=functools.partial(
                            inner_fn_requant,
                            scale=float(o_scale),
                            zero_point=int(o_zero_point),
                        ),
                        ranges=output_buf.get_size(),
                    )

                return output_buf

            assert x.get_dtype() in [torch.uint8, torch.int8]
            CppGemmTemplate.add_choices(
                choices,
                layout,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                has_bias=bias is not None,
                epilogue_creator=epilogue_creator,
                input_indices=[0, 3, 1, 2, 4, 5]
                if bias is None
                else [6, 0, 3, 1, 2, 4, 5],
            )
    if len(choices) == 0 or use_aten_gemm_kernels():
        kwargs = dict(
            output_scale=o_scale,
            output_zero_point=o_zero_point,
            output_dtype=output_dtype,
            post_op_name=attr,
            post_op_args=scalars,
            post_op_algorithm=algorithm,
        )
        if bias is None:
            kwargs["bias"] = None
        choices.append(
            aten_mkldnn_qlinear_unary.bind(
                (x, x_scale, x_zp, packed_weight, w_scale, w_zp)
                if bias is None
                else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias),
                layout,
                **kwargs,
            )
        )
    # this line is not needed and causes unnecessary errors
    #assert packed_weight.get_name() in V.graph.constants
    input_gen_fns = {
        3: lambda x: V.graph.constants[x.get_name()],  # packed weight
        4: lambda x: V.graph.constants[x.get_name()],  # weight scale
        5: lambda x: V.graph.constants[x.get_name()],  # weight zp
        6: lambda x: V.graph.constants[x.get_name()],  # bias
    }
    if isinstance(
        ir.InputsKernel.unwrap_storage_for_input(x_scale),
        ir.ConstantBuffer,
    ):
        # x is statically quantized
        input_gen_fns[1] = lambda x: V.graph.constants[x.get_name()]
    if isinstance(
        ir.InputsKernel.unwrap_storage_for_input(x_zp),
        ir.ConstantBuffer,
    ):
        input_gen_fns[2] = lambda x: V.graph.constants[x.get_name()]

    result = autotune_select_algorithm(
        "qlinear_unary",
        choices,
        [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
        if bias is None
        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
        layout,
        input_gen_fns=input_gen_fns,
    )
    if len(x_size) > 2:
        result = view(result, (*x_size[:-1], result.get_size()[-1]))
    return result
