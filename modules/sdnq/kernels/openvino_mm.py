import os
import torch
import openvino as ov
from openvino import opset16 as ov_ops
from openvino.properties import hint as ov_hints

core = ov.Core()

OV_DEVICE: str = os.environ.get("SDNQ_OPENVINO_DEVICE", "HETERO:NPU,CPU" if "NPU" in core.get_available_devices() else "CPU")
OV_COMPILED_CACHE: dict[tuple[str, tuple[int,int] | None, str, tuple[int,int] | None], tuple[ov.InferRequest, str]] = {}

if OV_DEVICE == "NPU":
    OV_DEVICE = "HETERO:NPU,CPU"
for ov_device in core.get_available_devices():
    core.set_property(ov_device, {ov_hints.execution_mode: ov_hints.ExecutionMode.ACCURACY})


def ov_mm(A: torch.Tensor, B: torch.Tensor, infer_request: ov.InferRequest, out_name: str) -> torch.FloatTensor:
    C = torch.empty((A.shape[0], B.shape[-1]), device="cpu", dtype=torch.float32)
    infer_request.set_tensor("A", ov.Tensor(A.detach().contiguous().to("cpu").numpy(), shared_memory=True))
    infer_request.set_tensor("B", ov.Tensor(B.detach().contiguous().to("cpu").numpy(), shared_memory=True))
    infer_request.set_tensor(out_name, ov.Tensor(C.numpy(), shared_memory=True))
    infer_request.infer()
    C = C.to(A.device)
    return C


@torch.library.custom_op("sdnq::openvino_int_mm", mutates_args=())
def openvino_int_mm(Tensor_A: torch.Tensor, Tensor_B: torch.Tensor) -> torch.Tensor:
    if "GPU" not in OV_DEVICE:
        cache_key = (OV_DEVICE, "int8", Tensor_A.shape, Tensor_B.shape)
    else:
        cache_key = (OV_DEVICE, "int8", None, None)
    infer_request, out_name = OV_COMPILED_CACHE.get(cache_key, (None, None))
    if infer_request is not None:
        return ov_mm(Tensor_A, Tensor_B, infer_request, out_name)

    if "GPU" not in OV_DEVICE:
        shape_a = ov.Shape(Tensor_A.shape)
        shape_b = ov.Shape(Tensor_B.shape)
    else:
        shape_a = ov.PartialShape([-1,-1])
        shape_b = ov.PartialShape([-1,-1])
    input_a = ov_ops.parameter(shape_a, ov.Type.i8, name="A")
    input_b = ov_ops.parameter(shape_b, ov.Type.i8, name="B")
    a = ov_ops.convert(input_a, ov.Type.f32)
    b = ov_ops.convert(input_b, ov.Type.f32)

    low  = ov_ops.constant(-128.0, dtype=ov.Type.f32)
    high = ov_ops.constant(127.0,  dtype=ov.Type.f32)
    a = ov_ops.fake_quantize(a, low, high, low, high, 256)
    b = ov_ops.fake_quantize(b, low, high, low, high, 256)

    # NPU uses FP16 x INT8 -> FP16 instead of INT8 x INT8 -> INT32 and FP16 output overflows
    if "NPU" in OV_DEVICE:
        fp16_scale = 0.25012213 * Tensor_B.shape[-2]
        in_scale = ov_ops.constant(fp16_scale ** 0.5, dtype=ov.Type.f32)
        out_scale = ov_ops.constant(fp16_scale, dtype=ov.Type.f32, name="out_scale_const")
        a = ov_ops.divide(a, in_scale)
        b = ov_ops.divide(b, in_scale)
        out = ov_ops.matmul(a, b, False, False)
        out = ov_ops.multiply(out, out_scale, name="out_scale")
    else:
        out = ov_ops.matmul(a, b, False, False)

    ov_model = ov.Model([out], [input_a, input_b], "ov_int8_mm")
    if "NPU" in OV_DEVICE: # NPU can't use FP32 for regular multiplications
        for node in ov_model.get_ops():
            if node.get_friendly_name() in {"out_scale", "out_scale_const"}:
                node.get_rt_info()["affinity"] = "CPU"
            else:
                node.get_rt_info()["affinity"] = "NPU"
    ov_model = core.compile_model(ov_model, OV_DEVICE)
    infer_request = ov_model.create_infer_request()
    out_name = ov_model.outputs[0]

    OV_COMPILED_CACHE[cache_key] = (infer_request, out_name)
    return ov_mm(Tensor_A, Tensor_B, infer_request, out_name)

@openvino_int_mm.register_fake
def openvino_int_mm_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A.to(dtype=torch.float32), B.to(dtype=torch.float32))


@torch.library.custom_op("sdnq::openvino_fp_mm", mutates_args=())
def openvino_fp_mm(Tensor_A: torch.Tensor, Tensor_B: torch.Tensor) -> torch.Tensor:
    mm_dtype = "fp16" if Tensor_B.dtype == torch.float16 else "fp8"
    if mm_dtype == "fp8":
        Tensor_A = Tensor_A.to(dtype=torch.float16)
        Tensor_B = Tensor_B.to(dtype=torch.float16)
    if "GPU" not in OV_DEVICE:
        cache_key = (OV_DEVICE, mm_dtype, Tensor_A.shape, Tensor_B.shape)
    else:
        cache_key = (OV_DEVICE, mm_dtype, None, None)
    infer_request, out_name = OV_COMPILED_CACHE.get(cache_key, (None, None))
    if infer_request is not None:
        return ov_mm(Tensor_A, Tensor_B, infer_request, out_name)

    if "GPU" not in OV_DEVICE:
        shape_a = ov.Shape(Tensor_A.shape)
        shape_b = ov.Shape(Tensor_B.shape)
    else:
        shape_a = ov.PartialShape([-1,-1])
        shape_b = ov.PartialShape([-1,-1])
    input_a = ov_ops.parameter(shape_a, ov.Type.f16, name="A")
    input_b = ov_ops.parameter(shape_b, ov.Type.f16, name="B")
    a = ov_ops.convert(input_a, ov.Type.f32)
    b = ov_ops.convert(input_b, ov.Type.f32)

    if mm_dtype == "fp8":
        low  = ov_ops.constant(-448.0, dtype=ov.Type.f32)
        high = ov_ops.constant(448.0,  dtype=ov.Type.f32)
        a = ov_ops.fake_quantize(a, low, high, low, high, 256)
        b = ov_ops.fake_quantize(b, low, high, low, high, 256)
        fp16_scale = 4 * Tensor_B.shape[-2]
    else:
        fp16_scale = 65536 * Tensor_B.shape[-2]

    in_scale = ov_ops.constant(fp16_scale**0.5, dtype=ov.Type.f32)
    out_scale = ov_ops.constant(fp16_scale, dtype=ov.Type.f32, name="out_scale_const")
    a = ov_ops.convert(ov_ops.divide(a, in_scale), ov.Type.f16)
    b = ov_ops.convert(ov_ops.divide(b, in_scale), ov.Type.f16)
    out = ov_ops.matmul(a, b, False, False, name="fp_mm")
    out = ov_ops.multiply(ov_ops.convert(out, ov.Type.f32), out_scale, name="out_scale")

    ov_model = ov.Model([out], [input_a, input_b], "ov_fp_mm")
    if "NPU" in OV_DEVICE: # NPU can't use FP32 for regular multiplications
        for node in ov_model.get_ops():
            if node.get_friendly_name() in {"out_scale", "out_scale_const"}:
                node.get_rt_info()["affinity"] = "CPU"
            else:
                node.get_rt_info()["affinity"] = "NPU"
    ov_model = core.compile_model(ov_model, OV_DEVICE)
    infer_request = ov_model.create_infer_request()
    out_name = ov_model.outputs[0]

    OV_COMPILED_CACHE[cache_key] = (infer_request, out_name)
    return ov_mm(Tensor_A, Tensor_B, infer_request, out_name)

@openvino_fp_mm.register_fake
def openvino_fp_mm_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A.to(dtype=torch.float32), B.to(dtype=torch.float32))
