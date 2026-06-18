import os
import torch
import openvino as ov
from openvino import opset16 as ov_ops
from openvino.properties import hint as ov_hints

core = ov.Core()

NPU_MUL = 32 # NPU uses FP16 x INT8 -> FP16 instead of INT8 x INT8 -> INT32 and FP16 output overflows
OV_DEVICE: str = os.environ.get("SDNQ_OPENVINO_DEVICE", "CPU")
OV_COMPILED_CACHE: dict[tuple[str, tuple[int,int] | None, tuple[int,int] | None], list[ov.InferRequest, str]] = {}
core.set_property(OV_DEVICE, {ov_hints.execution_mode: ov_hints.ExecutionMode.ACCURACY})


def ov_int_mm(A: torch.Tensor, B: torch.Tensor, infer_request: ov.InferRequest, out_name: str) -> torch.Tensor:
    C = torch.empty((A.shape[0], B.shape[-1]), device="cpu", dtype=torch.float32)
    infer_request.set_tensor("A", ov.Tensor(A.detach().contiguous().to("cpu").numpy(), shared_memory=True))
    infer_request.set_tensor("B", ov.Tensor(B.detach().contiguous().to("cpu").numpy(), shared_memory=True))
    infer_request.set_tensor(out_name, ov.Tensor(C.numpy(), shared_memory=True))
    infer_request.infer()
    C = C.to(A.device)
    if OV_DEVICE == "NPU":
        C.mul_(NPU_MUL**2)
    return C


@torch.library.custom_op("sdnq::openvino_int_mm", mutates_args=())
def openvino_int_mm(Tensor_A: torch.Tensor, Tensor_B: torch.Tensor) -> torch.Tensor:
    if OV_DEVICE in {"NPU", "CPU"}:
        cache_key = (OV_DEVICE, Tensor_A.shape, Tensor_B.shape)
    else:
        cache_key = (OV_DEVICE, None, None)
    infer_request, out_name = OV_COMPILED_CACHE.get(cache_key, (None, None))
    if infer_request is not None:
        return ov_int_mm(Tensor_A, Tensor_B, infer_request, out_name)

    if OV_DEVICE in {"NPU", "CPU"}:
        shape_a = ov.Shape(Tensor_A.shape)
        shape_b = ov.Shape(Tensor_B.shape)
    else:
        shape_a = ov.PartialShape([-1,-1])
        shape_b = ov.PartialShape([-1,-1])
    input_a = ov_ops.parameter(shape_a, ov.Type.i8, name="A")
    input_b = ov_ops.parameter(shape_b, ov.Type.i8, name="B")
    low  = ov_ops.constant(-128.0, dtype=ov.Type.f32)
    high = ov_ops.constant(127.0,  dtype=ov.Type.f32)

    a = ov_ops.fake_quantize(ov_ops.convert(input_a, ov.Type.f32), low, high, low, high, 256)
    b = ov_ops.fake_quantize(ov_ops.convert(input_b, ov.Type.f32), low, high, low, high, 256)
    if OV_DEVICE == "NPU":
        a = ov_ops.divide(a, ov_ops.constant(NPU_MUL, dtype=ov.Type.f32))
        b = ov_ops.divide(b, ov_ops.constant(NPU_MUL, dtype=ov.Type.f32))

    ov_model = ov.Model([ov_ops.matmul(a, b, False, False)], [input_a, input_b], "ov_int8_mm")
    ov_model = core.compile_model(ov_model, OV_DEVICE)
    infer_request = ov_model.create_infer_request()
    out_name = ov_model.outputs[0]

    OV_COMPILED_CACHE[cache_key] = (infer_request, out_name)
    return ov_int_mm(Tensor_A, Tensor_B, infer_request, out_name)

@openvino_int_mm.register_fake
def openvino_int_mm_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A.to(dtype=torch.float32), B.to(dtype=torch.float32))
