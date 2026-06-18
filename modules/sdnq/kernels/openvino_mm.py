import os
import torch
import openvino as ov
from openvino import opset16 as ov_ops


def build_openvino_int_mm(ov_device: str = "CPU"):
    core = ov.Core()
    input_a = ov_ops.parameter(ov.PartialShape([-1, -1]), ov.Type.i8, name="A")
    input_b = ov_ops.parameter(ov.PartialShape([-1, -1]), ov.Type.i8, name="B")
    a = ov_ops.fake_quantize(ov_ops.convert(input_a, ov.Type.f32), -128.0, 127.0, -128.0, 127.0, 256)
    b = ov_ops.fake_quantize(ov_ops.convert(input_b, ov.Type.f32), -128.0, 127.0, -128.0, 127.0, 256)
    ov_model = ov.Model([ov_ops.matmul(a, b, False, False)], [input_a, input_b], "ov_int8_mm")
    ov_model = core.compile_model(ov_model, ov_device)
    infer_request = ov_model.create_infer_request()
    out_name = ov_model.outputs[0]

    @torch.library.custom_op("sdnq::openvino_int_mm", mutates_args=())
    def openvino_int_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        C = torch.empty((A.shape[0], B.shape[-1]), device="cpu", dtype=torch.float32)
        infer_request.set_tensor("A", ov.Tensor(A.detach().contiguous().to("cpu").numpy(), shared_memory=True))
        infer_request.set_tensor("B", ov.Tensor(B.detach().contiguous().to("cpu").numpy(), shared_memory=True))
        infer_request.set_tensor(out_name, ov.Tensor(C.numpy(), shared_memory=True))
        infer_request.infer()
        return C.to(A.device)

    @openvino_int_mm.register_fake
    def openvino_int_mm_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.mm(A.to(dtype=torch.float32), B.to(dtype=torch.float32))

    return openvino_int_mm


openvino_int_mm = build_openvino_int_mm(ov_device=os.environ.get("SDNQ_OPENVINO_DEVICE", "CPU"))
