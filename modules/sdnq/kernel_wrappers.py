# pylint: disable=protected-access

import os
import sys
import torch

from modules import devices


if os.environ.get("SDNQ_ALLOW_FP8_MM", None) is None:
    if devices.backend == "cuda":
        is_fp8_mm_supported = bool(torch.cuda.get_device_capability(devices.device) >= (8,9))
    elif devices.backend == "rocm":
        gfx_version = devices.get_hip_agent().gfx_version
        is_fp8_mm_supported = bool(gfx_version >= 0x1200 or (gfx_version >= 0x940 and gfx_version < 0x1000))
    else:
        is_fp8_mm_supported = False
else:
    is_fp8_mm_supported = os.environ.get("SDNQ_ALLOW_FP8_MM", "0").lower() not in {"0", "false", "no"}

if os.environ.get("SDNQ_ALLOW_FP8_COMPILE", None) is None:
    if devices.backend == "cuda" and "linux" in sys.platform:
        is_fp8_compile_supported = bool(torch.cuda.get_device_capability(devices.device) >= (8,9)) # triton has no e4m3 conversions before sm_89
    else:
        is_fp8_compile_supported = True
else:
    is_fp8_compile_supported = os.environ.get("SDNQ_ALLOW_FP8_COMPILE", "0").lower() not in {"0", "false", "no"}

if devices.backend == "rocm":
    gfx_version = devices.get_hip_agent().gfx_version
    is_rdna2_and_older = bool(gfx_version < 0x940 or (gfx_version < 0x1100 and gfx_version >= 0x1000))
else:
    is_rdna2_and_older = False

if os.environ.get("SDNQ_USE_OPENVINO_MM", None) is None:
    use_openvino_mm = bool(devices.backend in {"cpu", "openvino"})
else:
    use_openvino_mm = bool(os.environ.get("SDNQ_USE_OPENVINO_MM", "0").lower() not in {"0", "false", "no"})

if os.environ.get("SDNQ_USE_TRITON_MM", None) is None:
    use_triton_mm = bool(is_rdna2_and_older or devices.backend in {"zluda", "ipex"})
else:
    use_triton_mm = bool(os.environ.get("SDNQ_USE_TRITON_MM", "0").lower() not in {"0", "false", "no"})

if os.environ.get("SDNQ_USE_TENSORWISE_FP8_MM", None) is None:
    # row-wise FP8 only exist on H100 hardware, sdnq will use software row-wise with tensorwise hardware with this setting
    use_tensorwise_fp8_matmul = bool(devices.backend != "cuda" or (devices.backend == "cuda" and torch.cuda.get_device_capability(devices.device) < (9,0)))
else:
    use_tensorwise_fp8_matmul = os.environ.get("SDNQ_USE_TENSORWISE_FP8_MM", "0").lower() not in {"0", "false", "no"}

if os.environ.get("SDNQ_USE_CONTIGUOUS_MM", None) is None:
    use_contiguous_int8_mm = bool(use_openvino_mm or is_rdna2_and_older or devices.backend in {"ipex", "mps", "openvino", "zluda"})
    use_contiguous_fp16_mm = bool(use_contiguous_int8_mm or devices.backend == "rocm")
else:
    use_contiguous_int8_mm = bool(os.environ.get("SDNQ_USE_CONTIGUOUS_MM", "0").lower() not in {"0", "false", "no"})
    use_contiguous_fp16_mm = use_contiguous_int8_mm


int_mm_func = None
fp_mm_func = None
fp8_mm_func = None
int_scaled_mm_func = None
fp_scaled_mm_func = None
fp8_scaled_mm_func = None

if use_openvino_mm:
    try:
        from .kernels.openvino_mm import openvino_int_mm, openvino_fp_mm
        int_mm_func = openvino_int_mm
        fp_mm_func = openvino_fp_mm
    except Exception:
        use_openvino_mm = False
elif use_triton_mm:
    try:
        from .kernels.triton_mm import sdnq_triton_mm
        from .kernels.triton_scaled_mm import sdnq_scaled_mm
        int_mm_func = sdnq_triton_mm
        fp_mm_func = sdnq_triton_mm
        int_scaled_mm_func = sdnq_scaled_mm
        fp_scaled_mm_func = sdnq_scaled_mm
        if is_fp8_mm_supported:
            fp8_mm_func = sdnq_triton_mm
            fp8_scaled_mm_func = sdnq_scaled_mm
            use_tensorwise_fp8_matmul = False
    except Exception:
        use_triton_mm = False

if fp_mm_func is None and os.environ.get("SDNQ_USE_TRITON_MM", "1").lower() not in {"0", "false", "no"}:
    try:
        from .kernels.triton_mm import sdnq_triton_mm
        from .kernels.triton_scaled_mm import sdnq_scaled_mm
        fp_mm_func = sdnq_triton_mm
        fp_scaled_mm_func = sdnq_scaled_mm
    except Exception:
        use_triton_mm = False


if int_mm_func is None:
    def int_mm_torch(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype = torch.int32) -> torch.FloatTensor:
        return torch._int_mm(a,b).to(dtype=out_dtype)
    int_mm_func = int_mm_torch

if fp_mm_func is None:
    if devices.backend == "cuda":
        def fp_mm_torch(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
            return torch.mm(a,b, out_dtype=out_dtype)
    else:
        def fp_mm_torch(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
            if b.dtype == torch.float8_e4m3fn:
                fp16_scale = 4 * b.shape[-2]
            else:
                fp16_scale = 65536 * b.shape[-2]
            in_scale = fp16_scale**0.5
            a = a.to(dtype=torch.float32).div_(in_scale).to(dtype=torch.float16)
            b = b.to(dtype=torch.float32).div_(in_scale).to(dtype=torch.float16)
            return torch.mm(a,b).to(dtype=torch.float32).mul_(fp16_scale).to(dtype=out_dtype)
    fp_mm_func = fp_mm_torch

if fp8_mm_func is None:
    def fp8_mm_torch(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
        dummy_input_scale = torch.ones(1, device=a.device, dtype=torch.float32)
        return torch._scaled_mm(a, b, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=out_dtype)
    fp8_mm_func = fp8_mm_torch


if int_scaled_mm_func is None:
    def int_scaled_mm_torch(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, bias: torch.FloatTensor | None = None, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
        if bias is None:
            return int_mm_func(a,b).to(dtype=scale_a.dtype).mul_(scale_a).mul_(scale_b).to(dtype=out_dtype)
        else:
            return torch.addcmul(bias, int_mm_func(a,b).to(dtype=scale_a.dtype).mul_(scale_a), scale_b).to(dtype=out_dtype)
    int_scaled_mm_func = int_scaled_mm_torch

if fp_scaled_mm_func is None:
    def fp_scaled_mm_torch(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, bias: torch.FloatTensor | None = None, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
        if bias is None:
            return fp_mm_func(a,b).to(dtype=scale_a.dtype).mul_(scale_a).mul_(scale_b).to(dtype=out_dtype)
        else:
            return torch.addcmul(bias, fp_mm_func(a,b).to(dtype=scale_a.dtype).mul_(scale_a), scale_b).to(dtype=out_dtype)
    fp_scaled_mm_func = fp_scaled_mm_torch

if fp8_scaled_mm_func is None:
    if use_tensorwise_fp8_matmul:
        def fp8_scaled_mm_torch(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, bias: torch.FloatTensor | None = None, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
            if bias is None:
                return fp8_mm_func(a,b, out_dtype=scale_a.dtype).mul_(scale_a).mul_(scale_b).to(dtype=out_dtype)
            else:
                return torch.addcmul(bias, fp8_mm_func(a,b, out_dtype=scale_a.dtype).mul_(scale_a), scale_b).to(dtype=out_dtype)
    else:
        def fp8_scaled_mm_torch(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, bias: torch.FloatTensor | None = None, out_dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
            if bias is not None and bias.ndim != 1:
                return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, bias=None, out_dtype=out_dtype).add_(bias)
            else:
                return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, bias=bias.to(dtype=out_dtype) if bias is not None else None, out_dtype=out_dtype)
    fp8_scaled_mm_func = fp8_scaled_mm_torch
