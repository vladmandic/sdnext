import os
from functools import wraps
from contextlib import nullcontext
import torch
import numpy as np

from modules import devices
from .device_prop import cache_size_dict

torch_version = torch.__version__[:4]
if torch_version[-1] not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
    torch_version = torch_version[:-1]
torch_version = torch_version.split(".")
torch_version[0], torch_version[1] = int(torch_version[0]), int(torch_version[1])

device_supports_fp64 = torch.xpu.has_fp64_dtype() if hasattr(torch.xpu, "has_fp64_dtype") else torch.xpu.get_device_properties(devices.device).has_fp64

# pylint: disable=protected-access, missing-function-docstring, line-too-long, unnecessary-lambda, no-else-return


@property
def is_cuda(self):
    return self.device.type == "xpu" or self.device.type == "cuda"


def check_device_type(device, device_type: str) -> bool:
    if device is None or type(device) not in {str, int, torch.device}:
        return False
    else:
        return bool(torch.device(device).type == device_type)


def check_cuda(device) -> bool:
    return bool(isinstance(device, int) or check_device_type(device, "cuda"))


def return_xpu(device): # keep the device instance type, aka return string if the input is string
    return devices.device if device is None else f"xpu:{device.split(':')[-1]}" if isinstance(device, str) and ":" in device else f"xpu:{device}" if isinstance(device, int) else torch.device(f"xpu:{device.index}" if device.index is not None else "xpu") if isinstance(device, torch.device) else "xpu"


# Autocast
original_autocast_init = torch.amp.autocast_mode.autocast.__init__
@wraps(torch.amp.autocast_mode.autocast.__init__)
def autocast_init(self, device_type=None, dtype=None, enabled=True, cache_enabled=None):
    if device_type is None or check_cuda(device_type):
        return original_autocast_init(self, device_type="xpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return original_autocast_init(self, device_type=device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


original_grad_scaler_init = torch.amp.grad_scaler.GradScaler.__init__
@wraps(torch.amp.grad_scaler.GradScaler.__init__)
def GradScaler_init(self, device: str = None, init_scale: float = 2.0**16, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000, enabled: bool = True):
    if device is None or check_cuda(device):
        return original_grad_scaler_init(self, device=return_xpu(device), init_scale=init_scale, growth_factor=growth_factor, backoff_factor=backoff_factor, growth_interval=growth_interval, enabled=enabled)
    else:
        return original_grad_scaler_init(self, device=device, init_scale=init_scale, growth_factor=growth_factor, backoff_factor=backoff_factor, growth_interval=growth_interval, enabled=enabled)


original_is_autocast_enabled = torch.is_autocast_enabled
@wraps(torch.is_autocast_enabled)
def torch_is_autocast_enabled(device_type=None):
    if device_type is None or check_cuda(device_type):
        return original_is_autocast_enabled(return_xpu(device_type))
    else:
        return original_is_autocast_enabled(device_type)


original_get_autocast_dtype = torch.get_autocast_dtype
@wraps(torch.get_autocast_dtype)
def torch_get_autocast_dtype(device_type=None):
    if device_type is None or check_cuda(device_type) or check_device_type(device_type, "xpu"):
        return torch.bfloat16
    else:
        return original_get_autocast_dtype(device_type)


# Latent Antialias CPU Offload:
# IPEX 2.5 and above has partial support but doesn't really work most of the time.
original_interpolate = torch.nn.functional.interpolate
@wraps(torch.nn.functional.interpolate)
def interpolate(tensor, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False): # pylint: disable=too-many-arguments
    if mode in {"bicubic", "bilinear"}:
        return_device = tensor.device
        return_dtype = tensor.dtype
        return original_interpolate(tensor.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(return_device, dtype=return_dtype)
    else:
        return original_interpolate(tensor, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)


# SwinIR BF16:
original_functional_pad = torch.nn.functional.pad
@wraps(torch.nn.functional.pad)
def functional_pad(input, pad, mode="constant", value=None):
    if mode == "reflect" and input.dtype == torch.bfloat16:
        return original_functional_pad(input.to(torch.float32), pad, mode=mode, value=value).to(dtype=torch.bfloat16)
    else:
        return original_functional_pad(input, pad, mode=mode, value=value)


# Diffusers FreeU
original_fft_fftn = torch.fft.fftn
@wraps(torch.fft.fftn)
def fft_fftn(input, s=None, dim=None, norm=None, *, out=None):
    return_dtype = input.dtype
    return original_fft_fftn(input.to(dtype=torch.float32), s=s, dim=dim, norm=norm, out=out).to(dtype=return_dtype)


# Diffusers FreeU
original_fft_ifftn = torch.fft.ifftn
@wraps(torch.fft.ifftn)
def fft_ifftn(input, s=None, dim=None, norm=None, *, out=None):
    return_dtype = input.dtype
    return original_fft_ifftn(input.to(dtype=torch.float32), s=s, dim=dim, norm=norm, out=out).to(dtype=return_dtype)


# Diffusers Float64 (Alchemist GPUs doesn't support 64 bit):
original_from_numpy = torch.from_numpy
@wraps(torch.from_numpy)
def from_numpy(ndarray):
    if ndarray.dtype == float:
        return original_from_numpy(ndarray.astype("float32"))
    else:
        return original_from_numpy(ndarray)


original_as_tensor = torch.as_tensor
@wraps(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    if check_cuda(device):
        device = return_xpu(device)
    if isinstance(data, np.ndarray) and data.dtype == float and not check_device_type(device, "cpu"):
        return original_as_tensor(data, dtype=torch.float32, device=device)
    else:
        return original_as_tensor(data, dtype=dtype, device=device)


original_torch_tensor = torch.tensor
@wraps(torch.tensor)
def torch_tensor(data, *args, dtype=None, device=None, **kwargs):
    global device_supports_fp64
    if check_cuda(device):
        device = return_xpu(device)
    if not device_supports_fp64 and check_device_type(device, "xpu"):
        if dtype == torch.float64:
            dtype = torch.float32
        elif dtype is None and (hasattr(data, "dtype") and (data.dtype == torch.float64 or data.dtype == float)):
            dtype = torch.float32
    return original_torch_tensor(data, *args, dtype=dtype, device=device, **kwargs)


torch.Tensor.original_Tensor_to = torch.Tensor.to
@wraps(torch.Tensor.to)
def Tensor_to(self, device=None, *args, **kwargs):
    global device_supports_fp64
    if check_cuda(device):
        device = return_xpu(device)
    if not device_supports_fp64:
        if kwargs.get("dtype", None) == torch.float64 and ((device is None and self.device.type == "xpu") or (device is not None and torch.device(device).type == "xpu")):
            kwargs["dtype"] = torch.float32
        elif device == torch.float64 and self.device.type == "xpu":
            device = torch.float32
    return self.original_Tensor_to(device, *args, **kwargs)


original_Tensor_cuda = torch.Tensor.cuda
@wraps(torch.Tensor.cuda)
def Tensor_cuda(self, device=None, *args, **kwargs):
    if device is None or check_cuda(device):
        return self.to(return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_cuda(self, device, *args, **kwargs)


original_Tensor_pin_memory = torch.Tensor.pin_memory
@wraps(torch.Tensor.pin_memory)
def Tensor_pin_memory(self, device=None, *args, **kwargs):
    if device is None or check_cuda(device):
        return original_Tensor_pin_memory(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_pin_memory(self, device, *args, **kwargs)


original_UntypedStorage_init = torch.UntypedStorage.__init__
@wraps(torch.UntypedStorage.__init__)
def UntypedStorage_init(*args, device=None, **kwargs):
    if check_cuda(device):
        return original_UntypedStorage_init(*args, device=return_xpu(device), **kwargs)
    else:
        return original_UntypedStorage_init(*args, device=device, **kwargs)


if torch_version[0] > 2 or (torch_version[0] == 2 and torch_version[1] >= 4):
    original_UntypedStorage_to = torch.UntypedStorage.to
    @wraps(torch.UntypedStorage.to)
    def UntypedStorage_to(self, *args, device=None, **kwargs):
        if check_cuda(device):
            return original_UntypedStorage_to(self, *args, device=return_xpu(device), **kwargs)
        else:
            return original_UntypedStorage_to(self, *args, device=device, **kwargs)

    original_UntypedStorage_cuda = torch.UntypedStorage.cuda
    @wraps(torch.UntypedStorage.cuda)
    def UntypedStorage_cuda(self, device=None, non_blocking=False, **kwargs):
        if device is None or check_cuda(device):
            return self.to(device=return_xpu(device), non_blocking=non_blocking, **kwargs)
        else:
            return original_UntypedStorage_cuda(self, device=device, non_blocking=non_blocking, **kwargs)


original_torch_empty = torch.empty
@wraps(torch.empty)
def torch_empty(*args, device=None, **kwargs):
    if check_cuda(device):
        return original_torch_empty(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_empty(*args, device=device, **kwargs)


original_torch_randn = torch.randn
@wraps(torch.randn)
def torch_randn(*args, device=None, dtype=None, **kwargs):
    if check_cuda(device):
        return original_torch_randn(*args, device=return_xpu(device), dtype=dtype, **kwargs)
    else:
        return original_torch_randn(*args, device=device, dtype=dtype, **kwargs)


original_torch_ones = torch.ones
@wraps(torch.ones)
def torch_ones(*args, device=None, **kwargs):
    if check_cuda(device):
        return original_torch_ones(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_ones(*args, device=device, **kwargs)


original_torch_zeros = torch.zeros
@wraps(torch.zeros)
def torch_zeros(*args, device=None, **kwargs):
    if check_cuda(device):
        return original_torch_zeros(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_zeros(*args, device=device, **kwargs)


original_torch_full = torch.full
@wraps(torch.full)
def torch_full(*args, device=None, **kwargs):
    if check_cuda(device):
        return original_torch_full(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_full(*args, device=device, **kwargs)


original_torch_arange = torch.arange
@wraps(torch.arange)
def torch_arange(*args, device=None, dtype=None, **kwargs):
    global device_supports_fp64
    if check_cuda(device):
        if not device_supports_fp64 and dtype == torch.float64:
            dtype = torch.float32
        return original_torch_arange(*args, device=return_xpu(device), dtype=dtype, **kwargs)
    else:
        if not device_supports_fp64 and check_device_type(device, "xpu") and dtype == torch.float64:
            dtype = torch.float32
        return original_torch_arange(*args, device=device, dtype=dtype, **kwargs)


original_torch_linspace = torch.linspace
@wraps(torch.linspace)
def torch_linspace(*args, device=None, dtype=None, **kwargs):
    global device_supports_fp64
    if check_cuda(device):
        if not device_supports_fp64 and dtype == torch.float64:
            dtype = torch.float32
        return original_torch_linspace(*args, device=return_xpu(device), dtype=dtype, **kwargs)
    else:
        if not device_supports_fp64 and check_device_type(device, "xpu") and dtype == torch.float64:
            dtype = torch.float32
        return original_torch_linspace(*args, device=device, dtype=dtype, **kwargs)


original_torch_eye = torch.eye
@wraps(torch.eye)
def torch_eye(*args, device=None, **kwargs):
    if check_cuda(device):
        return original_torch_eye(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_eye(*args, device=device, **kwargs)


original_torch_load = torch.load
@wraps(torch.load)
def torch_load(f, map_location=None, *args, **kwargs):
    if map_location is None or check_cuda(map_location):
        return original_torch_load(f, *args, map_location=return_xpu(map_location), **kwargs)
    else:
        return original_torch_load(f, *args, map_location=map_location, **kwargs)


@wraps(torch.cuda.synchronize)
def torch_cuda_synchronize(device=None):
    if check_cuda(device):
        return torch.xpu.synchronize(return_xpu(device))
    else:
        return torch.xpu.synchronize(device)


@wraps(torch.cuda.device)
def torch_cuda_device(device):
    if check_cuda(device):
        return torch.xpu.device(return_xpu(device))
    else:
        return torch.xpu.device(device)


@wraps(torch.cuda.set_device)
def torch_cuda_set_device(device):
    if check_cuda(device):
        torch.xpu.set_device(return_xpu(device))
    else:
        torch.xpu.set_device(device)


@wraps(torch.cuda.get_device_properties)
def get_device_properties(device=None):
    device_prop = torch.xpu.get_device_properties(device)
    new_keys = {
        "major": 12,
        "minor": 1,
        "multi_processor_count": device_prop.gpu_subslice_count,
    }
    if not hasattr(device_prop, "L2_cache_size"):
        new_keys["L2_cache_size"] = cache_size_dict.get(getattr(device_prop, "device_id", 0x56A0), cache_size_dict[0x0000])
    return DeviceProperties(device_prop, new_keys)


class DeviceProperties():
    def __init__(self, device_prop, new_keys):
        for key in dir(device_prop):
            if not key.startswith("__"):
                setattr(self, key, getattr(device_prop, key))
        for key, value in new_keys.items():
            setattr(self, key, value)


# torch.Generator has to be a class for isinstance checks
original_torch_Generator = torch.Generator
class torch_Generator(original_torch_Generator):
    def __new__(self, device=None):
        # can't hijack __init__ because of C override so use return super().__new__
        if check_cuda(device):
            return super().__new__(self, return_xpu(device))
        else:
            return super().__new__(self, device)


# Hijack Functions:
def ipex_hijacks():
    global device_supports_fp64
    if torch_version[0] > 2 or (torch_version[0] == 2 and torch_version[1] >= 4):
        torch.UntypedStorage.cuda = UntypedStorage_cuda
        torch.UntypedStorage.to = UntypedStorage_to
    torch.tensor = torch_tensor
    torch.Tensor.to = Tensor_to
    torch.Tensor.cuda = Tensor_cuda
    torch.Tensor.pin_memory = Tensor_pin_memory
    torch.UntypedStorage.__init__ = UntypedStorage_init
    torch.empty = torch_empty
    torch.randn = torch_randn
    torch.ones = torch_ones
    torch.zeros = torch_zeros
    torch.full = torch_full
    torch.arange = torch_arange
    torch.linspace = torch_linspace
    torch.eye = torch_eye
    torch.load = torch_load
    torch.cuda.synchronize = torch_cuda_synchronize
    torch.cuda.device = torch_cuda_device
    torch.cuda.set_device = torch_cuda_set_device
    torch.cuda.get_device_properties = get_device_properties

    torch.Generator = torch_Generator
    torch._C.Generator = torch_Generator

    torch.UntypedStorage.is_cuda = is_cuda
    torch.amp.autocast_mode.autocast.__init__ = autocast_init

    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.pad = functional_pad
    torch.fft.fftn = fft_fftn
    torch.fft.ifftn = fft_ifftn

    if not device_supports_fp64:
        torch.from_numpy = from_numpy
        torch.as_tensor = as_tensor

    try:
        import torchvision
        torchvision.transforms._functional_tensor.interpolate = interpolate
    except Exception:
        pass

    if os.environ.get("IPEX_FORCE_ATTENTION_SLICE", "0") == "0":
        if torch_version[0] > 2 or (torch_version[0] == 2 and torch_version[1] >= 7):
            use_dynamic_attention = False # torch 2.7 has flash atten support
        else:
            use_dynamic_attention = True
    else:
        use_dynamic_attention = bool(os.environ.get("IPEX_FORCE_ATTENTION_SLICE", "0") == "1")

    if use_dynamic_attention:
        from .attention import dynamic_scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = dynamic_scaled_dot_product_attention

    # AMP:
    torch.amp.grad_scaler.GradScaler.__init__ = GradScaler_init
    torch.is_autocast_enabled = torch_is_autocast_enabled
    torch.get_autocast_gpu_dtype = torch_get_autocast_dtype
    torch.get_autocast_dtype = torch_get_autocast_dtype

    if hasattr(torch.xpu, "amp"):
        if not hasattr(torch.xpu.amp, "custom_fwd"):
            torch.xpu.amp.custom_fwd = torch.cuda.amp.custom_fwd
            torch.xpu.amp.custom_bwd = torch.cuda.amp.custom_bwd
        if not hasattr(torch.xpu.amp, "GradScaler"):
            torch.xpu.amp.GradScaler = torch.amp.grad_scaler.GradScaler
        torch.cuda.amp = torch.xpu.amp
    else:
        if not hasattr(torch.amp, "custom_fwd"):
            torch.amp.custom_fwd = torch.cuda.amp.custom_fwd
            torch.amp.custom_bwd = torch.cuda.amp.custom_bwd
        torch.cuda.amp = torch.amp

    if not hasattr(torch.cuda.amp, "common"):
        torch.cuda.amp.common = nullcontext()
    torch.cuda.amp.common.amp_definitely_not_available = lambda: False

    return device_supports_fp64
