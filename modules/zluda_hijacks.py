import torch
from modules import rocm


_topk = torch.topk
def topk(input: torch.Tensor, *args, **kwargs): # pylint: disable=redefined-builtin
    device = input.device
    values, indices = _topk(input.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))


class DeviceProperties:
    PROPERTIES_OVERRIDE = {"regs_per_multiprocessor": 65535}
    internal: torch._C._CudaDeviceProperties

    def __init__(self, props: torch._C._CudaDeviceProperties):
        self.internal = props

    def __getattr__(self, name):
        if name in DeviceProperties.PROPERTIES_OVERRIDE:
            return DeviceProperties.PROPERTIES_OVERRIDE[name]
        return getattr(self.internal, name)


__get_device_properties = torch.cuda._get_device_properties # pylint: disable=protected-access
def torch_cuda__get_device_properties(device):
    return DeviceProperties(__get_device_properties(device))


def do_hijack():
    torch.version.hip = rocm.version
    torch.topk = topk

    torch.cuda._get_device_properties = torch_cuda__get_device_properties # pylint: disable=protected-access
    try:
        import triton
        _get_device_properties = triton.runtime.driver.active.utils.get_device_properties
        def triton_runtime_driver_active_utils_get_device_properties(device):
            props = _get_device_properties(device)
            props["mem_bus_width"] = 384
            return props
        triton.runtime.driver.active.utils.get_device_properties = triton_runtime_driver_active_utils_get_device_properties
    except Exception:
        pass
