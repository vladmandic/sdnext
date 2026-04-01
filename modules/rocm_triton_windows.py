import sys
import torch
from modules import shared, devices
from modules.logger import log
from modules.rocm import Agent


if sys.platform == "win32":
    MEM_BUS_WIDTH = {
        "AMD Radeon RX 9070 XT": 256,
        "AMD Radeon RX 9070": 256,
        "AMD Radeon RX 9060 XT": 192,
        "AMD Radeon RX 7900 XTX": 384,
        "AMD Radeon RX 7900 XT": 320,
        "AMD Radeon RX 7900 GRE": 256,
        "AMD Radeon RX 7800 XT": 256,
        "AMD Radeon RX 7700 XT": 192,
        "AMD Radeon RX 7700": 192,
        "AMD Radeon RX 7650 GRE": 128,
        "AMD Radeon RX 7600 XT": 128,
        "AMD Radeon RX 7600": 128,
        "AMD Radeon RX 7500 XT": 96,
        "AMD Radeon RX 6950 XT": 256,
        "AMD Radeon RX 6900 XT": 256,
        "AMD Radeon RX 6800 XT": 256,
        "AMD Radeon RX 6800": 256,
        "AMD Radeon RX 6750 XT": 192,
        "AMD Radeon RX 6700 XT": 192,
        "AMD Radeon RX 6700": 160,
        "AMD Radeon RX 6650 XT": 128,
        "AMD Radeon RX 6600 XT": 128,
        "AMD Radeon RX 6600": 128,
        "AMD Radeon RX 6500 XT": 64,
        "AMD Radeon RX 6400": 64,
    }

    class DeviceProperties:
        PROPERTIES_OVERRIDE = {
            # sometimes gcnArchName contains device name ("AMD Radeon RX ..."), not architecture name ("gfx...")
            "gcnArchName": "gfx0000",
        }
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

    _cuda_getCurrentRawStream = torch._C._cuda_getCurrentRawStream # pylint: disable=protected-access
    def torch__C__cuda_getCurrentRawStream(device):
        from modules import zluda_installer
        return zluda_installer.core.to_hip_stream(_cuda_getCurrentRawStream(device))

    def get_default_agent() -> Agent | None:
        if shared.devices.has_rocm():
            return devices.get_hip_agent()
        else:
            from modules import zluda_installer
            return zluda_installer.default_agent

    def apply_triton_patches():
        agent = get_default_agent()
        if agent is not None:
            DeviceProperties.PROPERTIES_OVERRIDE["gcnArchName"] = agent.name
        torch.cuda._get_device_properties = torch_cuda__get_device_properties # pylint: disable=protected-access
        if shared.devices.backend == "zluda":
            torch._C._cuda_getCurrentRawStream = torch__C__cuda_getCurrentRawStream # pylint: disable=protected-access
            torch._dynamo.device_interface.CudaInterface.get_raw_stream = staticmethod(torch__C__cuda_getCurrentRawStream) # pylint: disable=protected-access

        # Triton
        try:
            import triton
            _get_device_properties = triton.runtime.driver.active.utils.get_device_properties
            def triton_runtime_driver_active_utils_get_device_properties(device):
                props = _get_device_properties(device)
                name = torch.cuda.get_device_name()
                if shared.devices.has_zluda():
                    name = name[:-8]
                if props["mem_bus_width"] == 0: # Windows HIP SDK bug
                    if name in MEM_BUS_WIDTH:
                        props["mem_bus_width"] = MEM_BUS_WIDTH[name]
                    else:
                        props["mem_bus_width"] = 128
                        log.warning(f'[TRITON] defaulting mem_bus_width=128 for device "{name}".')
                return props
            triton.runtime.driver.active.utils.get_device_properties = triton_runtime_driver_active_utils_get_device_properties
        except Exception:
            pass
