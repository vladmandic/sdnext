import sys
from functools import wraps
import torch
from modules import shared


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
            "gcnArchName": "UNKNOWN ARCHITECTURE",
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
        from modules import zluda
        return zluda.core.to_hip_stream(_cuda_getCurrentRawStream(device))

    def get_default_agent_name():
        if shared.devices.backend == "rocm":
            device = shared.devices.get_optimal_device()
            return getattr(torch.cuda.get_device_properties(device), "gcnArchName", None)
        else:
            from modules import zluda
            if zluda.default_agent is None:
                return None
            return zluda.default_agent.name

    def apply_triton_patches():
        arch_name = get_default_agent_name()
        if arch_name is not None:
            DeviceProperties.PROPERTIES_OVERRIDE["gcnArchName"] = arch_name
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
                        shared.log.warning(f'[TRITON] defaulting mem_bus_width=128 for device "{name}".')
                return props
            triton.runtime.driver.active.utils.get_device_properties = triton_runtime_driver_active_utils_get_device_properties

            if 'Flash attention' in shared.opts.sdp_options:
                from modules.flash_attn_triton_amd import interface_fa
                sdpa_pre_flash_atten = torch.nn.functional.scaled_dot_product_attention
                @wraps(sdpa_pre_flash_atten)
                def sdpa_flash_atten(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
                    if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
                        if scale is None:
                            scale = query.shape[-1] ** (-0.5)
                        head_size_og = query.size(3)
                        if head_size_og % 8 != 0:
                            query = torch.nn.functional.pad(query, [0, 8 - head_size_og % 8])
                            key = torch.nn.functional.pad(key, [0, 8 - head_size_og % 8])
                            value = torch.nn.functional.pad(value, [0, 8 - head_size_og % 8])
                        query = query.transpose(1, 2)
                        out_padded = torch.zeros_like(query)
                        interface_fa.fwd(
                            query,
                            key.transpose(1, 2),
                            value.transpose(1, 2),
                            out_padded,
                            dropout_p,
                            scale,
                            is_causal,
                        )
                        return out_padded[..., :head_size_og].transpose(1, 2)
                    else:
                        return sdpa_pre_flash_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
                torch.nn.functional.scaled_dot_product_attention = sdpa_flash_atten
                shared.log.debug('Torch attention: type="triton flash attention"')
        except Exception:
            pass
