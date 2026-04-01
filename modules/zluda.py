import sys


PLATFORM = sys.platform
do_nothing = lambda _: None # pylint: disable=unnecessary-lambda-assignment


def test(device) -> Exception | None:
    import torch
    device = torch.device(device)
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        assert out.sum().is_nonzero()
        return None
    except Exception as e:
        return e


def zluda_init():
    try:
        import torch
        from modules.logger import log
        from modules import devices, zluda_installer
        from modules.shared import cmd_opts
        from modules.rocm_triton_windows import apply_triton_patches

        cmd_opts.device_id = None

        device = devices.get_optimal_device()
        result = test(device)
        if result is not None:
            log.warning(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            torch.cuda.is_available = lambda: False
            devices.cuda_ok = False
            devices.backend = 'cpu'
            devices.device = devices.cpu
            return False, result

        if not zluda_installer.default_agent.blaslt_supported:
            log.debug(f'ROCm: hipBLASLt unavailable agent={zluda_installer.default_agent}')

        apply_triton_patches()

        torch.backends.cudnn.enabled = zluda_installer.MIOpen_enabled
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            if not zluda_installer.MIOpen_enabled:
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_cudnn_sdp = do_nothing
        else:
            torch.backends.cuda.enable_cudnn_sdp = do_nothing
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_flash_sdp = torch.backends.cuda.enable_cudnn_sdp
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp = do_nothing
    except Exception as e:
        return False, e
    return True, None
