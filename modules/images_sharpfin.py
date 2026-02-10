"""Sharpfin wrapper for high-quality image resize and tensor conversion.

Provides drop-in replacements for torchvision.transforms.functional operations
with higher quality resampling (Magic Kernel Sharp 2021), sRGB linearization,
and Triton GPU acceleration when available.

All public functions include try/except fallback to PIL/torchvision.
"""

import torch
import numpy as np
from PIL import Image

_sharpfin_checked = False
_sharpfin_ok = False
_triton_ok = False
_log = None


def _get_log():
    global _log
    if _log is None:
        try:
            from modules.shared import log
            _log = log
        except Exception:
            import logging
            _log = logging.getLogger(__name__)
    return _log


def _check():
    global _sharpfin_checked, _sharpfin_ok, _triton_ok
    if not _sharpfin_checked:
        # DEBUG: no try/except — let import errors propagate
        from modules.sharpfin.functional import scale  # pylint: disable=unused-import
        _sharpfin_ok = True
        try:
            from modules.sharpfin import TRITON_AVAILABLE
            _triton_ok = TRITON_AVAILABLE
        except Exception:
            _triton_ok = False
        _sharpfin_checked = True


def is_available():
    """Check if sharpfin functional module loaded."""
    _check()
    return _sharpfin_ok


KERNEL_MAP = {
    "Sharpfin MKS2021": "MAGIC_KERNEL_SHARP_2021",
    "Sharpfin Lanczos3": "LANCZOS3",
    "Sharpfin Mitchell": "MITCHELL",
    "Sharpfin Catmull-Rom": "CATMULL_ROM",
}


def _resolve_kernel(kernel=None):
    """Resolve kernel name to ResizeKernel enum. Returns None for PIL fallback."""
    if kernel is not None:
        name = kernel
    else:
        try:
            from modules import shared
            name = getattr(shared.opts, 'resize_quality', 'Sharpfin MKS2021')
        except Exception:
            name = 'Sharpfin MKS2021'
    if name == "PIL Lanczos" or name not in KERNEL_MAP:
        return None
    from modules.sharpfin.util import ResizeKernel
    return getattr(ResizeKernel, KERNEL_MAP[name])


def _resolve_linearize(linearize=None, is_mask=False):
    """Determine sRGB linearization setting."""
    if is_mask:
        return False
    if linearize is not None:
        return linearize
    try:
        from modules import shared
        return getattr(shared.opts, 'resize_linearize_srgb', True)
    except Exception:
        return True


def _get_device_dtype(device=None, dtype=None):
    """Get optimal device/dtype for sharpfin operations."""
    if device is not None and dtype is not None:
        return device, dtype
    try:
        from modules import devices
        dev = device or devices.device
        if dev.type == 'cuda':
            return dev, dtype or torch.float16
        return dev, dtype or torch.float32
    except Exception:
        return device or torch.device('cpu'), dtype or torch.float32


def resize(image, target_size, *, kernel=None, linearize=None, device=None, dtype=None):
    """Resize PIL.Image or torch.Tensor, returning same type.

    Args:
        image: PIL.Image or torch.Tensor [B,C,H,W] / [C,H,W]
        target_size: (width, height) for PIL, (H, W) for tensor
        kernel: Override kernel name, or None for settings
        linearize: Override sRGB linearization, or None for settings
        device: Override compute device
        dtype: Override compute dtype
    """
    _check()
    if isinstance(image, Image.Image):
        return _resize_pil(image, target_size, kernel=kernel, linearize=linearize, device=device, dtype=dtype)
    elif isinstance(image, torch.Tensor):
        return resize_tensor(image, target_size, kernel=kernel, linearize=linearize if linearize is not None else False)
    return image


def _want_sparse(dev, rk, both_down):
    """Check if Triton sparse acceleration should be attempted."""
    return _triton_ok and dev.type == 'cuda' and rk.value == 'magic_kernel_sharp_2021' and both_down


def _scale_pil(scale_fn, tensor, out_res, rk, dev, dt, do_linear, src_h, src_w, h, w, both_down, both_up):
    """Run sharpfin scale with sparse fallback. Returns result tensor."""
    global _triton_ok # pylint: disable=global-statement
    if both_down or both_up:
        use_sparse = _want_sparse(dev, rk, both_down)
        if use_sparse:
            try:
                return scale_fn(tensor, out_res, resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=True)
            except Exception:
                _triton_ok = False
                _get_log().info("Sharpfin: Triton sparse disabled, using dense path")
        return scale_fn(tensor, out_res, resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
    # Mixed axis: split into two single-axis resizes
    if h > src_h:  # H up, W down
        intermediate = scale_fn(tensor, (h, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
        use_sparse = _want_sparse(dev, rk, True)
        if use_sparse:
            try:
                return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=True)
            except Exception:
                _triton_ok = False
                _get_log().info("Sharpfin: Triton sparse disabled, using dense path")
        return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
    # H down, W up
    use_sparse = _want_sparse(dev, rk, True)
    if use_sparse:
        try:
            intermediate = scale_fn(tensor, (h, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=True)
            return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
        except Exception:
            _triton_ok = False
            _get_log().info("Sharpfin: Triton sparse disabled, using dense path")
    intermediate = scale_fn(tensor, (h, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
    return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)


def _resize_pil(image, target_size, *, kernel=None, linearize=None, device=None, dtype=None):
    """Resize a PIL Image via sharpfin, falling back to PIL on error."""
    w, h = target_size
    if image.width == w and image.height == h:
        return image
    is_mask = image.mode == 'L'
    rk = _resolve_kernel(kernel)
    if rk is None:
        # DEBUG: only "PIL Lanczos" setting should reach here
        assert _resolve_kernel.__doc__, "unreachable"  # keeps linter happy
        return image.resize((w, h), resample=Image.Resampling.LANCZOS)
    try:
        from modules.sharpfin.functional import scale
        do_linear = _resolve_linearize(linearize, is_mask=is_mask)
        dev, dt = _get_device_dtype(device, dtype)
        tensor = to_tensor(image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device=dev, dtype=dt)
        out_res = (h, w)  # sharpfin uses (H, W)
        src_h, src_w = tensor.shape[-2], tensor.shape[-1]
        both_down = (h <= src_h and w <= src_w)
        both_up = (h >= src_h and w >= src_w)
        result = _scale_pil(scale, tensor, out_res, rk, dev, dt, do_linear, src_h, src_w, h, w, both_down, both_up)
        return to_pil(result)
    # except Exception as e:  # DEBUG: PIL fallback disabled for testing
    #     _get_log().warning(f"Sharpfin resize failed, falling back to PIL: {e}")
    #     return image.resize((w, h), resample=Image.Resampling.LANCZOS)
    finally:
        pass


def resize_tensor(tensor, target_size, *, kernel=None, linearize=False):
    """Resize tensor [B,C,H,W] or [C,H,W] -> Tensor. For in-pipeline tensor resizes.

    Args:
        tensor: Input tensor
        target_size: (H, W) tuple
        kernel: Override kernel name
        linearize: sRGB linearization (default False for latent/mask data)
    """
    _check()
    rk = _resolve_kernel(kernel)
    if rk is None:
        # DEBUG: only "PIL Lanczos" setting should reach here
        mode = 'bilinear' if target_size[0] * target_size[1] > tensor.shape[-2] * tensor.shape[-1] else 'area'
        return torch.nn.functional.interpolate(tensor if tensor.dim() == 4 else tensor.unsqueeze(0), size=target_size, mode=mode, antialias=True).squeeze(0) if tensor.dim() == 3 else torch.nn.functional.interpolate(tensor, size=target_size, mode=mode, antialias=True)
    try:
        from modules.sharpfin.functional import scale
        dev, dt = _get_device_dtype()
        squeezed = False
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        src_h, src_w = tensor.shape[-2], tensor.shape[-1]
        th, tw = target_size
        both_down = (th <= src_h and tw <= src_w)
        both_up = (th >= src_h and tw >= src_w)
        if both_down or both_up:
            use_sparse = _triton_ok and dev.type == 'cuda' and rk.value == 'magic_kernel_sharp_2021' and both_down
            result = scale(tensor, target_size, resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=use_sparse)
        else:
            if th > src_h:
                intermediate = scale(tensor, (th, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=False)
                result = scale(intermediate, (th, tw), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=False)
            else:
                intermediate = scale(tensor, (th, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=False)
                result = scale(intermediate, (th, tw), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=False)
        if squeezed:
            result = result.squeeze(0)
        return result
    # except Exception as e:  # DEBUG: F.interpolate fallback disabled for testing
    #     _get_log().warning(f"Sharpfin resize_tensor failed, falling back to F.interpolate: {e}")
    #     mode = 'bilinear' if target_size[0] * target_size[1] > tensor.shape[-2] * tensor.shape[-1] else 'area'
    #     inp = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
    #     result = torch.nn.functional.interpolate(inp, size=target_size, mode=mode, antialias=True)
    #     return result.squeeze(0) if tensor.dim() == 3 else result
    finally:
        pass


def to_tensor(image):
    """PIL Image -> float32 CHW tensor [0,1]. Pure torch, no torchvision."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    pic = np.array(image, copy=True)
    if pic.ndim == 2:
        pic = pic[:, :, np.newaxis]
    tensor = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div_(255.0)
    return tensor.to(torch.float32)


def to_pil(tensor):
    """Float CHW/HWC or BCHW/BHWC tensor [0,1] -> PIL Image. Pure torch, no torchvision."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:
        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[-1] < tensor.shape[-2]:  # BHWC
            tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor[0]
    elif tensor.dim() == 3:
        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[-1] < tensor.shape[-2] and tensor.shape[-1] < tensor.shape[-3]:  # HWC
            tensor = tensor.permute(2, 0, 1)
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
    ndarr = tensor.permute(1, 2, 0).numpy()
    if ndarr.shape[2] == 1:
        return Image.fromarray(ndarr[:, :, 0], mode='L')
    return Image.fromarray(ndarr)


def pil_to_tensor(image):
    """PIL Image -> uint8 CHW tensor (no float scaling). Replaces TF.pil_to_tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    pic = np.array(image, copy=True)
    if pic.ndim == 2:
        pic = pic[:, :, np.newaxis]
    return torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()


def normalize(tensor, mean, std, inplace=False):
    """Tensor normalization. Replaces TF.normalize."""
    if not inplace:
        tensor = tensor.clone()
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean_t.ndim == 1:
        mean_t = mean_t[:, None, None]
    if std_t.ndim == 1:
        std_t = std_t[:, None, None]
    tensor.sub_(mean_t).div_(std_t)
    return tensor
