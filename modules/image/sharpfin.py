"""Sharpfin wrapper for high-quality image resize and tensor conversion.

Provides drop-in replacements for torchvision.transforms.functional operations
with higher quality resampling (Magic Kernel Sharp 2021), sRGB linearization,
and Triton GPU acceleration when available.

Non-CUDA devices fall back to PIL/torch.nn.functional automatically.
"""

import sys
import torch
from PIL import Image
from modules.logger import log
from modules.image.convert import to_tensor, to_pil


_sharpfin_checked = False
_sharpfin_ok = False
_triton_ok = False


def check_sharpfin():
    global _sharpfin_checked, _sharpfin_ok, _triton_ok  # pylint: disable=global-statement
    if not _sharpfin_checked:
        _sharpfin_ok = True
        try:
            from modules.sharpfin import TRITON_AVAILABLE
            _triton_ok = TRITON_AVAILABLE
        except Exception:
            _triton_ok = False
        _sharpfin_checked = True


KERNEL_MAP = {
    "Sharpfin MKS2021": "MAGIC_KERNEL_SHARP_2021",
    "Sharpfin Lanczos3": "LANCZOS3",
    "Sharpfin Mitchell": "MITCHELL",
    "Sharpfin Catmull-Rom": "CATMULL_ROM",
}


def get_kernel(kernel=None):
    """Resolve kernel name to ResizeKernel enum. Returns None for PIL fallback."""
    if kernel is not None:
        name = kernel
    else:
        from modules import shared
        name = shared.opts.resize_quality
    if name == "PIL Lanczos" or name not in KERNEL_MAP:
        return None
    from modules.sharpfin.util import ResizeKernel
    return getattr(ResizeKernel, KERNEL_MAP[name])


def get_linearize(linearize=None, is_mask=False):
    """Determine sRGB linearization setting."""
    if is_mask:
        return False
    if linearize is not None:
        return linearize
    from modules import shared
    return shared.opts.resize_linearize_srgb


def allow_sharpfin(device=None):
    """Determine if sharpfin should be used based on device."""
    if device is None:
        from modules import devices
        device = devices.device
    # Sharpfin is optimized for CUDA with Triton, for other devices (CPU, MPS, OpenVINO), use torch/PIL optimized kernels
    return hasattr(device, 'type') and device.type == 'cuda'


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
    check_sharpfin()
    if isinstance(image, Image.Image):
        return resize_pil(image, target_size, kernel=kernel, linearize=linearize, device=device, dtype=dtype)
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
                log.info("Sharpfin: Triton sparse disabled, using dense path")
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
                log.info("Sharpfin: Triton sparse disabled, using dense path")
        return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
    # H down, W up
    use_sparse = _want_sparse(dev, rk, True)
    if use_sparse:
        try:
            intermediate = scale_fn(tensor, (h, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=True)
            return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
        except Exception:
            _triton_ok = False
            log.info("Sharpfin: Triton sparse disabled, using dense path")
    intermediate = scale_fn(tensor, (h, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)
    return scale_fn(intermediate, (h, w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=do_linear, use_sparse=False)


def resize_pil(image: Image.Image, target_size: tuple[int, int], *, kernel=None, linearize=None, device=None, dtype=None):
    """Resize a PIL Image via sharpfin, falling back to PIL on error."""
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    w, h = target_size
    is_mask = image.mode == 'L'

    if (image.width == w) and (image.height == h):
        log.debug(f'Resize image: skip={w}x{h} fn={fn}')
        return image

    from modules import devices
    dev = device if device is not None else devices.device
    if not allow_sharpfin(dev):
        log.debug(f'Resize image: method=PIL source={image.width}x{image.height} target={w}x{h} device={dev} fn={fn}')
        return image.resize((w, h), resample=Image.Resampling.LANCZOS)

    rk = get_kernel(kernel)
    if rk is None:
        log.debug(f'Resize image: method=PIL source={image.width}x{image.height} target={w}x{h} kernel=None fn={fn}')
        return image.resize((w, h), resample=Image.Resampling.LANCZOS)

    from modules.sharpfin.functional import scale
    dt = dtype or torch.float16
    do_linear = get_linearize(linearize, is_mask=is_mask)
    log.debug(f'Resize image: method=sharpfin source={image.width}x{image.height} target={w}x{h} kernel={rk} device={dev} linearize={do_linear} fn={fn}')
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


def resize_tensor(tensor: torch.Tensor, target_size: tuple[int, int], *, kernel=None, linearize=False):
    """Resize tensor [B,C,H,W] or [C,H,W] -> Tensor. For in-pipeline tensor resizes.

    Args:
        tensor: Input tensor
        target_size: (H, W) tuple
        kernel: Override kernel name
        linearize: sRGB linearization (default False for latent/mask data)
    """
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    check_sharpfin()
    from modules import devices
    dev = devices.device
    if not allow_sharpfin(dev):
        mode = 'bilinear' if (target_size[0] * target_size[1]) > (tensor.shape[-2] * tensor.shape[-1]) else 'area'
        log.debug(f'Resize tensor: method=torch mode={mode} shape={tensor.shape} target={target_size} fn={fn}')
        inp = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
        result = torch.nn.functional.interpolate(inp, size=target_size, mode=mode, antialias=True)
        return result.squeeze(0) if tensor.dim() == 3 else result
    rk = get_kernel(kernel)
    if rk is None:
        mode = 'bilinear' if (target_size[0] * target_size[1]) > (tensor.shape[-2] * tensor.shape[-1]) else 'area'
        log.debug(f'Resize tensor: method=torch mode={mode} shape={tensor.shape} target={target_size} kernel=None fn={fn}')
        inp = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
        result = torch.nn.functional.interpolate(inp, size=target_size, mode=mode, antialias=True)
        return result.squeeze(0) if tensor.dim() == 3 else result

    from modules.sharpfin.functional import scale
    dt = torch.float16
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
        log.debug(f'Resize tensor: method=sharpfin shape={tensor.shape} target={target_size} direction={both_up}:{both_down} kernel={rk} sparse={use_sparse} fn={fn}')
        result = scale(tensor, target_size, resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=use_sparse)
    else:
        log.debug(f'Resize tensor: method=sharpfin shape={tensor.shape} target={target_size} direction={both_up}:{both_down} kernel={rk} sparse=False fn={fn}')
        intermediate = scale(tensor, (th, src_w), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=False)
        result = scale(intermediate, (th, tw), resize_kernel=rk, device=dev, dtype=dt, do_srgb_conversion=linearize, use_sparse=False)
    if squeezed:
        result = result.squeeze(0)
    return result
