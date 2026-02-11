"""Sharpfin transform classes for torchvision integration.

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
Imports patched: absolute sharpfin.X -> relative .X, torchvision guarded.
"""

import torch
import torch.nn.functional as F

try:
    from torchvision.transforms.v2 import Transform
except ImportError:
    class Transform:
        _transformed_types = ()
        def __init__(self):
            pass

from .util import QuantHandling, ResizeKernel, SharpenKernel, srgb_to_linear, linear_to_srgb
from . import functional as SFF
from .cms import apply_srgb
import math
from typing import Any, Dict, Tuple
from PIL import Image
from .functional import _get_resize_kernel
from contextlib import nullcontext

try:
    from .triton_functional import downscale_sparse
except ImportError:
    downscale_sparse = None

# from Pytorch >= 2.6
set_stance = getattr(torch.compiler, "set_stance", None)

__all__ = ["ResizeKernel", "SharpenKernel", "QuantHandling"]

class Scale(Transform):
    """Rescaling transform supporting multiple algorithms with sRGB linearization."""
    _transformed_types = (torch.Tensor,)
    def __init__(self,
        out_res: tuple[int, int] | int,
        device: torch.device | str = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        out_dtype: torch.dtype | None = None,
        quantization: QuantHandling = QuantHandling.ROUND,
        generator: torch.Generator | None = None,
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        sharpen_kernel: SharpenKernel | None = None,
        do_srgb_conversion: bool = True,
        use_sparse: bool = False,
    ):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating point type")
        if dtype.itemsize == 1:
            raise ValueError("float8 types are not supported due to severe accuracy issues and limited function support. float16 or float32 is recommended.")
        if out_dtype is not None and not out_dtype.is_floating_point and out_dtype not in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            raise ValueError("out_dtype must be a torch float format or a torch unsigned int format")
        if use_sparse:
            assert device.type != 'cpu'
            if resize_kernel != ResizeKernel.MAGIC_KERNEL_SHARP_2021:
                raise NotImplementedError
        self.use_sparse = use_sparse

        if isinstance(out_res, int):
            out_res = (out_res, out_res)
        self.out_res = out_res
        self.device = device
        self.dtype = dtype
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.do_srgb_conversion = do_srgb_conversion

        if self.out_dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            match quantization:
                case QuantHandling.TRUNCATE:
                    self.quantize_function = lambda x: x.mul(torch.iinfo(self.out_dtype).max).to(self.out_dtype)
                case QuantHandling.ROUND:
                    self.quantize_function = lambda x: x.mul(torch.iinfo(self.out_dtype).max).round().to(self.out_dtype)
                case QuantHandling.STOCHASTIC_ROUND:
                    if generator is not None:
                        self.generator = torch.Generator(self.device)
                    else:
                        self.generator = generator
                    self.quantize_function = lambda x: SFF.stochastic_round(x, self.out_dtype, self.generator)
                case QuantHandling.BAYER:
                    self.bayer_matrix = torch.tensor(SFF.generate_bayer_matrix(16), dtype=self.dtype, device=self.device) / 255
                    self.quantize_function = lambda x: self.apply_bayer_matrix(x)
                case _:
                    raise ValueError(f"Unknown quantization handling type {quantization}")
        else:
            self.quantize_function = lambda x: x.to(dtype=out_dtype)

        self.resize_kernel, self.kernel_window = _get_resize_kernel(resize_kernel)

        match sharpen_kernel:
            case SharpenKernel.SHARP_2013:
                kernel = torch.tensor([-1, 6, -1], dtype=dtype, device=device) / 4
                self.sharp_2013_kernel = torch.outer(kernel, kernel).view(1, 1, 3, 3).expand(3, -1, -1, -1)
                self.sharpen_step = lambda x: SFF.sharpen_conv2d(x, self.sharp_2013_kernel, 1)
            case SharpenKernel.SHARP_2021:
                kernel = torch.tensor([-1, 6, -35, 204, -35, 6, -1], dtype=dtype, device=device) / 144
                self.sharp_2021_kernel = torch.outer(kernel, kernel).view(1, 1, 7, 7).expand(3, -1, -1, -1)
                self.sharpen_step = lambda x: SFF.sharpen_conv2d(x, self.sharp_2021_kernel, 3)
            case None:
                self.sharpen_step = lambda x: x
            case _:
                raise ValueError(f"Unknown sharpen kernel {sharpen_kernel}")

    def apply_bayer_matrix(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        b = self.bayer_matrix.repeat(1,1,math.ceil(H/16),math.ceil(W/16))[:,:,:H,:W]
        return (x*255 + b).to(self.out_dtype)

    @torch.compile(disable=False)
    def downscale(self, image: torch.Tensor, out_res: tuple[int, int]):
        H, W = out_res
        image = image.to(dtype=self.dtype)
        if self.do_srgb_conversion:
            image = srgb_to_linear(image)

        image = SFF._downscale_axis(image, W, self.kernel_window, self.resize_kernel, self.device, self.dtype)
        image = SFF._downscale_axis(image.mT, H, self.kernel_window, self.resize_kernel, self.device, self.dtype).mT

        image = self.sharpen_step(image)

        if self.do_srgb_conversion:
            image = linear_to_srgb(image)
        image = image.clamp(0,1)
        image = self.quantize_function(image)
        return image

    @torch.compile(disable=False)
    def downscale_sparse(self, image: torch.Tensor, out_res: tuple[int, int]):
        image = image.to(dtype=self.dtype)
        if downscale_sparse is not None:
            image = downscale_sparse(image, out_res)
        image = self.quantize_function(image)
        return image

    @torch.compile(disable=False)
    def upscale(self, image: torch.Tensor, out_res: tuple[int, int]):
        H, W = out_res
        image = image.to(dtype=self.dtype)
        if self.do_srgb_conversion:
            image = srgb_to_linear(image)

        image = self.sharpen_step(image)

        image = SFF._upscale_axis(image, W, self.kernel_window, self.resize_kernel, self.device, self.dtype)
        image = SFF._upscale_axis(image.mT, H, self.kernel_window, self.resize_kernel, self.device, self.dtype).mT

        if self.do_srgb_conversion:
            image = linear_to_srgb(image)
        image = image.clamp(0,1)
        image = self.quantize_function(image)
        return image

    def _transform(self, inpt: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        image = inpt.to(device=self.device)
        context_manager = (
            set_stance("force_eager") if set_stance and self.device.type == "cpu" else nullcontext()
        )
        with context_manager:
            if image.shape[-1] <= self.out_res[-1] and image.shape[-2] <= self.out_res[-2]:
                return self.upscale(image, self.out_res)
            elif image.shape[-1] >= self.out_res[-1] and image.shape[-2] >= self.out_res[-2]:
                if self.use_sparse:
                    return self.downscale_sparse(image, self.out_res)
                return self.downscale(image, self.out_res)
            else:
                raise ValueError("Mixed axis resizing (e.g. scaling one axis up and the other down) is not supported. File a bug report with your use case if needed.")

class ApplyCMS(Transform):
    """Apply color management to a PIL Image to standardize it to sRGB color space."""
    _transformed_types = (Image.Image,)

    def _transform(self, inpt: Image.Image, params: Dict[str, Any]) -> Image.Image:
        if not isinstance(inpt, Image.Image):
            raise TypeError(f"inpt should be PIL Image. Got {type(inpt)}")

        return apply_srgb(inpt)

class AlphaComposite(Transform):
    _transformed_types = (Image.Image,)
    def __init__(
        self,
        background: Tuple[int,int,int] = (255, 255, 255)
    ):
        super().__init__()
        self.background = background

    def _transform(self, inpt: Image.Image, params: Dict[str, Any]) -> Image.Image:
        if not isinstance(inpt, Image.Image):
            raise TypeError(f"inpt should be PIL Image. Got {type(inpt)}")
        if not inpt.has_transparency_data:
            return inpt

        bg = Image.new("RGB", inpt.size, self.background).convert('RGBA')

        return Image.alpha_composite(bg, inpt).convert('RGB')

class AspectRatioCrop(Transform):
    _transformed_types = (Image.Image,)
    def __init__(
        self,
        width: int,
        height: int,
    ):
        super().__init__()
        self.ref_width = width
        self.ref_height = height
        self.aspect_ratio = width / height

    def _transform(self, inpt: Image.Image, params: Dict[str, Any]) -> Image.Image:
        if not isinstance(inpt, Image.Image):
            raise TypeError(f"inpt should be PIL Image. Got {type(inpt)}")

        left, top, right, bottom = 0, 0, inpt.width, inpt.height
        inpt_ar = inpt.width / inpt.height

        if inpt_ar > self.aspect_ratio:
            result_width = int(round(inpt.height / self.ref_height * self.ref_width))
            crop_amt = (inpt.width - result_width) // 2
            left += crop_amt
            right -= crop_amt
        elif inpt_ar < self.aspect_ratio:
            result_height = int(round(inpt.width / self.ref_width * self.ref_height))
            crop_amt = (inpt.height - result_height) // 2
            top += crop_amt
            bottom -= crop_amt

        return inpt.crop((left, top, right, bottom))
