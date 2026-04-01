"""Sharpfin - High quality image resizing with GPU acceleration.

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
Provides Magic Kernel Sharp 2021 resampling, sRGB linearization,
and Triton sparse GPU acceleration.
"""

from .util import ResizeKernel, SharpenKernel, QuantHandling, srgb_to_linear, linear_to_srgb

try:
    from .functional import scale, _upscale, _downscale, _get_resize_kernel
    FUNCTIONAL_AVAILABLE = True
except Exception:
    FUNCTIONAL_AVAILABLE = False

try:
    from .triton_functional import downscale_sparse
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
