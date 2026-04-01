"""Sharpfin utility types and color space conversion functions.

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
"""

from enum import Enum
import torch


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        # Clamping is for protection against NaNs during backwards passes.
        ((torch.clamp(image, min=0.04045) + 0.055) / 1.055) ** 2.4
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        torch.clamp(1.055 * image ** (1 / 2.4) - 0.055, min=0.0, max=1.0)
    )


class ResizeKernel(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CATMULL_ROM = "catmull-rom"
    MITCHELL = "mitchell"
    B_SPLINE = "b-spline"
    LANCZOS2 = "lanczos2"
    LANCZOS3 = "lanczos3"
    MAGIC_KERNEL = "magic_kernel"
    MAGIC_KERNEL_SHARP_2013 = "magic_kernel_sharp_2013"
    MAGIC_KERNEL_SHARP_2021 = "magic_kernel_sharp_2021"


class SharpenKernel(Enum):
    SHARP_2013 = "sharp_2013"
    SHARP_2021 = "sharp_2021"


class QuantHandling(Enum):
    TRUNCATE = "truncate"
    ROUND = "round"
    STOCHASTIC_ROUND = "stochastic_round"
    BAYER = "bayer"
