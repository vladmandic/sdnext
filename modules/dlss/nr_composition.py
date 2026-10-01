from __future__ import annotations

import math

import numpy as np

from .nr_features import half


def compose_head(
    head: np.ndarray,
    color: np.ndarray,
    *,
    control_mask: np.ndarray | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    """RGB = colour + 0.25 * half(head[..., :3]), blended by mask red * intensity."""
    head = np.asarray(head, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    if head.shape[:2] != color.shape[:2] or head.shape[2] < 3 or color.shape[2] != 3:
        raise ValueError("head and colour must share height and width")
    residual = half(head[..., :3]) * np.float32(0.25)
    predicted = np.clip(color + residual, 0, 1)
    blend = np.float32(intensity)
    if control_mask is not None:
        control_mask = np.asarray(control_mask, dtype=np.float32)
        if control_mask.shape != color.shape:
            raise ValueError("control mask must match the colour image shape")
        blend = control_mask[..., :1] * np.float32(intensity)
    blend = np.clip(blend, 0, 1)
    return np.clip(color + blend * (predicted - color), 0, 1).astype(np.float32)


# pylint: disable=invalid-unary-operand-type
def gaussian_kernel(sigma: float) -> np.ndarray:
    extent = int(math.ceil(3 * sigma))
    offsets = np.arange(-extent, extent + 1, dtype=np.float32)
    kernel = np.exp(-offsets * offsets / np.float32(2 * sigma * sigma)).astype(np.float32)
    return kernel / kernel.sum()


def blend_mask(source: np.ndarray, output: np.ndarray, control_mask: np.ndarray) -> np.ndarray:
    """Apply the red-channel editing boundary once, after spatial filtering."""
    mask = np.asarray(control_mask, dtype=np.float32)
    if mask.shape != source.shape or output.shape != source.shape:
        raise ValueError("control mask and output must match the source image shape")
    blend = np.clip(mask[..., :1], 0, 1)
    return np.clip(source + blend * (output - source), 0, 1).astype(np.float32)


def blur(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable convolution with edge replication (vImage kvImageEdgeExtend)."""
    extent = (len(kernel) - 1) // 2
    padded = np.pad(plane, ((0, 0), (extent, extent)), mode="edge")
    horizontal = np.zeros_like(plane, dtype=np.float32)
    width = plane.shape[1]
    for index, weight in enumerate(kernel):
        horizontal += np.float32(weight) * padded[:, index : index + width]
    padded = np.pad(horizontal, ((extent, extent), (0, 0)), mode="edge")
    vertical = np.zeros_like(plane, dtype=np.float32)
    height = plane.shape[0]
    for index, weight in enumerate(kernel):
        vertical += np.float32(weight) * padded[index : index + height, :]
    return vertical


def resample(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize (height, width, channels) float32: exact box average for integer downscale, Lanczos otherwise."""
    image = np.asarray(image, dtype=np.float32)
    source_height, source_width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("resample target must be positive")
    if (width, height) == (source_width, source_height):
        return image
    if (
        source_width % width == 0 and source_height % height == 0
        and source_width // width == source_height // height and source_width // width > 1
    ):
        factor = source_width // width
        return image.reshape(height, factor, width, factor, image.shape[2]).mean(axis=(1, 3), dtype=np.float32)
    from PIL import Image

    planes = []
    for channel in range(image.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(image[..., channel]), mode="F")
        planes.append(np.asarray(plane.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32))
    return np.stack(planes, axis=-1)


def compose_detail(
    source: np.ndarray,
    output: np.ndarray,
    *,
    detail_strength: float = 1.0,
    colour_strength: float = 1.0,
    radius: float = 4.0,
) -> np.ndarray:
    """result = source + colour * lowpass(change) + detail * highpass(change), clamped to [0, 1]."""
    for value in (detail_strength, colour_strength, radius):
        if not math.isfinite(value):
            raise ValueError("strengths and radius must be finite")
    if radius <= 0:
        raise ValueError("radius must be positive")
    source = np.asarray(source, dtype=np.float32)
    output = np.asarray(output, dtype=np.float32)
    if source.shape != output.shape:
        raise ValueError("source and output must share a shape")
    if detail_strength == 1 and colour_strength == 1:
        return output
    kernel = gaussian_kernel(radius)
    result = np.empty_like(source)
    for channel in range(source.shape[2]):
        change = output[..., channel] - source[..., channel]
        low = blur(change, kernel)
        high = change - low
        result[..., channel] = np.clip(
            source[..., channel] + np.float32(colour_strength) * low + np.float32(detail_strength) * high, 0, 1
        )
    return result
