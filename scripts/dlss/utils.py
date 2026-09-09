from __future__ import annotations
import logging
import numpy as np


MAX_DIMENSION = 16_384

logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.DEBUG,
    filename='dlss.log',
    encoding='utf-8',
    filemode='a',
    format='%(asctime)s %(levelname)s %(message)s',
    # datefmt='%Y-%m-%d %H:%M:%S-%f',
    force=True,
)
log = logging.getLogger(__name__)
log.debug('DLSSInit')


class StandaloneError(RuntimeError):
    """Base error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidArrayError(StandaloneError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_array", message)


class VerificationError(StandaloneError):
    def __init__(self, message: str) -> None:
        super().__init__("verification_failed", message)


class ProcessingError(StandaloneError):
    def __init__(self, message: str, *, code: str = "processing_failed") -> None:
        super().__init__(code, message)


def validate_nchw(array: np.ndarray, *, name: str = "array") -> tuple[int, int, int, int]:
    if not isinstance(array, np.ndarray):
        raise InvalidArrayError(f"{name} must be a NumPy array.")
    if array.ndim != 4:
        raise InvalidArrayError(f"{name} must have shape (N, 3, H, W); got {array.shape}.")
    batch, channels, height, width = array.shape
    if batch < 1:
        raise InvalidArrayError(f"{name} must contain at least one image.")
    if channels != 3:
        raise InvalidArrayError(f"{name} must contain RGB data with C=3; got C={channels}.")
    if not 1 <= height <= MAX_DIMENSION or not 1 <= width <= MAX_DIMENSION:
        raise InvalidArrayError(
            f"{name} dimensions must be between 1 and {MAX_DIMENSION}; got {width}x{height}."
        )
    if array.dtype != np.uint8:
        raise InvalidArrayError(f"{name} must use dtype uint8; got {array.dtype}.")
    return batch, channels, height, width


def copy_nchw(array: np.ndarray, *, name: str = "array") -> np.ndarray:
    validate_nchw(array, name=name)
    return np.ascontiguousarray(array.copy())


def nchw_image_to_hwc(array: np.ndarray, index: int = 0, *, name: str = "array") -> np.ndarray:
    batch, _, _, _ = validate_nchw(array, name=name)
    if not 0 <= index < batch:
        raise InvalidArrayError(f"{name} image index {index} is outside batch size {batch}.")
    return np.ascontiguousarray(array[index].transpose(1, 2, 0))


def hwc_to_nchw(array: np.ndarray, *, name: str = "image") -> np.ndarray:
    if not isinstance(array, np.ndarray) or array.ndim != 3 or array.shape[2] != 3:
        raise InvalidArrayError(f"{name} must have HWC RGB shape (H, W, 3); got {getattr(array, 'shape', None)}.")
    if array.dtype != np.uint8:
        raise InvalidArrayError(f"{name} must use dtype uint8; got {array.dtype}.")
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def rgb_to_rgba(array: np.ndarray) -> np.ndarray:
    """Add opaque alpha only at the private native-worker boundary."""
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise InvalidArrayError("Native RGB input must have HWC uint8 shape with three channels.")
    result = np.empty((*array.shape[:2], 4), dtype=np.uint8)
    result[..., :3] = array
    result[..., 3] = 255
    return np.ascontiguousarray(result)


def rgba_to_rgb_nchw(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[2] != 4 or array.dtype != np.uint8:
        raise InvalidArrayError("Native RGBA output must have HWC uint8 shape with four channels.")
    return hwc_to_nchw(np.ascontiguousarray(array[..., :3]), name="native RGB output")


def srgb_to_worker(rgb: np.ndarray) -> np.ndarray:
    """Convert HWC sRGB RGB data to the RTX Video worker's gamma-2.2 RGBA data."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise InvalidArrayError("RGB input must have HWC uint8 shape with three channels.")
    rgba = rgb_to_rgba(rgb)
    values = rgba[..., :3].astype(np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    rgba[..., :3] = np.rint(np.clip(linear, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8) # pylint: disable=unsupported-assignment-operation
    return np.ascontiguousarray(rgba)


def worker_to_srgb_rgb(data: bytes | bytearray | memoryview, width: int, height: int) -> np.ndarray:
    """Convert packed worker RGBA output to an HWC sRGB RGB array."""
    rgba = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4).copy()
    values = (rgba[..., :3].astype(np.float32) / 255.0) ** 2.2
    rgb = np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1.0 / 2.4) - 0.055)
    rgba[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgba[..., :3])
