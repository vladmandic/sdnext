"""First-frame feature construction: the recovered 16-channel network input.
Channel layout (NHWC, float32 holding half-rounded values):
  0-2  deterministic Gaussian noise (per network pixel and frame index)
  3    constant 1
  4-6  scaled colour ((half(c) - 0.5) * 0.125), repeated in 7-9
  10   normalised style, 11 local tone, 12 local structure,
  13   skin structure, 14 automatic-mask structure, 15 zero
The network runs on a vendor-aligned extent (at least 320, multiples of 64)
the image sits at the origin and the extension mirrors the image without
repeating the edge (index 2*extent-2-x) while the noise is regenerated from the
network coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

MINIMUM_EXTENT = 320
EXTENT_MULTIPLE = 64
FEATURE_CHANNELS = 16


def half(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32)


@dataclass(frozen=True)
class NetworkGeometry:
    output_width: int
    output_height: int
    network_width: int
    network_height: int

    @staticmethod
    def aligned_extent(extent: int) -> int:
        minimum = max(MINIMUM_EXTENT, extent)
        return (minimum + EXTENT_MULTIPLE - 1) // EXTENT_MULTIPLE * EXTENT_MULTIPLE

    @classmethod
    def vendor_aligned(cls, width: int, height: int) -> "NetworkGeometry":
        return cls(width, height, cls.aligned_extent(width), cls.aligned_extent(height))

    @classmethod
    def identity(cls, width: int, height: int) -> "NetworkGeometry":
        return cls(width, height, width, height)

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output extent must be positive")
        if self.network_width < self.output_width or self.network_height < self.output_height:
            raise ValueError("network extent must cover the output")

    @property
    def is_identity(self) -> bool:
        return (self.network_width, self.network_height) == (self.output_width, self.output_height)

    @staticmethod
    def extended_indices(count: int, extent: int) -> np.ndarray:
        index = np.arange(count)
        return np.where(index < extent, index, np.maximum(0, 2 * extent - 2 - index))

    def source_rows(self) -> np.ndarray:
        return self.extended_indices(self.network_height, self.output_height)

    def source_columns(self) -> np.ndarray:
        return self.extended_indices(self.network_width, self.output_width)

    def crop(self, array: np.ndarray) -> np.ndarray:
        return array[: self.output_height, : self.output_width]


def _dynamic_shift_mix(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.uint32, copy=False)
    shift = (value >> np.uint32(28)) + np.uint32(4)
    mixed = value ^ (value >> shift)
    return (mixed * np.uint32(0x108EF2D9)).astype(np.uint32)


def _uniform24(value: np.ndarray) -> np.ndarray:
    mixed = _dynamic_shift_mix(value)
    bits = (mixed >> np.uint32(30)) ^ (mixed >> np.uint32(8))
    return (bits + np.uint32(1)).astype(np.float32) * np.float32(5.960464477539063e-8)


def deterministic_noise(height: int, width: int, frame_index: int = 0) -> np.ndarray:
    """The three half-rounded Gaussian noise channels for a network extent."""
    yy, xx = np.indices((height, width), dtype=np.uint32)
    seed = yy * np.uint32(0xD8163841)
    seed ^= xx * np.uint32(0x8DA6B343)
    seed ^= np.uint32((int(frame_index) * 0x9E3779B9) & 0xFFFFFFFF)
    seed ^= np.uint32(0x243F6A88)
    multiplied = _dynamic_shift_mix(seed)
    mixed = multiplied ^ (multiplied >> np.uint32(22))
    radius_a_uniform = _uniform24(mixed * np.uint32(0xCAA5B80D) + np.uint32(0x21DD796B))
    angle_b_uniform = _uniform24(mixed * np.uint32(0x83232C31) + np.uint32(0x3463E0AC))
    radius_b_uniform = _uniform24(mixed * np.uint32(0x2C9277B5) + np.uint32(0xAC564B05))
    angle_a_uniform = _uniform24(mixed * np.uint32(0xFA6DC5F9) + np.uint32(0x4712A88E))
    radius_a = np.sqrt(np.float32(-2) * np.log(radius_a_uniform)).astype(np.float32)
    radius_b = np.sqrt(np.float32(-2) * np.log(radius_b_uniform)).astype(np.float32)
    tau = np.float32(6.2831854820251465)
    angle_a = (tau * angle_a_uniform).astype(np.float32)
    angle_b = (tau * angle_b_uniform).astype(np.float32)
    return np.stack(
        (
            half(radius_b * np.cos(angle_a)),
            half(radius_b * np.sin(angle_a)),
            half(radius_a * np.cos(angle_b)),
        ),
        axis=-1,
    )


def scaled_color(color: np.ndarray) -> np.ndarray:
    sampled = half(color)
    centered = half(sampled - np.float32(0.5))
    return half(centered * np.float32(0.125))


def make_features(
    color: np.ndarray,
    *,
    frame_index: int = 0,
    geometry: NetworkGeometry | None = None,
    normalized_style: float = 0.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.0,
    skin_structure_strength: float | None = None,
    mask_structure_strength: float | None = None,
    control_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the (network_height, network_width, 16) float32 feature tensor.

    ``color`` is (height, width, 3) float32 in [0, 1]
    ``control_mask`` has the
    same shape (red: blend in the postprocessor, green: tone, blue: structure).
    """
    color = np.asarray(color, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color must be (height, width, 3)")
    height, width = color.shape[:2]
    if geometry is None:
        geometry = NetworkGeometry.identity(width, height)
    if (geometry.output_width, geometry.output_height) != (width, height):
        raise ValueError("geometry output extent must match the colour image")
    if control_mask is not None:
        control_mask = np.asarray(control_mask, dtype=np.float32)
        if control_mask.shape != color.shape:
            raise ValueError("control mask must match the colour image shape")
    style = half(normalized_style)
    tone = half(local_tone_strength)
    if control_mask is not None:
        structure = np.float32(0)
        skin_structure = np.float32(0)
        automatic_structure = np.float32(0)
    elif skin_structure_strength is not None or mask_structure_strength is not None:
        skin_val = skin_structure_strength if skin_structure_strength is not None else -1.0
        auto_val = mask_structure_strength if mask_structure_strength is not None else -1.0
        enabled = max(skin_val, auto_val) >= 0
        structure = half(1 if enabled else local_structure_strength)
        skin_structure = half(
            (skin_val if skin_val >= 0 else local_structure_strength)
            if enabled else -1
        )
        automatic_structure = half(
            (auto_val if auto_val >= 0 else local_structure_strength)
            if enabled else -1
        )
    else:
        structure = half(local_structure_strength)
        skin_structure = np.float32(-1)
        automatic_structure = np.float32(-1)
    rows = geometry.source_rows()
    columns = geometry.source_columns()
    extended = color[rows[:, None], columns[None, :], :]
    features = np.zeros((geometry.network_height, geometry.network_width, FEATURE_CHANNELS), dtype=np.float32)
    features[..., 0:3] = deterministic_noise(geometry.network_height, geometry.network_width, frame_index)
    features[..., 3] = 1
    scaled = scaled_color(extended)
    features[..., 4:7] = scaled
    features[..., 7:10] = scaled
    features[..., 10] = style
    if control_mask is not None:
        mask = control_mask[rows[:, None], columns[None, :], :]
        features[..., 11] = half(mask[..., 1] * np.float32(local_tone_strength))
        features[..., 12] = half(mask[..., 2] * np.float32(local_structure_strength))
    else:
        features[..., 11] = tone
        features[..., 12] = structure
    features[..., 13] = skin_structure
    features[..., 14] = automatic_structure
    return features


NR_PROFILES: dict[str, dict[str, float]] = {
    "Standard": {"normalized_style": 0.0, "local_tone_strength": 1.0, "local_structure_strength": 1.0},
    "Natural": {"normalized_style": 1.0 / 128, "local_tone_strength": 1.0, "local_structure_strength": 1.0},
    "Cinematic": {"normalized_style": 2.0 / 128, "local_tone_strength": 1.0, "local_structure_strength": 1.0},
}
