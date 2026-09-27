from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

# from .composition import blend_mask, compose_detail, compose_head # used for cpu fallback
from .nr_features import NR_PROFILES, NetworkGeometry, deterministic_noise, half, make_features, scaled_color
from .nr_gpu_ops import compose_detail_torch, compose_head_torch, compose_temporal_torch, make_features_torch, sample_history_torch


BLEND_SCALE = 0.73974609375  # half(blend_scale) of the recovered package


def normalize_pixel_motion(
    pixel_motion: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    effective_width: int,
    effective_height: int,
    jitter_dx: float = 0.0,
    jitter_dy: float = 0.0,
) -> np.ndarray:
    """Engine pixel motion (H, W, 2) -> normalised history-UV offsets, with an optional previous-minus-current jitter delta."""
    pixel_motion = np.asarray(pixel_motion, dtype=np.float32)
    if pixel_motion.ndim != 3 or pixel_motion.shape[2] != 2:
        raise ValueError("pixel motion must be (height, width, 2)")
    if effective_width <= 0 or effective_height <= 0:
        raise ValueError("effective extent must be positive")
    for value in (scale_x, scale_y, jitter_dx, jitter_dy):
        if not np.isfinite(value):
            raise ValueError("motion scale and jitter must be finite")
    out = np.empty_like(pixel_motion)
    out[..., 0] = pixel_motion[..., 0] * np.float32(scale_x / effective_width) + np.float32(jitter_dx / effective_width)
    out[..., 1] = pixel_motion[..., 1] * np.float32(scale_y / effective_height) + np.float32(jitter_dy / effective_height)
    return out


def _catmull_coordinates(normalized: np.ndarray, dimension: int):
    pixel = normalized * np.float32(dimension) - np.float32(0.5)
    base_index = np.floor(pixel)
    t = np.clip(pixel - base_index, 0, 1).astype(np.float32)
    square = t * t
    cube = square * t
    w0 = -0.5 * t + square - 0.5 * cube
    w1 = 1 - 2.5 * square + 1.5 * cube
    w2 = 0.5 * t + 2 * square - 1.5 * cube
    w3 = -0.5 * square + 0.5 * cube
    g = w1 + w2
    base = base_index + np.float32(0.5)
    lower, upper = np.float32(0.5), np.float32(dimension) - np.float32(0.5)
    return (
        np.clip(base - 1, lower, upper), np.clip(base + w2 / g, lower, upper), np.clip(base + 2, lower, upper),
        w0.astype(np.float32), w3.astype(np.float32), g.astype(np.float32),
    )


def _sample_linear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of (H, W, C) at pixel-centre coordinates (x, y), clamped to the edge."""
    height, width = image.shape[:2]
    px = x - np.float32(0.5)
    py = y - np.float32(0.5)
    x0 = np.clip(np.floor(px), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(py), 0, height - 1).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = np.clip(px - x0, 0, 1).astype(np.float32)[..., None]
    ty = np.clip(py - y0, 0, 1).astype(np.float32)[..., None]
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


def sample_history(history: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Five-tap Catmull-Rom approximation (cross of bilinear taps) at normalised coordinates (u, v)."""
    history = np.asarray(history, dtype=np.float32)
    height, width = history.shape[:2]
    x_outer0, x_middle, x_outer3, x_w0, x_w3, x_g = _catmull_coordinates(np.asarray(u, dtype=np.float32), width)
    y_outer0, y_middle, y_outer3, y_w0, y_w3, y_g = _catmull_coordinates(np.asarray(v, dtype=np.float32), height)
    weights = [x_w0 * y_g, x_g * y_w0, x_g * y_g, x_g * y_w3, x_w3 * y_g]
    taps = [
        _sample_linear(history, x_outer0, y_middle), _sample_linear(history, x_middle, y_outer0),
        _sample_linear(history, x_middle, y_middle), _sample_linear(history, x_middle, y_outer3),
        _sample_linear(history, x_outer3, y_middle),
    ]
    total = sum(w[..., None] * tap for w, tap in zip(weights, taps))
    return (total / sum(weights)[..., None]).astype(np.float32)


def _closest_depth_offsets(depth: np.ndarray, inverted: bool):
    """Per-pixel source coordinate of the closest of the pixel and its four diagonals (dormant vendor branch)."""
    height, width = depth.shape[:2]
    yy, xx = np.indices((height, width))
    best_x, best_y, best = xx.copy(), yy.copy(), depth[..., 0].copy()
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cx = np.clip(xx + dx, 0, width - 1)
        cy = np.clip(yy + dy, 0, height - 1)
        candidate = depth[cy, cx, 0]
        closer = candidate > best if inverted else candidate < best
        best_x = np.where(closer, cx, best_x)
        best_y = np.where(closer, cy, best_y)
        best = np.where(closer, candidate, best)
    return best_x, best_y


def make_temporal_features(
    color: np.ndarray,
    history: np.ndarray,
    motion: np.ndarray,
    *,
    frame_index: int,
    depth: np.ndarray | None = None,
    depth_guide: str = "observed",
    depth_inverted: bool = False,
    normalized_style: float = 0.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.0,
    skin_structure_strength: float | None = None,
    mask_structure_strength: float | None = None,
    control_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Logical-size (H, W, 16) features: first-frame layout with reprojected history in channels 7-9."""
    color = np.asarray(color, dtype=np.float32)
    history = np.asarray(history, dtype=np.float32)
    motion = np.asarray(motion, dtype=np.float32)
    height, width = color.shape[:2]
    if history.shape != color.shape:
        raise ValueError("history must match the colour shape")
    if motion.shape != (height, width, 2):
        raise ValueError("motion must be (height, width, 2)")
    features = make_features(
        color, frame_index=frame_index, normalized_style=normalized_style, local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength, skin_structure_strength=skin_structure_strength,
        mask_structure_strength=mask_structure_strength, control_mask=control_mask,
    )
    yy, xx = np.indices((height, width))
    if depth_guide == "closest":
        if depth is None:
            raise ValueError("closest-depth guide needs a depth map")
        sx, sy = _closest_depth_offsets(np.asarray(depth, dtype=np.float32).reshape(height, width, -1), depth_inverted)
        sampled_motion = motion[sy, sx]
    elif depth_guide == "observed":
        sampled_motion = motion
    else:
        raise ValueError("depth_guide must be 'observed' or 'closest'")
    u = (xx.astype(np.float32) + np.float32(0.5)) / np.float32(width) + sampled_motion[..., 0]
    v = (yy.astype(np.float32) + np.float32(0.5)) / np.float32(height) + sampled_motion[..., 1]
    features[..., 7:10] = scaled_color(sample_history(history, u, v))
    return features


def extend_features(features: np.ndarray, geometry: NetworkGeometry, frame_index: int) -> np.ndarray:
    """Mirror logical features onto the network extent and regenerate the noise in the extension."""
    if geometry.is_identity:
        return features
    rows, columns = geometry.source_rows(), geometry.source_columns()
    extended = features[rows[:, None], columns[None, :], :].copy()
    noise = deterministic_noise(geometry.network_height, geometry.network_width, frame_index)
    outside = (rows[:, None] != np.arange(geometry.network_height)[:, None]) | (columns[None, :] != np.arange(geometry.network_width)[None, :])
    extended[..., 0:3] = np.where(outside[..., None], noise, extended[..., 0:3])
    return extended


def compose_temporal(
    head: np.ndarray,
    color: np.ndarray,
    features: np.ndarray,
    *,
    blend_scale: float = BLEND_SCALE,
    control_mask: np.ndarray | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    """predicted + alpha * (history - predicted), alpha = clamp(sigmoid(half(logit)) * half(blend_scale))."""
    head = np.asarray(head, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if head.shape[:2] != color.shape[:2] or features.shape[:2] != color.shape[:2] or features.shape[2] != 16:
        raise ValueError("head, colour and features must share height and width; features need 16 channels")
    logit = half(head[..., 3:4])
    alpha = np.clip(1 / (1 + np.exp(-logit)) * half(blend_scale), 0, 1)
    predicted = np.clip(color + half(head[..., :3]) * np.float32(0.25), 0, 1)
    history = features[..., 7:10] * np.float32(8) + np.float32(0.5)
    temporal = predicted + alpha * (history - predicted)
    if control_mask is None and intensity == 1:
        return temporal.astype(np.float32)
    blend = np.float32(intensity) if control_mask is None else np.asarray(control_mask, dtype=np.float32)[..., :1] * np.float32(intensity)
    blend = np.clip(blend, 0, 1)
    return np.clip(color + blend * (temporal - color), 0, 1).astype(np.float32)


class FlowMotionEstimator:
    """Dense optical flow (OpenCV DIS) from the current frame to the previous one, as normalised history-UV offsets."""

    def __init__(self, preset: str = "ultrafast", target_dim: int = 960):
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("optical-flow motion needs OpenCV: pip install 'mlxdlss[video]'") from error
        presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
        self._cv2 = cv2
        self._flow = cv2.DISOpticalFlow_create(presets.get(preset, cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST))
        self.target_dim = max(240, int(target_dim))

    def __call__(self, current: np.ndarray | torch.Tensor, previous: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(current, torch.Tensor):
            current = current.detach().cpu().numpy()
        if isinstance(previous, torch.Tensor):
            previous = previous.detach().cpu().numpy()

        h, width = current.shape[:2]
        factor = max(1, int(round(max(h, width) / self.target_dim)))

        if factor > 1:
            curr_sub = current[::factor, ::factor]
            prev_sub = previous[::factor, ::factor]
            gc = (np.clip(curr_sub[..., 0] * 0.2126 + curr_sub[..., 1] * 0.7152 + curr_sub[..., 2] * 0.0722, 0, 1) * 255).astype(np.uint8)
            gp = (np.clip(prev_sub[..., 0] * 0.2126 + prev_sub[..., 1] * 0.7152 + prev_sub[..., 2] * 0.0722, 0, 1) * 255).astype(np.uint8)
            flow_small = self._flow.calc(gc, gp, None)
            flow = self._cv2.resize(flow_small * float(factor), (width, h), interpolation=self._cv2.INTER_LINEAR)
        else:
            gc = (np.clip(current[..., 0] * 0.2126 + current[..., 1] * 0.7152 + current[..., 2] * 0.0722, 0, 1) * 255).astype(np.uint8)
            gp = (np.clip(previous[..., 0] * 0.2126 + previous[..., 1] * 0.7152 + previous[..., 2] * 0.0722, 0, 1) * 255).astype(np.uint8)
            flow = self._flow.calc(gc, gp, None)

        return normalize_pixel_motion(flow, scale_x=1, scale_y=1, effective_width=width, effective_height=h)


def zero_motion(current: np.ndarray, previous: np.ndarray) -> np.ndarray: # pylint: disable=unused-argument
    return np.zeros((*current.shape[:2], 2), dtype=np.float32)


@dataclass
class DLSSNRTemporalOptions:
    profile: str = "standard"
    scale: float = 1.0
    blend_scale: float = BLEND_SCALE
    intensity: float = 1.0
    scene_cut_threshold: float = 0.3
    detail_strength: float = 1.0
    colour_strength: float = 1.0
    detail_radius: float = 4.0
    normalized_style: float | None = None
    local_tone_strength: float | None = None
    local_structure_strength: float | None = None
    skin_structure_strength: float | None = None
    mask_structure_strength: float | None = None

    def __str__(self) -> str:
        return f"DLSSNRTemporalOptions(profile={self.profile} scale={self.scale} blend={self.blend_scale} intensity={self.intensity} scene={self.scene_cut_threshold} detail={self.detail_strength} colour={self.colour_strength} radius={self.detail_radius} normalized={self.normalized_style} tone={self.local_tone_strength} structure={self.local_structure_strength} skin={self.skin_structure_strength} mask={self.mask_structure_strength})"


class DLSSNRTemporalSession:
    """Frame-sequence processor with display history, motion reprojection and the learned blend.

    ``motion`` is a callable ``(current, previous) -> (H, W, 2)`` normalised offsets (default: optical flow),
    or engine motion passed per frame to ``process``. A scene cut (mean absolute luma change above the
    threshold) or ``reset`` clears the history and restarts the noise frame index, like the Swift backend.
    """

    def __init__(self, pipeline, *, options: DLSSNRTemporalOptions | None = None, motion: Callable | str = "flow"):
        self.pipeline = pipeline
        self.options = options or DLSSNRTemporalOptions()
        if self.options.profile not in NR_PROFILES:
            raise ValueError(f"profile must be one of {tuple(NR_PROFILES)}")
        if motion == "flow":
            self.motion = FlowMotionEstimator(preset="ultrafast", target_dim=960)
        elif motion == "zero":
            self.motion = zero_motion
        elif callable(motion):
            self.motion = motion
        else:
            raise ValueError("motion must be 'flow', 'zero' or a callable")
        self.history: np.ndarray | torch.Tensor | None = None
        self.previous: np.ndarray | torch.Tensor | None = None
        self.frame_index = 0
        self.scene_cuts = 0

    def __str__(self) -> str:
        return f"DLSSNRTemporalSession(options={self.options} motion={self.motion.__class__.__name__})"

    def reset(self) -> None:
        self.history = None
        self.previous = None
        self.frame_index = 0

    def _controls(self) -> dict[str, float]:
        controls = dict(NR_PROFILES[self.options.profile])
        for key in ("normalized_style", "local_tone_strength", "local_structure_strength", "skin_structure_strength", "mask_structure_strength"):
            value = getattr(self.options, key)
            if value is not None:
                controls[key] = value
        return controls

    def __call__(self, frame: np.ndarray | torch.Tensor, *, motion: np.ndarray | torch.Tensor | None = None, control_mask: np.ndarray | torch.Tensor | None = None) -> np.ndarray:
        device = getattr(self.pipeline, "device", "cpu")

        frame_t = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(np.asarray(frame, dtype=np.float32)).to(device)
        if self.options.scale != 1.0:
            h_proc = int(round(frame_t.shape[0] * self.options.scale))
            w_proc = int(round(frame_t.shape[1] * self.options.scale))
            frame_t = torch.nn.functional.interpolate(
                frame_t.permute(2, 0, 1).unsqueeze(0),
                size=(h_proc, w_proc),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
        height, width = frame_t.shape[:2]
        geometry = NetworkGeometry.vendor_aligned(width, height)
        controls = self._controls()
        intensity = self.options.intensity
        control_mask_t = control_mask if (control_mask is None or isinstance(control_mask, torch.Tensor)) else torch.from_numpy(np.asarray(control_mask, dtype=np.float32)).to(device)

        if self.previous is not None and (self.previous.shape != frame_t.shape or self._is_scene_cut(frame_t)):
            self.reset()
            self.scene_cuts += 1

        if self.history is None:
            features_t = make_features_torch(
                frame_t,
                frame_index=self.frame_index,
                geometry=geometry,
                control_mask=control_mask_t,
                dtype=self.pipeline.dtype,
                **controls,
            )
            head_raw = self.pipeline._forward_model(features_t) # pylint: disable=protected-access
            head = head_raw.squeeze(0)[:height, :width]
            output = compose_head_torch(head, frame_t, control_mask=control_mask_t, intensity=intensity)
        else:
            if motion is None:
                motion = self.motion(frame_t, self.previous)
            motion_t = motion if isinstance(motion, torch.Tensor) else torch.from_numpy(np.asarray(motion, dtype=np.float32)).to(device)

            # Coordinate mesh on GPU
            yy = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).expand(height, width)
            xx = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).expand(height, width)
            u = (xx + 0.5) / float(width) + motion_t[..., 0]
            v = (yy + 0.5) / float(height) + motion_t[..., 1]

            hist_t = self.history if isinstance(self.history, torch.Tensor) else torch.from_numpy(np.asarray(self.history, dtype=np.float32)).to(device)
            sampled_hist = sample_history_torch(hist_t, u, v)

            features_t = make_features_torch(
                frame_t,
                frame_index=self.frame_index,
                geometry=geometry,
                control_mask=control_mask_t,
                history_reprojected=sampled_hist,
                dtype=self.pipeline.dtype,
                **controls,
            )
            head_raw = self.pipeline._forward_model(features_t) # pylint: disable=protected-access
            head = head_raw.squeeze(0)[:height, :width]
            output = compose_temporal_torch(head, frame_t, features_t, blend_scale=self.options.blend_scale, control_mask=control_mask_t, intensity=intensity)

        if control_mask_t is None:
            self.history = output
        else:
            blend = torch.clamp(control_mask_t[..., :1], 0.0, 1.0)
            self.history = torch.clamp(frame_t + blend * (output - frame_t), 0.0, 1.0)

        self.previous = frame_t
        self.frame_index += 1

        detailed = compose_detail_torch(
            frame_t,
            output,
            detail_strength=self.options.detail_strength,
            colour_strength=self.options.colour_strength,
            radius=self.options.detail_radius,
        )
        if control_mask_t is not None:
            blend = torch.clamp(control_mask_t[..., :1], 0.0, 1.0)
            detailed = torch.clamp(frame_t + blend * (detailed - frame_t), 0.0, 1.0)

        detailed = torch.clamp(detailed, 0.0, 1.0)
        return detailed.detach().cpu().numpy()

    def _is_scene_cut(self, frame: np.ndarray | torch.Tensor) -> bool:
        if self.options.scene_cut_threshold <= 0:
            return False
        if isinstance(frame, torch.Tensor) and isinstance(self.previous, torch.Tensor):
            f_sub = frame[::8, ::8]
            p_sub = self.previous[::8, ::8]
            luma_curr = f_sub[..., 0] * 0.2126 + f_sub[..., 1] * 0.7152 + f_sub[..., 2] * 0.0722
            luma_prev = p_sub[..., 0] * 0.2126 + p_sub[..., 1] * 0.7152 + p_sub[..., 2] * 0.0722
            return float(torch.abs(luma_curr - luma_prev).mean().item()) > self.options.scene_cut_threshold
        f_curr = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else frame
        f_prev = self.previous.detach().cpu().numpy() if isinstance(self.previous, torch.Tensor) else self.previous
        c_sub = f_curr[::8, ::8]
        p_sub = f_prev[::8, ::8]
        l_c = c_sub[..., 0] * 0.2126 + c_sub[..., 1] * 0.7152 + c_sub[..., 2] * 0.0722
        l_p = p_sub[..., 0] * 0.2126 + p_sub[..., 1] * 0.7152 + p_sub[..., 2] * 0.0722
        return float(np.abs(l_c - l_p).mean()) > self.options.scene_cut_threshold
