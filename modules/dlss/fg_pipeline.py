from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .fg_gpu_ops import box2_torch, compose_fg_torch, photometric_error_torch, warp_torch
from .fg_model import NeuralFrameGenModel, _up2

FG_PROFILES = {
    "None": {},
    "2x": {"factor": 2, "mode": "fps"},
    "3x": {"factor": 3, "mode": "fps"},
    "4x": {"factor": 4, "mode": "fps"},
    "8x": {"factor": 8, "mode": "fps"},
    "2x slowmo": {"factor": 2, "mode": "slowmo"},
    "3x slowmo": {"factor": 3, "mode": "slowmo"},
    "4x slowmo": {"factor": 4, "mode": "slowmo"},
    "8x slowmo": {"factor": 8, "mode": "slowmo"},
}

_SPEC_PATH = pathlib.Path(__file__).with_name("fg_spec.json")


def load_fg_weight_spec() -> dict[str, dict[str, Any]]:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))["tensors"]


def validate_fg_weights(weights: dict[str, torch.Tensor]) -> None:
    spec = load_fg_weight_spec()
    missing = sorted(set(spec) - set(weights))
    if missing:
        raise ValueError(f"DLSS FrameGen weights are missing {len(missing)} tensors, first: {missing[:3]}")
    for name, entry in spec.items():
        if list(weights[name].shape) != entry["shape"]:
            raise ValueError(f"{name}: expected shape {entry['shape']}, got {list(weights[name].shape)}")


def load_fg_weights(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load FrameGen weights from .safetensors."""
    from safetensors.torch import load_file

    weights = load_file(str(path))
    validate_fg_weights(weights)
    return weights


@dataclass
class DLSSFGOptions:
    profile: str = "2x"
    factor: int = 2
    mode: str = "fps"
    scene_cut_threshold: float = 0.4

    def __post_init__(self) -> None:
        if self.profile in FG_PROFILES:
            preset = FG_PROFILES[self.profile]
            if "factor" in preset and self.factor == 2:
                self.factor = preset["factor"]
            if "mode" in preset and self.mode == "fps":
                self.mode = preset["mode"]


    def __str__(self) -> str:
        return f"DLSSFGOptions(profile={self.profile} factor={self.factor} mode={self.mode} scene={self.scene_cut_threshold})"

class DLSSFGPipeline:
    path: str = ""
    dtype: torch.dtype = torch.float32

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype | str | None = None,
        graphs: bool = False,
    ):
        validate_fg_weights(weights)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.graphs = graphs
        self.model = NeuralFrameGenModel(weights).eval()

        if dtype is not None:
            if dtype == "fast":
                self.dtype = torch.float16 if self.device.type != "cpu" else torch.float32
            elif isinstance(dtype, str):
                self.dtype = getattr(torch, dtype)
            else:
                self.dtype = dtype
        else:
            self.dtype = torch.float16 if (self.device.type != "cpu" and torch.cuda.is_available()) else torch.float32

        self.model = self.model.to(self.device, self.dtype)

    def __str__(self) -> str:
        return f'DLSSFGPipeline(model="{self.path}" device={self.device} dtype={self.dtype} graphs={self.graphs})'

    @classmethod
    def from_safetensors(
        cls,
        path: str | pathlib.Path,
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype | str | None = None,
        graphs: bool = False,
    ) -> "DLSSFGPipeline":
        cls.path = str(path)
        return cls(load_fg_weights(path), device=device, dtype=dtype, graphs=graphs)

    def _to_tensor(self, frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            x = frame
            if x.ndim == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1)
            if x.ndim == 3:
                x = x.unsqueeze(0)
        else:
            arr = np.asarray(frame)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr))
            if x.ndim == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
        return x.to(self.device, self.dtype)

    @torch.no_grad()
    def synthesize(self, a_full: torch.Tensor, b_full: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Half-resolution export [N, 8, h, w]: flowA.xy, flowB.xy, mask logit, residual.rgb."""
        a = box2_torch(a_full)
        b = box2_torch(b_full)
        err = photometric_error_torch(a, b)
        zero = torch.zeros_like(err)
        phases = torch.as_tensor(t, device=a.device, dtype=a.dtype).reshape(-1, 1, 1, 1)
        phase = torch.zeros_like(err) + phases
        cand_a = torch.cat([a, err], 1)
        cand_b = torch.cat([b, err], 1)

        f0, m0, r0 = self.model.forward_block0(torch.cat([cand_a, cand_b, zero, phase], 1))
        f0, m0, r0 = _up2(f0), _up2(m0), _up2(r0)
        warped_a = warp_torch(cand_a, 2 * f0[:, 0], 2 * f0[:, 1])
        warped_b = warp_torch(cand_b, 2 * f0[:, 2], 2 * f0[:, 3])

        f1, m1, r1 = self.model.forward_block1(torch.cat([warped_a, warped_b, f0, m0, zero, r0, phase], 1))
        return torch.cat([2 * f0 + f1, m0 + m1, r0 + r1], 1)

    @torch.no_grad()
    def compose(self, a_full: torch.Tensor, b_full: torch.Tensor, export: torch.Tensor) -> torch.Tensor:
        """Full-resolution frames [N, 3, H, W] from the half-res export."""
        return compose_fg_torch(a_full, b_full, export)

    @torch.no_grad()
    def interpolate(
        self,
        a: np.ndarray | torch.Tensor,
        b: np.ndarray | torch.Tensor,
        t: float = 0.5,
    ) -> np.ndarray:
        """Interpolate one intermediate frame between a and b at relative phase t in (0, 1)."""
        a_t = self._to_tensor(a)
        b_t = self._to_tensor(b)
        if a_t.shape[2:] != b_t.shape[2:]:
            raise ValueError(f"Frame sizes differ: {a_t.shape[2:]} vs {b_t.shape[2:]}")
        export = self.synthesize(a_t, b_t, t)
        out = self.compose(a_t, b_t, export)
        return out.squeeze(0).permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()

    @torch.no_grad()
    def generate(
        self,
        a: np.ndarray | torch.Tensor,
        b: np.ndarray | torch.Tensor,
        factor: int = 2,
    ) -> list[np.ndarray]:
        """Generate factor - 1 intermediate frames between a and b in a single batch."""
        if factor < 2:
            raise ValueError("factor must be >= 2")
        a_t = self._to_tensor(a)
        b_t = self._to_tensor(b)
        n = factor - 1
        phases = torch.tensor([k / factor for k in range(1, factor)], device=a_t.device, dtype=a_t.dtype)
        a_n, b_n = a_t.expand(n, -1, -1, -1), b_t.expand(n, -1, -1, -1)
        export = self.synthesize(a_n, b_n, phases)
        out = self.compose(a_n, b_n, export)
        frames = out.permute(0, 2, 3, 1).float().clamp(0.0, 1.0).cpu().numpy()
        return [frames[i] for i in range(n)]


class DLSSFGSession:
    """Stateful frame interpolation session for streaming video frames."""

    def __init__(self, pipeline: DLSSFGPipeline, options: DLSSFGOptions | None = None, **kwargs: Any):
        self.pipeline = pipeline
        if options is not None:
            self.options = options
        else:
            self.options = DLSSFGOptions(**kwargs)
        self.prev_frame: np.ndarray | None = None

    def __str__(self) -> str:
        return f"DLSSFGSession(options={self.options})"

    def reset(self) -> None:
        self.prev_frame = None

    def __call__(self, frame: np.ndarray) -> list[np.ndarray]:
        """Process incoming frame. Returns list of output frames: [intermediates..., frame]."""
        curr_frame = np.asarray(frame, dtype=np.float32)
        if self.prev_frame is None:
            self.prev_frame = curr_frame
            return [curr_frame]

        # Check scene cut threshold
        diff = np.mean(np.abs(curr_frame - self.prev_frame))
        if diff > self.options.scene_cut_threshold:
            # Scene cut: repeat frames rather than interpolating across scene boundary
            intermediates = [self.prev_frame for _ in range(self.options.factor - 1)]
        else:
            intermediates = self.pipeline.generate(self.prev_frame, curr_frame, factor=self.options.factor)

        self.prev_frame = curr_frame
        return intermediates + [curr_frame]
