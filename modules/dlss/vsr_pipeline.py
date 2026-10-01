from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .vsr_gpu_ops import compose_vsr_torch, prepare_vsr_input_torch, process_vsr_output_torch
from .vsr_model import NeuralVSRModel

# Standard RTX VSR Quality Presets
VSR_PROFILES = {
    "Low": {"quality": 1, "scale": 2.0, "detail_strength": 0.25},
    "Medium": {"quality": 2, "scale": 2.0, "detail_strength": 0.5},
    "High": {"quality": 3, "scale": 2.0, "detail_strength": 0.75},
    "Ultra": {"quality": 4, "scale": 2.0, "detail_strength": 1.0},
}

_SPEC_PATH = pathlib.Path(__file__).with_name("vsr_spec.json")


def load_vsr_weight_spec() -> dict[str, dict[str, Any]]:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))["tensors"]


def validate_vsr_weights(weights: dict[str, torch.Tensor]) -> None:
    spec = load_vsr_weight_spec()
    missing = sorted(set(spec) - set(weights))
    if missing:
        raise ValueError(f"DLSS VSR weights are missing {len(missing)} tensors, first: {missing[:3]}")
    for name, entry in spec.items():
        if list(weights[name].shape) != entry["shape"]:
            raise ValueError(f"{name}: expected shape {entry['shape']}, got {list(weights[name].shape)}")


def load_vsr_weights(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load VSR weights from .safetensors."""
    from safetensors.torch import load_file

    weights = load_file(str(path))
    validate_vsr_weights(weights)
    return weights


@dataclass
class DLSSVSROptions:
    profile: str = "Ultra"
    scale: float = 2.0
    detail_strength: float = 1.0
    colour_strength: float = 1.0
    detail_radius: float = 4.0
    target_width: int | None = None
    target_height: int | None = None

    def __post_init__(self) -> None:
        if self.profile in VSR_PROFILES:
            preset = VSR_PROFILES[self.profile]
            if "scale" in preset and self.scale == 2.0:
                self.scale = preset["scale"]
            if "detail_strength" in preset and self.detail_strength == 1.0:
                self.detail_strength = preset["detail_strength"]


class DLSSVSRPipeline:
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
        validate_vsr_weights(weights)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.graphs = graphs
        self._graph_cache: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self.model = NeuralVSRModel(weights).eval()

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
        return f'DLSSVSRPipeline(model="{self.path}" device={self.device} dtype={self.dtype} graphs={self.graphs})'

    @classmethod
    def from_safetensors(
        cls,
        path: str | pathlib.Path,
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype | str | None = None,
        graphs: bool = False,
    ) -> "DLSSVSRPipeline":
        cls.path = str(path)
        return cls(load_vsr_weights(path), device=device, dtype=dtype, graphs=graphs)

    def _forward_model(self, x_unshuffle: torch.Tensor) -> torch.Tensor:
        """Run the neural super resolution model, optionally accelerated with CUDA graphs."""
        if not self.graphs or self.device.type == "cpu" or not torch.cuda.is_available():
            return self.model(x_unshuffle)

        extent_key = (x_unshuffle.shape[2], x_unshuffle.shape[3])
        if extent_key not in self._graph_cache:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(device=self.device))
            static_in = x_unshuffle.detach().clone()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _ = self.model(static_in)
            torch.cuda.current_stream(device=self.device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_out = self.model(static_in)
            self._graph_cache[extent_key] = (graph, static_in, static_out)

        graph, static_in, static_out = self._graph_cache[extent_key]
        static_in.copy_(x_unshuffle)
        graph.replay()
        return static_out.clone()

    def _to_tensor(self, image: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if isinstance(image, torch.Tensor):
            x = image
            if x.ndim == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1)
            if x.ndim == 3:
                x = x.unsqueeze(0)
        else:
            arr = np.asarray(image)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr))
            if x.ndim == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
        h, w = x.shape[2], x.shape[3]
        return x.to(self.device, self.dtype), h, w

    @torch.no_grad()
    def enhance( # pylint: disable=unused-argument
        self,
        image: np.ndarray | torch.Tensor,
        *,
        profile: str = "ultra", # pylint: disable=unused-argument
        scale: float = 2.0,
        detail_strength: float = 1.0,
        colour_strength: float = 1.0,
        detail_radius: float = 4.0,
        target_width: int | None = None,
        target_height: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Upscale an image with 2x neural super-resolution and optional resample to target scale."""
        rgb_t, orig_h, orig_w = self._to_tensor(image)
        unshuf_t = prepare_vsr_input_torch(rgb_t, dtype=self.dtype)
        out_proj = self._forward_model(unshuf_t)
        neural_2x = process_vsr_output_torch(out_proj, orig_h, orig_w)

        output = compose_vsr_torch(
            rgb_t,
            neural_2x,
            scale=scale,
            target_width=target_width,
            target_height=target_height,
            detail_strength=detail_strength,
            colour_strength=colour_strength,
            detail_radius=detail_radius,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        out_np = output.squeeze(0).permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()
        return out_np


class DLSSVSRSession:
    """Session runner for DLSS-VSR super-resolution."""

    def __init__(self, pipeline: DLSSVSRPipeline, **options: Any):
        self.pipeline = pipeline
        self.options = DLSSVSROptions(**options)

    def __str__(self) -> str:
        return f"DLSSVSRSession(options={self.options})"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = self.pipeline.enhance(image, **self.options.__dict__)
        return np.clip(result, 0.0, 1.0)
