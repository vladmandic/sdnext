from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import nr_model as reference
from .nr_composition import blend_mask, compose_detail, compose_head, resample
from .nr_features import NR_PROFILES, NetworkGeometry, make_features
from .nr_gpu_ops import compose_detail_torch, compose_head_torch, make_features_torch


WEIGHT_FORMATS = tuple(f"dlssnr-logical-v{v}" for v in range(8, 19))
_SPEC_PATH = pathlib.Path(__file__).with_name("nr_spec.json")


def load_weight_spec() -> dict[str, dict[str, Any]]:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))["tensors"]


def validate_weights(weights: dict[str, torch.Tensor]) -> None:
    spec = load_weight_spec()
    missing = sorted(set(spec) - set(weights))
    if missing:
        raise ValueError(f"weights are missing {len(missing)} tensors, first: {missing[:3]}")
    for name, entry in spec.items():
        if list(weights[name].shape) != entry["shape"]:
            raise ValueError(f"{name}: expected shape {entry['shape']}, got {list(weights[name].shape)}")


def load_weights(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load logical safetensors (``dlssnr-logical-v8`` … ``v18``, fully logical)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if metadata.get("format") not in WEIGHT_FORMATS:
            raise ValueError(f"unsupported weight format {metadata.get('format')!r}; expected one of {WEIGHT_FORMATS[0]} … {WEIGHT_FORMATS[-1]}")
        if metadata.get("fully_logical") != "true":
            raise ValueError("weights must declare fully_logical=true")
        weights = {name: source.get_tensor(name) for name in source.keys()}
    validate_weights(weights)
    return weights


class DLSSNRPipeline:
    path = ''
    dtype = torch.float32
    """Runs the recovered transformer on a single RGB frame.

    ``precision='reference'`` computes in float32 with the E4M3/half rounding
    points of the recovered graph (bit-faithful to the reference); ``'fast'``
    runs the same graph in float16 on GPU devices.
    """

    def __init__(self, weights: dict[str, torch.Tensor], *, device: str | torch.device = "auto", dtype: torch.dtype | None = None, graphs: bool = False):
        self.dtype = dtype
        validate_weights(weights)
        self.device = device
        self.graphs = graphs
        self._graph_cache: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self.model = reference.NeuralRenderingModel(weights).eval()
        if dtype is not None:
            if dtype == "fast":
                self.dtype = torch.float16
            elif isinstance(dtype, str):
                self.dtype = getattr(torch, dtype)
            else:
                self.dtype = dtype
            self.model = self.model.to(self.dtype)
        self.model = self.model.to(self.device)

    def __str__(self) -> str:
        return f'DLSSNRPipeline(model="{self.path}" device={self.device} dtype={self.dtype} graphs={self.graphs})'

    @classmethod
    def from_safetensors(cls, path: str | pathlib.Path, *, device: str | torch.device = "auto", dtype: torch.dtype | None = None, graphs: bool = False) -> "DLSSNRPipeline":
        cls.path = str(path)
        return cls(load_weights(path), device=device, dtype=dtype, graphs=graphs)

    def _forward_model(self, features: torch.Tensor) -> torch.Tensor:
        """Run the neural rendering model, optionally accelerated via CUDA graphs for static shapes."""
        if not self.graphs or self.device == "cpu" or str(self.device) == "cpu" or not torch.cuda.is_available():
            return self.model(features)

        # Key on spatial extent
        extent_key = (features.shape[1], features.shape[2])
        if extent_key not in self._graph_cache:
            # Warm up on a private stream and capture
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(device=self.device))
            static_input = features.detach().clone()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _ = self.model(static_input)
            torch.cuda.current_stream(device=self.device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_output = self.model(static_input)
            self._graph_cache[extent_key] = (graph, static_input, static_output)

        graph, static_in, static_out = self._graph_cache[extent_key]
        static_in.copy_(features)
        graph.replay()
        return static_out.clone()

    @torch.no_grad()
    def run_features(self, features: np.ndarray) -> np.ndarray:
        """Raw network: (height, width, 16) features -> (height, width, 4) head."""
        return self.run_features_batch(np.asarray(features, dtype=np.float32)[None])[0]

    @torch.no_grad()
    def run_features_batch(self, features: np.ndarray) -> np.ndarray:
        """Raw network on a batch: (count, height, width, 16) -> (count, height, width, 4)."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 4 or features.shape[3] != 16:
            raise ValueError("features must be (count, height, width, 16)")
        if features.shape[1] % 64 or features.shape[2] % 64:
            raise ValueError("feature extent must be a multiple of 64 (use NetworkGeometry.vendor_aligned)")
        tensor = torch.from_numpy(np.ascontiguousarray(features)).to(self.device, self.dtype)
        head = self._forward_model(tensor)
        return head.detach().cpu().numpy()

    def _controls(self, profile, normalized_style, local_tone_strength, local_structure_strength, skin_structure_strength=None, mask_structure_strength=None):
        if profile not in NR_PROFILES:
            raise ValueError(f"profile must be one of {tuple(NR_PROFILES)}")
        controls = dict(NR_PROFILES[profile])
        if normalized_style is not None:
            controls["normalized_style"] = normalized_style
        if local_tone_strength is not None:
            controls["local_tone_strength"] = local_tone_strength
        if local_structure_strength is not None:
            controls["local_structure_strength"] = local_structure_strength
        if skin_structure_strength is not None:
            controls["skin_structure_strength"] = skin_structure_strength
        if mask_structure_strength is not None:
            controls["mask_structure_strength"] = mask_structure_strength
        return controls

    def prepare(
        self,
        image: np.ndarray,
        *,
        profile: str = "standard",
        scale: float = 1.0,
        processing_scale: float = 1.0,
        frame_index: int = 0,
        control_mask: np.ndarray | None = None,
        normalized_style: float | None = None,
        local_tone_strength: float | None = None,
        local_structure_strength: float | None = None,
        skin_structure_strength: float | None = None,
        mask_structure_strength: float | None = None,
    ) -> "PreparedFrame":
        """Resample to the processing scale and build the network features (no network call)."""
        controls = self._controls(profile, normalized_style, local_tone_strength, local_structure_strength, skin_structure_strength, mask_structure_strength)
        if not 1 <= processing_scale <= 4:
            raise ValueError("processing_scale must be within [1, 4]")
        if control_mask is not None and processing_scale != 1:
            raise ValueError("a control mask requires processing_scale=1")
        is_gpu = self.device != "cpu" and str(self.device) != "cpu"

        if is_gpu:
            if isinstance(image, torch.Tensor):
                source_t = image.to(self.device, dtype=torch.float32)
            else:
                source_t = torch.from_numpy(np.asarray(image, dtype=np.float32)).to(self.device)
            if scale != 1:
                h_proc = int(round(source_t.shape[0] * scale))
                w_proc = int(round(source_t.shape[1] * scale))
                source_t = torch.nn.functional.interpolate(
                    source_t.permute(2, 0, 1).unsqueeze(0),
                    size=(h_proc, w_proc),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

            processing_t = source_t
            if processing_scale != 1:
                h_proc = int(round(source_t.shape[0] * processing_scale))
                w_proc = int(round(source_t.shape[1] * processing_scale))
                processing_t = torch.nn.functional.interpolate(
                    source_t.permute(2, 0, 1).unsqueeze(0),
                    size=(h_proc, w_proc),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

            h_proc, w_proc = processing_t.shape[:2]
            geometry = NetworkGeometry.vendor_aligned(w_proc, h_proc)

            control_mask_t = None
            if control_mask is not None:
                if isinstance(control_mask, torch.Tensor):
                    control_mask_t = control_mask.to(self.device, dtype=torch.float32)
                else:
                    control_mask_t = torch.from_numpy(np.asarray(control_mask, dtype=np.float32)).to(self.device)

            features_t = make_features_torch(
                processing_t,
                frame_index=frame_index,
                geometry=geometry,
                control_mask=control_mask_t,
                dtype=self.dtype,
                **controls,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return PreparedFrame(
                source=source_t,
                processing=processing_t,
                features=features_t,
                geometry=geometry,
                control_mask=control_mask_t,
            )
        else:
            source = np.asarray(image, dtype=np.float32)
            if source.ndim != 3 or source.shape[2] != 3:
                raise ValueError("image must be (height, width, 3)")
            processing = source
            if processing_scale != 1:
                processing = resample(
                    source,
                    int(round(source.shape[1] * processing_scale)),
                    int(round(source.shape[0] * processing_scale)),
                )
            geometry = NetworkGeometry.vendor_aligned(processing.shape[1], processing.shape[0])
            features = make_features(processing, frame_index=frame_index, geometry=geometry, control_mask=control_mask, **controls)
            return PreparedFrame(
                source=source,
                processing=processing,
                features=features,
                geometry=geometry,
                control_mask=control_mask,
            )
    def finish(
        self,
        prepared: "PreparedFrame",
        head: np.ndarray | torch.Tensor,
        *,
        detail_strength: float = 1.0,
        colour_strength: float = 1.0,
        detail_radius: float = 4.0,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Compose the head over the frame, resample back and apply the detail/colour split."""
        # GPU postprocessing path
        if isinstance(head, torch.Tensor) and isinstance(prepared.processing, torch.Tensor):
            # head is (1, net_h, net_w, 4) -> squeeze batch if present
            head_t = head.squeeze(0) if head.ndim == 4 else head
            # Crop to actual processing extent
            head_cropped = head_t[: prepared.processing.shape[0], : prepared.processing.shape[1]]
            composed = compose_head_torch(
                head_cropped,
                prepared.processing,
                control_mask=prepared.control_mask,
                intensity=intensity,
            )
            if composed.shape[:2] != prepared.source.shape[:2]:
                composed = torch.nn.functional.interpolate(
                    composed.permute(2, 0, 1).unsqueeze(0),
                    size=(prepared.source.shape[0], prepared.source.shape[1]),
                    mode="bicubic",
                    align_corners=True,
                    antialias=True,
                ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

            output = compose_detail_torch(
                prepared.source,
                composed,
                detail_strength=detail_strength,
                colour_strength=colour_strength,
                radius=detail_radius,
            )
            if prepared.control_mask is not None:
                blend = torch.clamp(prepared.control_mask[..., :1], 0.0, 1.0)
                output = torch.clamp(prepared.source + blend * (output - prepared.source), 0.0, 1.0)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return output.detach().cpu().numpy()

        mask = prepared.control_mask
        head_np = head.detach().cpu().numpy() if isinstance(head, torch.Tensor) else head
        composed = compose_head(
            prepared.geometry.crop(head_np), prepared.processing,
            intensity=intensity
        )
        if composed.shape[:2] != prepared.source.shape[:2]:
            composed = resample(composed, prepared.source.shape[1], prepared.source.shape[0])
        output = compose_detail(prepared.source, composed, detail_strength=detail_strength, colour_strength=colour_strength, radius=detail_radius)
        if mask is not None:
            # Filtering an already masked residual spreads the change outside
            # the mask. Filter first, then mask once (also for soft masks).
            # Intensity stays before detail/clipping, matching unmasked runs.
            output = blend_mask(prepared.source, output, mask)
        return output

    def enhance(
        self,
        image: np.ndarray | torch.Tensor,
        *,
        profile: str = "standard",
        scale: float = 1.0,
        processing_scale: float = 1.0,
        detail_strength: float = 1.0,
        colour_strength: float = 1.0,
        detail_radius: float = 4.0,
        intensity: float = 1.0,
        frame_index: int = 0,
        control_mask: np.ndarray | torch.Tensor | None = None,
        normalized_style: float | None = None,
        local_tone_strength: float | None = None,
        local_structure_strength: float | None = None,
        skin_structure_strength: float | None = None,
        mask_structure_strength: float | None = None,
    ) -> np.ndarray:
        """Enhance one (height, width, 3) float32 RGB frame in [0, 1]."""
        prepared = self.prepare(
            image, profile=profile, scale=scale, processing_scale=processing_scale, frame_index=frame_index, control_mask=control_mask,
            normalized_style=normalized_style, local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            mask_structure_strength=mask_structure_strength,
        )
        if isinstance(prepared.features, torch.Tensor):
            head = self._forward_model(prepared.features)
        else:
            head = self.run_features(prepared.features)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self.finish(prepared, head, detail_strength=detail_strength, colour_strength=colour_strength, detail_radius=detail_radius, intensity=intensity)


@dataclass
class PreparedFrame:
    source: np.ndarray | torch.Tensor
    processing: np.ndarray | torch.Tensor
    features: np.ndarray | torch.Tensor
    geometry: NetworkGeometry
    control_mask: np.ndarray | torch.Tensor | None


@dataclass
class DLSSNROptions:
    profile: str = "standard"
    scale: float = 1.0
    processing_scale: float = 1.0
    detail_strength: float = 1.0
    colour_strength: float = 1.0
    detail_radius: float = 4.0
    intensity: float = 1.0
    control_mask: np.ndarray | None = None
    normalized_style: float | None = None
    local_tone_strength: float | None = None
    local_structure_strength: float | None = None
    skin_structure_strength: float | None = None
    mask_structure_strength: float | None = None


class DLSSNRSession:
    def __init__(self, pipeline: DLSSNRPipeline, **enhance_options: Any):
        self.pipeline = pipeline
        self.options = DLSSNROptions(**enhance_options)

    def __str__(self) -> str:
        return f"DLSSNRSession(options={self.options})"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = self.pipeline.enhance(image, **self.options.__dict__)
        result = np.clip(result, 0, 1)
        return result
