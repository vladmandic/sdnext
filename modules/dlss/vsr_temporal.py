from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .vsr_pipeline import DLSSVSRPipeline, VSR_PROFILES


@dataclass
class DLSSVSRTemporalOptions:
    profile: str = "ultra"
    scale: float = 2.0
    detail_strength: float = 1.0
    colour_strength: float = 1.0
    detail_radius: float = 4.0
    scene_cut_threshold: float = 0.4
    target_width: int | None = None
    target_height: int | None = None

    def __post_init__(self) -> None:
        if self.profile in VSR_PROFILES:
            preset = VSR_PROFILES[self.profile]
            if "scale" in preset and self.scale == 2.0:
                self.scale = preset["scale"]
            if "detail_strength" in preset and self.detail_strength == 1.0:
                self.detail_strength = preset["detail_strength"]

    def __str__(self) -> str:
        return f"DLSSVSRTemporalOptions(profile={self.profile} scale={self.scale} detail={self.detail_strength} colour={self.colour_strength} radius={self.detail_radius} scene={self.scene_cut_threshold} width={self.target_width} height={self.target_height})"

class DLSSVSRTemporalSession:
    """Frame-by-frame temporal super-resolution session for video processing."""

    def __init__(
        self,
        pipeline: DLSSVSRPipeline,
        options: DLSSVSRTemporalOptions | None = None,
        **kwargs: Any,
    ):
        self.pipeline = pipeline
        if options is not None:
            self.options = options
        else:
            self.options = DLSSVSRTemporalOptions(**kwargs)
        self.frame_index = 0
        self.prev_frame: np.ndarray | None = None

    def __str__(self) -> str:
        return f"DLSSVSRTemporalSession(options={self.options})"

    def reset(self) -> None:
        self.frame_index = 0
        self.prev_frame = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        curr_frame = np.asarray(frame, dtype=np.float32)
        out = self.pipeline.enhance(
            curr_frame,
            profile=self.options.profile,
            scale=self.options.scale,
            detail_strength=self.options.detail_strength,
            colour_strength=self.options.colour_strength,
            detail_radius=self.options.detail_radius,
            target_width=self.options.target_width,
            target_height=self.options.target_height,
        )
        self.frame_index += 1
        self.prev_frame = curr_frame
        return np.clip(out, 0.0, 1.0)
