from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.jobs import JobController, active_job
from src.core.runtime import (
    DLSSFrameSession,
    prepare_runtime,
    resolve_native_settings,
    resolve_output_size,
    resolve_runtime_ai_gpu,
    resolve_upscaling_mode,
    resize_fit,
)
from src.neural_rendering.image.models import ImageConversionOptions

from .utils import StandaloneError, nchw_image_to_hwc, rgba_to_rgb_nchw, validate_nchw, rgb_to_rgba, log


@dataclass(frozen=True, slots=True)
class RenderOptions:
    ai_gpu_uuid: str = "auto"
    nr_style: str = "Default"
    nr_intensity: float = 1.0
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    upscaling_factor: float = 1.0
    warmup_frames: int = 0
    nr_preset: str = "Default"
    automatic_mask: bool = False
    dlss_model_preset: str = "Default"

    def source_options(self) -> ImageConversionOptions:
        return ImageConversionOptions(
            ai_gpu_uuid=self.ai_gpu_uuid,
            nr_style=self.nr_style,
            nr_intensity=self.nr_intensity,
            local_tone_strength=self.local_tone_strength,
            local_structure_strength=self.local_structure_strength,
            skin_structure_strength=self.skin_structure_strength,
            upscaling_factor=self.upscaling_factor,
            warmup_frames=self.warmup_frames,
            nr_preset=self.nr_preset,
            automatic_mask=self.automatic_mask,
            dlss_model_preset=self.dlss_model_preset,
        )

    def validate(self) -> None:
        if isinstance(self.warmup_frames, bool) or not isinstance(self.warmup_frames, int) or self.warmup_frames < 0:
            raise ValueError("warmup_frames must be a non-negative integer.")
        if not isinstance(self.automatic_mask, bool):
            raise ValueError("automatic_mask must be a boolean.")
        options = self.source_options()
        resolve_upscaling_mode(options.upscaling_factor)
        resolve_native_settings(options)


class DLSSNeuralRenderer:
    """RGB NCHW adapter for still-image DLSS Neural Rendering."""

    def __init__(self) -> None:
        log.info('DLSSNeuralRenderer: init')
        self.diagnostics: dict[str, Any] = {}
        self.last_report: dict[str, Any] = {}

    def __call__(
        self,
        images: np.ndarray,
        options: RenderOptions | None = None,
        *,
        controller: JobController | None = None,
    ) -> np.ndarray:
        log.info('DLSSNeuralRenderer: call')
        options = options or RenderOptions()
        options.validate()
        batch, _channels, height, width = validate_nchw(images, name="images")
        log.debug(f'DLSSNeuralRenderer: input={images.shape}')
        if width < 64 or height < 64:
            raise StandaloneError("invalid_dimensions", "NeuralRender: invalid resolution")
        output_width, output_height = resolve_output_size(width, height, options.upscaling_factor)
        own_controller = controller or JobController()
        log.debug(f'DLSSNeuralRenderer: controller={own_controller}')
        outputs: list[np.ndarray] = []
        try:
            with active_job(own_controller) as active_controller:
                prepared = prepare_runtime()
                log.debug(f'DLSSNeuralRenderer: runtime={prepared}')
                gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, options.ai_gpu_uuid)
                log.debug(f'DLSSNeuralRenderer: gpu={gpu}')
                factor, mode = resolve_upscaling_mode(options.upscaling_factor)
                native_settings = resolve_native_settings(options.source_options())
                session_diagnostics: list[dict[str, Any]] = []
                session = DLSSFrameSession(
                    input_width=width,
                    input_height=height,
                    output_width=output_width,
                    output_height=output_height,
                    frame_count=batch,
                    warmup_frames=options.warmup_frames,
                    factor=factor,
                    mode=mode,
                    native_settings=native_settings,
                    gpu=gpu,
                    runtime_bundle=prepared.runtime_bundle,
                    controller=active_controller,
                )
                log.debug(f'DLSSNeuralRenderer: session={session}')
                for index in range(batch):
                    if active_controller.cancel.is_set():
                        raise StandaloneError("cancelled", "NeuralRender: cancelled.")
                    try:
                        rgb = nchw_image_to_hwc(images, index, name="images")
                        rgba = rgb_to_rgba(rgb)
                        render_input = resize_fit(rgba, session.render_width, session.render_height)
                        motion = np.zeros((session.render_height, session.render_width, 2), dtype=np.float16)
                        log.debug(f'DLSSNeuralRenderer: index={index} processes={render_input.shape}')
                        processed, _ = session.process(
                            index=index,
                            rgba=render_input,
                            motion=motion,
                            reset=True,
                            pts=0,
                        )
                        log.debug(f'DLSSNeuralRenderer: index={index} processed={processed.shape}')
                        outputs.append(rgba_to_rgb_nchw(processed)[0])
                        session_diagnostics.append({
                            "render_width": session.render_width,
                            "render_height": session.render_height,
                            "applied_dlss_model_preset": session.applied_dlss_model_preset,
                            "worker_logs": session.worker_logs,
                            "completed_frames": session.completed_frames,
                        })
                        for l in session.worker_logs or []:
                            log.debug(f'DLSSNeuralRenderer worker: {l}')
                    except Exception as e:
                        log.error(f'DLSSNeuralRenderer: exception {e}')
                        if session is not None and not session.closed:
                            session.abort()
                        raise
                session.close()
                self.diagnostics = {
                    "gpu": dict(gpu),
                    "runtime_bundle": prepared.runtime_bundle,
                    "sessions": session_diagnostics,
                }
        except StandaloneError:
            raise
        except Exception as exc:
            log.error(f'DLSSNeuralRenderer: unexpected exception {exc}')
            raise StandaloneError("processing_failed", f"NeuralRender failed: {exc}") from exc
        result = np.ascontiguousarray(np.stack(outputs, axis=0))
        self.last_report = {
            "input_shape": tuple(images.shape),
            "output_shape": tuple(result.shape),
            "completed_images": batch,
        }
        return result


__all__ = ["DLSSNeuralRenderer", "RenderOptions"]
