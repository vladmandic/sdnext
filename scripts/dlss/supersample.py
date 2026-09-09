from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.jobs import JobController, active_job
from src.upscale.image.models import ImageUpscaleOptions, output_size as source_output_size
from src.upscale.video.models import UpscaleOptions as NativeUpscaleOptions
from src.upscale.video.native import RTXVideoSession, probe_capabilities

from .utils import StandaloneError, nchw_image_to_hwc, hwc_to_nchw, validate_nchw, srgb_to_worker, worker_to_srgb_rgb, log


@dataclass(frozen=True, slots=True)
class UpscaleOptions:
    vsr_quality: int = 4
    size_mode: str = "Scale factor"
    scale_factor: float = 2.0
    width: int = 3840
    height: int = 2160
    aspect_lock: bool = True
    ai_gpu_uuid: str = "auto"

    def source_options(self) -> ImageUpscaleOptions:
        return ImageUpscaleOptions(
            vsr_quality=self.vsr_quality,
            size_mode=self.size_mode,
            scale_factor=self.scale_factor,
            width=self.width,
            height=self.height,
            aspect_lock=self.aspect_lock,
            ai_gpu_uuid=self.ai_gpu_uuid,
        )

    def validate(self) -> None:
        source = self.source_options()
        source.validate()


class DLSSSuperSample:
    """RGB NCHW adapter for the native RTX Video Super Resolution worker."""

    def __init__(self) -> None:
        log.info('DLSSSuperSample: init')
        self.diagnostics: dict[str, Any] = {}
        self.last_report: dict[str, Any] = {}

    def __call__(
        self,
        images: np.ndarray,
        options: UpscaleOptions | None = None,
        *,
        controller: JobController | None = None,
    ) -> np.ndarray:
        log.info('DLSSSuperSample: call')
        options = options or UpscaleOptions()
        options.validate()
        batch, _, height, width = validate_nchw(images, name="images")
        source = options.source_options()
        output_width, output_height = source_output_size(width, height, source)
        own_controller = controller or JobController()
        log.debug(f'DLSSSuperSample: controller={own_controller}')
        outputs: list[np.ndarray] = []
        try:
            with active_job(own_controller) as active_controller:
                capabilities = probe_capabilities(options.ai_gpu_uuid, controller=active_controller)
                log.debug(f'DLSSSuperSample: capabilities={capabilities}')
                native_options = NativeUpscaleOptions(
                    vsr_enabled=True,
                    vsr_quality=int(options.vsr_quality),
                    ai_gpu_uuid=options.ai_gpu_uuid,
                )
                native_options.validate(for_render=False)
                with RTXVideoSession(
                    width,
                    height,
                    output_width,
                    output_height,
                    native_options,
                    1,
                    capabilities,
                    active_controller,
                ) as session:
                    log.debug(f'DLSSSuperSample: session={session}')
                    for index in range(batch):
                        if active_controller.cancel.is_set():
                            raise StandaloneError("cancelled", "Upscale was cancelled.")
                        frame = nchw_image_to_hwc(images, index, name="images")
                        worker_input = srgb_to_worker(frame)
                        worker_output = session.process_frame(worker_input)
                        rgb = worker_to_srgb_rgb(worker_output, output_width, output_height)
                        log.debug(f'DLSSSuperSample: processed={rgb.shape}')
                        outputs.append(hwc_to_nchw(rgb, name="upscaled RGB output")[0])
                    self.diagnostics = {
                        "gpu": dict(capabilities.gpu),
                        "sdk_version": capabilities.sdk_version,
                        "worker_version": capabilities.worker_version,
                        "completed_frames": session.completed_frames,
                        "last_results": session.last_results,
                    }
        except StandaloneError:
            raise
        except Exception as exc:
            log.error(f'DLSSSuperSample: unexpected exception {exc}')
            raise StandaloneError("processing_failed", f"RTX Video upscaling failed: {exc}") from exc
        result = np.ascontiguousarray(np.stack(outputs, axis=0))
        self.last_report = {
            "input_shape": tuple(images.shape),
            "output_shape": tuple(result.shape),
            "completed_images": batch,
        }
        return result


__all__ = ["DLSSSuperSample", "UpscaleOptions"]
