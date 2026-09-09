from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from src.core.jobs import JobController, active_job
from src.frame_interpolation.capabilities import probe_frame_interpolation_capabilities
from src.frame_interpolation.guides import DLSSGGuideGenerator
from src.frame_interpolation.models import ENGINE_CHOICES, resolve_target_rate
from src.frame_interpolation.native import DirectDLSSGSession
from src.frame_interpolation.scheduler import choose_interpolation_plan, output_frame_count

from .utils import StandaloneError, nchw_image_to_hwc, rgba_to_rgb_nchw, validate_nchw, rgb_to_rgba, log


@dataclass(frozen=True, slots=True)
class InterpolationOptions:
    ai_gpu_uuid: str = "auto"
    engine: str = "Auto"

    def validate(self) -> None:
        if self.engine not in ENGINE_CHOICES:
            raise ValueError(f"Unknown frame interpolation engine: {self.engine!r}.")


@dataclass(frozen=True, slots=True)
class _TimedFrame:
    rgba: np.ndarray
    timestamp: Fraction


class _Stage:
    def __init__(self, session: DirectDLSSGSession, width: int, height: int) -> None:
        self.session = session
        self.guides = DLSSGGuideGenerator(width, height)
        self.previous: _TimedFrame | None = None

    def push(self, frame: _TimedFrame) -> list[_TimedFrame]:
        previous = self.previous
        guide = self.guides.process(frame.rgba, force_reset=previous is not None and frame.timestamp <= previous.timestamp)
        self.previous = frame
        generated = self.session.process_frame(
            frame.rgba,
            guide.motion,
            frame.timestamp,
            reset=previous is None or guide.reset,
        )
        result: list[_TimedFrame] = []
        if previous is not None and not guide.reset:
            interval = frame.timestamp - previous.timestamp
            count = len(generated)
            for index, rgba in enumerate(generated, start=1):
                result.append(_TimedFrame(
                    rgba,
                    previous.timestamp + interval * Fraction(index, count + 1),
                ))
        result.append(frame)
        return result


class DLSSFrameGen:
    """In-memory RGB NCHW constant-frame-rate DLSS frame interpolation."""

    def __init__(self) -> None:
        log.info('DLSSFrameGen: init')
        self.diagnostics: dict[str, Any] = {}
        self.last_report: dict[str, Any] = {}

    def __call__(
        self,
        frames: np.ndarray,
        source_fps: str | int | float | Fraction,
        target_fps: str | int | float | Fraction,
        options: InterpolationOptions | None = None,
        *,
        controller: JobController | None = None,
    ) -> np.ndarray:
        log.info('DLSSFrameGen: call')
        options = options or InterpolationOptions()
        options.validate()
        batch, _, height, width = validate_nchw(frames, name="frames")
        if width < 64 or height < 64:
            raise StandaloneError("invalid_dimensions", "FrameGen: invalid resolution")
        source_rate = resolve_target_rate(source_fps)
        target_rate = resolve_target_rate(target_fps)
        own_controller = controller or JobController()
        log.debug(f'DLSSFrameGen: controller={own_controller}')
        try:
            with active_job(own_controller) as active_controller:
                capabilities = probe_frame_interpolation_capabilities(options.ai_gpu_uuid)
                log.debug(f'DLSSFrameGen: capabilities={capabilities}')
                if not capabilities.available:
                    raise StandaloneError("feature_unavailable", "FrameGen: unavailable. " + capabilities.detail,
                    )
                plan = choose_interpolation_plan(
                    source_rate,
                    target_rate,
                    options.engine,
                    capabilities.native_multiplier,
                    cfr=True,
                )
                source_frames = [
                    _TimedFrame(rgb_to_rgba(nchw_image_to_hwc(frames, index, name="frames")), Fraction(index, 1) / source_rate)
                    for index in range(batch)
                ]
                if plan.generated_per_interval == 0:
                    result = self._resample_source(source_frames, target_rate, source_rate)
                else:
                    result = self._generate(source_frames, plan, active_controller, width, height)
                expected = output_frame_count(Fraction(batch, 1) / source_rate, target_rate)
                if len(result) != expected:
                    log.error(f'DLSSFrameGen: result length={len(result)} expected={expected}')
                    raise StandaloneError("invalid_native_output", f"FrameGen: interpolation produced {len(result)} frames; expected {expected}.")
                output = np.stack([rgba_to_rgb_nchw(item.rgba)[0] for item in result], axis=0)
                log.debug(f'DLSSFrameGen: output={output.shape}')
                self.diagnostics = {
                    "gpu": capabilities.gpu,
                    "driver": capabilities.driver,
                    "runtime_version": capabilities.runtime_version,
                    "worker_version": capabilities.worker_version,
                    "selected_path": plan.path,
                    "native_multiplier": plan.native_multiplier,
                    "cascade_stages": plan.cascade_stages,
                }
        except StandaloneError:
            raise
        except Exception as exc:
            log.error(f'DLSSFrameGen: unexpected exception {exc}')
            raise StandaloneError("processing_failed", f"FrameGen: failed: {exc}") from exc
        self.last_report = {
            "input_shape": tuple(frames.shape),
            "output_shape": tuple(output.shape),
            "source_fps": str(source_rate),
            "target_fps": str(target_rate),
        }
        return np.ascontiguousarray(output)

    @staticmethod
    def _resample_source(frames: list[_TimedFrame], target_rate: Fraction, source_rate: Fraction) -> list[_TimedFrame]:
        count = output_frame_count(Fraction(len(frames), 1) / source_rate, target_rate)
        result: list[_TimedFrame] = []
        for index in range(count):
            ideal = Fraction(index, 1) / target_rate
            selected = min(frames, key=lambda frame, target=ideal: abs(frame.timestamp - target))
            result.append(_TimedFrame(selected.rgba.copy(), ideal))
        return result

    @staticmethod
    def _generate(source_frames, plan, controller, width, height) -> list[_TimedFrame]:
        sessions: list[DirectDLSSGSession] = []
        stages: list[_Stage] = []
        try:
            stage_count = plan.cascade_stages or 1
            for stage_index in range(stage_count):
                generated_count = (
                    plan.generated_per_interval
                    if plan.path == "Native DLSSG"
                    else 1
                )
                expected_frames = (
                    len(source_frames)
                    if stage_index == 0
                    else max(1, (len(source_frames) - 1) * (1 << stage_index) + 1)
                )
                session = DirectDLSSGSession(
                    width,
                    height,
                    expected_frames,
                    generated_count,
                    controller,
                )
                sessions.append(session)
                stages.append(_Stage(session, width, height))
            candidates: list[_TimedFrame] = []
            for source in source_frames:
                if controller.cancel.is_set():
                    raise StandaloneError("cancelled", "FrameGen: cancelled")
                items = [source]
                for stage in stages:
                    next_items: list[_TimedFrame] = []
                    for item in items:
                        next_items.extend(stage.push(item))
                    items = next_items
                candidates.extend(items)
            duration = Fraction(len(source_frames), 1) / plan.source_rate
            count = output_frame_count(duration, plan.target_rate)
            result: list[_TimedFrame] = []
            for index in range(count):
                ideal = Fraction(index, 1) / plan.target_rate
                selected = min(candidates, key=lambda frame, target=ideal: abs(frame.timestamp - target))
                result.append(_TimedFrame(selected.rgba.copy(), ideal))
            return result
        finally:
            for session in reversed(sessions):
                try:
                    session.close()
                except (OSError, RuntimeError, ValueError):
                    session.abort()


__all__ = ["DLSSFrameGen", "InterpolationOptions"]
