from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from src.core.gpu_selection import resolve_ai_gpu
from src.core.gpu_detection import detect_gpus
from src.core.paths import ADDON, DLSS_SUPERRES, FFMPEG, FFPROBE, HOST_DXGI, NEURAL_RUNTIME, WORKER
from src.core.runtime import validate_runtime_files

from .utils import log


@dataclass(frozen=True, slots=True)
class VerifyOptions:
    level: str = "basic"
    check_neural: bool = True
    check_upscale: bool = True
    check_interpolation: bool = True

    def validate(self) -> None:
        if self.level not in {"basic", "deep"}:
            raise ValueError("Verification level must be 'basic' or 'deep'.")
        for name in ("check_neural", "check_upscale", "check_interpolation"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        log.debug(f'DLSSVerify: check="{self.name}" passed={self.passed} detail="{self.detail}"')
        result = {"name": self.name, "passed": self.passed, "detail": self.detail}
        return result


@dataclass(frozen=True, slots=True)
class VerificationReport:
    ok: bool
    level: str
    python: str
    platform: str
    gpu: dict[str, Any] | None
    paths: dict[str, str]
    checks: tuple[VerificationCheck, ...]
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> tuple[VerificationCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        log.info(f'DLSSVerify: level="{self.level}" python="{self.python}" platform="{self.platform}"')
        log.info(f'DLSSVerify: paths={self.paths}')
        log.info(f'DLSSVerify: gpu={self.gpu}')
        log.debug(f'DLSSVerify: diagnostics={self.diagnostics}')
        return {
            "ok": self.ok,
            "python": self.python,
            "platform": self.platform,
            "gpu": self.gpu,
            "paths": self.paths,
            "checks": [check.to_dict() for check in self.checks],
            "diagnostics": self.diagnostics,
        }


_RUNTIME_PATHS = {
    "ffmpeg": FFMPEG,
    "ffprobe": FFPROBE,
    "worker": WORKER,
    "host_dxgi": HOST_DXGI,
    "dlss_addon": ADDON,
    "dlss_superres": DLSS_SUPERRES,
    "dlss_neural": NEURAL_RUNTIME,
}


class DLSSVerify:
    """Perform side-effect-free runtime preflight checks by default."""

    def __init__(self) -> None:
        self.last_report: VerificationReport | None = None

    def __call__(self, gpu_uuid: str = "auto", options: VerifyOptions | None = None) -> VerificationReport:
        options = options or VerifyOptions()
        options.validate()
        checks: list[VerificationCheck] = []
        diagnostics: list[str] = []
        selected_gpu: dict[str, Any] | None = None

        checks.extend(self._check_files())
        checks.append(self._check_import("numpy"))
        checks.append(self._check_import("PIL"))
        checks.append(self._check_import("cv2"))
        checks.append(self._check_import("av"))

        try:
            gpus = detect_gpus()
            selected_gpu = resolve_ai_gpu(gpus, gpu_uuid)
            checks.append(VerificationCheck("gpu", True, self._gpu_detail(selected_gpu)))
        except (OSError, RuntimeError, ValueError) as exc:
            checks.append(VerificationCheck("gpu", False, str(exc)))
            diagnostics.append(str(exc))

        try:
            validate_runtime_files()
            checks.append(VerificationCheck("runtime", True, "Required runtime files are present."))
        except (OSError, RuntimeError, ValueError) as exc:
            checks.append(VerificationCheck("runtime", False, str(exc)))
            diagnostics.append(str(exc))

        if options.level == "deep":
            checks.extend(self._deep_checks(options, gpu_uuid, selected_gpu))
        else:
            checks.append(VerificationCheck("deep_capabilities", True, "Deep capability checks were not requested."))

        filtered = tuple(check for check in checks if self._feature_enabled(check.name, options))
        report = VerificationReport(
            ok=all(check.passed or check.status == "not_run" for check in filtered),
            level=options.level,
            python=platform.python_version(),
            platform=sys.platform,
            gpu=selected_gpu,
            paths={name: str(path) for name, path in _RUNTIME_PATHS.items()},
            checks=filtered,
            diagnostics=tuple(diagnostics),
        )
        self.last_report = report
        return report

    @staticmethod
    def _check_files() -> list[VerificationCheck]:
        return [
            VerificationCheck(
                name=f"file:{name}",
                passed=path.is_file(),
                detail=str(path),
            )
            for name, path in _RUNTIME_PATHS.items()
        ]

    @staticmethod
    def _check_import(name: str) -> VerificationCheck:
        available = importlib.util.find_spec(name) is not None
        return VerificationCheck(
            name=f"dependency:{name}",
            passed=available,
            detail="available" if available else "not installed",
        )

    @staticmethod
    def _gpu_detail(gpu: dict[str, Any]) -> str:
        return f"{gpu.get('name', 'NVIDIA GPU')} driver={gpu.get('driver', 'unknown')} uuid={gpu.get('uuid', 'unknown')}"

    @staticmethod
    def _feature_enabled(name: str, options: VerifyOptions) -> bool:
        if name.startswith("neural:"):
            return options.check_neural
        if name.startswith("upscale:"):
            return options.check_upscale
        if name.startswith("interpolation:"):
            return options.check_interpolation
        return True

    @staticmethod
    def _deep_checks(options: VerifyOptions, gpu_uuid: str, selected_gpu: dict[str, Any] | None) -> list[VerificationCheck]:
        del selected_gpu
        checks: list[VerificationCheck] = []
        if options.check_neural:
            try:
                from src.core.runtime import prepare_runtime

                prepared = prepare_runtime()
                checks.append(VerificationCheck("neural:runtime", True, f"Prepared {len(prepared.warmed_files)} runtime components."))
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                checks.append(VerificationCheck("neural:runtime", False, str(exc)))
        if options.check_upscale:
            try:
                from src.upscale.video.native import probe_capabilities

                capabilities = probe_capabilities(gpu_uuid)
                checks.append(VerificationCheck("upscale:capability", bool(capabilities.vsr.get("available")), str(capabilities.vsr)))
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                checks.append(VerificationCheck("upscale:capability", False, str(exc)))
        if options.check_interpolation:
            try:
                from src.frame_interpolation.capabilities import probe_frame_interpolation_capabilities

                capabilities = probe_frame_interpolation_capabilities(gpu_uuid)
                checks.append(VerificationCheck("interpolation:capability", capabilities.available, capabilities.detail or f"native_multiplier={capabilities.native_multiplier}"))
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                checks.append(VerificationCheck("interpolation:capability", False, str(exc)))
        return checks
