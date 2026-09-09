from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# when launched directly as the stdio bridge subprocess, keep the real stdout clean of stray
# prints from native imports/libraries below so only explicit JSON response lines reach the pipe
_stdio_stdout = None
if __name__ == "__main__":
    _stdio_stdout = sys.stdout
    sys.stdout = sys.stderr

import base64
import ctypes
import dataclasses
import json
import multiprocessing as mp
import os
import threading
import uuid
from queue import Empty
from typing import Any

import numpy as np

from .utils import StandaloneError, log
from .framegen import DLSSFrameGen, InterpolationOptions
from .render import DLSSNeuralRenderer, RenderOptions
from .supersample import DLSSSuperSample, UpscaleOptions
from .verify import DLSSVerify, VerifyOptions

_SUPPORTED_COMMANDS = {
    "status",
    "verify",
    "render",
    "upscale",
    "framegen",
    "cancel",
    "reset",
    "shutdown",
}


def _response(request_id: str, *, status: str, result: Any = None, error: dict[str, str] | None = None, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "result": result,
        "error": error,
        "diagnostics": diagnostics or {},
    }


def _coerce_options(options: Any, *, default: Any, option_type: type[Any]) -> Any:
    if options is None:
        return default
    if isinstance(options, option_type):
        return options
    if isinstance(options, dict):
        return option_type(**options)
    raise TypeError(f"Expected {option_type.__name__} or dict, got {type(options).__name__}")


def _set_shared_text(buffer: Any, value: str) -> None:
    with buffer.get_lock():
        for index in range(len(buffer)):
            buffer[index] = "\0"
        for index, character in enumerate(value[: len(buffer) - 1]):
            buffer[index] = character


def _get_shared_text(buffer: Any) -> str:
    with buffer.get_lock():
        return "".join(buffer).split("\0", 1)[0]


def _dispatch_command(command: str, request_id: str, args: tuple[Any, ...], kwargs: dict[str, Any], *, busy: Any = None, current_request_id: Any = None, current_command: Any = None) -> dict[str, Any]: # pylint: disable=unused-argument
    log.debug(f'DLSSController dispatch: command={command} id={request_id}')
    if command == "status":
        is_busy = bool(busy.value) if busy is not None else False
        active_request_id = _get_shared_text(current_request_id) if current_request_id is not None else ""
        active_command = _get_shared_text(current_command) if current_command is not None else ""
        return _response(
            request_id,
            status="ok",
            result={
                "ready": True,
                "busy": is_busy,
                "pid": os.getpid(),
                "id": active_request_id if is_busy else "",
                "job": active_command if is_busy else "idle",
            },
            diagnostics={"controller": "ready"},
        )

    if command == "reset":
        return _response(request_id, status="ok", result={"reset": True}, diagnostics={"controller": "reset"})

    if command == "cancel":
        target_id = str(kwargs.get("request_id") or "")
        return _response(request_id, status="ok", result={"cancelled": bool(target_id), "target_request_id": target_id}, diagnostics={"controller": "cancelled"})

    if command == "verify":
        gpu_uuid = str(kwargs.get("gpu_uuid", "auto"))
        options = _coerce_options(kwargs.get("options"), default=VerifyOptions(), option_type=VerifyOptions)
        result = DLSSVerify()(gpu_uuid, options)
        return _response(request_id, status="ok", result={"ok": result.ok, "report": result.to_dict()}, diagnostics={"gpu": result.gpu or {}})

    if command == "render":
        images = kwargs.get("images")
        if images is None:
            raise StandaloneError("invalid_arguments", "Missing required 'images' argument for render command.")
        options = _coerce_options(kwargs.get("options"), default=RenderOptions(), option_type=RenderOptions)
        result = DLSSNeuralRenderer()(np.asarray(images), options)
        return _response(request_id, status="ok", result=result, diagnostics={"shape": list(result.shape)})

    if command == "upscale":
        images = kwargs.get("images")
        if images is None:
            raise StandaloneError("invalid_arguments", "Missing required 'images' argument for upscale command.")
        options = _coerce_options(kwargs.get("options"), default=UpscaleOptions(), option_type=UpscaleOptions)
        result = DLSSSuperSample()(np.asarray(images), options)
        return _response(request_id, status="ok", result=result, diagnostics={"shape": list(result.shape)})

    if command == "framegen":
        frames = kwargs.get("frames")
        if frames is None:
            raise StandaloneError("invalid_arguments", "Missing required 'frames' argument for framegen command.")
        source_fps = kwargs.get("source_fps")
        target_fps = kwargs.get("target_fps")
        if source_fps is None or target_fps is None:
            raise StandaloneError("invalid_arguments", "framegen requires both 'source_fps' and 'target_fps'.")
        options = _coerce_options(kwargs.get("options"), default=InterpolationOptions(), option_type=InterpolationOptions)
        result = DLSSFrameGen()(np.asarray(frames), source_fps, target_fps, options)
        return _response(request_id, status="ok", result=result, diagnostics={"shape": list(result.shape)})

    raise StandaloneError("invalid_arguments", f"Unsupported controller command: {command!r}")


def _controller_worker(request_queue: mp.Queue, response_queue: mp.Queue, busy: Any, current_request_id: Any, current_command: Any) -> None:
    worker_lock = threading.Lock()
    while True:
        try:
            request = request_queue.get(timeout=0.25)
        except Empty:
            continue

        if not isinstance(request, dict):
            response_queue.put(_response(str(uuid.uuid4()), status="error", error={"code": "invalid_arguments", "message": "Controller request must be a dict."}))
            continue

        request_id = str(request.get("request_id") or uuid.uuid4())
        command = str(request.get("command") or "").strip().lower()
        args = tuple(request.get("args", ()))
        kwargs = dict(request.get("kwargs", {}))

        if command == "shutdown":
            log.debug(f'DLSSController shutdown: id={request_id}')
            response_queue.put(_response(request_id, status="ok", result={"shutdown": True}, diagnostics={"controller": "shutdown"}))
            return

        if command not in _SUPPORTED_COMMANDS:
            log.warning(f'DLSSController: command={command} id={request_id} unsupported')
            response_queue.put(_response(request_id, status="error", error={"code": "invalid_arguments", "message": f"Unsupported command: {command!r}"}, diagnostics={"controller": "invalid_command"}))
            continue

        try:
            if command != "status":
                busy.value = True
                _set_shared_text(current_request_id, request_id)
                _set_shared_text(current_command, command)
            with worker_lock:
                response = _dispatch_command(command, request_id, args, kwargs, busy=busy, current_request_id=current_request_id, current_command=current_command)
            response_queue.put(response)
        except StandaloneError as exc:
            log.error(f'DLSSController: StandaloneError command={command} id={request_id} code={exc.code} message={exc.message}')
            response_queue.put(_response(request_id, status="error", error={"code": exc.code, "message": exc.message}, diagnostics={"controller": "error"}))
        except Exception as exc:  # pragma: no cover - defensive catch for controller safety
            log.error(f'DLSSController: unexpected exception command={command} id={request_id} error={exc}')
            response_queue.put(_response(request_id, status="error", error={"code": "processing_failed", "message": str(exc)}, diagnostics={"controller": "error"}))
        finally:
            if command != "status":
                busy.value = False
                _set_shared_text(current_request_id, "")
                _set_shared_text(current_command, "")


class ControllerProcess(mp.Process):
    def __init__(self, request_queue: mp.Queue | None = None, response_queue: mp.Queue | None = None, *, busy: Any = None, current_request_id: Any = None, current_command: Any = None, ctx: mp.context.BaseContext | None = None) -> None:
        self.ctx = ctx or mp.get_context("spawn")
        self.request_queue = request_queue or self.ctx.Queue()
        self.response_queue = response_queue or self.ctx.Queue()
        self.busy = busy or self.ctx.Value("b", False)
        self.current_request_id = current_request_id or self.ctx.Array(ctypes.c_wchar, 256)
        self.current_command = current_command or self.ctx.Array(ctypes.c_wchar, 64)
        super().__init__(target=_controller_worker, args=(self.request_queue, self.response_queue, self.busy, self.current_request_id, self.current_command))


class ControllerClient:
    """Simple client wrapper for callers that want a long-lived controller process."""

    def __init__(self, request_queue: mp.Queue | None = None, response_queue: mp.Queue | None = None, *, process: ControllerProcess | None = None, timeout: float = 30.0, ctx: mp.context.BaseContext | None = None) -> None:
        self.ctx = ctx or mp.get_context("spawn")
        self.request_queue = request_queue or self.ctx.Queue()
        self.response_queue = response_queue or self.ctx.Queue()
        self.timeout = timeout
        self.process = process
        self.busy = process.busy if process is not None else self.ctx.Value("b", False)
        self.current_request_id = process.current_request_id if process is not None else self.ctx.Array(ctypes.c_wchar, 256)
        self.current_command = process.current_command if process is not None else self.ctx.Array(ctypes.c_wchar, 64)
        self._pending: dict[str, dict[str, Any]] = {}

    def start(self) -> "ControllerClient":
        if self.process is None or not self.process.is_alive():
            self.process = ControllerProcess(self.request_queue, self.response_queue, busy=self.busy, current_request_id=self.current_request_id, current_command=self.current_command, ctx=self.ctx)
            self.process.start()
            log.info(f'DLSSController: pid={self.process.pid} started')
        return self

    def _send_and_wait(self, command: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.process is None or not self.process.is_alive():
            self.start()
        request_id = str(uuid.uuid4())
        request = {"request_id": request_id, "command": command, "args": list(args), "kwargs": kwargs}
        self.request_queue.put(request)

        while True:
            try:
                response = self.response_queue.get(timeout=self.timeout)
            except Empty as exc:
                raise TimeoutError(f"Controller request timed out for command {command!r}.") from exc
            if response.get("request_id") == request_id:
                return response
            self._pending[response.get("request_id", str(uuid.uuid4()))] = response

    def status(self) -> dict[str, Any]:
        if self.process is None or not self.process.is_alive():
            self.start()
        is_busy = bool(self.busy.value)
        active_request_id = _get_shared_text(self.current_request_id)
        active_command = _get_shared_text(self.current_command)
        return _response(
            str(uuid.uuid4()),
            status="ok",
            result={
                "ready": self.process.is_alive(),
                "busy": is_busy,
                "pid": self.process.pid,
                "id": active_request_id if is_busy else "",
                "job": active_command if is_busy else "idle",
            },
            diagnostics={"controller": "busy" if is_busy else "ready"},
        )

    def verify(self, *, gpu_uuid: str = "auto", options: VerifyOptions | dict[str, Any] | None = None) -> dict[str, Any]:
        return self._send_and_wait("verify", gpu_uuid=gpu_uuid, options=options)

    def render(self, *, images: np.ndarray, options: RenderOptions | dict[str, Any] | None = None) -> dict[str, Any]:
        return self._send_and_wait("render", images=images, options=options)

    def upscale(self, *, images: np.ndarray, options: UpscaleOptions | dict[str, Any] | None = None) -> dict[str, Any]:
        return self._send_and_wait("upscale", images=images, options=options)

    def framegen(self, *, frames: np.ndarray, source_fps: float | str, target_fps: float | str, options: InterpolationOptions | dict[str, Any] | None = None) -> dict[str, Any]:
        return self._send_and_wait("framegen", frames=frames, source_fps=source_fps, target_fps=target_fps, options=options)

    def cancel(self, request_id: str) -> dict[str, Any]:
        return self._send_and_wait("cancel", request_id=request_id)

    def reset(self) -> dict[str, Any]:
        return self._send_and_wait("reset")

    def shutdown(self) -> dict[str, Any]:
        if self.process is None or not self.process.is_alive():
            return {"request_id": "shutdown", "status": "ok", "result": {"shutdown": True}, "error": None, "diagnostics": {}}
        response = self._send_and_wait("shutdown")
        if self.process.is_alive():
            self.process.join(timeout=5.0)
        return response

    def close(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)

    def __enter__(self) -> "ControllerClient":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def start_controller(*, request_queue: mp.Queue | None = None, response_queue: mp.Queue | None = None, timeout: float = 30.0, ctx: mp.context.BaseContext | None = None) -> ControllerClient:
    client = ControllerClient(request_queue=request_queue, response_queue=response_queue, timeout=timeout, ctx=ctx)
    return client.start()


# ---- stdio bridge -----------------------------------------------------------------
# Used only when this module is launched directly as a subprocess, e.g.
# `python.exe app/controller.py`, to drive the same dispatch logic over a single
# stdin/stdout JSON-lines protocol instead of multiprocessing queues. This allows an
# external caller running a different Python interpreter (for example WSL/Linux Python
# invoking the packaged Windows embedded python.exe) to reuse one long-lived worker.


def _stdio_encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": str(arr.dtype), "shape": list(arr.shape), "data": base64.b64encode(arr.tobytes()).decode("ascii")}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _stdio_encode(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _stdio_encode(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _stdio_encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stdio_encode(v) for v in value]
    return value


def _stdio_decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__"):
        data = base64.b64decode(value["data"])
        return np.frombuffer(data, dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {k: _stdio_decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stdio_decode(v) for v in value]
    return value


def _stdio_write(payload: dict[str, Any]) -> None:
    _stdio_stdout.write(json.dumps(payload) + "\n")
    _stdio_stdout.flush()


def _stdio_error(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {"request_id": request_id, "status": "error", "result": None, "error": {"code": code, "message": message}, "diagnostics": {}}


def _stdio_main() -> None:
    # long-lived worker: one JSON request per line on stdin, one JSON response per line on the real stdout
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _stdio_write(_stdio_error("", "invalid_arguments", f"Malformed request: {exc}"))
            continue

        request_id = str(request.get("request_id") or "")
        command = str(request.get("command") or "").strip().lower()
        args = tuple(request.get("args", ()))
        try:
            kwargs = _stdio_decode(dict(request.get("kwargs", {})))
        except Exception as exc:
            _stdio_write(_stdio_error(request_id, "invalid_arguments", f"Failed to decode request payload: {exc}"))
            continue

        if command == "shutdown":
            _stdio_write({"request_id": request_id, "status": "ok", "result": {"shutdown": True}, "error": None, "diagnostics": {}})
            return
        if command not in _SUPPORTED_COMMANDS:
            _stdio_write(_stdio_error(request_id, "invalid_arguments", f"Unsupported command: {command!r}"))
            continue

        try:
            response = _dispatch_command(command, request_id, args, kwargs)
        except StandaloneError as exc:
            response = _stdio_error(request_id, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive catch for controller safety
            response = _stdio_error(request_id, "processing_failed", str(exc))
        _stdio_write(_stdio_encode(response))


if __name__ == "__main__":
    _stdio_main()

    def __enter__(self) -> "ControllerClient":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: # pylint: disable=unused-argument
        self.close()


__all__ = [
    "ControllerClient",
    "ControllerProcess",
    "start_controller",
]
