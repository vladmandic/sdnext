"""Capture and report native MIOpen convolution selections."""

import atexit
import os
import re
import sys
import threading
import time

from modules.logger import log


_ALGORITHM_PATTERN = re.compile(r"FW Chosen Algorithm:\s*([^,\s]+)")
_CHOSEN_PATTERN = re.compile(r"FW Chosen Algorithm:\s*([^,\s]+)\s*,\s*[^,]*,\s*([0-9.eE+-]+)")
_MIOPEN_PREFIX = "MIOpen(HIP):"
_logging_capture = None


def _forward_stderr(fd, line):
    data = line if line.endswith(b"\n") else line + b"\n"
    while data:
        written = os.write(fd, data)
        data = data[written:]


def _process_line(line, saved_stderr):
    text = line.decode(errors="replace").rstrip("\r\n")
    if not text.lstrip().startswith(_MIOPEN_PREFIX):
        _forward_stderr(saved_stderr, line)
        return
    match = _CHOSEN_PATTERN.search(text)
    if match:
        log.info(f'MIOpen: algorithm={match.group(1)} time={float(match.group(2)):.3f}')


class MIOpenLogRedirect:
    """Redirect native MIOpen diagnostics into structured informational logs."""

    def __init__(self):
        self.read_fd = -1
        self.saved_stderr = -1
        self.saved_python_stderr = sys.stderr
        self.safe_stderr = None
        self.reader = None

    def __enter__(self):
        write_fd = -1
        try:
            self.read_fd, write_fd = os.pipe()
            self.saved_stderr = os.dup(2)
            self.saved_python_stderr = sys.stderr
            self.safe_stderr = os.fdopen(os.dup(self.saved_stderr), "w", encoding=getattr(sys.stderr, "encoding", None) or "utf-8", buffering=1)
            os.dup2(write_fd, 2)
            os.close(write_fd)
            write_fd = -1
            sys.stderr = self.safe_stderr

            def read_output():
                pending = b""
                while True:
                    chunk = os.read(self.read_fd, 4096)
                    if not chunk:
                        break
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        _process_line(line + b"\n", self.saved_stderr)
                if pending:
                    _process_line(pending, self.saved_stderr)

            self.reader = threading.Thread(target=read_output, daemon=True)
            self.reader.start()
            return self
        except Exception:
            if write_fd >= 0:
                os.close(write_fd)
            if self.saved_stderr >= 0:
                os.dup2(self.saved_stderr, 2)
            sys.stderr = self.saved_python_stderr
            if self.safe_stderr is not None:
                self.safe_stderr.close()
            if self.reader is not None:
                self.reader.join()
            if self.saved_stderr >= 0:
                os.close(self.saved_stderr)
            if self.read_fd >= 0:
                os.close(self.read_fd)
            raise

    def __exit__(self, _exc_type, _exc_value, _traceback):
        os.dup2(self.saved_stderr, 2)
        sys.stderr = self.saved_python_stderr
        self.safe_stderr.close()
        self.reader.join()
        os.close(self.saved_stderr)
        os.close(self.read_fd)
        return False


def start_miopen_logging():
    """Start filtering native MIOpen diagnostics without changing the environment."""
    global _logging_capture  # pylint: disable=global-statement
    if _logging_capture is None:
        try:
            _logging_capture = MIOpenLogRedirect()
            _logging_capture.__enter__()
        except Exception as err:
            log.warning(f'MIOpen logging: failed to start: {err}')
            _logging_capture = None


def stop_miopen_logging():
    """Stop filtering native MIOpen diagnostics and restore stderr."""
    global _logging_capture  # pylint: disable=global-statement
    if _logging_capture is not None:
        _logging_capture.__exit__(None, None, None)
        _logging_capture = None

atexit.register(stop_miopen_logging)


class MIOpenLogCapture:
    """Capture one native MIOpen operation and log its selected algorithm and time."""

    def __init__(self, operation: str = "convolution", repeats: int = 1):
        self.operation = operation
        self.repeats = max(1, repeats)
        self.lines: list[str] = []
        self.elapsed_ms = 0.0
        self.algorithms: list[str] = []
        self.read_fd = -1
        self.saved_stderr = -1
        self.saved_python_stderr = sys.stderr
        self.safe_stderr = None
        self.start = 0.0
        self.reader = None

    def __enter__(self):
        write_fd = -1
        try:
            self.read_fd, write_fd = os.pipe()
            self.saved_stderr = os.dup(2)
            self.saved_python_stderr = sys.stderr
            self.safe_stderr = os.fdopen(
                os.dup(self.saved_stderr),
                "w",
                encoding=getattr(sys.stderr, "encoding", None) or "utf-8",
                buffering=1,
                )
            os.dup2(write_fd, 2)
            os.close(write_fd)
            write_fd = -1
            sys.stderr = self.safe_stderr
            self.lines = []
            self.start = time.perf_counter()

            def read_output():
                chunks = []
                while True:
                    chunk = os.read(self.read_fd, 4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                self.lines.extend(b"".join(chunks).decode(errors="replace").splitlines())

            self.reader = threading.Thread(target=read_output, daemon=True)
            self.reader.start()
            return self
        except Exception:
            if write_fd >= 0:
                os.close(write_fd)
            if self.saved_stderr >= 0:
                os.dup2(self.saved_stderr, 2)
            sys.stderr = self.saved_python_stderr
            if self.safe_stderr is not None:
                self.safe_stderr.close()
            if self.reader is not None:
                self.reader.join()
            if self.saved_stderr >= 0:
                os.close(self.saved_stderr)
            if self.read_fd >= 0:
                os.close(self.read_fd)
            raise

    def __exit__(self, _exc_type, _exc_value, _traceback):
        os.dup2(self.saved_stderr, 2)
        sys.stderr = self.saved_python_stderr
        self.safe_stderr.close()
        os.close(self.saved_stderr)

        self.reader.join()
        os.close(self.read_fd)
        return False


def capture_miopen(operation: str = "convolution", repeats: int = 1):
    """Return a side-effect-free context manager for one MIOpen operation."""
    return MIOpenLogCapture(operation=operation, repeats=repeats)
