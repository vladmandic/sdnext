import os
import json
import shutil
import base64
import select
import threading
import subprocess
import time
import uuid
import numpy as np
from PIL import Image
from modules.logger import log
from modules.errors import display


debug = os.environ.get('SD_DLSS_DEBUG', None) is not None


def image_to_nchw(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert('RGB'), dtype=np.uint8) # HWC
    return np.ascontiguousarray(arr.transpose(2, 0, 1))[np.newaxis, ...] # 1CHW


def images_to_nchw(images: list) -> np.ndarray:
    # last image is reference, skip all images that do not have same dimensions as the last image
    if len(images) > 1:
        ref_size = images[-1].size
        images = [image for image in images if image.size == ref_size]
    return np.concatenate([image_to_nchw(image) for image in images], axis=0)


def nchw_to_images(arr) -> list:
    arr = np.asarray(arr)
    return [Image.fromarray(arr[i].transpose(1, 2, 0), 'RGB') for i in range(arr.shape[0])]


def _encode_value(value):
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return { '__ndarray__': True, 'dtype': str(arr.dtype), 'shape': list(arr.shape), 'data': base64.b64encode(arr.tobytes()).decode('ascii') }
    return value


def _decode_value(value):
    if isinstance(value, dict) and value.get('__ndarray__'):
        data = base64.b64decode(value['data'])
        return np.frombuffer(data, dtype=value['dtype']).reshape(value['shape'])
    if isinstance(value, dict):
        return { k: _decode_value(v) for k, v in value.items() }
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


class DLSSController:
    """Persistent stdio bridge to the DLSS package's long-lived controller worker (app/controller.py)."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.pkg_path: str | None = None
        self.lock = threading.Lock()

    def get_python(self, pkg_path: str):
        python_exe = os.path.join(pkg_path, 'bin', 'python-3.13.15-embed-amd64', 'python.exe')
        if not os.path.exists(python_exe):
            log.error(f'DLSS: path={pkg_path} python={python_exe} not found')
            return None
        return python_exe

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self):
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                line = json.dumps({ 'request_id': str(uuid.uuid4()), 'command': 'shutdown', 'args': [], 'kwargs': {} }) + '\n'
                process.stdin.write(line.encode('utf-8'))
                process.stdin.flush()
        except Exception:
            pass
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5.0)
        except Exception:
            pass

    def ensure_installed(self, pkg_path: str) -> bool:
        if self.is_alive() and self.pkg_path == pkg_path:
            return True
        self.stop()
        python_exe = self.get_python(pkg_path)
        if not python_exe:
            return False
        # create _sdnext directory if it doesn't exist
        sdnext_path = os.path.join(pkg_path, '_sdnext')
        if not os.path.exists(sdnext_path):
            if debug:
                log.trace(f'DLSS install: create folder="{sdnext_path}"')
            try:
                os.makedirs(sdnext_path, exist_ok=True)
            except Exception as e:
                log.error(f'DLSS install: failed to create folder: {e}')
                display(e, 'DLSS')
                return False
        files_to_copy = ['__init__.py', 'controller_srv.py', 'utils.py', 'verify.py', 'render.py', 'supersample.py', 'framegen.py']
        for file_name in files_to_copy:
            # src path is current path of this file
            src = os.path.join(os.path.dirname(__file__), file_name)
            dst = os.path.join(sdnext_path, file_name)
            # not exist or newer
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                if debug:
                    log.trace(f'DLSS install: copy src="{src}" "{dst}"')
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    log.error(f'DLSS install: failed to copy {file_name}: {e}')
                    display(e, 'DLSS')
                    return False
        return True

    def ensure_started(self, pkg_path: str) -> bool:
        if self.is_alive() and self.pkg_path == pkg_path:
            return True
        self.stop()
        python_exe = self.get_python(pkg_path)
        if not python_exe:
            return False
        env = {
            'GRADIO_ANALYTICS_ENABLED': 'False',
            'PYTHONNOUSERSITE': '1',
            'PYTHONIOENCODING': 'utf-8'
        }
        if debug:
            env['SD_DLSS_DEBUG'] = 'True'
            log.trace(f'DLSS controller start: env={env}')
        try:
            self.process = subprocess.Popen( # pylint: disable=consider-using-with
                [python_exe, '-m', '_sdnext.controller_srv'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=pkg_path,
                env=env,
                bufsize=0,
            )
        except Exception as e:
            log.error(f'DLSS controller start: {e}')
            display(e, 'DLSS')
            self.process = None
            return False
        self.pkg_path = pkg_path
        response = self._send({ 'request_id': str(uuid.uuid4()), 'command': 'status', 'args': [], 'kwargs': {} }, timeout=30.0)
        if response is None or response.get('status') != 'ok':
            log.error(f'DLSS controller start: response={response}')
            self.stop()
            return False
        if debug:
            log.trace(f'DLSS controller start: result={response.get("result")}')
        return True

    def _send(self, request: dict, timeout: float = 300.0):
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            return None
        try:
            process.stdin.write((json.dumps(request) + '\n').encode('utf-8'))
            process.stdin.flush()
        except Exception as e:
            log.error(f'DLSS: failed to send request: {e}')
            display(e, 'DLSS')
            self.process = None
            return None
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                log.error(f'DLSS controller: timeout={timeout}')
                return None
            try:
                ready, _, _ = select.select([process.stdout], [], [], remaining)
            except Exception:
                ready = [process.stdout] # select() is not supported on pipes on some platforms: fall back to a blocking read
            if not ready:
                log.error(f'DLSS controller: timeout={timeout}')
                return None
            try:
                raw = process.stdout.readline()
            except Exception as e:
                log.error(f'DLSS controller read: {e}')
                display(e, 'DLSS')
                self.process = None
                return None
            if not raw:
                stderr = process.stderr.read().decode('utf-8', errors='ignore') if process.stderr else ''
                log.error(f'DLSS controller process: stderr="{stderr.strip()}"')
                self.process = None
                return None
            try:
                return json.loads(raw.decode('utf-8'))
            except Exception:
                if debug:
                    log.trace(f'DLSS controller stray output: {raw!r}')
                continue # skip any non-JSON noise emitted before the JSON response line

    def call(self, pkg_path: str, command: str, kwargs: dict, timeout: float = 600.0) -> dict:
        with self.lock:
            if not self.ensure_installed(pkg_path):
                return { 'status': 'error', 'result': None, 'error': { 'code': 'not_installed', 'message': 'controller is not installed' } }
            if not self.ensure_started(pkg_path):
                return { 'status': 'error', 'result': None, 'error': { 'code': 'not_ready', 'message': 'controller failed to start' } }
            encoded_kwargs = { key: _encode_value(value) for key, value in kwargs.items() }
            request = { 'request_id': str(uuid.uuid4()), 'command': command, 'args': [], 'kwargs': encoded_kwargs }
            if debug:
                log.trace(f'DLSS controller request: command={command} timeout={timeout}')
            response = self._send(request, timeout=timeout)
            if response is None:
                return { 'status': 'error', 'result': None, 'error': { 'code': 'not_ready', 'message': 'controller is not responding' } }
            if isinstance(response.get('result'), (dict, list)):
                response['result'] = _decode_value(response['result'])
            return response


controller = DLSSController()
