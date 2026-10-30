#!/usr/bin/env python
"""
Offline regression suite for the video save layer: ``modules.video`` (the legacy writer every
generation script and DLSS use), ``modules.video_models.video_save`` (the Video tab writer), the
readers, and the gradio preview conversion hook in ``modules.gr_tempdir``.

Encodes are real: every case that claims something about an output file opens the file again
with av and reads the stream header, the colour tags and the decoded planes. Quality and colour
verdicts are made against floors measured in the same run (an in-process cv2 mp4v control for
quality, a numpy BT.709/BT.601 oracle for colour) with a positive and a negative control each, so
a run whose controls fail reports its own instrument as broken instead of passing.

Covers:

- the platform: library versions, cv2 backends, ffmpeg on PATH, every codec preset opened for real
- the writer failure paths: unknown codec, rejected pixel format, container that refuses the
  codec, av missing, mid-encode exception, output equal to input, registration order
- pixel formats per codec and the chroma crop of odd frame sizes
- the legacy default codec, its quality against the old cv2 writer, and the label map
- fractional frame rates and the 16-bit time base limit of mpeg4
- BT.709 conversion and tagging, measured per patch against the oracle
- the readers agreeing on one fixture, audio muxing, metadata round trip, paths
- the gradio preview hook: probe and convert behaviour with and without ffmpeg on PATH

Cases for symbols the checked-out tree does not have yet report ABSENT, not FAIL. A returned
string is a skip reason; without av every case that encodes skips.

No running server required.

Usage:
    python test/test-video-save.py [--report PATH] [--clips DIR] [--keep]
"""

import os
import sys
import argparse

cli = argparse.ArgumentParser(description='video save regression suite')
cli.add_argument('--report', default=os.environ.get('SD_TEST_REPORT', ''), help='write the JSON report here')
cli.add_argument('--clips', default=os.environ.get('SD_TEST_CLIPS', ''), help='write the browser gate clips here')
cli.add_argument('--keep', action='store_true', help='keep the temporary directory')
cli.add_argument('--spawn', action='store_true', default=os.environ.get('SD_TEST_SPAWN', '') != '', help='run cases in spawned interpreters instead of forked children (always on where fork is missing)')
cli.add_argument('--child-start', type=int, default=-1, help=argparse.SUPPRESS)
cli.add_argument('--child-results', default='', help=argparse.SUPPRESS)
cli.add_argument('--child-state', default='', help=argparse.SUPPRESS)
cli.add_argument('--work-dir', default='', help=argparse.SUPPRESS)
flags = cli.parse_args()

script_file = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.dirname(script_file))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

for env_name in ('SD_DATADIR', 'SD_CONFIG'):
    env_value = os.environ.get(env_name, '')
    if env_value and not os.path.realpath(env_value).startswith(os.path.realpath(script_dir)):
        print(f'{env_name}={env_value} points outside {script_dir}; refusing to run against another install')
        sys.exit(2)

import tempfile  # pylint: disable=wrong-import-position
work_dir = flags.work_dir or tempfile.mkdtemp(prefix='sdnext-video-save-')
os.environ['GRADIO_TEMP_DIR'] = os.path.join(work_dir, 'gradio-default')
os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
sys.argv = [sys.argv[0]]
modules.cmd_args.parse_args()
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

import io  # pylint: disable=wrong-import-position
import re  # pylint: disable=wrong-import-position
import json  # pylint: disable=wrong-import-position
import time  # pylint: disable=wrong-import-position
import math  # pylint: disable=wrong-import-position
import wave  # pylint: disable=wrong-import-position
import struct  # pylint: disable=wrong-import-position
import shutil  # pylint: disable=wrong-import-position
import hashlib  # pylint: disable=wrong-import-position
import inspect  # pylint: disable=wrong-import-position
import logging  # pylint: disable=wrong-import-position
import platform  # pylint: disable=wrong-import-position
import threading  # pylint: disable=wrong-import-position
import subprocess  # pylint: disable=wrong-import-position
import contextlib  # pylint: disable=wrong-import-position
from fractions import Fraction  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import cv2  # pylint: disable=wrong-import-position
from PIL import Image  # pylint: disable=wrong-import-position

from modules import shared, devices  # pylint: disable=wrong-import-position
from modules.logger import log  # pylint: disable=wrong-import-position
from modules import video  # pylint: disable=wrong-import-position
from modules.video_models import video_save, video_codecs, video_utils  # pylint: disable=wrong-import-position
try:
    from modules.video_models import video_writer  # pylint: disable=wrong-import-position
except ImportError:  # trees before the shared writer keep its helpers in video_save
    video_writer = video_save
try:
    from modules.video_models import video_reader  # pylint: disable=wrong-import-position
except ImportError:  # trees before the reader had its own module keep probe_media in video_utils
    video_reader = video_utils
from modules import gr_tempdir  # pylint: disable=wrong-import-position

devices.torch_gc = lambda *args, **kwargs: None  # state.begin/end call it; nothing here holds device memory

try:
    import av
except Exception:  # pylint: disable=broad-except
    av = None


class Absent(Exception):
    """The symbol under test is not in this tree."""


def need(obj, name: str):
    value = getattr(obj, name, None)
    if value is None:
        raise Absent(f'{getattr(obj, "__name__", obj)}.{name}')
    return value


def need_param(fn, name: str):
    if name not in inspect.signature(fn).parameters:
        raise Absent(f'{fn.__module__}.{fn.__name__}({name}=)')


def need_av():
    if av is None:
        return 'av missing: it installs with the requirements at startup'
    return None


# ============================================================
# harness
# ============================================================

results: dict[str, dict] = {}
report: dict = {'verdicts': {}, 'measurements': {}, 'platform': {}}
current_group = ''


def category(name: str):
    global current_group  # pylint: disable=global-statement
    current_group = name
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'absent': 0, 'tests': []}
        report['verdicts'][name] = {}
    return name


def verdict(name: str, status: str, detail: str = '', **invariants):
    entry = {'status': status}
    if detail:
        entry['detail'] = detail
    for key, value in sorted(invariants.items()):
        entry[key] = value
    report['verdicts'][current_group][name] = entry


def measure(name: str, **values):
    report['measurements'].setdefault(current_group, {})[name] = dict(sorted(values.items()))


def record(cat: str, passed: bool, name: str, detail: str = ''):
    status = 'PASS' if passed else 'FAIL'
    results[cat]['passed' if passed else 'failed'] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if passed:
        log.info(msg)
    else:
        log.error(msg)
    existing = report['verdicts'][cat].get(name, {})
    existing['status'] = status
    if detail and 'detail' not in existing:
        existing['detail'] = detail
    report['verdicts'][cat][name] = existing


def skip(cat: str, name: str, reason: str):
    results[cat]['skipped'] += 1
    results[cat]['tests'].append(('SKIP', name))
    log.warning(f'  SKIP: {name} ({reason})')
    report['verdicts'][cat][name] = {'status': 'SKIP', 'detail': reason}


def absent(cat: str, name: str, symbol: str):
    results[cat]['absent'] += 1
    results[cat]['tests'].append(('ABSENT', name))
    log.warning(f'  ABSENT: {name} ({symbol})')
    report['verdicts'][cat][name] = {'status': 'ABSENT', 'detail': symbol}


CASE_TIMEOUT = int(os.environ.get('SD_TEST_CASE_TIMEOUT', '90'))


def hung_child_diagnostics(pid: int) -> dict:
    """What a stuck child is waiting on; the parent is its ancestor, so /proc/<pid>/syscall is readable under yama scope 1."""
    out: dict = {}
    if not os.path.isdir(f'/proc/{pid}'):
        return {'proc': 'unavailable'}
    try:
        with open(f'/proc/{pid}/wchan', encoding='utf8') as f:
            out['wchan'] = f.read().strip()
        with open(f'/proc/{pid}/status', encoding='utf8') as f:
            out['threads'] = next((line.split(':')[1].strip() for line in f if line.startswith('Threads')), None)
        with open(f'/proc/{pid}/syscall', encoding='utf8') as f:
            syscall = f.read().split()
        out['syscall'] = syscall[:4]
        if syscall and syscall[0] == '202':
            addr = int(syscall[1], 16)
            with open(f'/proc/{pid}/maps', encoding='utf8') as f:
                for line in f:
                    parts = line.split()
                    lo, hi = (int(x, 16) for x in parts[0].split('-'))
                    if lo <= addr < hi:
                        out['futex_in'] = parts[5] if len(parts) > 5 else parts[1]
                        break
    except Exception as e:  # pylint: disable=broad-except
        out['error'] = f'{type(e).__name__}: {e}'
    return out


def run_case_outcome(fn) -> dict:
    """Run one case in this process and describe what happened; the parent turns it into the verdict line."""
    outcome: dict = {}
    try:
        ret = fn()
        outcome['ret'] = ret if isinstance(ret, (bool, str)) or ret is None else True
    except Absent as e:
        outcome['absent'] = str(e)
    except AssertionError as e:
        outcome['assert'] = str(e)
    except Exception as e:  # pylint: disable=broad-except
        outcome['exception'] = f'{type(e).__name__}: {e}'
    return outcome


def apply_outcome(cat: str, name: str, outcome: dict):
    # crash and hang details are platform specific, so they go to measurements and the verdict text stays comparable
    if 'hang' in outcome:
        report['measurements'].setdefault(cat, {})[f'{name}_hang'] = outcome['diag']
        record(cat, False, name, f'process hung for {outcome["hang"]}s')
    elif 'crash' in outcome:
        report['measurements'].setdefault(cat, {})[f'{name}_crash'] = outcome['crash']
        record(cat, False, name, 'process crashed')
    elif 'absent' in outcome:
        absent(cat, name, outcome['absent'])
    elif 'assert' in outcome:
        record(cat, False, name, outcome['assert'])
    elif 'exception' in outcome:
        record(cat, False, name, f'exception: {outcome["exception"]}')
    elif outcome.get('ret') is False:
        record(cat, False, name)
    elif isinstance(outcome.get('ret'), str):
        skip(cat, name, outcome['ret'])
    else:
        record(cat, True, name)


CASES: list[tuple[str, object]] = []  # (group, case), filled from GROUPS at the end of the file
SPAWN = flags.spawn or not hasattr(os, 'fork')
SPAWN_STARTUP = int(os.environ.get('SD_TEST_SPAWN_STARTUP', '180'))  # a spawned child imports torch and modules.shared before its first case


def run_cases_here(start: int, results_path: str):
    """Child side: run every case from `start`, appending one JSON line per finished case to `results_path`."""
    import faulthandler
    faulthandler.enable()
    group = None
    with open(results_path, 'a', encoding='utf8') as out:
        for index in range(start, len(CASES)):
            cat, fn = CASES[index]
            category(cat)
            if cat != group:
                log.warning(f'--- {cat} ---')
                group = cat
            faulthandler.dump_traceback_later(max(1, CASE_TIMEOUT - 3))  # a hung case prints its stack before the parent kills it
            t0 = time.time()
            outcome = run_case_outcome(fn)
            faulthandler.cancel_dump_traceback_later()
            line = {
                'index': index,
                'ms': round(1000 * (time.time() - t0)),
                'outcome': outcome,
                'state': {'FLOOR': FLOOR, 'PRESET_OPEN': PRESET_OPEN, 'verdicts': report['verdicts'].get(cat, {}), 'measurements': report['measurements'].get(cat, {}), 'platform': report['platform']},
            }
            sys.stdout.flush()
            sys.stderr.flush()
            out.write(json.dumps(line, default=str) + '\n')
            out.flush()


class ForkedChild:
    def __init__(self, start: int, results_path: str):
        self.status = None
        self.pid = os.fork()
        if self.pid == 0:
            try:
                run_cases_here(start, results_path)
            finally:
                os._exit(0)

    def alive(self) -> bool:
        if self.status is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.status = status
        return self.status is None

    def kill(self):
        os.kill(self.pid, 9)

    def exit_info(self) -> dict:
        if self.status is None:
            _pid, self.status = os.waitpid(self.pid, 0)
        if os.WIFSIGNALED(self.status):
            return {'crash': {'signal': os.WTERMSIG(self.status)}}
        return {'exit': os.WEXITSTATUS(self.status)}


class SpawnedChild:
    def __init__(self, start: int, results_path: str):
        state_path = os.path.join(work_dir, f'state-{start}.json')
        with open(state_path, 'w', encoding='utf8') as f:
            json.dump({'FLOOR': FLOOR, 'PRESET_OPEN': PRESET_OPEN, 'platform': report['platform']}, f, default=str)
        cmd = [sys.executable, '-X', 'faulthandler', script_file, '--work-dir', work_dir, '--child-start', str(start), '--child-results', results_path, '--child-state', state_path]
        if flags.clips:
            cmd += ['--clips', os.path.abspath(flags.clips)]
        self.proc = subprocess.Popen(cmd, cwd=script_dir)  # pylint: disable=consider-using-with

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self):
        self.proc.kill()

    def exit_info(self) -> dict:
        code = self.proc.wait()
        if code < 0:
            return {'crash': {'signal': -code}}
        if os.name == 'nt' and code >= 0xC0000000:
            return {'crash': {'ntstatus': hex(code)}}  # 0xc0000005 is an access violation
        return {'exit': code}


def merge_line(line: dict):
    cat = CASES[line['index']][0]
    category(cat)
    state = line.get('state', {})
    FLOOR.update(state.get('FLOOR', {}))
    PRESET_OPEN.update(state.get('PRESET_OPEN', {}))
    for key, value in state.get('verdicts', {}).items():
        report['verdicts'].setdefault(cat, {}).setdefault(key, value)  # the parent owns the status once recorded
    for key, value in state.get('measurements', {}).items():
        report['measurements'].setdefault(cat, {}).setdefault(key, value)
    report['platform'].update(state.get('platform', {}))
    report['measurements'].setdefault('case_ms', {})[CASES[line['index']][1].__name__] = line.get('ms')
    apply_outcome(cat, CASES[line['index']][1].__name__, line['outcome'])


def supervise(start: int) -> int:
    """Run cases from `start` in one child; returns the index after the last case it accounted for.

    A child that dies or stops reporting for CASE_TIMEOUT seconds is killed, and the case it was on is recorded as
    crashed or hung; the caller starts the next child after it."""
    results_path = os.path.join(work_dir, f'results-{start}.jsonl')
    with open(results_path, 'w', encoding='utf8'):
        pass
    child = SpawnedChild(start, results_path) if SPAWN else ForkedChild(start, results_path)
    index = start
    offset = 0
    pending = ''
    deadline = time.time() + CASE_TIMEOUT + (SPAWN_STARTUP if SPAWN else 0)
    hung = None
    while index < len(CASES):
        running = child.alive()  # checked before reading, so lines written just before an exit are still read
        with open(results_path, encoding='utf8') as f:
            f.seek(offset)
            chunk = f.read()
            offset = f.tell()
        pending += chunk
        while '\n' in pending:
            text, pending = pending.split('\n', 1)
            line = json.loads(text)
            merge_line(line)
            index = line['index'] + 1
            deadline = time.time() + CASE_TIMEOUT
        if not running or index >= len(CASES):
            break
        if time.time() > deadline:
            pid = child.pid if isinstance(child, ForkedChild) else child.proc.pid
            hung = hung_child_diagnostics(pid)
            time.sleep(3)  # the child's faulthandler watchdog prints its stack just before this
            child.kill()
            break
        time.sleep(0.1)
    info = child.exit_info()
    if index < len(CASES):
        cat, fn = CASES[index]
        category(cat)
        if hung is not None:
            apply_outcome(cat, fn.__name__, {'hang': CASE_TIMEOUT, 'diag': hung})
        elif 'crash' in info:
            apply_outcome(cat, fn.__name__, {'crash': info['crash']})
        else:
            apply_outcome(cat, fn.__name__, {'exception': f'child exited with status {info.get("exit")} before finishing the case'})
        index += 1
    return index


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):  # pylint: disable=redefined-outer-name
        self.records.append(record)

    def messages(self, level: int = logging.DEBUG) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]

    def find(self, *needles: str, level: int = logging.DEBUG) -> list[str]:
        return [m for m in self.messages(level) if all(n in m for n in needles)]


@contextlib.contextmanager
def capture_log():
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


@contextlib.contextmanager
def opts(**overrides):
    """shared.opts.data overrides, restored on exit."""
    previous = {k: shared.opts.data.get(k, None) for k in overrides}
    missing = [k for k in overrides if k not in shared.opts.data and k not in shared.opts.data_labels]
    assert not missing, f'unknown options: {missing}'
    shared.opts.data.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                shared.opts.data.pop(key, None)
            else:
                shared.opts.data[key] = value


@contextlib.contextmanager
def patched(obj, name: str, value):
    """Set an attribute for the block; an inherited attribute (a bound method) is deleted again, not pinned."""
    own = name in getattr(obj, '__dict__', {})
    previous = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, previous)
        else:
            try:
                delattr(obj, name)
            except AttributeError:
                pass


class OutputsRecorder:
    """Records shared.state.outputs calls with whether the named file existed at that moment."""
    def __init__(self):
        self.calls: list[tuple[str, bool, int]] = []
        self.original = shared.state.outputs

    def __call__(self, results_):
        items = results_ if isinstance(results_, list) else [results_]
        for item in items:
            exists = isinstance(item, str) and os.path.isfile(item)
            size = os.path.getsize(item) if exists else 0
            self.calls.append((str(item), exists, size))
        return self.original(results_)


out_dir = os.path.join(work_dir, 'video')
temp_dir = os.path.join(work_dir, 'gradio-temp')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


# ============================================================
# fixtures: synthetic, numpy only
# ============================================================

def synthetic_canvas(width: int = 1536, height: int = 864, seed: int = 7) -> np.ndarray:
    """Three incommensurate sinusoids, seeded low-frequency noise and hard-edged colour rectangles."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 0.5 + 0.25 * np.sin(x / 37.0) + 0.25 * np.sin(y / 23.0 + x / 211.0)
    g = 0.5 + 0.25 * np.sin(x / 53.0 + y / 71.0) + 0.25 * np.cos(y / 29.0)
    b = 0.5 + 0.25 * np.cos(x / 41.0 - y / 97.0) + 0.25 * np.sin((x + y) / 61.0)
    noise = rng.normal(0.0, 0.08, size=(height // 32 + 1, width // 32 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    canvas = np.stack([r + noise, g - noise, b + 0.5 * noise], axis=-1)
    for i, colour in enumerate([(220, 40, 40), (40, 200, 60), (40, 60, 220), (230, 210, 40), (255, 255, 255), (0, 0, 0)]):
        x0 = 120 + i * 220
        canvas[200:360, x0:x0 + 140] = np.array(colour, dtype=np.float32) / 255.0
    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def pan_frames(count: int = 48, width: int = 1024, height: int = 576) -> list[np.ndarray]:
    canvas = synthetic_canvas()
    return [np.ascontiguousarray(canvas[8 + 3 * i: 8 + 3 * i + height, 16 + 6 * i: 16 + 6 * i + width]) for i in range(count)]


CHART_PATCHES = {
    'red': (220, 40, 40),
    'green': (40, 200, 60),
    'blue': (40, 60, 220),
    'yellow': (230, 210, 40),
    'skin': (200, 150, 120),
    'grey': (128, 128, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}


def chart_frame(width: int = 1280, height: int = 720) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Eight flat patches in a 4x2 grid; returns the frame and each patch centre (x, y)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pw, ph = width // 4, height // 2
    centres = {}
    for i, (name, colour) in enumerate(CHART_PATCHES.items()):
        x, y = (i % 4) * pw, (i // 4) * ph
        img[y:y + ph, x:x + pw] = colour
        centres[name] = (x + pw // 2, y + ph // 2)
    return img, centres


def flat_frames(count: int, width: int, height: int, step: int = 0) -> list[np.ndarray]:
    """Small frames for encoder-open probes; a drifting gradient so an encoder cannot skip them."""
    y, x = np.mgrid[0:height, 0:width]
    frames = []
    for i in range(count):
        r = ((x * 255) // max(1, width - 1) + i * step) % 256
        g = ((y * 255) // max(1, height - 1) + i * step) % 256
        b = ((x + y + i * step) * 255 // max(1, width + height - 2)) % 256
        frames.append(np.stack([r, g, b], axis=-1).astype(np.uint8))
    return frames


def odd_frames(count: int = 4) -> list[np.ndarray]:
    return flat_frames(count, 513, 385, step=7)


def small_frames(count: int = 4) -> list[np.ndarray]:
    return flat_frames(count, 256, 144, step=11)


def make_wav(name: str, seconds: float = 1.0, rate: int = 32000, channels: int = 1) -> str:
    fn = os.path.join(work_dir, name)
    with wave.open(fn, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        count = int(seconds * rate) * channels
        handle.writeframes(struct.pack('<' + 'h' * count, *([0] * count)))
    return fn


def sine(seconds: float, rate: int, channels: int = 1, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    wave_ = (0.3 * np.sin(2 * math.pi * hz * t)).astype(np.float32)
    return np.stack([wave_] * channels, axis=0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape, f'shape mismatch {a.shape} vs {b.shape}'
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def mean_psnr(decoded: list[np.ndarray], source: list[np.ndarray]) -> float:
    assert len(decoded) == len(source), f'frame count {len(decoded)} vs {len(source)}'
    return float(np.mean([psnr(d, s) for d, s in zip(decoded, source, strict=True)]))


def file_hash(fn: str) -> str:
    with open(fn, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def out_path(name: str) -> str:
    return os.path.join(out_dir, name)


# ============================================================
# colour oracle: numpy, not swscale
# ============================================================

MATRICES = {'bt709': (0.2126, 0.0722), 'bt601': (0.299, 0.114)}


def rgb_to_ycbcr(rgb, matrix: str, full: bool) -> np.ndarray:
    """8-bit Y'CbCr of an RGB triple or array, under the named matrix and range."""
    kr, kb = MATRICES[matrix]
    kg = 1.0 - kr - kb
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    y = kr * arr[..., 0] + kg * arr[..., 1] + kb * arr[..., 2]
    pb = (arr[..., 2] - y) / (2.0 * (1.0 - kb))
    pr = (arr[..., 0] - y) / (2.0 * (1.0 - kr))
    if full:
        out = np.stack([255.0 * y, 128.0 + 255.0 * pb, 128.0 + 255.0 * pr], axis=-1)
    else:
        out = np.stack([16.0 + 219.0 * y, 128.0 + 224.0 * pb, 128.0 + 224.0 * pr], axis=-1)
    return np.clip(out, 0.0, 255.0)


def ycbcr_to_rgb(ycc, matrix: str, full: bool) -> np.ndarray:
    """8-bit RGB a decoder produces from Y'CbCr when it applies the named matrix and range."""
    kr, kb = MATRICES[matrix]
    kg = 1.0 - kr - kb
    arr = np.asarray(ycc, dtype=np.float64)
    if full:
        y, pb, pr = arr[..., 0] / 255.0, (arr[..., 1] - 128.0) / 255.0, (arr[..., 2] - 128.0) / 255.0
    else:
        y, pb, pr = (arr[..., 0] - 16.0) / 219.0, (arr[..., 1] - 128.0) / 224.0, (arr[..., 2] - 128.0) / 224.0
    r = y + 2.0 * (1.0 - kr) * pr
    b = y + 2.0 * (1.0 - kb) * pb
    g = (y - kr * r - kb * b) / kg
    return np.clip(np.stack([r, g, b], axis=-1) * 255.0, 0.0, 255.0)


def predicted_patches(matrix: str, full: bool) -> dict[str, np.ndarray]:
    return {name: rgb_to_ycbcr(colour, matrix, full) for name, colour in CHART_PATCHES.items()}


def patch_shift(full: bool = False) -> dict[str, float]:
    """Predicted max-channel distance between the two matrices per patch; what the paired test can see."""
    a, b = predicted_patches('bt709', full), predicted_patches('bt601', full)
    return {name: float(np.max(np.abs(a[name] - b[name]))) for name in CHART_PATCHES}


def plane_array(plane) -> np.ndarray:
    buf = np.frombuffer(plane, dtype=np.uint8)
    rows = buf.size // plane.line_size
    return buf.reshape(rows, plane.line_size)[:plane.height, :plane.width]


def frame_yuv444(frame) -> np.ndarray:
    """Decoded frame as three 8-bit full-size planes (3, h, w); chroma upsampled, matrix untouched."""
    if frame.format.name != 'yuv444p':
        frame = frame.reformat(format='yuv444p')
    return np.stack([plane_array(frame.planes[i]) for i in range(3)], axis=0)


def sample_patches(yuv: np.ndarray, centres: dict[str, tuple[int, int]], radius: int = 4) -> dict[str, np.ndarray]:
    out = {}
    for name, (cx, cy) in centres.items():
        block = yuv[:, cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        out[name] = block.reshape(3, -1).mean(axis=1)
    return out


def patch_errors(measured: dict[str, np.ndarray], predicted: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.max(np.abs(measured[name] - predicted[name]))) for name in measured}


# ============================================================
# av helpers: test-local encode and decode
# ============================================================

def colour_constants():
    from av.video.reformatter import Colorspace, ColorRange
    return {'bt709': (Colorspace.ITU709, ColorRange.MPEG), 'bt709_full': (Colorspace.ITU709, ColorRange.JPEG), 'bt601': (Colorspace.ITU601, ColorRange.MPEG), 'bt601_full': (Colorspace.ITU601, ColorRange.JPEG)}


STREAM_TAGS = {'bt709': {'colorspace': 1, 'color_range': 1, 'color_primaries': 1, 'color_trc': 1}, 'rgb': {'colorspace': 0, 'color_range': 2, 'color_primaries': 1, 'color_trc': 1}}


def encode_frames(fn: str, codec: str, pix_fmt: str, options: dict | None, rate, frames: list[np.ndarray], convert: str | None = None, tags: dict | None = None, threads: int | None = 1):
    """Encode RGB frames the way the writers do (from_ndarray then reformat); optional explicit matrix and tags."""
    opts_ = dict(options or {})
    if threads is not None and codec in ('libx264', 'libx264rgb', 'libx265', 'mpeg4'):
        opts_.setdefault('threads', str(threads))
    container = av.open(fn, mode='w')
    try:
        stream = container.add_stream(codec, rate=rate, options=opts_)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = pix_fmt
        for key, value in (tags or {}).items():
            setattr(stream.codec_context, key, value)
        kwargs = {}
        if convert is not None:
            space, crange = colour_constants()[convert]
            kwargs = {'dst_colorspace': space, 'dst_color_range': crange}
        for i, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame = frame.reformat(width=stream.width, height=stream.height, format=pix_fmt, **kwargs)
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux_one(packet)
        for packet in stream.encode():
            container.mux_one(packet)
    finally:
        container.close()
    return fn


def encode_planes(fn: str, codec: str, planes: np.ndarray, count: int, rate, tags: dict | None = None, options: dict | None = None):
    """Encode prebuilt yuv444p planes (3, h, w) so no matrix conversion happens on the way in."""
    container = av.open(fn, mode='w')
    try:
        stream = container.add_stream(codec, rate=rate, options=dict(options or {}))
        stream.width, stream.height = planes.shape[2], planes.shape[1]
        stream.pix_fmt = 'yuv444p'
        for key, value in (tags or {}).items():
            setattr(stream.codec_context, key, value)
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(planes, format='yuv444p')
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux_one(packet)
        for packet in stream.encode():
            container.mux_one(packet)
    finally:
        container.close()
    return fn


def stream_info(fn: str) -> dict:
    with av.open(fn) as container:
        s = container.streams.video[0]
        cc = s.codec_context
        info = {
            'codec': cc.name,
            'pix_fmt': cc.pix_fmt,
            'width': cc.width,
            'height': cc.height,
            'profile': cc.profile,
            'colorspace': int(cc.colorspace),
            'color_range': int(cc.color_range),
            'color_primaries': int(cc.color_primaries),
            'color_trc': int(cc.color_trc),
            'average_rate': str(s.average_rate) if s.average_rate is not None else None,
            'time_base': str(s.time_base),
            'frames': s.frames,
            'container': container.format.name,
            'audio': [a.codec_context.name for a in container.streams.audio],
            'duration': (float(container.duration) / 1000000) if container.duration is not None else None,
        }
    return info


def same_timing(src: str, res: str) -> tuple[bool, str]:
    def timing(fn: str) -> tuple[int, float]:
        with av.open(fn) as container:
            frames = sum(1 for _ in container.decode(video=0))
            return frames, (float(container.duration) / 1000000) if container.duration is not None else 0.0
    (src_frames, src_duration), (res_frames, res_duration) = timing(src), timing(res)
    period = src_duration / max(src_frames, 1)
    ok = res_frames == src_frames and abs(res_duration - src_duration) <= period
    return ok, f'frames {res_frames}/{src_frames} duration {res_duration:.3f}/{src_duration:.3f}s'


def decode_rgb(fn: str) -> list[np.ndarray]:
    with av.open(fn) as container:
        return [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]


def decode_yuv(fn: str, count: int = 1) -> list[np.ndarray]:
    out = []
    with av.open(fn) as container:
        for f in container.decode(video=0):
            out.append(frame_yuv444(f))
            if len(out) >= count:
                break
    return out


def tags_of(info: dict) -> dict:
    return {k: info[k] for k in ('colorspace', 'color_range', 'color_primaries', 'color_trc')}


def saved_info(res, err) -> tuple[dict, str | None]:
    """Stream info of a writer's result, or the reason there is none: an exception, no path, or a path without a file."""
    if err is not None:
        return {}, f'raised {type(err).__name__}: {err}'
    if not res or not isinstance(res, str):
        return {}, f'returned {res!r}'
    if not os.path.isfile(res):
        return {}, f'returned {res!r} but no file exists'
    try:
        return stream_info(res), None
    except Exception as e:  # pylint: disable=broad-except
        return {}, f'file unreadable: {type(e).__name__}: {e}'


def is_untagged(info: dict) -> bool:
    return info['colorspace'] == 2 or info['color_range'] == 0


def cv2_control(fn: str, frames: list[np.ndarray], fps: float) -> str:
    """The writer every legacy caller had before the move to av: OpenCV mp4v through its ffmpeg backend."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(fn, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    assert writer.isOpened(), 'cv2 mp4v writer did not open through CAP_FFMPEG'
    backend = writer.getBackendName()
    assert backend == 'FFMPEG', f'cv2 writer backend {backend}'
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return fn


def cv2_mpeg4_options(fps: float, width: int, height: int) -> dict:
    """OpenCV's own mpeg4 parameters: bitrate fps*w*h, gop 12, no B frames (cap_ffmpeg_impl.hpp)."""
    return {'b': str(int(fps * width * height)), 'g': '12', 'bf': '0'}


def legacy_save(frames, name: str, **kwargs):
    """modules.video.save_video with p=None, sync=True and an explicit filename under the suite's output dir."""
    kwargs.setdefault('video_type', 'mp4')
    kwargs.setdefault('duration', len(frames) / 24.0)
    kwargs.setdefault('sync', True)
    return video.save_video(None, frames, filename=out_path(name), **kwargs)


# ============================================================
# platform
# ============================================================

FLOOR: dict[str, float] = {}
PRESET_OPEN: dict[str, dict] = {}


def git_head() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short=9', 'HEAD'], cwd=script_dir, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:  # pylint: disable=broad-except
        return ''


def binary_version(name: str) -> str | None:
    path = shutil.which(name)
    if path is None:
        return None
    try:
        first = subprocess.run([path, '-version'], capture_output=True, text=True, check=False).stdout.splitlines()
        return first[0] if first else path
    except Exception:  # pylint: disable=broad-except
        return path


def cv2_video_io() -> list[str]:
    lines = cv2.getBuildInformation().splitlines()
    out = []
    grab = False
    for line in lines:
        if line.strip().startswith('Video I/O'):
            grab = True
            continue
        if grab:
            if not line.strip() or not line.startswith(' '):
                break
            out.append(line.strip())
    return out


def test_versions():
    import gradio
    info = {
        'python': platform.python_version(),
        'os': platform.platform(),
        'commit': git_head(),
        'av': getattr(av, '__version__', None),
        'av_libraries': {k: '.'.join(str(x) for x in v) for k, v in av.library_versions.items()} if av is not None else None,
        'cv2': cv2.__version__,
        'cv2_backends': sorted(cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()),
        'cv2_writer_backends': sorted(cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getWriterBackends()),
        'cv2_video_io': cv2_video_io(),
        'numpy': np.__version__,
        'gradio': gradio.__version__,
        'ffmpeg': binary_version('ffmpeg'),
        'ffprobe': binary_version('ffprobe'),
        'nvidia_encode_library': any(os.path.exists(p) for p in ('/usr/lib/wsl/lib/libnvidia-encode.so.1', '/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1')),
    }
    try:
        import torch
        info['torch'] = torch.__version__
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # pylint: disable=broad-except
        info['torch'] = None
    report['platform'].update(info)
    verdict('test_versions', 'PASS', av=info['av'], commit=info['commit'])
    return True


def test_presets_open():
    """Every codecs_config preset opened for real with the DLSS default pixel format; the result is the report."""
    if need_av():
        return need_av()
    hw_bits = (0x40000, 0x80000)
    frames = small_frames(4)
    for name, cfg in video_codecs.codecs_config.items():
        entry = {'available': name in av.codecs_available, 'ext': cfg['ext'], 'options': cfg['options']}
        if entry['available']:
            try:
                c = av.Codec(name, mode='w')
                formats = c.video_formats
                entry['video_formats'] = sorted(f.name for f in formats) if formats else []
                entry['hardware'] = bool((c.capabilities & hw_bits[0]) or (c.capabilities & hw_bits[1]))
            except Exception as e:  # pylint: disable=broad-except
                entry['codec_error'] = f'{type(e).__name__}: {e}'
            fn = out_path(f'preset_{name}.{cfg["ext"]}')
            t0 = time.time()
            try:
                encode_frames(fn, name, 'yuv420p', video_writer.parse_options(cfg['options']), 24, frames, threads=None)
                info = stream_info(fn)
                entry['open'] = 'ok'
                entry['stored_pix_fmt'] = info['pix_fmt']
            except Exception as e:  # pylint: disable=broad-except
                entry['open'] = f'{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}'
            entry['ms'] = round(1000 * (time.time() - t0))
        PRESET_OPEN[name] = entry
    report['platform']['presets'] = PRESET_OPEN
    opened = sorted(k for k, v in PRESET_OPEN.items() if v.get('open') == 'ok')
    verdict('test_presets_open', 'PASS', opened=opened, refused=sorted(k for k, v in PRESET_OPEN.items() if v.get('available') and v.get('open') != 'ok'), unavailable=sorted(k for k, v in PRESET_OPEN.items() if not v.get('available')))
    assert 'libx264' in opened and 'mpeg4' in opened, f'the suite needs libx264 and mpeg4; opened={opened}'
    return True


# ============================================================
# floors and controls
# ============================================================

PAN = {}
CHART = {}


def pan():
    if 'frames' not in PAN:
        PAN['frames'] = pan_frames()
    return PAN['frames']


def chart(size: str = '720'):
    if size not in CHART:
        w, h = (1280, 720) if size == '720' else (512, 512)
        CHART[size] = chart_frame(w, h)
    return CHART[size]


def kbps(fn: str, seconds: float) -> float:
    return round(os.path.getsize(fn) * 8 / seconds / 1000)


def test_quality_floor():
    """Floor: two encodes of the same frames with the same encoder and settings; the old cv2 writer is the reference,
    and av mpeg4 at OpenCV's own parameters is recorded as an admissibility check, not as noise."""
    if need_av():
        return need_av()
    frames = pan()
    fps = 24.0
    seconds = len(frames) / fps
    h, w = frames[0].shape[:2]
    t0 = time.time()
    fn_cv2 = cv2_control(out_path('control_cv2_mp4v.mp4'), frames, fps)
    t1 = time.time()
    fn_av = encode_frames(out_path('control_av_mpeg4_cv2params.mp4'), 'mpeg4', 'yuv420p', cv2_mpeg4_options(fps, w, h), 24, frames)
    t2 = time.time()
    fn_repeat = encode_frames(out_path('control_av_mpeg4_cv2params_repeat.mp4'), 'mpeg4', 'yuv420p', cv2_mpeg4_options(fps, w, h), 24, frames)
    psnr_cv2 = mean_psnr(decode_rgb(fn_cv2), frames)
    psnr_av = mean_psnr(decode_rgb(fn_av), frames)
    psnr_repeat = mean_psnr(decode_rgb(fn_repeat), frames)
    floor = abs(psnr_av - psnr_repeat)
    FLOOR['quality'] = floor
    FLOOR['quality_margin'] = max(0.5, 4 * floor)
    FLOOR['psnr_cv2'] = psnr_cv2
    measure('quality_floor', psnr_cv2=round(psnr_cv2, 2), psnr_av_cv2params=round(psnr_av, 2), psnr_av_repeat=round(psnr_repeat, 2), floor_db=round(floor, 3), margin_db=round(FLOOR['quality_margin'], 2), cv2_params_match_db=round(psnr_cv2 - psnr_av, 2), kbps_cv2=kbps(fn_cv2, seconds), kbps_av=kbps(fn_av, seconds), ms_cv2=round(1000 * (t1 - t0)), ms_av=round(1000 * (t2 - t1)))
    verdict('test_quality_floor', 'PASS', floor_db=round(floor, 3), margin_db=round(FLOOR['quality_margin'], 2), cv2_params_match_db=round(psnr_cv2 - psnr_av, 2))
    return True


def test_quality_negative_control():
    """av mpeg4 with no options (the legacy default after bc0074f53) must fail the quality bar by a wide margin."""
    if need_av():
        return need_av()
    if 'quality_margin' not in FLOOR:
        return 'quality floor not measured'
    frames = pan()
    fn = encode_frames(out_path('control_av_mpeg4_noopts.mp4'), 'mpeg4', 'yuv420p', {}, 24, frames)
    p = mean_psnr(decode_rgb(fn), frames)
    gap = FLOOR['psnr_cv2'] - p
    measure('quality_negative_control', psnr_mpeg4_noopts=round(p, 1), gap_db=round(gap, 2), required_db=round(4 * FLOOR['quality_margin'], 2), kbps=kbps(fn, len(frames) / 24.0))
    verdict('test_quality_negative_control', 'PASS', gap_db=round(gap, 2))
    assert gap >= 4 * FLOOR['quality_margin'], f'negative control gap {gap:.2f} dB below 4*margin {4 * FLOOR["quality_margin"]:.2f}; the floor cannot discriminate'
    return True


def test_quality_fixture_guard():
    """The fixture separates the codecs the suite compares: x264 at the preset beats mpeg4 without options by more than the margin."""
    if need_av():
        return need_av()
    if 'quality_margin' not in FLOOR:
        return 'quality floor not measured'
    frames = pan()
    fn = encode_frames(out_path('control_av_x264_preset.mp4'), 'libx264', 'yuv420p', video_writer.parse_options(video_codecs.get_codec_options('libx264')), 24, frames)
    p_x264 = mean_psnr(decode_rgb(fn), frames)
    fn2 = out_path('control_av_mpeg4_noopts.mp4')
    p_mpeg4 = mean_psnr(decode_rgb(fn2), frames) if os.path.isfile(fn2) else mean_psnr(decode_rgb(encode_frames(fn2, 'mpeg4', 'yuv420p', {}, 24, frames)), frames)
    measure('quality_fixture_guard', psnr_x264_preset=round(p_x264, 1), psnr_mpeg4_noopts=round(p_mpeg4, 1), kbps_x264=kbps(fn, len(frames) / 24.0))
    verdict('test_quality_fixture_guard', 'PASS', separation_db=round(p_x264 - p_mpeg4, 2))
    assert p_x264 - p_mpeg4 > FLOOR['quality_margin'], f'x264 {p_x264:.1f} vs mpeg4 {p_mpeg4:.1f} within margin {FLOOR["quality_margin"]:.2f}'
    return True


def colour_verdict(info: dict, yuv: np.ndarray, centres: dict) -> tuple[bool, str, dict]:
    """Per-patch paired matrix test against the oracle under the stream's own tags; untagged fails before any decode."""
    if is_untagged(info):
        return False, f'untagged colorspace={info["colorspace"]} range={info["color_range"]}', {}
    tag = {1: 'bt709', 5: 'bt601', 6: 'bt601'}.get(info['colorspace'])
    if tag is None:
        return False, f'colorspace {info["colorspace"]} not covered by the oracle', {}
    full = info['color_range'] == 2
    floor = FLOOR.get('colour', 0.0)
    measured = sample_patches(yuv, centres)
    err_tag = patch_errors(measured, predicted_patches(tag, full))
    other = 'bt601' if tag == 'bt709' else 'bt709'
    err_other = patch_errors(measured, predicted_patches(other, full))
    shift = patch_shift(full)
    discriminating = [n for n in CHART_PATCHES if shift[n] > 3 * floor]
    bad = [n for n in discriminating if not err_tag[n] < err_other[n] - 2 * floor]
    pred = predicted_patches(tag, full)
    range_bad = [n for n in ('black', 'white') if abs(measured[n][0] - pred[n][0]) > 2 + floor]
    detail = {'tag': tag, 'full': full, 'discriminating': discriminating, 'err_tagged': {k: round(v, 2) for k, v in err_tag.items()}, 'err_other': {k: round(v, 2) for k, v in err_other.items()}, 'bad': bad, 'range_bad': range_bad}
    ok = not bad and not range_bad and len(discriminating) >= 2
    reason = '' if ok else f'matrix mismatch on {bad}; range off on {range_bad}; discriminating={discriminating}'
    return ok, reason, detail


def test_colour_floor():
    """Floor: matched-matrix error of a tagged BT.709 x264 encode at the preset, max over the eight patches."""
    if need_av():
        return need_av()
    img, centres = chart('720')
    fn = encode_frames(out_path('control_chart_bt709_x264.mp4'), 'libx264', 'yuv420p', video_writer.parse_options(video_codecs.get_codec_options('libx264')), 24, [img] * 12, convert='bt709', tags=STREAM_TAGS['bt709'])
    info = stream_info(fn)
    assert not is_untagged(info), f'control encode lost its tags: {tags_of(info)}'
    measured = sample_patches(decode_yuv(fn)[0], centres)
    err = patch_errors(measured, predicted_patches('bt709', False))
    floor = max(err.values())
    FLOOR['colour'] = floor
    shift = patch_shift(False)
    discriminating = [n for n in CHART_PATCHES if shift[n] > 3 * floor]
    measure('colour_floor', floor_levels=round(floor, 2), err_per_patch={k: round(v, 2) for k, v in err.items()}, predicted_shift={k: round(v, 1) for k, v in shift.items()}, discriminating=discriminating)
    verdict('test_colour_floor', 'PASS', floor_levels=round(floor, 2), discriminating=discriminating)
    assert len(discriminating) >= 2, f'floor {floor:.2f} leaves fewer than two discriminating patches: {discriminating}'
    return True


def test_colour_positive_control():
    """numpy-built BT.709 planes, encoded losslessly and tagged, pass the paired test."""
    if need_av():
        return need_av()
    if 'colour' not in FLOOR:
        return 'colour floor not measured'
    img, centres = chart('720')
    planes = np.ascontiguousarray(np.rint(rgb_to_ycbcr(img, 'bt709', False)).astype(np.uint8).transpose(2, 0, 1))
    fn = encode_planes(out_path('control_chart_bt709_numpy_ffv1.mkv'), 'ffv1', planes, 4, 24, tags=STREAM_TAGS['bt709'])
    info = stream_info(fn)
    ok, reason, detail = colour_verdict(info, decode_yuv(fn)[0], centres)
    verdict('test_colour_positive_control', 'PASS', tags=tags_of(info), **{k: detail.get(k) for k in ('bad', 'range_bad')})
    assert ok, reason
    return True


def test_colour_negative_controls():
    """An untagged BT.601 encode fails on tags; a BT.709-tagged encode converted with BT.601 fails on the matrix."""
    if need_av():
        return need_av()
    if 'colour' not in FLOOR:
        return 'colour floor not measured'
    img, centres = chart('720')
    options = video_writer.parse_options(video_codecs.get_codec_options('libx264'))
    fn_untagged = encode_frames(out_path('control_chart_untagged_601_x264.mp4'), 'libx264', 'yuv420p', options, 24, [img] * 12, convert='bt601')
    ok_untagged, reason_untagged, _ = colour_verdict(stream_info(fn_untagged), decode_yuv(fn_untagged)[0], centres)
    fn_lying = encode_frames(out_path('control_chart_lying_601_as_709.mp4'), 'libx264', 'yuv420p', options, 24, [img] * 12, convert='bt601', tags=STREAM_TAGS['bt709'])
    ok_lying, reason_lying, detail_lying = colour_verdict(stream_info(fn_lying), decode_yuv(fn_lying)[0], centres)
    verdict('test_colour_negative_controls', 'PASS', untagged=reason_untagged, mistagged=reason_lying, mistagged_bad=detail_lying.get('bad'))
    assert not ok_untagged, 'untagged BT.601 encode passed the colour verdict'
    assert not ok_lying, 'BT.601-converted stream tagged BT.709 passed the colour verdict'
    assert 'untagged' in reason_untagged, reason_untagged
    assert detail_lying.get('bad'), f'mistagged stream failed for another reason: {reason_lying}'
    return True


def test_log_capture_sees_writer_lines():
    """Cases assert on log lines; the capture must see a writer error and must see nothing when nothing is logged."""
    with capture_log() as logs:
        call_and_catch(video.save_video_atomic, small_frames(1), out_path('log_control.mp4'), video_type='mp4', mp4_opt='stats=/dev/null')
    with capture_log() as quiet:
        pass
    found = logs.find('stats', level=logging.ERROR)
    verdict('test_log_capture_sees_writer_lines', 'PASS', found=found[:1], quiet=len(quiet.records))
    assert found, f'capture missed the rejected-option error; saw {logs.messages()[:3]}'
    assert not quiet.records, f'capture saw records with nothing logged: {quiet.messages()[:3]}'
    return True


def test_lossless_rgb_roundtrip():
    if need_av():
        return need_av()
    frames = small_frames(4)
    fn = encode_frames(out_path('control_x264rgb_lossless.mp4'), 'libx264rgb', 'rgb24', {'crf': '0', 'preset': 'fast'}, 24, frames)
    decoded = decode_rgb(fn)
    assert len(decoded) == len(frames), f'{len(decoded)} frames decoded'
    exact = all(np.array_equal(d, s) for d, s in zip(decoded, frames, strict=True))
    verdict('test_lossless_rgb_roundtrip', 'PASS', exact=exact, pix_fmt=stream_info(fn)['pix_fmt'])
    assert exact, 'lossless RGB round trip is not bit exact'
    return True


# ============================================================
# writer failure paths
# ============================================================

def call_and_catch(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # pylint: disable=broad-except
        return None, e


@contextlib.contextmanager
def av_missing():
    """Make `import av` fail."""
    saved = sys.modules.get('av', None)
    sys.modules['av'] = None
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop('av', None)
        else:
            sys.modules['av'] = saved


def test_sync_unknown_codec_returns_none():
    if need_av():
        return need_av()
    rec = OutputsRecorder()
    with patched(shared.state, 'outputs', rec):
        res, err = call_and_catch(legacy_save, small_frames(4), 'fail_unknown_codec.mp4', mp4_codec='nonexistent')
    fn = out_path('fail_unknown_codec.mp4')
    verdict('test_sync_unknown_codec_returns_none', 'PASS', returned=res, raised=repr(err) if err else None, file_exists=os.path.exists(fn), registered=[c[0] for c in rec.calls])
    assert err is None, f'sync save raised {err!r}'
    assert res is None, f'sync save returned {res!r} for a codec that cannot open'
    assert not os.path.exists(fn), 'a file was left behind'
    assert not rec.calls, f'registered outputs for a failed save: {rec.calls}'
    return True


def test_stream_rejected_pix_fmt_raises_with_reason():
    """DLSS streams into the writer; a pixel format the codec rejects must raise and name both."""
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, small_frames(1), 'fail_stream_pix.mp4', stream=True, mp4_codec='libx264rgb', mp4_pix='yuv420p')
    fn = out_path('fail_stream_pix.mp4')
    verdict('test_stream_rejected_pix_fmt_raises_with_reason', 'PASS', returned=repr(res) if res is not None else None, raised=f'{type(err).__name__}: {err}' if err else None, file_exists=os.path.exists(fn))
    assert err is not None, f'stream save returned {res!r} instead of raising'
    assert 'libx264rgb' in str(err) and 'yuv420p' in str(err), f'reason does not name codec and pixel format: {err}'
    assert type(err).__name__ == 'VideoWriteError', f'raised {type(err).__name__}'
    assert not os.path.exists(fn), 'a file was left behind'
    return True


def test_container_refusing_codec_removes_its_file():
    """libx264 into webm: the header is refused after the file is created; the writer removes what it created."""
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, small_frames(4), 'fail_x264.webm', video_type='webm', mp4_codec='libx264', mp4_ext='webm')
    fn = out_path('fail_x264.webm')
    verdict('test_container_refusing_codec_removes_its_file', 'PASS', returned=res, raised=repr(err) if err else None, file_exists=os.path.exists(fn), size=os.path.getsize(fn) if os.path.exists(fn) else None)
    assert err is None, f'sync save raised {err!r}'
    assert res is None, f'sync save returned {res!r} for a container that refused the codec'
    assert not os.path.exists(fn), 'the refused container file was left behind'
    return True


def test_av_missing_sync_returns_none_stream_raises():
    rec = OutputsRecorder()
    with av_missing(), patched(shared.state, 'outputs', rec):
        res_sync, err_sync = call_and_catch(legacy_save, small_frames(2), 'fail_noav_sync.mp4', mp4_codec='libx264')
        res_stream, err_stream = call_and_catch(legacy_save, small_frames(1), 'fail_noav_stream.mp4', stream=True, mp4_codec='libx264')
    verdict('test_av_missing_sync_returns_none_stream_raises', 'PASS', sync_returned=res_sync, sync_raised=repr(err_sync) if err_sync else None, stream_raised=f'{type(err_stream).__name__}: {err_stream}' if err_stream else None, registered=[c[0] for c in rec.calls])
    assert err_sync is None and res_sync is None, f'sync without av: returned {res_sync!r} raised {err_sync!r}'
    assert err_stream is not None, f'stream without av returned {res_stream!r} instead of raising'
    assert not rec.calls, f'registered outputs without av: {rec.calls}'
    return True


def test_zero_byte_mp4_metadata_is_empty():
    fn = out_path('zero.mp4')
    with open(fn, 'wb'):
        pass
    res, err = call_and_catch(video.get_video_metadata, fn)
    info = video.get_video_info(fn)
    verdict('test_zero_byte_mp4_metadata_is_empty', 'PASS', metadata=res, raised=repr(err) if err else None, info_error='error' in info)
    assert err is None, f'get_video_metadata raised {err!r} on a zero-byte file'
    assert res == {}, f'metadata {res!r} for a zero-byte file'
    return True


def test_process_missing_video_reports_reason():
    """The Process tab with a video path that does not exist returns no video and says why in info."""
    from modules import postprocessing
    missing = out_path('process_missing_input.mp4')
    with capture_log() as logs:
        res, err = call_and_catch(postprocessing.run_postprocessing, 3, None, [], '', '', missing, False)
    _outputs, video_out, info, _params = res if res is not None else (None, None, None, None)
    verdict('test_process_missing_video_reports_reason', 'PASS', raised=repr(err) if err else None, video=video_out, info=info, unpack_error=bool(logs.find('unpack')))
    assert err is None, f'run_postprocessing raised {err!r}'
    assert video_out is None, f'returned video {video_out!r} for a missing input'
    assert info and 'not found' in info, f'info {info!r} does not name the missing file'
    assert not logs.find('unpack'), f'logged {logs.find("unpack")[:1]}'
    return True


def test_process_tab_shows_failure_text_whole():
    """A failed Process video run shows its error text as written; parsed as infotext it lost the codec and accepted formats."""
    import html
    from modules import postprocessing, ui_postprocessing
    reason = 'codec=libx264rgb pix_fmt=yuv420p: pixel format yuv420p not supported, libx264rgb accepts bgr0, bgr24, rgb24'
    original = postprocessing.run_postprocessing
    postprocessing.run_postprocessing = lambda *args, **kwargs: ([], None, reason, {})
    try:
        res, err = call_and_catch(ui_postprocessing.submit_process, 'task(video-save-test)', 3, None, [], '', '', out_path('process_refused.mp4'), False, True)
    finally:
        postprocessing.run_postprocessing = original
    shown = res[3] if res is not None else ''
    verdict('test_process_tab_shows_failure_text_whole', 'PASS', raised=repr(err) if err else None, whole=html.escape(reason) in shown)
    assert err is None, f'submit_process raised {err!r}'
    assert html.escape(reason) in shown, f'shown {shown!r}'
    return True


def test_output_equal_to_source_refused():
    if need_av():
        return need_av()
    need_param(video.save_video, 'source')
    src = encode_frames(out_path('source_input.mp4'), 'libx264', 'yuv420p', {}, 24, small_frames(4))
    before = file_hash(src)
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, small_frames(4), 'source_input.mp4', mp4_codec='libx264', source=src)
    verdict('test_output_equal_to_source_refused', 'PASS', returned=res, raised=repr(err) if err else None, unchanged=file_hash(src) == before, log=logs.find('source_input.mp4', level=logging.WARNING)[:1])
    assert err is None and res is None, f'save over its own input returned {res!r} raised {err!r}'
    assert file_hash(src) == before, 'the input file was modified'
    assert logs.find('source_input.mp4', level=logging.WARNING), 'no log line names the refused overwrite'
    return True


def dlss_module():
    try:
        from scripts.dlss import dlss_run as module
    except Exception as e:  # pylint: disable=broad-except
        return None, f'scripts.dlss.dlss_run import failed: {type(e).__name__}: {e}'
    return module, ''


def test_dlss_none_kwargs_resolve_to_ui_values():
    """API and Enso jobs pass None for the codec controls; they must resolve to what the UI shows."""
    module, reason = dlss_module()
    if module is None:
        return reason
    resolve = need(module, 'resolve_video_options')
    codec, ext, pix, opt = resolve(None, None, None, None)
    verdict('test_dlss_none_kwargs_resolve_to_ui_values', 'PASS', resolved=[codec, ext, pix, opt])
    assert (codec, ext) == ('libx264', 'mp4'), f'{codec} {ext}'
    assert pix in ('yuv420p', None), f'pix {pix!r}'
    assert opt == video_codecs.get_codec_options('libx264'), f'options {opt!r}'
    codec, ext, pix, opt = resolve('prores_ks', None, None, '')
    assert ext == 'mov', f'prores_ks ext {ext}'
    assert opt == '', 'an explicit empty option string means no options'
    return True


def test_gif_failure_returns_none():
    blocker = out_path('not_a_dir')
    with open(blocker, 'w', encoding='utf8') as f:
        f.write('x')
    images = [Image.fromarray(f) for f in small_frames(3)]
    res, err = call_and_catch(video.save_video, None, images, filename=os.path.join(blocker, 'anim.gif'), video_type='gif', duration=1.0, sync=True)
    verdict('test_gif_failure_returns_none', 'PASS', returned=res, raised=repr(err) if err else None)
    assert err is None, f'gif save raised {err!r}'
    assert res is None, f'gif save returned {res!r} for an unwritable path'
    return True


def test_mid_encode_exception_returns_none():
    if need_av():
        return need_av()
    frames = small_frames(6)
    frames[3] = np.zeros((144, 256, 3), dtype=np.float32)  # refused by av as an rgb24 frame
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, frames, 'fail_mid_encode.mp4', mp4_codec='libx264')
    partial = out_path('fail_mid_encode.mp4')
    partial_frames = len(decode_rgb(partial)) if os.path.isfile(partial) else None
    verdict('test_mid_encode_exception_returns_none', 'PASS', returned=res, raised=repr(err) if err else None, errors=logs.find('fail_mid_encode', level=logging.ERROR)[:1], partial_frames=partial_frames)
    assert err is None, f'sync save raised {err!r}'
    assert res is None, f'sync save returned {res!r} after a frame failed mid-encode'
    assert logs.find('fail_mid_encode', level=logging.ERROR), 'no error line names the file'
    assert partial_frames == 3, f'the frames written before the failure: {partial_frames}'
    return True


def video_packets(fn: str) -> list[bytes]:
    with av.open(fn) as container:
        return [bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size]


def test_both_paths_write_the_same_stream():
    """The legacy writer, the Video tab writer and a direct BT.709 encode of the same RGB frames with the same options give identical packets; repeat encodes are packet-identical, so the floor is 0."""
    if need_av():
        return need_av()
    frames = small_frames(12)
    preset = video_codecs.get_codec_options('libx264')
    legacy, legacy_err = call_and_catch(legacy_save, frames, 'paths_legacy.mp4', mp4_codec='libx264', mp4_opt=preset)
    tab = out_path('paths_tab.mp4')
    _res, tab_err = call_and_catch(video_save.atomic_save_video, tab, tensor_of(frames), fps=24, codec='libx264', options=preset)
    ref = encode_frames(out_path('paths_ref.mp4'), 'libx264', 'yuv420p', video_writer.parse_options(preset), 24, frames, convert='bt709', tags=STREAM_TAGS['bt709'], threads=None)
    assert legacy_err is None and legacy, f'legacy save: returned {legacy!r} raised {legacy_err!r}'
    assert tab_err is None and os.path.isfile(tab), f'Video tab save raised {tab_err!r}'
    packets = {name: video_packets(fn) for name, fn in (('legacy', legacy), ('video_tab', tab), ('reference', ref))}
    first_diff = {name: next((i for i, (a, b) in enumerate(zip(packets[name], packets['reference'])) if a != b), None) for name in ('legacy', 'video_tab')}
    measure('both_paths', **{name: len(value) for name, value in packets.items()}, first_differing_packet=str(first_diff))
    verdict('test_both_paths_write_the_same_stream', 'PASS', packets={name: len(value) for name, value in packets.items()}, first_differing_packet=first_diff)
    assert packets['video_tab'] == packets['reference'], f'Video tab packets differ from the reference from packet {first_diff["video_tab"]}'
    assert packets['legacy'] == packets['reference'], f'legacy packets differ from the reference from packet {first_diff["legacy"]}'
    return True


def test_legacy_save_accepts_grayscale_frames():
    """A clip of 2-D frames (grayscale images) encodes as the same frames repeated over three channels."""
    if need_av():
        return need_av()
    gray = [np.full((144, 256), 16 + 18 * i, dtype=np.uint8) for i in range(6)]
    rgb = [np.repeat(frame[:, :, None], 3, axis=2) for frame in gray]
    res_gray, err_gray = call_and_catch(legacy_save, gray, 'gray_2d.mp4', mp4_codec='libx264')
    res_rgb, err_rgb = call_and_catch(legacy_save, rgb, 'gray_rgb.mp4', mp4_codec='libx264')
    verdict('test_legacy_save_accepts_grayscale_frames', 'PASS', gray_returned=res_gray, gray_raised=repr(err_gray) if err_gray else None, rgb_returned=res_rgb)
    assert err_gray is None and res_gray and os.path.isfile(res_gray), f'2-D frames: returned {res_gray!r} raised {err_gray!r}'
    assert err_rgb is None and res_rgb, f'control: returned {res_rgb!r} raised {err_rgb!r}'
    assert video_packets(res_gray) == video_packets(res_rgb), 'the 2-D clip differs from the same frames as RGB'
    return True


def test_stream_writer_failure_leaves_partial_file():
    """Stream mode (DLSS): a frame that fails mid-stream releases the writer and leaves a file holding the frames written before it."""
    if need_av():
        return need_av()
    frames = small_frames(4)
    res, err = call_and_catch(legacy_save, frames[:1], 'stream_partial.mp4', stream=True, mp4_codec='libx264')
    assert err is None and isinstance(res, tuple), f'stream open: returned {res!r} raised {err!r}'
    fn, writer = res
    video.save_video_frames(writer, frames[1:3])
    _res, write_err = call_and_catch(video.save_video_frames, writer, [np.zeros((144, 256, 3), dtype=np.float32)])
    _res, release_err = call_and_catch(writer.release)
    decoded = len(decode_rgb(fn)) if os.path.isfile(fn) else None
    verdict('test_stream_writer_failure_leaves_partial_file', 'PASS', write_raised=type(write_err).__name__ if write_err else None, release_raised=repr(release_err) if release_err else None, decoded=decoded)
    assert write_err is not None, 'a float frame was accepted'
    assert release_err is None, f'release after the failure raised {release_err!r}'
    assert decoded == 3, f'partial file holds {decoded} frames, 3 were written'
    return True


def test_write_audio_without_stream_refused():
    """Audio needs its stream declared when the writer opens, before the header; a writer opened without one refuses audio by name."""
    if need_av():
        return need_av()
    open_writer = need(video_writer, 'open_video_writer')
    writer = open_writer(out_path('no_audio_stream.mp4'), 256, 144, 24, 'libx264')
    try:
        _res, err = call_and_catch(writer.write_audio, np.zeros((2, 2400), dtype=np.float32), 24000)
    finally:
        writer.release()
    verdict('test_write_audio_without_stream_refused', 'PASS', raised=f'{type(err).__name__}: {err}' if err else None)
    assert type(err).__name__ == 'VideoWriteError' and 'audio' in str(err), f'raised {err!r}'
    return True


def test_sync_success_registers_after_open():
    if need_av():
        return need_av()
    rec = OutputsRecorder()
    with patched(shared.state, 'outputs', rec):
        res = legacy_save(small_frames(4), 'ok_register.mp4', mp4_codec='libx264')
    fn = out_path('ok_register.mp4')
    verdict('test_sync_success_registers_after_open', 'PASS', returned=res, calls=rec.calls)
    assert res == fn, f'returned {res!r}'
    assert len(rec.calls) == 1, f'registered {len(rec.calls)} times: {rec.calls}'
    assert rec.calls[0][1] and rec.calls[0][2] > 0, f'registered before the file existed: {rec.calls}'
    return True


def test_threaded_save_joined():
    if need_av():
        return need_av()
    started = []

    class RecordingThread(threading.Thread):
        def start(self):
            started.append(self)
            super().start()

    with patched(threading, 'Thread', RecordingThread), capture_log() as logs:
        res_ok = legacy_save(small_frames(4), 'threaded_ok.mp4', sync=False, mp4_codec='libx264')
        res_bad = legacy_save(small_frames(4), 'threaded_bad.mp4', sync=False, mp4_codec='nonexistent')
        for t in started:
            t.join(timeout=60)
    ok_fn, bad_fn = out_path('threaded_ok.mp4'), out_path('threaded_bad.mp4')
    verdict('test_threaded_save_joined', 'PASS', threads=len(started), returned=[res_ok, res_bad], ok_exists=os.path.isfile(ok_fn), bad_exists=os.path.exists(bad_fn), errors=logs.find('threaded_bad', level=logging.ERROR)[:1])
    assert len(started) == 2, f'{len(started)} threads started'
    assert res_ok == ok_fn and os.path.isfile(ok_fn), 'threaded save did not produce its file'
    assert not os.path.exists(bad_fn), 'threaded failure left a file'
    assert logs.find('threaded_bad', level=logging.ERROR), 'threaded failure was not logged'
    return True


# ============================================================
# pixel formats
# ============================================================

def tensor_of(frames: list[np.ndarray]):
    import torch
    return torch.from_numpy(np.stack(frames))


def test_presets_declare_pix_fmt():
    get_pix = need(video_codecs, 'get_codec_pix_fmt')
    missing = [name for name, cfg in video_codecs.codecs_config.items() if not cfg.get('pix_fmt')]
    assert not missing, f'presets without pix_fmt: {missing}'
    unsupported = []
    for name in video_codecs.codecs_config:
        pix = get_pix(name)
        formats = PRESET_OPEN.get(name, {}).get('video_formats')
        if formats and pix not in formats:
            unsupported.append(f'{name}:{pix} not in {formats}')
    verdict('test_presets_declare_pix_fmt', 'PASS', pix_fmts={k: get_pix(k) for k in video_codecs.codecs_config})
    assert not unsupported, f'preset pix_fmt the encoder does not accept: {unsupported}'
    return True


def test_table_pix_fmt_opens_in_legacy_loop():
    if need_av():
        return need_av()
    out = {}
    for codec, ext in (('libx264rgb', 'mp4'), ('prores_ks', 'mov'), ('huffyuv', 'avi')):
        if not PRESET_OPEN.get(codec, {}).get('available', True):
            out[codec] = 'unavailable'
            continue
        res, err = call_and_catch(legacy_save, small_frames(4), f'pix_table_{codec}.{ext}', video_type=ext, mp4_codec=codec, mp4_ext=ext, mp4_pix=None)
        info, reason = saved_info(res, err)
        out[codec] = info['pix_fmt'] if reason is None else reason
    verdict('test_table_pix_fmt_opens_in_legacy_loop', 'PASS', stored=out)
    bad = {k: v for k, v in out.items() if v in (None, '') or str(v).startswith(('returned', 'raised', 'file unreadable'))}
    assert not bad, f'table pixel format did not open: {bad}'
    return True


def test_table_pix_fmt_opens_in_video_tab_loop():
    if need_av():
        return need_av()
    fn = out_path('pix_table_videotab_prores.mov')
    res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of(small_frames(4)), fps=24, codec='prores_ks', pix_fmt=None)
    info = stream_info(fn) if os.path.isfile(fn) and err is None else {}
    verdict('test_table_pix_fmt_opens_in_video_tab_loop', 'PASS', raised=repr(err) if err else None, stored=info.get('pix_fmt'))
    assert err is None, f'atomic_save_video(pix_fmt=None) raised {err!r}'
    assert info.get('codec') == 'prores', f'stored {info}'
    assert res in (None, True), f'returned {res!r}'  # True once the loop reports success, None before
    return True


def test_rejected_pix_fmt_names_the_list():
    if need_av():
        return need_av()
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, small_frames(2), 'pix_rejected.mp4', mp4_codec='libx264rgb', mp4_pix='yuv420p')
    lines = logs.find('libx264rgb', level=logging.ERROR)
    verdict('test_rejected_pix_fmt_names_the_list', 'PASS', returned=res, raised=repr(err) if err else None, lines=lines[:2])
    assert err is None and res is None, f'returned {res!r} raised {err!r}'
    assert any('yuv420p' in m and ('rgb24' in m or 'bgr0' in m) for m in lines), f'no line names the rejected format and the accepted list: {lines}'
    return True


def test_empty_pix_fmt_refused_naming_default():
    if need_av():
        return need_av()
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, small_frames(2), 'pix_empty.mp4', mp4_codec='libx264', mp4_pix='')
    lines = logs.find('yuv420p', level=logging.ERROR)
    verdict('test_empty_pix_fmt_refused_naming_default', 'PASS', returned=res, raised=repr(err) if err else None, lines=lines[:2])
    assert err is None and res is None, f'returned {res!r} raised {err!r}'
    assert lines, 'no error line names the default pixel format'
    return True


def test_out_of_table_codec_defaults_yuv420p():
    if need_av():
        return need_av()
    if 'mpeg2video' not in av.codecs_available:
        return 'mpeg2video encoder not in this av build'
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, small_frames(4), 'pix_oot_mpeg2.ts', video_type='ts', mp4_codec='mpeg2video', mp4_ext='ts', mp4_pix=None)
    info, reason = saved_info(res, err)
    lines = logs.find('mpeg2video', 'yuv420p')
    verdict('test_out_of_table_codec_defaults_yuv420p', 'PASS', returned=res, reason=reason, stored=info.get('pix_fmt'), lines=lines[:1])
    assert reason is None, reason
    assert info.get('pix_fmt') == 'yuv420p', f'stored {info.get("pix_fmt")}'
    assert lines, 'no log line says the out-of-table codec got yuv420p'
    return True


def test_ext_none_follows_codec_when_container_refuses():
    """ffv1 asked into mp4 with no explicit extension: the writer picks the codec's container and says so."""
    if need_av():
        return need_av()
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, small_frames(4), 'ext_follow_ffv1', video_type='mp4', mp4_codec='ffv1', mp4_ext=None)
    lines = logs.find('ffv1', 'mkv')
    verdict('test_ext_none_follows_codec_when_container_refuses', 'PASS', returned=res, raised=repr(err) if err else None, lines=lines[:1], leftovers=sorted(f for f in os.listdir(out_dir) if f.startswith('ext_follow_ffv1')))
    assert err is None and res, f'returned {res!r} raised {err!r}'
    assert res.endswith('.mkv') and os.path.isfile(res), f'returned {res!r}'
    assert stream_info(res)['codec'] == 'ffv1'
    assert lines, 'no log line explains the container change'
    return True


def decoded_size(fn: str) -> tuple[int, int]:
    info = stream_info(fn)
    return info['width'], info['height']


def test_odd_size_cropped_under_yuv420p_legacy():
    if need_av():
        return need_av()
    with capture_log() as logs:
        res, err = call_and_catch(legacy_save, odd_frames(), 'odd_legacy_x264.mp4', mp4_codec='libx264', mp4_pix='yuv420p')
    size = decoded_size(res) if res and not err else None
    lines = logs.find('513', '512', level=logging.WARNING)
    verdict('test_odd_size_cropped_under_yuv420p_legacy', 'PASS', returned=res, raised=repr(err) if err else None, size=size, lines=lines[:1])
    assert err is None and res, f'returned {res!r} raised {err!r}'
    assert size == (512, 384), f'decoded {size}'
    assert lines, 'no warning names both sizes'
    return True


def test_odd_size_cropped_under_yuv420p_video_tab():
    if need_av():
        return need_av()
    fn = out_path('odd_videotab_x264.mp4')
    with capture_log() as logs:
        _res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of(odd_frames()), fps=24, codec='libx264', pix_fmt='yuv420p')
    size = decoded_size(fn) if err is None and os.path.isfile(fn) else None
    lines = logs.find('513', '512', level=logging.WARNING)
    verdict('test_odd_size_cropped_under_yuv420p_video_tab', 'PASS', raised=repr(err) if err else None, size=size, lines=lines[:1])
    assert err is None, f'raised {err!r}'
    assert size == (512, 384), f'decoded {size}'
    assert lines, 'no warning names both sizes'
    return True


def test_odd_size_kept_without_subsampling():
    """The crop follows the chroma factor: none for RGB and 4:4:4, width only for 4:2:2."""
    if need_av():
        return need_av()
    out = {}
    for label, codec, pix in (('rgb24', 'libx264rgb', 'rgb24'), ('yuv444p', 'libx264', 'yuv444p'), ('yuv422p', 'libx264', 'yuv422p')):
        res, err = call_and_catch(legacy_save, odd_frames(), f'odd_{label}.mp4', mp4_codec=codec, mp4_pix=pix)
        out[label] = decoded_size(res) if res and not err else f'returned {res!r} raised {err!r}'
    verdict('test_odd_size_kept_without_subsampling', 'PASS', sizes={k: str(v) for k, v in out.items()})
    assert out['rgb24'] == (513, 385), f'rgb24 {out["rgb24"]}'
    assert out['yuv444p'] == (513, 385), f'yuv444p {out["yuv444p"]}'
    assert out['yuv422p'] == (512, 385), f'yuv422p {out["yuv422p"]}'
    return True


# ============================================================
# legacy defaults
# ============================================================

def plays_by_shape(info: dict) -> bool:
    return info['codec'] == 'h264' and info['pix_fmt'] in ('yuv420p', 'yuvj420p') and '4:4:4' not in str(info['profile'])


def test_legacy_default_is_h264_yuv420p():
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, pan()[:12], 'default_legacy.mp4', video_type='MP4')
    info = stream_info(res) if res and not err else {}
    verdict('test_legacy_default_is_h264_yuv420p', 'PASS', returned=res, raised=repr(err) if err else None, codec=info.get('codec'), pix_fmt=info.get('pix_fmt'), profile=info.get('profile'))
    assert err is None and res, f'returned {res!r} raised {err!r}'
    assert res.endswith('.mp4'), res
    assert plays_by_shape(info), f'default output is {info.get("codec")} {info.get("pix_fmt")} {info.get("profile")}'
    return True


def test_legacy_default_quality_matches_old_writer():
    if need_av():
        return need_av()
    if 'quality_margin' not in FLOOR:
        return 'quality floor not measured'
    frames = pan()
    res, err = call_and_catch(legacy_save, frames, 'default_legacy_pan.mp4', video_type='MP4', duration=len(frames) / 24.0)
    assert err is None and res, f'returned {res!r} raised {err!r}'
    p = mean_psnr(decode_rgb(res), frames)
    measure('legacy_default_quality', psnr=round(p, 1), psnr_cv2=round(FLOOR['psnr_cv2'], 1), margin_db=round(FLOOR['quality_margin'], 2), kbps=kbps(res, len(frames) / 24.0))
    verdict('test_legacy_default_quality_matches_old_writer', 'PASS', psnr=round(p, 1))
    assert p >= FLOOR['psnr_cv2'] - FLOOR['quality_margin'], f'default {p:.1f} dB below the old writer {FLOOR["psnr_cv2"]:.1f} dB by more than {FLOOR["quality_margin"]:.2f}'
    return True


def test_explicit_mpeg4_gets_table_preset():
    if need_av():
        return need_av()
    if 'quality_margin' not in FLOOR:
        return 'quality floor not measured'
    frames = pan()
    res, err = call_and_catch(legacy_save, frames, 'default_mpeg4_preset.mp4', video_type='MP4', mp4_codec='mpeg4', mp4_opt=None, duration=len(frames) / 24.0)
    assert err is None and res, f'returned {res!r} raised {err!r}'
    p = mean_psnr(decode_rgb(res), frames)
    measure('explicit_mpeg4_quality', psnr=round(p, 1), kbps=kbps(res, len(frames) / 24.0))
    verdict('test_explicit_mpeg4_gets_table_preset', 'PASS', psnr=round(p, 1))
    assert p >= FLOOR['psnr_cv2'] - FLOOR['quality_margin'], f'mpeg4 with mp4_opt=None gave {p:.1f} dB, below the old writer {FLOOR["psnr_cv2"]:.1f} dB'
    return True


LEGACY_LABELS = {'MP4/MP4V': 'mpeg4', 'AVI/DIVX': 'mpeg4', 'MP4/AVC1': 'libx264', 'MP4/JVT3': 'libx264', 'MKV/H264': 'libx264', 'MJPEG/MJPG': 'mjpeg', 'MP4': 'libx264', 'mov/avc1': 'libx264'}  # no two keys differ only in case; JSON readers such as PowerShell reject that


def test_legacy_labels_map_to_codecs():
    codec_for = need(video, 'codec_from_video_type')
    got = {label: codec_for(label) for label in LEGACY_LABELS}
    with capture_log() as logs:
        unknown = codec_for('MPG/MPG1')
    verdict('test_legacy_labels_map_to_codecs', 'PASS', mapped=got, unknown=unknown, warning=logs.find('MPG1', level=logging.WARNING)[:1])
    assert got == LEGACY_LABELS, f'label map {got}'
    assert unknown == 'libx264', f'unknown half mapped to {unknown}'
    assert logs.find('MPG1', level=logging.WARNING), 'unknown label did not warn'
    return True


def test_legacy_labels_encode():
    if need_av():
        return need_av()
    expected = {'MP4/AVC1': ('h264', '.mp4'), 'MKV/H264': ('h264', '.mkv'), 'AVI/DIVX': ('mpeg4', '.avi'), 'MP4/MP4V': ('mpeg4', '.mp4')}
    got = {}
    for label in expected:
        res, err = call_and_catch(legacy_save, small_frames(4), f'label_{label.replace("/", "_")}', video_type=label)
        got[label] = (stream_info(res)['codec'], os.path.splitext(res)[1]) if res and not err else f'returned {res!r} raised {err!r}'
    others = {}
    for label in ('MJPEG/MJPG', 'MPG/MPG1', 'AVR/AVR1', 'AVI/RGBA'):
        res, err = call_and_catch(legacy_save, small_frames(4), f'label_{label.replace("/", "_")}', video_type=label)
        if err is not None:
            others[label] = f'raised {type(err).__name__}'
        elif res and os.path.isfile(res):
            info_, info_err = call_and_catch(stream_info, res)
            others[label] = info_['codec'] if info_err is None else f'unreadable: {type(info_err).__name__}'
        else:
            others[label] = 'no file'
    measure('legacy_labels_other', **{k.replace('/', '_'): str(v) for k, v in others.items()})
    verdict('test_legacy_labels_encode', 'PASS', got={k: str(v) for k, v in got.items()})
    assert got == expected, f'{got}'
    return True


RESOLVED_VIDEO_TYPES = {
    # video_type: (kind, codec, ext, warns)
    'None': ('none', None, None, False),
    'GIF': ('gif', None, 'gif', False),
    'PNG': ('png', None, 'png', False),
    'MP4': ('video', 'libx264', 'mp4', False),
    'mp4': ('video', 'libx264', 'mp4', False),
    'LIBX264': ('video', 'libx264', 'mp4', False),
    'MP4/MP4V': ('video', 'mpeg4', 'mp4', False),
    'MP4/AVC1': ('video', 'libx264', 'mp4', False),
    'MP4/JVT3': ('video', 'libx264', 'mp4', False),
    'MKV/H264': ('video', 'libx264', 'mkv', False),
    'AVI/DIVX': ('video', 'mpeg4', 'avi', False),
    'AVI/RGBA': ('video', 'libx264', 'mp4', True),
    'MJPEG/MJPG': ('video', 'mjpeg', 'avi', True),
    'MPG/MPG1': ('video', 'libx264', 'mp4', True),
    'AVR/AVR1': ('video', 'libx264', 'mp4', True),
}


def test_resolve_video_type_aliases():
    """Every "Video format" value a saved setting, an API caller or a queued task can hold: codec keys in any case, bare containers, the legacy EXT/FOURCC labels."""
    resolve = need(video_codecs, 'resolve_video_type')
    expected = dict(RESOLVED_VIDEO_TYPES)
    expected.update({key: ('video', key, cfg['ext'], False) for key, cfg in video_codecs.codecs_config.items()})
    got = {}
    for value in expected:
        with capture_log() as logs:
            target = resolve(value)
        got[value] = (target.kind, target.codec, target.ext, bool(logs.messages(level=logging.WARNING)))
    wrong = {value: (got[value], expected[value]) for value in expected if got[value] != expected[value]}
    verdict('test_resolve_video_type_aliases', 'PASS', resolved={k: str(v) for k, v in got.items()})
    assert not wrong, f'resolved differently: {wrong}'
    return True


def test_explicit_codec_overrides_label_container():
    """An explicit codec is checked against the container the label names, and an explicit extension is used as given."""
    resolve = need(video_codecs, 'resolve_video_type')
    cases = {
        ('mp4', 'ffv1', None): ('ffv1', 'mkv'),
        ('MP4/AVC1', 'mpeg4', 'avi'): ('mpeg4', 'avi'),
        ('libvpx-vp9', 'libx264', None): ('libx264', 'mp4'),
        ('MKV/H264', 'libvpx-vp9', None): ('libvpx-vp9', 'mkv'),
    }
    got = {}
    for (value, codec, ext), _expected in cases.items():
        with capture_log() as logs:
            target = resolve(value, codec, ext)
        got[(value, codec, ext)] = (target.codec, target.ext)
        assert not logs.find('has no encoder'), f'{value} with an explicit codec warned about its label'
    verdict('test_explicit_codec_overrides_label_container', 'PASS', resolved={str(k): str(v) for k, v in got.items()})
    assert got == cases, f'{got}'
    return True


def test_extension_rule():
    """One extension rule, with and without an explicit extension: the container's extension is appended unless the name already ends with it."""
    if need_av():
        return need_av()
    names = {'ext_rule_a.mp4': 'ext_rule_a.mp4', 'ext_rule_b.MP4': 'ext_rule_b.MP4', 'ext_rule_v1.5': 'ext_rule_v1.5.mp4', 'ext_rule_c': 'ext_rule_c.mp4'}
    got = {}
    for mp4_ext in (None, 'mp4'):
        for name in names:
            res, err = call_and_catch(legacy_save, small_frames(2), name, mp4_codec='libx264', mp4_ext=mp4_ext)
            got[f'{name}|{mp4_ext}'] = os.path.basename(res) if res else repr(err)
    verdict('test_extension_rule', 'PASS', **got)
    wrong = {key: value for key, value in got.items() if value != names[key.split('|')[0]]}
    assert not wrong, f'names differ: {wrong}'
    return True


def test_codec_name_saves_in_its_container():
    """A codec key as the "Video format" writes that codec in its own container."""
    if need_av():
        return need_av()
    cases = {'prores_ks': ('prores', '.mov')}
    if 'libvpx-vp9' in av.codecs_available:
        cases['libvpx-vp9'] = ('vp9', '.webm')
    got = {}
    for value in cases:
        res, err = call_and_catch(legacy_save, small_frames(4), f'codec_name_{value}', video_type=value)
        got[value] = (stream_info(res)['codec'], os.path.splitext(res)[1]) if res and not err else f'returned {res!r} raised {err!r}'
    verdict('test_codec_name_saves_in_its_container', 'PASS', **{k: str(v) for k, v in got.items()})
    assert got == cases, f'{got}'
    return True


def test_odd_size_under_default():
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, odd_frames(), 'odd_default.mp4', video_type='MP4')
    size = decoded_size(res) if res and not err else None
    verdict('test_odd_size_under_default', 'PASS', returned=res, raised=repr(err) if err else None, size=size)
    assert err is None and res, f'returned {res!r} raised {err!r}'
    assert size == (512, 384), f'decoded {size}'
    return True


def test_none_codec_kwargs_give_default():
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, small_frames(4), 'default_none_kwargs.mp4', video_type='MP4', mp4_codec=None, mp4_pix=None, mp4_opt=None)
    info, reason = saved_info(res, err)
    verdict('test_none_codec_kwargs_give_default', 'PASS', returned=res, reason=reason, codec=info.get('codec'), pix_fmt=info.get('pix_fmt'))
    assert reason is None, reason
    assert plays_by_shape(info), f'{info.get("codec")} {info.get("pix_fmt")} {info.get("profile")}'
    return True


# ============================================================
# frame rates
# ============================================================

def test_video_rate_fractions():
    video_rate = need(video_writer, 'video_rate')
    assert video_rate(29.97002997) == Fraction(30000, 1001), video_rate(29.97002997)
    assert video_rate(24) == Fraction(24, 1)
    assert video_rate(Fraction(24000, 1001)) == Fraction(24000, 1001)
    long_rate = video_rate(30000 / 1000.01)
    assert long_rate.denominator <= 65535, f'{long_rate}'
    assert abs(float(long_rate) - 30000 / 1000.01) < 1e-3, f'{long_rate}'
    _res, err = call_and_catch(video_rate, 0)
    assert isinstance(err, ValueError) and 'fps' in str(err), f'fps=0 gave {err!r}'
    verdict('test_video_rate_fractions', 'PASS', long_rate=str(long_rate))
    return True


def test_mpeg4_opens_at_dlss_long_rate():
    """DLSS derives fps from a two-decimal duration; 30000/1000.01 needs a time base mpeg4 can carry."""
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, small_frames(4), 'rate_mpeg4_long.mp4', mp4_codec='mpeg4', duration=1000.01, frames_hint=30000)
    info = stream_info(res) if res and not err else {}
    verdict('test_mpeg4_opens_at_dlss_long_rate', 'PASS', returned=res, raised=repr(err) if err else None, rate=info.get('average_rate'), time_base=info.get('time_base'))
    assert err is None and res, f'returned {res!r} raised {err!r}'
    assert info.get('codec') == 'mpeg4'
    return True


def test_video_tab_keeps_fractional_rate():
    if need_av():
        return need_av()
    fn = out_path('rate_videotab_2997.mp4')
    _res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of(small_frames(8)), fps=29.97, codec='libx264')
    info = stream_info(fn) if err is None and os.path.isfile(fn) else {}
    rate = float(Fraction(info['average_rate'])) if info.get('average_rate') else None
    verdict('test_video_tab_keeps_fractional_rate', 'PASS', raised=repr(err) if err else None, rate=info.get('average_rate'))
    assert err is None, f'raised {err!r}'
    assert rate is not None and abs(rate - 29.97) < 0.005, f'stored rate {rate}'
    return True


def test_fps_zero_refused_by_name():
    if need_av():
        return need_av()
    fn = out_path('rate_zero.mp4')
    _res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of(small_frames(2)), fps=0, codec='libx264')
    verdict('test_fps_zero_refused_by_name', 'PASS', raised=f'{type(err).__name__}: {err}' if err else None, file_exists=os.path.exists(fn))
    assert err is not None, 'fps=0 did not raise'
    assert 'fps' in str(err), f'refusal does not name fps: {err!r}'
    assert not os.path.exists(fn), 'a file was left behind'
    return True


def test_dlss_output_fps_rule():
    module, reason = dlss_module()
    if module is None:
        return reason
    rule = need(module, 'dlss_output_fps')
    assert rule(24.0, 1, 'fps') == 24.0
    assert rule(24.0, 2, 'fps') == 48.0
    assert rule(24.0, 2, 'slowmo') == 24.0
    assert rule(29.97, 4, 'fps') == 29.97 * 4
    verdict('test_dlss_output_fps_rule', 'PASS')
    return True


def test_video_tab_returns_none_video_on_failure():
    if need_av():
        return need_av()
    import torch
    pixels = (tensor_of(small_frames(4)).float() / 127.5 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)  # 1 c t h w in [-1, 1]
    with opts(outdir_video=out_dir, outdir_samples='', save_to_dirs=False, save_txt=False):
        res, err = call_and_catch(video_save.save_video, None, pixels=pixels, mp4_codec='nonexistent', mp4_thumb=False, mp4_fps=24)
    frames_, output_video, thumb = res if res is not None else (None, None, None)
    verdict('test_video_tab_returns_none_video_on_failure', 'PASS', raised=repr(err) if err else None, frames=frames_, output_video=output_video, thumb=thumb)
    assert err is None, f'raised {err!r}'
    assert output_video is None, f'returned a video path for a failed encode: {output_video!r} exists={os.path.exists(output_video or "")}'
    del torch
    return True


# ============================================================
# colour
# ============================================================

def test_default_outputs_are_tagged_bt709():
    if need_av():
        return need_av()
    img, _centres = chart('720')
    res, err = call_and_catch(legacy_save, [img] * 12, 'colour_default_legacy_tags.mp4', video_type='MP4')
    legacy_tags = tags_of(stream_info(res)) if res and not err else {'error': repr(err)}
    fn = out_path('colour_default_videotab_tags.mp4')
    _res2, err2 = call_and_catch(video_save.atomic_save_video, fn, tensor_of([img] * 12), fps=24)
    tab_tags = tags_of(stream_info(fn)) if err2 is None and os.path.isfile(fn) else {'error': repr(err2)}
    verdict('test_default_outputs_are_tagged_bt709', 'PASS', legacy=legacy_tags, video_tab=tab_tags)
    assert legacy_tags == STREAM_TAGS['bt709'], f'legacy default tags {legacy_tags}'
    assert tab_tags == STREAM_TAGS['bt709'], f'video tab default tags {tab_tags}'
    return True


def colour_case(name: str, fn: str, centres: dict):
    info = stream_info(fn)
    ok, reason, detail = colour_verdict(info, decode_yuv(fn)[0], centres)
    measure(name, tags=tags_of(info), **{k: detail.get(k) for k in ('err_tagged', 'err_other') if k in detail})
    verdict(name, 'PASS', tags=tags_of(info), bad=detail.get('bad'), range_bad=detail.get('range_bad'))
    assert ok, reason
    return True


def test_legacy_default_colour_720():
    if need_av():
        return need_av()
    if 'colour' not in FLOOR:
        return 'colour floor not measured'
    img, centres = chart('720')
    res, err = call_and_catch(legacy_save, [img] * 12, 'colour_default_legacy_720.mp4', video_type='MP4')
    assert err is None and res, f'returned {res!r} raised {err!r}'
    return colour_case('test_legacy_default_colour_720', res, centres)


def test_legacy_default_colour_512():
    if need_av():
        return need_av()
    if 'colour' not in FLOOR:
        return 'colour floor not measured'
    img, centres = chart('512')
    res, err = call_and_catch(legacy_save, [img] * 12, 'colour_default_legacy_512.mp4', video_type='MP4')
    assert err is None and res, f'returned {res!r} raised {err!r}'
    return colour_case('test_legacy_default_colour_512', res, centres)


def test_video_tab_default_colour():
    if need_av():
        return need_av()
    if 'colour' not in FLOOR:
        return 'colour floor not measured'
    img, centres = chart('720')
    fn = out_path('colour_default_videotab_720.mp4')
    _res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of([img] * 12), fps=24)
    assert err is None, f'raised {err!r}'
    return colour_case('test_video_tab_default_colour', fn, centres)


def test_rgb_pix_fmt_tagged_full_and_exact():
    if need_av():
        return need_av()
    frames = small_frames(4)
    res, err = call_and_catch(legacy_save, frames, 'colour_x264rgb.mp4', mp4_codec='libx264rgb', mp4_pix='rgb24', mp4_opt=None)
    assert err is None and res, f'returned {res!r} raised {err!r}'
    info = stream_info(res)
    decoded = decode_rgb(res)
    exact = len(decoded) == len(frames) and all(np.array_equal(d, s) for d, s in zip(decoded, frames, strict=True))
    verdict('test_rgb_pix_fmt_tagged_full_and_exact', 'PASS', tags=tags_of(info), pix_fmt=info['pix_fmt'], exact=exact)
    assert tags_of(info) == STREAM_TAGS['rgb'], f'rgb tags {tags_of(info)}'
    assert exact, 'rgb round trip is not bit exact'
    return True


def test_prores_tagged_and_within_oracle():
    if need_av():
        return need_av()
    if 'colour' not in FLOOR:
        return 'colour floor not measured'
    if not PRESET_OPEN.get('prores_ks', {}).get('available', True):
        return 'prores_ks not in this av build'
    img, centres = chart('720')
    res, err = call_and_catch(legacy_save, [img] * 4, 'colour_prores.mov', video_type='mov', mp4_codec='prores_ks', mp4_ext='mov')
    assert err is None and res, f'returned {res!r} raised {err!r}'
    return colour_case('test_prores_tagged_and_within_oracle', res, centres)


def test_mpeg4_in_mp4_carries_colr():
    if need_av():
        return need_av()
    img, _centres = chart('512')
    res, err = call_and_catch(legacy_save, [img] * 4, 'colour_mpeg4_colr.mp4', mp4_codec='mpeg4')
    assert err is None and res, f'returned {res!r} raised {err!r}'
    info = stream_info(res)
    verdict('test_mpeg4_in_mp4_carries_colr', 'PASS', tags=tags_of(info))
    assert tags_of(info) == STREAM_TAGS['bt709'], f'mpeg4 tags {tags_of(info)}'
    return True


# ============================================================
# readers
# ============================================================

READER = {}


def reader_fixture():
    if 'h264' not in READER:
        READER['h264'] = encode_frames(out_path('reader_fixture.mp4'), 'libx264', 'yuv420p', {'crf': '18'}, 24, pan(), convert='bt709', tags=STREAM_TAGS['bt709'])
    return READER['h264']


def test_readers_agree_on_h264_fixture():
    if need_av():
        return need_av()
    fn = reader_fixture()
    frames, fps, duration, w, h, codec, frame = video.get_video_params(fn)
    info = video.get_video_info(fn)
    decoded = video.read_video(fn)
    streamed = sum(1 for _ in video.read_generator(fn))
    probe = video_reader.probe_media(fn, 'video')
    out = {'params': [frames, fps, duration, w, h, codec, frame is None], 'info': {k: info.get(k) for k in ('frames', 'fps', 'width', 'height', 'codec', 'format', 'duration', 'backend')}, 'read_video': [len(decoded), list(decoded[0].shape) if decoded else None], 'read_generator': streamed, 'probe': [probe.fps, probe.frames, probe.width, probe.height, probe.duration] if probe else None}
    measure('readers_h264', **{k: str(v) for k, v in out.items()})
    verdict('test_readers_agree_on_h264_fixture', 'PASS', **{k: str(v) for k, v in out.items()})
    assert (frames, fps, duration, w, h) == (48, 24.0, 2.0, 1024, 576), f'get_video_params {out["params"]}'
    assert info.get('frames') == 48 and info.get('width') == 1024 and info.get('height') == 576, f'get_video_info {out["info"]}'
    assert len(decoded) == 48 and decoded[0].shape == (576, 1024, 3), f'read_video {out["read_video"]}'
    assert streamed == 48, f'read_generator {streamed}'
    assert probe is not None and abs(probe.fps - 24.0) < 1e-6 and probe.width == 1024 and probe.height == 576, f'probe_media {out["probe"]}'
    return True


def test_readers_fractional_rate():
    if need_av():
        return need_av()
    fn = encode_frames(out_path('reader_23976.mp4'), 'libx264', 'yuv420p', {'crf': '18'}, Fraction(24000, 1001), small_frames(24))
    _frames, fps, duration, _w, _h, _codec, _frame = video.get_video_params(fn)
    probe = video_reader.probe_media(fn, 'video')
    cap = cv2.VideoCapture(fn)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    measure('readers_fractional_rate', get_video_params_fps=fps, get_video_params_duration=duration, probe_fps=probe.fps if probe else None, cv2_raw_fps=raw_fps, int_truncation=int(raw_fps))
    verdict('test_readers_fractional_rate', 'PASS', get_video_params_fps=fps, probe_fps=probe.fps if probe else None)
    assert abs(fps - 24000 / 1001) < 1e-9, f'get_video_params fps {fps}'
    assert probe is not None and abs(probe.fps - 24000 / 1001) < 1e-9, f'probe fps {probe.fps if probe else None}'
    return True


def encode_reader_fixture(fn: str, frames: list[np.ndarray], codec: str = 'libx264', pix_fmt: str = 'yuv420p', options: dict | None = None, rate=24,
                          container_format: str | None = None, live: bool = False, rotation: int | None = None, tags: dict | None = None, convert: str | None = None, audio_seconds: float = 0.0) -> str:
    """Test-local encode with what the readers must handle: no cues or duration (live), raw streams, display rotation, a soundtrack longer than the video."""
    open_args = {}
    if container_format:
        open_args['format'] = container_format
    if live:
        open_args['options'] = {'live': '1'}
    container = av.open(fn, mode='w', **open_args)
    try:
        stream = container.add_stream(codec, rate=rate, options=dict(options or {}))
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = pix_fmt
        for key, value in (tags or {}).items():
            setattr(stream.codec_context, key, value)
        if rotation is not None:
            stream.set_display_rotation(rotation)
        audio = None
        if audio_seconds:
            audio = container.add_stream('aac', rate=24000)
            audio.layout = 'mono'
        reformat = {}
        if convert is not None:
            space, crange = colour_constants()[convert]
            reformat = {'dst_colorspace': space, 'dst_color_range': crange}
        for i, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(format=pix_fmt, **reformat)
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None:
            samples = (0.1 * np.sin(np.arange(int(24000 * audio_seconds)) * 2 * np.pi * 440 / 24000)).astype(np.float32)
            video_writer.add_audio_tensor(container, audio, samples, 24000)
    finally:
        container.close()
    return fn


VP9_FAST = {'crf': '30', 'b': '0', 'deadline': 'realtime', 'cpu-used': '8'}


def test_readers_per_container():
    """Frame counts in each container the writers produce: the header count (mp4, mov, avi), the video stream's own duration (ts), the packets (mkv, webm); never the container duration."""
    if need_av():
        return need_av()
    rate = Fraction(24000, 1001)
    frames = pan_frames(48, 320, 192)
    cases = [('mp4', 'libx264', {}, 'header'), ('mov', 'libx264', {}, 'header'), ('avi', 'mpeg4', {}, 'header'), ('ts', 'libx264', {}, 'duration'), ('mkv', 'libx264', {}, None)]
    if 'libvpx-vp9' in av.codecs_available:
        cases.append(('webm', 'libvpx-vp9', VP9_FAST, None))
    out = {}
    for ext, codec, options, source in cases:
        fn = encode_reader_fixture(out_path(f'reader_container.{ext}'), frames, codec=codec, options=options, rate=rate)
        with av.open(fn) as container:
            stream = container.streams.video[0]
            header = {'stream_frames': stream.frames, 'stream_duration': stream.duration}
        cap = cv2.VideoCapture(fn)
        cv2_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else None
        cap.release()
        probe = video_reader.probe_media(fn, 'video')
        out[ext] = {**header, 'frames_source': probe.frames_source, 'probe_frames': probe.frames, 'read': len(video.read_video(fn)), 'params': video.get_video_params(fn)[0], 'cv2_count': cv2_count, 'expected_source': source}
    measure('readers_per_container', **{k: str(v) for k, v in out.items()})
    verdict('test_readers_per_container', 'PASS', **{k: str({x: y for x, y in v.items() if x != 'cv2_count'}) for k, v in out.items()})
    for ext, o in out.items():
        if o['expected_source'] != 'header':
            assert o['stream_frames'] == 0, f'{ext}: fixture carries a frame count {o["stream_frames"]}, the estimate path is not exercised'
        if o['expected_source'] is None:
            assert o['stream_duration'] is None, f'{ext}: fixture carries a stream duration {o["stream_duration"]}'
        assert o['frames_source'] == o['expected_source'], f'{ext}: frames from {o["frames_source"]}, expected {o["expected_source"]}'
        assert o['read'] == 48, f'{ext}: read {o["read"]} of 48 frames'
        assert o['params'] == 48, f'{ext}: get_video_params reports {o["params"]} frames'
    return True


def test_reader_codec_names():
    """Codec and pixel format are the libav names (vp9, mpeg4, yuv420p), not fourcc codes."""
    if need_av():
        return need_av()
    cases = [('mp4', 'libx264', {}, 'h264'), ('avi', 'mpeg4', {}, 'mpeg4')]
    if 'libvpx-vp9' in av.codecs_available:
        cases.append(('webm', 'libvpx-vp9', VP9_FAST, 'vp9'))
    out = {}
    for ext, codec, options, name in cases:
        fn = encode_reader_fixture(out_path(f'reader_codec.{ext}'), small_frames(6), codec=codec, options=options)
        info = video.get_video_info(fn)
        out[ext] = {'params': video.get_video_params(fn)[5], 'info': info.get('codec'), 'format': info.get('format'), 'expected': name}
    verdict('test_reader_codec_names', 'PASS', **{k: str(v) for k, v in out.items()})
    for ext, o in out.items():
        assert o['params'] == o['expected'] and o['info'] == o['expected'], f'{ext}: codec {o}'
        assert o['format'] == 'yuv420p', f'{ext}: format {o["format"]}'
    return True


def test_count_frames_without_header_or_duration():
    """A clip with no frame count and no duration (a live matroska or webm, a raw h264 stream) is counted from its packets; every number get_video_params returns is finite and formats."""
    if need_av():
        return need_av()
    count_frames = need(video_reader, 'count_frames')
    frames = small_frames(12)
    fixtures = {
        'live_mkv': encode_reader_fixture(out_path('live.mkv'), frames, live=True),
        'raw_h264': encode_reader_fixture(out_path('raw.h264'), frames, container_format='h264'),
    }
    if 'libvpx-vp9' in av.codecs_available:
        fixtures['live_webm'] = encode_reader_fixture(out_path('live.webm'), frames, codec='libvpx-vp9', options=VP9_FAST, live=True)
    out = {}
    for name, fn in fixtures.items():
        probe = video_reader.probe_media(fn, 'video')
        params = video.get_video_params(fn)
        out[name] = {'probe_frames': probe.frames, 'probe_duration': probe.duration, 'counted': count_frames(fn), 'params': params[:6]}
    verdict('test_count_frames_without_header_or_duration', 'PASS', **{k: str(v) for k, v in out.items()})
    for name, o in out.items():
        assert o['probe_frames'] is None and o['probe_duration'] is None, f'{name}: fixture has a count or duration in its header: {o}'
        assert o['counted'] == 12, f'{name}: counted {o["counted"]} of 12'
        frames_, fps, duration = o['params'][:3]
        assert frames_ == 12, f'{name}: get_video_params frames {frames_}'
        assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in o['params'][:5]), f'{name}: {o["params"]}'
        assert fps > 0 and 0 < duration < 1, f'{name}: fps {fps} duration {duration}'
        assert f'{fps:.2f} {duration:.2f}' and 1 + frames_ // 2 == 7  # the gallery's format and the Control tab's frame_count arithmetic
    return True


def test_frames_estimate_ignores_trailing_audio():
    """A matroska clip whose soundtrack outlasts its 12 frames: the count comes from the packets, not from the container duration, which includes the audio."""
    if need_av():
        return need_av()
    fn = encode_reader_fixture(out_path('trailing_audio.mkv'), small_frames(12), audio_seconds=2.0)
    with av.open(fn) as container:
        stream = container.streams.video[0]
        fixture = {'stream_frames': stream.frames, 'stream_duration': stream.duration, 'container_seconds': container.duration / 1000000 if container.duration else None}
    probe = video_reader.probe_media(fn, 'video')
    params = video.get_video_params(fn)
    measure('trailing_audio', container_estimate=round((fixture['container_seconds'] or 0) * 24), **{k: str(v) for k, v in fixture.items()})
    verdict('test_frames_estimate_ignores_trailing_audio', 'PASS', probe_frames=probe.frames, params_frames=params[0])
    assert fixture['stream_frames'] == 0 and fixture['stream_duration'] is None, f'fixture carries a count or a stream duration: {fixture}'
    assert (fixture['container_seconds'] or 0) > 1.5, f'fixture soundtrack is not longer than the video: {fixture}'
    assert probe.frames is None, f'probe estimated {probe.frames} frames from the container duration'
    assert params[0] == 12, f'get_video_params reports {params[0]} frames for 12'
    return True


def block_mean(img: np.ndarray, centre: tuple[int, int], radius: int = 4) -> np.ndarray:
    cx, cy = centre
    return img[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].reshape(-1, img.shape[-1]).astype(np.float64).mean(axis=0)


def test_reader_decodes_with_stream_tags():
    """The reader converts each stream with its own matrix and range tags. Oracle: the numpy conversion of the same decoded planes, so encoder loss cancels; floor: the reader's error on the BT.709 clip; the other reading must miss by more than floor + 1 on the patches it moves. RGB streams round trip exactly."""
    if need_av():
        return need_av()
    img, centres = chart('720')
    frames = [img] * 2
    fixtures = {
        'bt709_tagged': (encode_reader_fixture(out_path('tags_bt709.mp4'), frames, convert='bt709', tags=STREAM_TAGS['bt709']), ('bt709', False), ('bt601', False)),
        'bt601_untagged': (encode_reader_fixture(out_path('tags_untagged.mp4'), frames, convert='bt601'), ('bt601', False), ('bt709', False)),
    }
    if 'libvpx-vp9' in av.codecs_available:
        full_tags = {'colorspace': 1, 'color_range': 2, 'color_primaries': 1, 'color_trc': 1}
        fixtures['bt709_full_vp9'] = (encode_reader_fixture(out_path('tags_full.webm'), frames, codec='libvpx-vp9', options={'crf': '10', 'b': '0', 'deadline': 'realtime', 'cpu-used': '8'}, convert='bt709_full', tags=full_tags), ('bt709', True), ('bt709', False))
    out = {}
    for name, (fn, right, wrong) in fixtures.items():
        decoded = video.read_video(fn)[0]
        with av.open(fn) as container:
            ycc = frame_yuv444(next(container.decode(video=0))).transpose(1, 2, 0)
        oracle, other = ycbcr_to_rgb(ycc, *right), ycbcr_to_rgb(ycc, *wrong)
        out[name] = {
            'error': {patch: float(np.max(np.abs(block_mean(decoded, c) - block_mean(oracle, c)))) for patch, c in centres.items()},
            'error_other': {patch: float(np.max(np.abs(block_mean(decoded, c) - block_mean(other, c)))) for patch, c in centres.items()},
            'separation': {patch: float(np.max(np.abs(block_mean(oracle, c) - block_mean(other, c)))) for patch, c in centres.items()},
        }
    floor = max(out['bt709_tagged']['error'].values())
    bar = floor + 1
    rgb_fn = encode_reader_fixture(out_path('tags_rgb.mp4'), frames, codec='libx264rgb', pix_fmt='rgb24', options={'crf': '0', 'preset': 'ultrafast'})
    rgb_exact = np.array_equal(video.read_video(rgb_fn)[0], img)
    measure('reader_colour', floor=round(floor, 2), **{k: str({x: round(y, 2) for x, y in v['error'].items()}) for k, v in out.items()})
    verdict('test_reader_decodes_with_stream_tags', 'PASS', floor=round(floor, 2), rgb_exact=rgb_exact, fixtures=sorted(out))
    assert floor <= 3, f'reader is {floor:.2f} levels off the oracle on the BT.709 clip, beyond rounding'
    for name, o in out.items():
        assert max(o['error'].values()) <= bar, f'{name}: {o["error"]} against floor {floor:.2f} + 1'
        moved = [patch for patch, sep in o['separation'].items() if sep > 3 * bar]
        assert len(moved) >= 2, f'{name}: the other reading moves only {moved}, the fixture cannot tell them apart'
        assert all(o['error_other'][patch] > bar for patch in moved), f'{name}: the other reading also fits on {moved}: {o["error_other"]}'
    assert rgb_exact, 'libx264rgb clip does not round trip exactly'
    return True


def rotation_fixture() -> np.ndarray:
    """192x320 grey with three different corners, so every rotation and flip is distinguishable."""
    img = np.full((192, 320, 3), 128, dtype=np.uint8)
    img[:48, :64] = (220, 40, 40)
    img[:48, -64:] = (40, 200, 60)
    img[-48:, :64] = (40, 60, 220)
    return img


def test_reader_rotation_follows_display_matrix():
    """A clip with a display rotation reads upright, the way players show it, and every reported size is the displayed one."""
    if need_av():
        return need_av()
    if not hasattr(av.video.stream.VideoStream, 'set_display_rotation'):
        return f'av {av.__version__} cannot write a display rotation for the fixture'
    src = rotation_fixture()
    out = {}
    for angle in (90, 180, 270):
        for ext in ('mp4', 'mkv'):
            fn = encode_reader_fixture(out_path(f'rotated_{angle}.{ext}'), [src] * 3, codec='libx264rgb', pix_fmt='rgb24', options={'crf': '0', 'preset': 'ultrafast'}, rotation=angle)
            expected = np.rot90(src, k=angle // 90)
            frames = video.read_video(fn)
            params = video.get_video_params(fn, capture=True)
            headers = video.get_video_params(fn)
            info = video.get_video_info(fn)
            display = (expected.shape[1], expected.shape[0])
            out[f'{ext}_{angle}'] = {
                'frame': bool(frames) and np.array_equal(frames[0], expected),
                'contiguous': bool(frames) and frames[0].flags['C_CONTIGUOUS'],
                'params_size': (params[3], params[4]) == display,
                'params_size_without_capture': (headers[3], headers[4]) == display,
                'info_size': (info.get('width'), info.get('height')) == display,
                'capture': params[6] is not None and np.array_equal(np.asarray(params[6]), expected),
            }
    verdict('test_reader_rotation_follows_display_matrix', 'PASS', **{k: str(v) for k, v in out.items()})
    failed = {k: [check for check, ok in v.items() if not ok] for k, v in out.items() if not all(v.values())}
    assert not failed, f'rotation checks failed: {failed}'
    return True


def test_params_without_capture_read_headers_only():
    """get_video_params without capture decodes nothing where av reports the display rotation from the header (the gallery's case of a thumbnail already on disk), and its sizes still follow the rotation."""
    if need_av():
        return need_av()
    if not hasattr(av.video.stream.VideoStream, 'set_display_rotation'):
        return f'av {av.__version__} cannot write a display rotation for the fixture'
    need(video_reader, 'probe_video')
    fn = encode_reader_fixture(out_path('headers_only.mp4'), [rotation_fixture()] * 3, rotation=90)
    with av.open(fn) as container:
        header_side_data = hasattr(container.streams.video[0].codec_context, 'coded_side_data')
    if not header_side_data:
        return f'av {av.__version__} reports the display rotation on decoded frames only'

    def decode_refused(*args, **kwargs):
        raise AssertionError('a frame was decoded')

    with patched(video_reader, 'probe_video', decode_refused):
        params, err = call_and_catch(video.get_video_params, fn)
    verdict('test_params_without_capture_read_headers_only', 'PASS', raised=repr(err) if err else None, size=(params[3], params[4]) if params else None)
    assert err is None, f'get_video_params raised {err!r}'
    assert (params[3], params[4]) == (192, 320), f'size {params[3]}x{params[4]}'
    assert params[0] == 3 and params[6] is None, f'{params[:6]}'
    return True


def test_iter_frames_skip_and_limit():
    """skip_frames keeps frame 0 and every (skip + 1)th after it; num_frames stops after that many kept frames. Each frame carries its index as its level."""
    if need_av():
        return need_av()
    iter_frames = need(video_reader, 'iter_frames')
    frames = [np.full((64, 64, 3), 16 + 18 * i, dtype=np.uint8) for i in range(12)]
    fn = encode_reader_fixture(out_path('indexed.mp4'), frames, codec='libx264rgb', pix_fmt='rgb24', options={'crf': '0', 'preset': 'ultrafast'})

    def indices(**kwargs):
        return [int(round((int(frame[32, 32, 0]) - 16) / 18)) for frame in iter_frames(fn, **kwargs)]

    out = {'all': indices(), 'skip1': indices(skip_frames=1), 'skip1_n3': indices(num_frames=3, skip_frames=1), 'n4': indices(num_frames=4)}
    verdict('test_iter_frames_skip_and_limit', 'PASS', **{k: str(v) for k, v in out.items()})
    assert out['all'] == list(range(12)), out['all']
    assert out['skip1'] == [0, 2, 4, 6, 8, 10], out['skip1']
    assert out['skip1_n3'] == [0, 2, 4], out['skip1_n3']
    assert out['n4'] == [0, 1, 2, 3], out['n4']
    return True


def test_ltx_condition_frames_keep_frame_zero():
    """LTX condition video with skip 1: frames 0, 2, 4 reach the condition, the same rule as the Control tab input."""
    if need_av():
        return need_av()
    from modules.ltx import ltx_util
    frames = [np.full((64, 64, 3), 16 + 18 * i, dtype=np.uint8) for i in range(12)]
    fn = encode_reader_fixture(out_path('ltx_indexed.mp4'), frames, codec='libx264rgb', pix_fmt='rgb24', options={'crf': '0', 'preset': 'ultrafast'})

    class Recorder:
        def __init__(self, video=None, frame_index=0, strength=1.0, **kwargs):
            self.video = video

    with patched(ltx_util, '_condition_cls', lambda family: Recorder):
        conditions = ltx_util.get_conditions(64, 64, 1.0, None, None, fn, 3, 1, family='0.9')
    indices = [int(round((np.asarray(f)[32, 32, 0] - 16) / 18)) for f in conditions[0].video] if conditions else []
    verdict('test_ltx_condition_frames_keep_frame_zero', 'PASS', indices=indices)
    assert indices == [0, 2, 4], f'condition frames {indices}'
    return True


def test_first_frame_equals_read_video_first():
    """get_video_params(capture=True) returns the same first frame read_video does."""
    if need_av():
        return need_av()
    fn = reader_fixture()
    params = video.get_video_params(fn, capture=True)
    first = video.read_video(fn)[0]
    verdict('test_first_frame_equals_read_video_first', 'PASS', captured=params[6] is not None)
    assert params[6] is not None and np.array_equal(np.asarray(params[6]), first), 'captured frame differs from the first decoded frame'
    return True


GET_VIDEO_INFO_KEYS = {'file', 'size', 'container', 'frames', 'fps', 'width', 'height', 'duration', 'codec', 'format', 'bitrate_kbps', 'audio_streams', 'audio_channels', 'audio_sample_rate', 'audio'}


def test_get_video_info_exact_keys():
    """The info panel's keys for an h264 clip with an aac soundtrack, with their values."""
    if need_av():
        return need_av()
    fn = encode_reader_fixture(out_path('info_keys.mp4'), small_frames(24), audio_seconds=1.0)
    info = video.get_video_info(fn)
    verdict('test_get_video_info_exact_keys', 'PASS', keys=sorted(info))
    assert set(info) == GET_VIDEO_INFO_KEYS, f'missing {sorted(GET_VIDEO_INFO_KEYS - set(info))} extra {sorted(set(info) - GET_VIDEO_INFO_KEYS)}'
    assert (info['frames'], info['width'], info['height'], info['codec'], info['format']) == (24, 256, 144, 'h264', 'yuv420p'), info
    assert abs(info['fps'] - 24) < 1e-9 and info['duration'] == 1.0, info
    assert (info['audio_streams'], info['audio_channels'], info['audio_sample_rate'], info['audio']) == (1, 1, 24000, '1x1x24000'), info
    return True


def test_unreadable_file_contracts():
    """An empty file and an mp4 cut before its index: probe None, get_video_params logs and raises naming the path, get_video_info reports the error, the frame readers raise."""
    if need_av():
        return need_av()
    read_frames = need(video_reader, 'read_frames')
    empty = out_path('unreadable_empty.mp4')
    with open(empty, 'wb'):
        pass
    whole = reader_fixture()
    cut = out_path('unreadable_cut.mp4')
    with open(whole, 'rb') as src, open(cut, 'wb') as dst:
        dst.write(src.read()[:os.path.getsize(whole) * 2 // 5])
    out = {}
    for name, fn in (('empty', empty), ('cut', cut)):
        with capture_log() as logs:
            _res, params_err = call_and_catch(video.get_video_params, fn)
        _res, read_err = call_and_catch(video.read_video, fn)
        _res, frames_err = call_and_catch(read_frames, fn)
        out[name] = {'probe': video_reader.probe_media(fn, 'video'), 'params': repr(params_err), 'logged': bool(logs.find(os.path.basename(fn), level=logging.ERROR)), 'info_error': 'error' in video.get_video_info(fn), 'read_video': type(read_err).__name__, 'read_frames': type(frames_err).__name__}
    verdict('test_unreadable_file_contracts', 'PASS', **{k: str(v) for k, v in out.items()})
    for name, o in out.items():
        assert o['probe'] is None, f'{name}: probe {o["probe"]}'
        assert 'RuntimeError' in o['params'] and 'unreadable' in o['params'], f'{name}: get_video_params {o["params"]}'
        assert o['logged'], f'{name}: no error line names the file'
        assert o['info_error'], f'{name}: get_video_info has no error'
        assert o['read_video'] == 'RuntimeError' and o['read_frames'] == 'RuntimeError', f'{name}: readers {o}'
    return True


def test_corrupt_stream_stops_with_warning():
    """A stream damaged part way: the reader returns the frames before the damage and logs a warning instead of raising."""
    if need_av():
        return need_av()
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (128, 160, 3), dtype=np.uint8) for _ in range(48)]
    clean = encode_reader_fixture(out_path('corrupt_src.avi'), frames, codec='mpeg4')
    with open(clean, 'rb') as f:
        data = f.read()
    damaged, decodable = None, None
    for seed in range(1, 9):  # the first corruption that makes libav itself raise
        r = np.random.default_rng(seed)
        bad = bytearray(data)
        for k in r.integers(len(bad) * 2 // 5, len(bad) * 7 // 10, 400):
            bad[k] = int(r.integers(0, 255))
        fn = out_path(f'corrupt_{seed}.avi')
        with open(fn, 'wb') as f:
            f.write(bad)
        count = 0
        try:
            with av.open(fn) as container:
                for _ in container.decode(video=0):
                    count += 1
        except av.error.FFmpegError:
            damaged, decodable = fn, count
            break
    if damaged is None:
        return 'no corruption of the fixture made libav raise'
    with capture_log() as logs:
        read, err = call_and_catch(video.read_video, damaged)
    warned = logs.find(os.path.basename(damaged), level=logging.WARNING)
    verdict('test_corrupt_stream_stops_with_warning', 'PASS', decodable=decodable, read=len(read) if read is not None else None, raised=repr(err) if err else None, warned=bool(warned))
    assert err is None, f'reader raised {err!r}'
    assert len(read) == decodable, f'reader returned {len(read)} frames, {decodable} decode before the damage'
    assert warned, 'no warning names the damaged file'
    return True


def test_frames_are_contiguous():
    """Frames come out C-contiguous at widths whose decoded rows are padded, so torch.from_numpy takes them."""
    if need_av():
        return need_av()
    import torch
    fn = encode_reader_fixture(out_path('width98.mp4'), flat_frames(3, 98, 64, step=9))
    frames = video.read_video(fn)
    _res, err = call_and_catch(torch.from_numpy, frames[0])
    verdict('test_frames_are_contiguous', 'PASS', contiguous=[f.flags['C_CONTIGUOUS'] for f in frames], from_numpy=repr(err) if err else None)
    assert all(f.flags['C_CONTIGUOUS'] for f in frames), 'a frame is not contiguous'
    assert err is None, f'torch.from_numpy refused the frame: {err!r}'
    return True


def test_unchanged_callers():
    """Callers of get_video_params keep working on a rotated clip: the Civitai and Video tab thumbnails are upright and the Control input line names the displayed size."""
    if need_av():
        return need_av()
    if not hasattr(av.video.stream.VideoStream, 'set_display_rotation'):
        return f'av {av.__version__} cannot write a display rotation for the fixture'
    from modules.civitai import video_helper
    from modules import ui_control_helpers
    fn = encode_reader_fixture(out_path('callers_rotated.mp4'), [rotation_fixture()] * 3, codec='libx264rgb', pix_fmt='rgb24', options={'crf': '0', 'preset': 'ultrafast'}, rotation=90)
    civitai_frame = video_helper.save_video_frame(fn)
    civitai_thumb = os.path.splitext(fn)[0] + '.thumb.jpg'
    civitai_size = Image.open(civitai_thumb).size if os.path.isfile(civitai_thumb) else None
    os.remove(civitai_thumb)
    tab_thumb = video_save.save_thumbnail(fn)
    tab_size = Image.open(tab_thumb).size if tab_thumb and os.path.isfile(tab_thumb) else None
    message = ui_control_helpers.get_video(fn)
    verdict('test_unchanged_callers', 'PASS', civitai=civitai_size, video_tab=tab_size, control=message)
    assert civitai_frame is not None and civitai_size == (192, 320), f'Civitai thumbnail {civitai_size}'
    assert tab_size == (192, 320), f'Video tab thumbnail {tab_size}'
    assert 'Size 192x320' in message, f'Control input line: {message}'
    return True


def test_reader_decode_speed_measured():
    """Measurement only: decode time per 1080p frame, reader against cv2, five warm repeats after a discarded first run."""
    if need_av():
        return need_av()
    fn = encode_reader_fixture(out_path('speed_1080.mp4'), flat_frames(24, 1920, 1080, step=5), options={'preset': 'ultrafast'})

    def reader_run():
        return len(video.read_video(fn))

    def cv2_run():
        cap = cv2.VideoCapture(fn)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1
        cap.release()
        return count

    result = {}
    for name, run in (('reader', reader_run), ('cv2', cv2_run)):
        times = []
        for _ in range(6):
            t0 = time.perf_counter()
            count = run()
            times.append((time.perf_counter() - t0) * 1000 / max(1, count))
        warm = sorted(times[1:])
        result[name] = {'min': round(warm[0], 2), 'median': round(warm[len(warm) // 2], 2), 'max': round(warm[-1], 2), 'frames': count}
    measure('reader_decode_speed_1080', **{k: str(v) for k, v in result.items()})
    verdict('test_reader_decode_speed_measured', 'PASS')
    return True


# ============================================================
# audio
# ============================================================

def test_video_tab_muxes_audio_tensor():
    if need_av():
        return need_av()
    import torch
    fn = out_path('audio_tensor.mp4')
    audio = torch.from_numpy(sine(1.0, 24000, channels=2))
    _res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of(small_frames(24)), audio=audio, fps=24, codec='libx264', sample_rate=24000)
    info = stream_info(fn) if err is None and os.path.isfile(fn) else {}
    tolerance = 2 * 1024 / 24000
    verdict('test_video_tab_muxes_audio_tensor', 'PASS', raised=repr(err) if err else None, audio=info.get('audio'), duration=info.get('duration'))
    assert err is None, f'raised {err!r}'
    assert info.get('audio') == ['aac'], f'audio streams {info.get("audio")}'
    assert info.get('duration') is not None and abs(info['duration'] - 1.0) <= tolerance, f'duration {info.get("duration")} not within two aac frames of 1.0 s'
    del torch
    return True


def test_video_tab_muxes_audio_packets():
    if need_av():
        return need_av()
    wav = make_wav('audio_packets.wav', seconds=1.0, rate=32000)
    with av.open(wav) as container:
        frames = list(container.decode(audio=0))
    fn = out_path('audio_packets.mp4')
    _res, err = call_and_catch(video_save.atomic_save_video, fn, tensor_of(small_frames(24)), audio={'sr': 32000, 'layout': 'mono', 'frames': frames}, fps=24, codec='libx264')
    info = stream_info(fn) if err is None and os.path.isfile(fn) else {}
    tolerance = 2 * 1024 / 32000
    verdict('test_video_tab_muxes_audio_packets', 'PASS', raised=repr(err) if err else None, audio=info.get('audio'), duration=info.get('duration'), packets=len(frames))
    assert err is None, f'raised {err!r}'
    assert info.get('audio') == ['aac'], f'audio streams {info.get("audio")}'
    assert info.get('duration') is not None and abs(info['duration'] - 1.0) <= tolerance, f'duration {info.get("duration")}'
    return True


# ============================================================
# metadata
# ============================================================

def test_metadata_roundtrip():
    if need_av():
        return need_av()
    with opts(image_metadata=True):
        res, err = call_and_catch(legacy_save, small_frames(4), 'metadata_mp4.mp4', mp4_codec='libx264', metadata={'title': 'suite title', 'comment': 'suite comment'})
        res_mkv, err_mkv = call_and_catch(legacy_save, small_frames(4), 'metadata_mkv.mkv', video_type='mkv', mp4_codec='libx264', mp4_ext='mkv', metadata={'title': 'suite title', 'comment': 'suite comment'})
    assert err is None and res, f'returned {res!r} raised {err!r}'
    meta = video.get_video_metadata(res)
    meta_mkv = video.get_video_metadata(res_mkv) if res_mkv and not err_mkv else {'error': repr(err_mkv)}
    measure('metadata_mkv', **{k: str(v) for k, v in meta_mkv.items()})
    verdict('test_metadata_roundtrip', 'PASS', mp4={k: meta.get(k) for k in ('title', 'comment', 'encoder')}, mkv_keys=sorted(meta_mkv))
    assert meta.get('title') == 'suite title' and meta.get('comment') == 'suite comment', f'mp4 metadata {meta}'
    return True


# ============================================================
# gradio preview hook
# ============================================================

def processing_utils():
    import gradio.processing_utils as pu
    gr_tempdir.patch_video_conversion()
    return pu


def simulate_format_video(path: str) -> str:
    """What gradio's Video component does with an output path (components/video.py, _format_video)."""
    pu = processing_utils()
    if pu.ffmpeg_installed() and not pu.video_is_playable(path):
        return pu.convert_video_to_playable_mp4(path)
    return path


@contextlib.contextmanager
def binaries_absent():
    """No ffmpeg or ffprobe on PATH: which() finds nothing and ffmpy raises as it does for a missing executable."""
    from ffmpy import FFExecutableNotFoundError
    pu = processing_utils()

    class MissingExecutable:
        def __init__(self, *args, **kwargs):
            raise FFExecutableNotFoundError('executable not found on PATH')

    with patched(shutil, 'which', lambda *args, **kwargs: None), patched(pu, 'FFmpeg', MissingExecutable), patched(pu, 'FFprobe', MissingExecutable):
        yield


def preview_sources() -> dict[str, str]:
    if 'sources' not in READER:
        src = {}
        src['mkv_ffv1'] = encode_frames(out_path('preview_src_ffv1.mkv'), 'ffv1', 'yuv420p', {}, 24, small_frames(6), threads=None)
        src['mov_prores'] = encode_frames(out_path('preview_src_prores.mov'), 'prores_ks', 'yuv422p10le', {'profile': '3'}, 24, small_frames(6), threads=None) if 'prores_ks' in av.codecs_available else None
        src['mp4_h264'] = encode_frames(out_path('preview_src_h264.mp4'), 'libx264', 'yuv420p', {}, 24, small_frames(6), convert='bt709', tags=STREAM_TAGS['bt709'])
        src['mp4_x264rgb'] = encode_frames(out_path('preview_src_x264rgb.mp4'), 'libx264rgb', 'rgb24', {}, 24, small_frames(6))
        src['mp4_h264_444'] = encode_frames(out_path('preview_src_h264_444.mp4'), 'libx264', 'yuv444p', {}, 24, small_frames(6))
        src['webm_vp9'] = encode_frames(out_path('preview_src_vp9.webm'), 'libvpx-vp9', 'yuv420p', {'crf': '30', 'b': '0'}, 24, small_frames(6), threads=None) if 'libvpx-vp9' in av.codecs_available else None
        src['mp4_av1'] = encode_frames(out_path('preview_src_av1.mp4'), 'libsvtav1', 'yuv420p', {'preset': '10', 'crf': '35'}, 24, small_frames(6), threads=None) if 'libsvtav1' in av.codecs_available else None
        READER['sources'] = src
    return READER['sources']


def temp_files() -> set[str]:
    return set(os.listdir(temp_dir)) if os.path.isdir(temp_dir) else set()


def test_option_off_does_not_probe_or_convert():
    pu = processing_utils()
    with opts(ui_video_convert=False):
        installed = pu.ffmpeg_installed()
    verdict('test_option_off_does_not_probe_or_convert', 'PASS', ffmpeg_installed=installed)
    assert installed is False, 'with the option off gradio still reaches the probe and the converter'
    return True


def test_option_on_converts_without_binaries():
    if need_av():
        return need_av()
    src = preview_sources()
    out = {}
    timings = {}
    with opts(ui_video_convert=True, temp_dir=temp_dir), binaries_absent():
        for label in ('mkv_ffv1', 'mov_prores'):
            fn = src.get(label)
            if fn is None:
                out[label] = 'unavailable'
                continue
            before = (file_hash(fn), os.path.getmtime(fn))
            with capture_log() as logs:
                res, err = call_and_catch(simulate_format_video, fn)
            info = stream_info(res) if res and os.path.isfile(res) and err is None else {}
            timings[label] = same_timing(fn, res) if info else (False, 'no preview')
            out[label] = {'converted': res != fn, 'same_timing': timings[label][0], 'in_temp': bool(res) and os.path.realpath(os.path.dirname(res)) == os.path.realpath(temp_dir), 'ext': os.path.splitext(res or '')[1], 'codec': info.get('codec'), 'pix_fmt': info.get('pix_fmt'), 'tags': tags_of(info) if info else None, 'source_unchanged': (file_hash(fn), os.path.getmtime(fn)) == before, 'raised': repr(err) if err else None, 'log': logs.find('preview', level=logging.INFO)[:1]}
    verdict('test_option_on_converts_without_binaries', 'PASS', **{k: str(v) for k, v in out.items()})
    for label, v in out.items():
        if v == 'unavailable':
            continue
        assert v['raised'] is None, f'{label}: {v["raised"]}'
        assert v['converted'] and v['in_temp'] and v['ext'] == '.mp4', f'{label}: not converted into the temp dir: {v}'
        assert v['codec'] == 'h264' and v['pix_fmt'] == 'yuv420p' and v['tags'] == STREAM_TAGS['bt709'], f'{label}: preview is {v["codec"]} {v["pix_fmt"]} {v["tags"]}'
        assert v['source_unchanged'], f'{label}: the source was modified'
        assert v['same_timing'], f'{label}: preview timing differs from the source: {timings[label][1]}'
    return True


def test_option_on_serves_playable_as_is():
    if need_av():
        return need_av()
    src = preview_sources()
    out = {}
    with opts(ui_video_convert=True, temp_dir=temp_dir), binaries_absent():
        for label in ('mp4_h264', 'webm_vp9', 'mp4_av1'):
            fn = src.get(label)
            if fn is None:
                out[label] = 'unavailable'
                continue
            res, err = call_and_catch(simulate_format_video, fn)
            out[label] = {'served_as_is': res == fn, 'raised': repr(err) if err else None}
    verdict('test_option_on_serves_playable_as_is', 'PASS', **{k: str(v) for k, v in out.items()})
    for label, v in out.items():
        if v != 'unavailable':
            assert v['raised'] is None and v['served_as_is'], f'{label}: {v}'
    return True


def test_option_on_converts_unplayable_h264():
    if need_av():
        return need_av()
    src = preview_sources()
    out = {}
    timings = {}
    with opts(ui_video_convert=True, temp_dir=temp_dir), binaries_absent():
        for label in ('mp4_x264rgb', 'mp4_h264_444'):
            fn = src[label]
            res, err = call_and_catch(simulate_format_video, fn)
            info = stream_info(res) if res and os.path.isfile(res) and err is None else {}
            timings[label] = same_timing(fn, res) if info else (False, 'no preview')
            out[label] = {'converted': res != fn, 'same_timing': timings[label][0], 'pix_fmt': info.get('pix_fmt'), 'raised': repr(err) if err else None}
    verdict('test_option_on_converts_unplayable_h264', 'PASS', **{k: str(v) for k, v in out.items()})
    for label, v in out.items():
        assert v['raised'] is None and v['converted'] and v['pix_fmt'] == 'yuv420p', f'{label}: {v}'
        assert v['same_timing'], f'{label}: preview timing differs from the source: {timings[label][1]}'
    return True


def test_preview_audio_passthrough_and_drop():
    if need_av():
        return need_av()
    import torch
    fn_aac = out_path('preview_src_ffv1_aac.mkv')
    _res, err = call_and_catch(video_save.atomic_save_video, fn_aac, tensor_of(small_frames(24)), audio=torch.from_numpy(sine(1.0, 24000)), fps=24, codec='ffv1', pix_fmt='yuv420p', sample_rate=24000)
    assert err is None, f'fixture with aac raised {err!r}'
    fn_pcm = out_path('preview_src_ffv1_pcm.mkv')
    container = av.open(fn_pcm, mode='w')
    try:
        vs = container.add_stream('ffv1', rate=24)
        vs.width, vs.height, vs.pix_fmt = 256, 144, 'yuv420p'
        audio_stream = container.add_stream('pcm_s16le', rate=24000)
        audio_stream.layout = 'mono'
        for i, rgb in enumerate(small_frames(24)):
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(format='yuv420p')
            frame.pts = i
            for packet in vs.encode(frame):
                container.mux_one(packet)
        pcm = (sine(1.0, 24000) * 32767).astype(np.int16)
        aframe = av.AudioFrame.from_ndarray(pcm, format='s16', layout='mono')
        aframe.sample_rate = 24000
        aframe.pts = 0
        for packet in audio_stream.encode(aframe):
            container.mux_one(packet)
        for packet in audio_stream.encode():
            container.mux_one(packet)
        for packet in vs.encode():
            container.mux_one(packet)
    finally:
        container.close()
    out = {}
    with opts(ui_video_convert=True, temp_dir=temp_dir), binaries_absent():
        for label, fn in (('aac', fn_aac), ('pcm', fn_pcm)):
            with capture_log() as logs:
                res, err = call_and_catch(simulate_format_video, fn)
            info = stream_info(res) if res and os.path.isfile(res) and err is None else {}
            out[label] = {'converted': res != fn, 'audio': info.get('audio'), 'raised': repr(err) if err else None, 'log': logs.find('audio')[:1]}
    verdict('test_preview_audio_passthrough_and_drop', 'PASS', **{k: str(v) for k, v in out.items()})
    assert out['aac']['raised'] is None and out['aac']['converted'] and out['aac']['audio'] == ['aac'], f'aac source: {out["aac"]}'
    assert out['pcm']['raised'] is None and out['pcm']['converted'] and out['pcm']['audio'] == [], f'pcm source: {out["pcm"]}'
    assert out['pcm']['log'], 'dropping the pcm track was not logged'
    del torch
    return True


def test_preview_unopenable_file_served_as_is():
    fn = out_path('preview_zero.mkv')
    with open(fn, 'wb'):
        pass
    with opts(ui_video_convert=True, temp_dir=temp_dir), binaries_absent(), capture_log() as logs:
        res, err = call_and_catch(simulate_format_video, fn)
    verdict('test_preview_unopenable_file_served_as_is', 'PASS', returned=res, raised=repr(err) if err else None, log=logs.find('preview_zero')[:1])
    assert err is None and res == fn, f'returned {res!r} raised {err!r}'
    return True


def test_preview_av_import_failure_does_not_raise():
    src = preview_sources() if av is not None else {'mkv_ffv1': out_path('preview_zero.mkv')}
    fn = src['mkv_ffv1']
    with opts(ui_video_convert=True, temp_dir=temp_dir), av_missing(), binaries_absent(), capture_log() as logs:
        res, err = call_and_catch(simulate_format_video, fn)
    warnings_ = logs.messages(logging.WARNING)
    verdict('test_preview_av_import_failure_does_not_raise', 'PASS', returned=res == fn, raised=repr(err) if err else None, warnings=warnings_[:2])
    assert err is None, f'raised {err!r} with av and ffprobe both missing'
    assert res == fn, f'returned {res!r}'
    return True


def test_cli_fallback_leaves_no_copy():
    if need_av():
        return need_av()
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        return 'ffmpeg or ffprobe not on PATH'
    fn = preview_sources()['mkv_ffv1']
    before = temp_files()
    with opts(ui_video_convert=True, temp_dir=temp_dir), av_missing():
        res, err = call_and_catch(simulate_format_video, fn)
    new_files = sorted(temp_files() - before)
    verdict('test_cli_fallback_leaves_no_copy', 'PASS', returned=res != fn, raised=repr(err) if err else None, new_files=new_files)
    assert err is None, f'raised {err!r}'
    assert res != fn and res.endswith('.mp4'), f'CLI fallback did not convert: {res!r}'
    assert len(new_files) == 1, f'the CLI fallback left extra files in the temp dir: {new_files}'
    return True


def test_cleanup_tmpdr_removes_all_video_extensions():
    folder = os.path.join(work_dir, 'cleanup')
    os.makedirs(folder, exist_ok=True)
    exts = sorted({e for cfg in video_codecs.codecs_config.values() for e in cfg['allowed_exts']} | {'png', 'mp4', 'webm'})
    for ext in exts:
        with open(os.path.join(folder, f'leftover.{ext}'), 'wb') as f:
            f.write(b'0')
    with opts(temp_dir=folder, clean_temp_dir_at_start=True):
        gr_tempdir.cleanup_tmpdr()
    remaining = sorted(os.listdir(folder))
    verdict('test_cleanup_tmpdr_removes_all_video_extensions', 'PASS', remaining=remaining)
    assert not remaining, f'cleanup left {remaining}'
    return True


# ============================================================
# paths
# ============================================================

def test_non_ascii_output_path():
    if need_av():
        return need_av()
    res, err = call_and_catch(legacy_save, small_frames(4), 'ünïcode_漢字.mp4', mp4_codec='libx264')
    cap = cv2.VideoCapture(res) if res else None
    cv2_ok = bool(cap and cap.isOpened())
    if cap:
        cap.release()
    av_ok = bool(res) and stream_info(res)['codec'] == 'h264'
    verdict('test_non_ascii_output_path', 'PASS', returned=res, raised=repr(err) if err else None, cv2_opened=cv2_ok, av_opened=av_ok)
    assert err is None and res and os.path.isfile(res), f'returned {res!r} raised {err!r}'
    assert av_ok, 'av could not open the non-ASCII path'
    assert cv2_ok, 'cv2 could not open the non-ASCII path'
    return True


def test_long_path_truncation():
    if need_av():
        return need_av()
    name = ('long_' + 'x' * 300) + '.mp4'
    res, err = call_and_catch(legacy_save, small_frames(2), name, mp4_codec='libx264')
    measure('long_path', requested=len(os.path.join(out_dir, name)), written=len(res) if res else None)
    verdict('test_long_path_truncation', 'PASS', returned_length=len(res) if res else None, raised=repr(err) if err else None)
    assert err is None and res and os.path.isfile(res), f'returned {res!r} raised {err!r}'
    assert res.endswith('.mp4'), res
    return True


def test_output_folder_kept_as_configured():
    if need_av():
        return need_av()
    folder = os.path.join(work_dir, 'configured_')
    pixels = (tensor_of(small_frames(4)).float() / 127.5 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)  # 1 c t h w in [-1, 1]
    with opts(outdir_video=folder, outdir_samples='', save_to_dirs=False, save_txt=False):
        legacy, legacy_err = call_and_catch(video.save_video, None, small_frames(4), filename='folder_legacy.mp4', video_type='mp4', duration=4 / 24.0, sync=True, mp4_codec='libx264')
        tab, tab_err = call_and_catch(video_save.save_video, None, pixels=pixels, mp4_codec='libx264', mp4_thumb=False, mp4_fps=24)
    tab_video = tab[1] if tab is not None else None
    verdict('test_output_folder_kept_as_configured', 'PASS', legacy=legacy, legacy_raised=repr(legacy_err) if legacy_err else None, video_tab=tab_video, video_tab_raised=repr(tab_err) if tab_err else None)
    for res, err in ((legacy, legacy_err), (tab_video, tab_err)):
        assert err is None and res and os.path.isfile(res), f'returned {res!r} raised {err!r}'
        assert os.path.dirname(res) == folder, f'saved to "{res}", configured "{folder}"'
    return True


def test_frame_export_to_non_ascii_folder():
    """The Video tab frames export writes every frame into a folder named with non-ASCII characters, in RGB order."""
    frames = small_frames(4)
    pixels = (tensor_of(frames).float() / 127.5 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)  # 1 c t h w in [-1, 1]
    folder = os.path.join(work_dir, 'кадры_帧')
    with opts(outdir_video=folder, outdir_samples='', save_to_dirs=False, save_txt=False, samples_filename_pattern='frames'):
        _res, err = call_and_catch(video_save.save_video, None, pixels=pixels, mp4_video=False, mp4_frames=True, mp4_thumb=False, mp4_fps=24)
    written = sorted(f for f in os.listdir(folder) if f.endswith('.jpg')) if os.path.isdir(folder) else []
    control = []
    for frame in frames:  # the same frame through an in-memory jpeg at the export's quality
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format='JPEG', quality=95)
        control.append(float(np.abs(np.asarray(Image.open(buffer).convert('RGB'), dtype=np.int16) - frame).mean()))
    bar = 3 * max(control) + 1
    errors_ = [float(np.abs(np.asarray(Image.open(os.path.join(folder, f)).convert('RGB'), dtype=np.int16) - frame).mean()) for f, frame in zip(written, frames)]
    measure('frame_export', control_mean_error=[round(e, 2) for e in control], export_mean_error=[round(e, 2) for e in errors_], bar=round(bar, 2))
    verdict('test_frame_export_to_non_ascii_folder', 'PASS', raised=repr(err) if err else None, written=len(written))
    assert err is None, f'raised {err!r}'
    assert len(written) == len(frames), f'{len(written)} of {len(frames)} frames written to "{folder}"'
    assert all(e <= bar for e in errors_), f'exported frames differ from the source: {errors_} against {bar:.2f}'
    return True


ORDER_FOLDERS = (('control', 'cat', 'cat'), ('case', 'cat#1', 'cat_1')) # (label, generated folder, as sanitized); the control is a name sanitize keeps


def numbered_save(base: str, sanitized: str, save):
    """Seed <base>/<sanitized>/00001-probe.mp4, restart the process-wide numbering, save once.

    Returns (result, raised, seed path, seed intact)."""
    from modules.image import namegen
    folder = os.path.join(base, sanitized)
    os.makedirs(folder, exist_ok=True)
    seed = os.path.join(folder, '00001-probe.mp4')
    with open(seed, 'wb') as f:
        f.write(b'seed')
    with patched(namegen, 'seq', 0):
        res, err = call_and_catch(save)
    with open(seed, 'rb') as f:
        intact = f.read() == b'seed'
    return res, err, seed, intact


def check_numbered_saves(name: str, out: dict, bases: dict):
    verdict(name, 'PASS', **{label: str(o) for label, o in out.items()})
    for label, generated, sanitized in ORDER_FOLDERS:
        o = out[label]
        res = o['returned']
        assert o['raised'] is None and res and os.path.isfile(res), f'{label}: returned {res!r} raised {o["raised"]}'
        assert os.path.dirname(res) == os.path.join(bases[label], sanitized), f'{label}: saved to "{res}"'
        assert o['seed_intact'], f'{label}: numbering restarted and the save overwrote "{o["seed"]}"'
        assert not o['unsanitized_made'], f'{label}: created the unsanitized folder "{generated}"'
    return True


def test_video_tab_numbers_in_the_sanitized_folder():
    """The generated folder is sanitized before the file is numbered and the folder created, so numbering scans the folder the file lands in and no unsanitized sibling is made."""
    if need_av():
        return need_av()
    pixels = (tensor_of(small_frames(4)).float() / 127.5 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)  # 1 c t h w in [-1, 1]
    out, bases = {}, {}
    for label, generated, sanitized in ORDER_FOLDERS:
        base = bases[label] = os.path.join(work_dir, f'order_tab_{label}')
        def save(base=base, generated=generated):
            with opts(outdir_video=base, outdir_samples='', save_to_dirs=True, directories_filename_pattern=generated, samples_filename_pattern='[seq]-probe', save_txt=False):
                return video_save.save_video(None, pixels=pixels, mp4_codec='libx264', mp4_thumb=False, mp4_fps=24)[1]
        res, err, seed, intact = numbered_save(base, sanitized, save)
        out[label] = {'returned': res, 'raised': repr(err) if err else None, 'seed': seed, 'seed_intact': intact, 'unsanitized_made': generated != sanitized and os.path.isdir(os.path.join(base, generated))}
    return check_numbered_saves('test_video_tab_numbers_in_the_sanitized_folder', out, bases)


def test_legacy_numbers_in_the_sanitized_folder():
    """The legacy writer sanitizes the name before numbering it, so a pattern with a folder sanitize renames cannot restart at 00001 over an existing file."""
    if need_av():
        return need_av()
    out, bases = {}, {}
    for label, generated, sanitized in ORDER_FOLDERS:
        base = bases[label] = os.path.join(work_dir, f'order_legacy_{label}')
        def save(base=base, generated=generated):
            with opts(outdir_video=base, outdir_samples='', samples_filename_pattern=f'{generated}/[seq]-probe'):
                return video.save_video(None, small_frames(4), video_type='mp4', duration=4 / 24.0, sync=True, mp4_codec='libx264')
        res, err, seed, intact = numbered_save(base, sanitized, save)
        out[label] = {'returned': res, 'raised': repr(err) if err else None, 'seed': seed, 'seed_intact': intact, 'unsanitized_made': generated != sanitized and os.path.isdir(os.path.join(base, generated))}
    return check_numbered_saves('test_legacy_numbers_in_the_sanitized_folder', out, bases)


# ============================================================
# browser gate clips
# ============================================================

def ffmpeg_cli_bt709(fn: str, img: np.ndarray, frames: int = 12) -> str | None:
    """The known-good: ffmpeg CLI with an explicit bt709 scale and tags."""
    exe = shutil.which('ffmpeg')
    if exe is None:
        return None
    h, w = img.shape[:2]
    cmd = [exe, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '24', '-i', '-',
           '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-crf', '18',
           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv', fn]
    proc = subprocess.run(cmd, input=img.tobytes() * frames, capture_output=True, check=False)
    return fn if proc.returncode == 0 else None


CLIP_ROLES = {
    # label: (role, size, expected table in predicted_rgb_after_decode, or the expected outcome)
    'pan_default': ('playback', '1024x576', None),
    'pan_bt709_x264': ('playback', '1024x576', None),
    'chart_default_720': ('default', '1280x720', 'correct'),
    'chart_default_512': ('default', '512x512', 'correct'),
    'chart_videotab_default': ('default', '1280x720', 'correct'),
    'chart_bt709_ffmpeg': ('positive_control', '1280x720', 'correct'),
    'chart_untagged_601_720': ('negative_control', '1280x720', 'bt601_decoded_as_bt709'),
    'chart_untagged_601_576': ('negative_control', '1024x576', 'bt601_decoded_as_bt709'),
    'chart_x264rgb': ('negative_control', '1280x720', 'wrong_unpredicted'),
    'chart_mpeg4': ('refused', '1280x720', 'refused'),
}

CLIP_SCORING = {
    'floor': 'per-patch error of the positive_control against "correct" in that browser; without the control every verdict is "no floor"',
    'default': 'every patch within floor + 3 levels of "correct" and closer to "correct" than to bt601_decoded_as_bt709',
    'negative_control': 'closer to its expected table than to "correct"; wrong_unpredicted means it must differ from "correct" by more than floor + 3 and the raw values are reported',
    'refused': 'the video element reports an error or never reaches canplaythrough',
    'playback': 'plays to the end; no chart',
    'sample': '9x9 mean at each centre, canvas drawImage with colorSpace srgb, plus the screenshot pixels',
}


def test_clips_written():
    if not flags.clips:
        return 'no --clips directory given'
    if need_av():
        return need_av()
    clips = os.path.abspath(flags.clips)
    os.makedirs(clips, exist_ok=True)
    written = {}

    def copy(label: str, src: str | None):
        role, size, expect = CLIP_ROLES[label]
        entry = {'file': None, 'role': role, 'size': size, 'expect': expect}
        if src and os.path.isfile(src):
            dst = os.path.join(clips, f'{label}{os.path.splitext(src)[1]}')
            shutil.copy2(src, dst)
            info = stream_info(dst)
            entry.update({'file': os.path.basename(dst), 'codec': info['codec'], 'pix_fmt': info['pix_fmt'], 'profile': info['profile'], 'tags': tags_of(info), 'untagged': is_untagged(info)})
        written[label] = entry

    img720, c720 = chart('720')
    img512, c512 = chart('512')
    img576, _ = chart_frame(1024, 576)
    copy('pan_default', legacy_save(pan(), 'clip_pan_default.mp4', video_type='MP4', duration=2.0))
    copy('chart_default_720', legacy_save([img720] * 24, 'clip_chart_default_720.mp4', video_type='MP4'))
    copy('chart_default_512', legacy_save([img512] * 24, 'clip_chart_default_512.mp4', video_type='MP4'))
    fn = out_path('clip_chart_videotab_default.mp4')
    call_and_catch(video_save.atomic_save_video, fn, tensor_of([img720] * 24), fps=24)
    copy('chart_videotab_default', fn if os.path.isfile(fn) else None)
    copy('chart_bt709_ffmpeg', ffmpeg_cli_bt709(out_path('clip_chart_bt709_ffmpeg.mp4'), img720, 24))
    options = video_writer.parse_options(video_codecs.get_codec_options('libx264'))
    copy('pan_bt709_x264', encode_frames(out_path('clip_pan_bt709_x264.mp4'), 'libx264', 'yuv420p', options, 24, pan(), convert='bt709', tags=STREAM_TAGS['bt709']))
    copy('chart_untagged_601_720', encode_frames(out_path('clip_chart_untagged_601_720.mp4'), 'libx264', 'yuv420p', options, 24, [img720] * 24, convert='bt601'))
    copy('chart_untagged_601_576', encode_frames(out_path('clip_chart_untagged_601_576.mp4'), 'libx264', 'yuv420p', options, 24, [img576] * 24, convert='bt601'))
    copy('chart_x264rgb', encode_frames(out_path('clip_chart_x264rgb.mp4'), 'libx264rgb', 'rgb24', {'crf': '0'}, 24, [img720] * 24))
    copy('chart_mpeg4', encode_frames(out_path('clip_chart_mpeg4.mp4'), 'mpeg4', 'yuv420p', {}, 24, [img720] * 24))
    manifest = {
        'patches': CHART_PATCHES,
        'centres': {'1280x720': c720, '512x512': c512, '1024x576': chart_frame(1024, 576)[1]},
        'scoring': CLIP_SCORING,
        'predicted_rgb_after_decode': {
            'correct': {k: list(v) for k, v in CHART_PATCHES.items()},
            'bt601_decoded_as_bt709': {k: [round(float(x), 1) for x in ycbcr_to_rgb(rgb_to_ycbcr(v, 'bt601', False), 'bt709', False)] for k, v in CHART_PATCHES.items()},
            'bt709_decoded_as_bt601': {k: [round(float(x), 1) for x in ycbcr_to_rgb(rgb_to_ycbcr(v, 'bt709', False), 'bt601', False)] for k, v in CHART_PATCHES.items()},
        },
        'predicted_yuv': {m: {k: [round(float(x), 2) for x in v] for k, v in predicted_patches(m, False).items()} for m in MATRICES},
        'clips': written,
        'commit': report['platform'].get('commit'),
    }
    with open(os.path.join(clips, 'chart.json'), 'w', encoding='utf8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    verdict('test_clips_written', 'PASS', clips=sorted(k for k, v in written.items() if v['file']), missing=sorted(k for k, v in written.items() if not v['file']))
    return True


# ============================================================
# runner
# ============================================================

GROUPS = [
    ('platform', [test_versions, test_presets_open]),
    ('floor', [test_log_capture_sees_writer_lines, test_quality_floor, test_quality_negative_control, test_quality_fixture_guard, test_colour_floor, test_colour_positive_control, test_colour_negative_controls, test_lossless_rgb_roundtrip]),
    ('writer failure', [test_sync_unknown_codec_returns_none, test_stream_rejected_pix_fmt_raises_with_reason, test_container_refusing_codec_removes_its_file, test_av_missing_sync_returns_none_stream_raises, test_zero_byte_mp4_metadata_is_empty, test_output_equal_to_source_refused, test_dlss_none_kwargs_resolve_to_ui_values, test_gif_failure_returns_none, test_mid_encode_exception_returns_none, test_sync_success_registers_after_open, test_threaded_save_joined, test_both_paths_write_the_same_stream, test_legacy_save_accepts_grayscale_frames, test_stream_writer_failure_leaves_partial_file, test_write_audio_without_stream_refused]),
    ('pixel format', [test_presets_declare_pix_fmt, test_table_pix_fmt_opens_in_legacy_loop, test_table_pix_fmt_opens_in_video_tab_loop, test_rejected_pix_fmt_names_the_list, test_empty_pix_fmt_refused_naming_default, test_out_of_table_codec_defaults_yuv420p, test_ext_none_follows_codec_when_container_refuses, test_odd_size_cropped_under_yuv420p_legacy, test_odd_size_cropped_under_yuv420p_video_tab, test_odd_size_kept_without_subsampling]),
    ('defaults', [test_legacy_default_is_h264_yuv420p, test_legacy_default_quality_matches_old_writer, test_explicit_mpeg4_gets_table_preset, test_legacy_labels_map_to_codecs, test_legacy_labels_encode, test_odd_size_under_default, test_none_codec_kwargs_give_default, test_resolve_video_type_aliases, test_explicit_codec_overrides_label_container, test_extension_rule, test_codec_name_saves_in_its_container]),
    ('rate', [test_video_rate_fractions, test_mpeg4_opens_at_dlss_long_rate, test_video_tab_keeps_fractional_rate, test_fps_zero_refused_by_name, test_dlss_output_fps_rule, test_video_tab_returns_none_video_on_failure]),
    ('colour', [test_default_outputs_are_tagged_bt709, test_legacy_default_colour_720, test_legacy_default_colour_512, test_video_tab_default_colour, test_rgb_pix_fmt_tagged_full_and_exact, test_prores_tagged_and_within_oracle, test_mpeg4_in_mp4_carries_colr]),
    ('readers', [test_readers_agree_on_h264_fixture, test_readers_fractional_rate, test_readers_per_container, test_reader_codec_names, test_count_frames_without_header_or_duration, test_frames_estimate_ignores_trailing_audio, test_reader_decodes_with_stream_tags, test_reader_rotation_follows_display_matrix, test_params_without_capture_read_headers_only, test_iter_frames_skip_and_limit, test_ltx_condition_frames_keep_frame_zero, test_first_frame_equals_read_video_first, test_get_video_info_exact_keys, test_unreadable_file_contracts, test_corrupt_stream_stops_with_warning, test_frames_are_contiguous, test_unchanged_callers, test_reader_decode_speed_measured]),
    ('audio', [test_video_tab_muxes_audio_tensor, test_video_tab_muxes_audio_packets]),
    ('metadata', [test_metadata_roundtrip]),
    ('gradio', [test_option_off_does_not_probe_or_convert, test_option_on_converts_without_binaries, test_option_on_serves_playable_as_is, test_option_on_converts_unplayable_h264, test_preview_audio_passthrough_and_drop, test_preview_unopenable_file_served_as_is, test_preview_av_import_failure_does_not_raise, test_cli_fallback_leaves_no_copy, test_cleanup_tmpdr_removes_all_video_extensions]),
    ('process', [test_process_missing_video_reports_reason, test_process_tab_shows_failure_text_whole]),
    ('paths', [test_non_ascii_output_path, test_long_path_truncation, test_output_folder_kept_as_configured, test_video_tab_numbers_in_the_sanitized_folder, test_legacy_numbers_in_the_sanitized_folder, test_frame_export_to_non_ascii_folder]),
    ('clips', [test_clips_written]),
]


CASES.extend((name, fn) for name, fns in GROUPS for fn in fns)


def suite_opts():
    return opts(outdir_video=out_dir, outdir_samples='', temp_dir=temp_dir, save_images_add_number=False, samples_filename_pattern='', image_metadata=True)


def normalised_report_text() -> str:
    """The report with the per-run work dir replaced by <work> and its separators made '/', so two runs diff by content."""
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    for raw in {work_dir, os.path.realpath(work_dir)}:
        for encoded in (json.dumps(raw)[1:-1], json.dumps(raw.replace('\\', '\\\\'))[1:-1]):
            text = text.replace(encoded, '<work>')
    return re.sub(r'<work>[^"\' ]*', lambda m: re.sub(r'(\\\\)+', '/', m.group(0)), text)


def run_all():
    """Supervisor: never encodes or decodes itself. A process that has used swscale cannot fork a child that uses it
    (the child deadlocks on inherited state; av 19.0.2 with ffmpeg 9.0.2), and an encoder that fails to open can
    take its process down, so every case runs in a child and a dead or silent child is replaced from the next case."""
    log.warning('=== Video save layer ===')
    log.info(f'work dir: {work_dir} av={getattr(av, "__version__", None)} runner={"spawn" if SPAWN else "fork"}')
    report['platform']['runner'] = {'mode': 'spawn' if SPAWN else 'fork', 'case_timeout_s': CASE_TIMEOUT}
    report['platform']['flags'] = {'clips': bool(flags.clips)}
    if not SPAWN:
        pan()  # numpy fixtures built once and inherited by every forked child
        chart('720')
        chart('512')
    children = 0
    with suite_opts():
        index = 0
        while index < len(CASES):
            index = supervise(index)
            children += 1
    timings: dict[str, float] = {}
    for name, fn in CASES:
        timings[name] = round(timings.get(name, 0.0) + (report['measurements'].get('case_ms', {}).get(fn.__name__) or 0) / 1000, 2)
    report['measurements']['timings_s'] = timings
    report['platform']['runner']['children'] = children

    log.warning('=== Results ===')
    totals = {'passed': 0, 'failed': 0, 'skipped': 0, 'absent': 0}
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped, {info['absent']} absent [{status}]")
        for key in totals:
            totals[key] += info[key]
    log.warning(f"Total: {totals['passed']} passed, {totals['failed']} failed, {totals['skipped']} skipped, {totals['absent']} absent")
    report['measurements']['totals'] = totals
    if flags.report:
        os.makedirs(os.path.dirname(os.path.abspath(flags.report)) or '.', exist_ok=True)
        with open(flags.report, 'w', encoding='utf8') as f:
            f.write(normalised_report_text())
        log.info(f'report: {os.path.abspath(flags.report)}')
    return totals['failed'] == 0


def run_child():
    """Spawned child: take the supervisor's state, run from the given case, exit without interpreter teardown."""
    with open(flags.child_state, encoding='utf8') as f:
        state = json.load(f)
    FLOOR.update(state.get('FLOOR', {}))
    PRESET_OPEN.update(state.get('PRESET_OPEN', {}))
    report['platform'].update(state.get('platform', {}))
    with suite_opts():
        run_cases_here(flags.child_start, flags.child_results)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    if flags.child_start >= 0:
        run_child()
    t_start = time.time()
    try:
        all_ok = run_all()
    finally:
        if flags.keep:
            log.info(f'kept: {work_dir}')
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            if os.path.isdir(work_dir):
                log.warning(f'leftovers in {work_dir}')
    log.warning(f'Total time: {time.time() - t_start:.2f}s')
    sys.exit(0 if all_ok else 1)
