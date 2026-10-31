#!/usr/bin/env python
"""
Offline tests for gallery media: which files the gallery treats as video.

Every container a codec preset writes gets a video thumbnail from ``/sdapi/v1/browser/thumb``
and the video preview in the Gradio gallery, and the frontend lists it.

Covers:

- the thumb route, called as the server registers it, on one clip per container, plus an image and an uppercase extension
- ``ui_gallery.read_media`` taking the video branch for every container (only ts reached the image branch before)
- the frontend's extension list in ``ui/gallery.ts`` naming every container
- ``gr_tempdir.cleanup_tmpdr`` removing preview copies of every container and nothing else
- ``video_codecs.is_video_file`` on extensions, cases and paths
- every container served with a video media type once ``api.mime`` registered its types

No running server required.

Usage:
    python test/test-gallery-media.py
"""

import os
import re
import sys
import json
import time
import shutil
import asyncio
import logging
import tempfile
import contextlib
from types import SimpleNamespace

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

import numpy as np                                                   # pylint: disable=wrong-import-position
from PIL import Image, PngImagePlugin                                # pylint: disable=wrong-import-position
from modules import shared                                           # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position
from modules.video_models.video_codecs import codecs_config          # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-gallery-media-', suffix='-work')
CONTAINERS = sorted({ext for cfg in codecs_config.values() for ext in cfg['allowed_exts']})
ENCODERS = {'webm': 'libvpx-vp9', 'avi': 'mpeg4'} # libx264 for the rest
OPTIONS = {'libx264': {'crf': '23', 'preset': 'veryfast'}, 'libvpx-vp9': {'deadline': 'realtime', 'cpu-used': '8'}, 'mpeg4': {}}
WIDTH, HEIGHT, FRAMES = 160, 90, 16
clips: dict[str, str] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, status: str, name: str, detail: str = ''):
    results[cat][{'PASS': 'passed', 'FAIL': 'failed', 'SKIP': 'skipped'}[status]] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if status == 'PASS':
        log.info(msg)
    elif status == 'SKIP':
        log.warning(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if isinstance(ok, str):
            record(cat, 'SKIP', name, ok)
        else:
            record(cat, 'FAIL' if ok is False else 'PASS', name)
    except AssertionError as e:
        record(cat, 'FAIL', name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, 'FAIL', name, f'exception: {type(e).__name__}: {e}')


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):  # pylint: disable=redefined-outer-name
        self.records.append(record)

    def messages(self, level: int = logging.DEBUG) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


@contextlib.contextmanager
def capture_log():
    """Records of the sd logger; a probe warning is logged first and must land, so an empty capture means nothing was logged."""
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        log.warning('gallery-media capture probe')
        assert handler.messages(logging.WARNING) == ['gallery-media capture probe'], 'log capture missed its probe'
        handler.records.clear()
        yield handler
    finally:
        logger.removeHandler(handler)


@contextlib.contextmanager
def demo(allowed: list[str]):
    previous = shared.demo
    shared.demo = SimpleNamespace(allowed_paths=list(allowed))
    try:
        yield
    finally:
        shared.demo = previous


def textured(i: int) -> np.ndarray:
    """A moving gradient over seeded noise: compresses to well over 1 KB, which the image thumbnailer needs to try a file at all."""
    rng = np.random.default_rng(i)
    x = np.linspace(0, 255, WIDTH)[None, :] * np.ones((HEIGHT, 1))
    y = np.linspace(0, 255, HEIGHT)[:, None] * np.ones((1, WIDTH))
    return np.stack([(x + 10 * i) % 256, (y + 5 * i) % 256, rng.integers(0, 256, (HEIGHT, WIDTH))], axis=-1).astype(np.uint8)


def make_clips() -> str | None:
    try:
        import av
    except ImportError:
        return 'av not installed'
    for ext in CONTAINERS:
        codec = ENCODERS.get(ext, 'libx264')
        fn = os.path.join(work_dir, f'clip.{ext}')
        with av.open(fn, 'w') as container:
            stream = container.add_stream(codec, rate=24, options=OPTIONS[codec])
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
            for i in range(FRAMES):
                for packet in stream.encode(av.VideoFrame.from_ndarray(textured(i), format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        size = os.path.getsize(fn)
        assert size >= 1024, f'fixture {fn} is {size} bytes'
        clips[ext] = fn
    return None


def make_image() -> str:
    fn = os.path.join(work_dir, 'image.png')
    info = PngImagePlugin.PngInfo()
    info.add_text('parameters', 'gallery media probe')
    Image.fromarray(textured(0)).save(fn, pnginfo=info)
    assert os.path.getsize(fn) >= 1024, 'image fixture under 1 KB'
    return fn


def thumb_route():
    """The thumb handler as the server registers it."""
    from modules.api import gallery
    routes = {}
    app = SimpleNamespace(websocket=lambda path: (lambda fn: fn))
    api = SimpleNamespace(app=app, credentials={}, add_api_route=lambda path, fn, **kwargs: routes.__setitem__(path, fn))
    gallery.register_api(api)
    return routes['/sdapi/v1/browser/thumb']


def thumb(fn: str, exif: bool = False) -> dict:
    with demo([work_dir]):
        get_thumb = thumb_route()
        response = asyncio.run(get_thumb(file=fn, exif=exif))
    return json.loads(response.body)


# ============================================================
# thumb route
# ============================================================

def test_thumb_every_container():
    if not clips:
        return 'av not installed'
    failed = {}
    for ext, fn in clips.items():
        with capture_log() as captured:
            content = thumb(fn)
        problems = []
        if not str(content.get('data', '')).startswith('data:image/jpeg;base64,'):
            problems.append(f'no thumbnail, keys={sorted(content)}')
        if (content.get('width'), content.get('height')) != (WIDTH, HEIGHT):
            problems.append(f"size {content.get('width')}x{content.get('height')}")
        if not str(content.get('exif', '')).startswith('Codec: '):
            problems.append(f"exif {content.get('exif')!r}")
        warnings = captured.messages(logging.WARNING)
        if warnings:
            problems.append(f'logged {warnings}')
        if problems:
            failed[ext] = problems
    assert not failed, f'containers without a video thumbnail: {failed}'


def test_thumb_image_and_uppercase_extension():
    image = make_image()
    content = thumb(image, exif=True)
    assert str(content.get('data', '')).startswith('data:image/jpeg;base64,'), f'image thumbnail missing: {sorted(content)}'
    assert content.get('exif') == 'gallery media probe', f"image exif {content.get('exif')!r}"
    if 'mp4' not in clips:
        return 'av not installed'
    upper = os.path.join(work_dir, 'upper.MP4')
    shutil.copyfile(clips['mp4'], upper)
    content = thumb(upper)
    assert str(content.get('exif', '')).startswith('Codec: '), f"uppercase mp4 not read as video: {content.get('exif')!r}"


# ============================================================
# gradio gallery preview
# ============================================================

def test_read_media_every_container():
    """Before the shared list, ts was the only preset container outside read_media's own tuple."""
    if not clips:
        return 'av not installed'
    from modules import ui_gallery
    images = []
    for ext, fn in clips.items():
        updates = ui_gallery.read_media(fn)
        if not (isinstance(updates[1], dict) and updates[1].get('visible') is True and updates[1].get('value') == fn):
            images.append(ext)
    assert not images, f'containers previewed as images: {images}'
    updates = ui_gallery.read_media(make_image())
    assert updates[1].get('visible') is False, f'image previewed as video: {updates[1]}'


# ============================================================
# frontend list
# ============================================================

def test_frontend_lists_every_container():
    source = open(os.path.join(script_dir, 'ui', 'gallery.ts'), encoding='utf8').read()
    match = re.search(r'const SUPPORTED_EXTENSIONS(?::[^=]+)?\s*=\s*\[([^\]]*)\]', source, re.DOTALL)
    assert match is not None, 'SUPPORTED_EXTENSIONS array literal not found in ui/gallery.ts'
    tokens = [t.strip() for t in match.group(1).split(',') if t.strip()]
    malformed = [t for t in tokens if not re.fullmatch(r"""['"][a-z0-9]+['"]""", t)]
    assert not malformed, f'SUPPORTED_EXTENSIONS holds entries that are not plain extension strings: {malformed}'
    listed = {t[1:-1] for t in tokens}
    missing = sorted(set(CONTAINERS) - listed)
    assert not missing, f'ui/gallery.ts does not list {missing}'


# ============================================================
# temp cleanup
# ============================================================

def test_cleanup_removes_every_container():
    from modules import gr_tempdir
    temp_dir = os.path.join(work_dir, 'temp')
    os.makedirs(temp_dir)
    names = [f'copy.{ext}' for ext in CONTAINERS] + ['copy.png', 'notes.txt', 'upper.MKV']
    for name in names:
        with open(os.path.join(temp_dir, name), 'wb') as f:
            f.write(b'x')
    previous = (shared.opts.data.get('temp_dir', None), shared.opts.data.get('clean_temp_dir_at_start', None))
    shared.opts.data['temp_dir'] = temp_dir
    shared.opts.data['clean_temp_dir_at_start'] = True
    try:
        assert os.path.isdir(shared.opts.temp_dir), 'temp folder missing, cleanup would fall back to the real one'
        gr_tempdir.cleanup_tmpdr()
    finally:
        shared.opts.data['temp_dir'], shared.opts.data['clean_temp_dir_at_start'] = previous
    left = sorted(os.listdir(temp_dir))
    assert left == ['notes.txt', 'upper.MKV'], f'left after cleanup: {left}'


# ============================================================
# helper
# ============================================================

def test_is_video_file():
    from modules.video_models.video_codecs import is_video_file
    wrong = []
    for ext in CONTAINERS:
        for name in (f'clip.{ext}', f'CLIP.{ext.upper()}', f'Clip.{ext.capitalize()}', f'/out/v1.2/clip.{ext}'):
            if not is_video_file(name):
                wrong.append(name)
    for name in ('image.png', 'log.json', 'notes.txt', 'noext', 'ts', 'mp4', '.mp4', 'folder.mp4/image.png', 'clip.tar.gz', 'clip.mpg', 'clip.mjpeg'):
        if is_video_file(name):
            wrong.append(name)
    assert not wrong, f'misclassified: {wrong}'


# ============================================================
# media types
# ============================================================

def test_every_container_has_a_video_media_type():
    import mimetypes
    from modules.api import mime
    mime.register()
    types = {ext: mimetypes.guess_type(f'clip.{ext}')[0] for ext in CONTAINERS}
    wrong = {ext: t for ext, t in types.items() if not (t or '').startswith('video/')}
    assert not wrong, f'containers without a video media type: {wrong}'
    assert types['ts'] == 'video/mp2t', f"ts served as {types['ts']}"


def run_all():
    log.warning('=== Gallery media ===')
    reason = make_clips()
    if reason:
        log.warning(f'  clips: {reason}')
    cat = category('thumb')
    for fn in [
        test_thumb_every_container,
        test_thumb_image_and_uppercase_extension,
    ]:
        run_test(cat, fn)
    cat = category('preview')
    run_test(cat, test_read_media_every_container)
    cat = category('frontend')
    run_test(cat, test_frontend_lists_every_container)
    cat = category('cleanup')
    run_test(cat, test_cleanup_removes_every_container)
    cat = category('helper')
    run_test(cat, test_is_video_file)
    cat = category('mime')
    run_test(cat, test_every_container_has_a_video_media_type)

    log.warning('=== Results ===')
    total = {'passed': 0, 'failed': 0, 'skipped': 0}
    for cat_name, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        for key in total:
            total[key] += info[key]
    log.warning(f"Total: {total['passed']} passed, {total['failed']} failed, {total['skipped']} skipped")
    return total['failed'] == 0


if __name__ == '__main__':
    t0 = time.time()
    try:
        ok = run_all()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
