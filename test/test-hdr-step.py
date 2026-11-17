#!/usr/bin/env python
"""
Offline unit tests for the HDR step in scripts.hdr.hdr_process.

``convert_hdr`` writes the HDR version of an image beside an output path its caller saved and
returns the written path; ``hdr_support`` answers before a run what would be written, or why
nothing would be.

Covers:

- ``hdr_support`` per image file format, a missing executable, a package installed on first use
- ``convert_hdr`` naming, the PNG cICP chunk, the caller's output left untouched
- ``HDRUnavailable`` codes for an unsupported format, a missing encoder and a failed install

No running server required.

Usage:
    python test/test-hdr-step.py
"""

import os
import sys
import shutil
import struct
import tempfile
import types
import importlib.util

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

import numpy as np                                                   # pylint: disable=wrong-import-position
from PIL import Image                                                # pylint: disable=wrong-import-position
import modules                                                       # pylint: disable=wrong-import-position
from modules.logger import log                                       # pylint: disable=wrong-import-position
from scripts.hdr import hdr_process                                  # pylint: disable=wrong-import-position
from scripts.hdr.hdr_process import HDRUnavailable, convert_hdr, hdr_support  # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
tmp = tempfile.mkdtemp(prefix='sdnext-hdr-test-')


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'tests': []}
    return name


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


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {type(e).__name__}: {e}')


class patched:
    def __init__(self, target, name, value):
        self.target, self.name, self.value = target, name, value

    def __enter__(self):
        self.original = getattr(self.target, self.name)
        setattr(self.target, self.name, self.value)

    def __exit__(self, *args):
        setattr(self.target, self.name, self.original)


def without_modules(*names):
    original = importlib.util.find_spec
    return patched(hdr_process.importlib.util, 'find_spec', lambda name, *args: None if name in names else original(name, *args))


def sample_image() -> Image.Image:
    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    return Image.fromarray(np.stack([np.tile(gradient, (48, 1))] * 3, axis=2))


def saved_output(name: str) -> tuple[str, bytes]:
    output = os.path.join(tmp, name)
    sample_image().save(output)
    with open(output, 'rb') as file:
        return output, file.read()


def png_chunks(path: str) -> dict[bytes, bytes]:
    with open(path, 'rb') as file:
        data = file.read()
    chunks, offset = {}, 8
    while offset < len(data):
        length, kind = struct.unpack('>I4s', data[offset:offset + 8])
        chunks.setdefault(kind, data[offset + 8:offset + 8 + length])
        offset += length + 12
    return chunks


def unavailable(fn) -> HDRUnavailable:
    try:
        fn()
    except HDRUnavailable as e:
        return e
    raise AssertionError('HDRUnavailable not raised')


# ============================================================
# hdr_support
# ============================================================

def test_png_is_written_by_opencv():
    support = hdr_support('png')
    assert (support.available, support.hdr_format, support.extension, support.encoder) == (True, 'png-cicp', 'png', 'opencv'), support
    assert (support.installs, support.replaces_output, support.code, support.reason) == (False, False, None, None), support


def test_format_spelling_is_normalized():
    assert hdr_support('.TIF ').extension == 'tiff'
    assert hdr_support('jpeg').hdr_format == 'jpeg-gain-map'


def test_unsupported_format_names_itself():
    for samples_format in ('webp', 'jp2', 'heif'):
        support = hdr_support(samples_format)
        assert (support.available, support.code, support.extension) == (False, 'format-unsupported', None), support
        assert samples_format in support.reason, support.reason


def test_jpg_needs_the_executable():
    with patched(hdr_process.shutil, 'which', lambda name: None):
        support = hdr_support('jpg')
        assert (support.available, support.code, support.encoder, support.installs) == (False, 'encoder-missing', 'ultrahdr_app', False), support
    with patched(hdr_process.shutil, 'which', lambda name: f'/usr/bin/{name}'):
        support = hdr_support('jpg')
        assert (support.available, support.code, support.extension) == (True, None, 'jpg'), support


def test_missing_package_is_installed_on_first_use():
    with without_modules('imagecodecs', 'OpenEXR'):
        for samples_format, encoder in (('avif', 'imagecodecs'), ('jxl', 'imagecodecs'), ('exr', 'OpenEXR')):
            support = hdr_support(samples_format)
            assert (support.available, support.installs, support.encoder) == (True, True, encoder), support


def test_missing_tifffile_is_not_installed():
    with without_modules('tifffile'):
        support = hdr_support('tiff')
        assert (support.available, support.code, support.installs) == (False, 'encoder-missing', False), support


def test_only_exr_replaces_the_output():
    replaced = [samples_format for samples_format in ('jpg', 'png', 'tiff', 'avif', 'jxl', 'exr') if hdr_support(samples_format).replaces_output]
    assert replaced == ['exr'], replaced


def test_default_format_is_the_setting():
    stub = types.SimpleNamespace(opts=types.SimpleNamespace(samples_format='tiff'))
    had = hasattr(modules, 'shared')
    original = getattr(modules, 'shared', None)
    modules.shared = stub
    try:
        assert hdr_support().samples_format == 'tiff'
    finally:
        if had:
            modules.shared = original
        else:
            del modules.shared


# ============================================================
# convert_hdr
# ============================================================

def test_png_lands_beside_the_output():
    output, before = saved_output('00012-castle.png')
    written = convert_hdr(sample_image(), output, samples_format='png')
    assert written == os.path.join(tmp, '00012-castle-hdr.png'), written
    chunks = png_chunks(written)
    assert chunks[b'cICP'] == bytes((9, 16, 0, 1)), chunks.get(b'cICP')
    assert chunks[b'IHDR'][8] == 16, f'bit depth {chunks[b"IHDR"][8]}'
    with open(output, 'rb') as file:
        assert file.read() == before, 'output changed'


def test_gamut_reaches_the_file():
    output, _before = saved_output('gamut.png')
    written = convert_hdr(sample_image(), output, gamut='DCI-P3', samples_format='png')
    assert png_chunks(written)[b'cICP'][0] == 12


def test_extension_follows_the_hdr_format():
    output, _before = saved_output('00013-castle.png')
    written = convert_hdr(sample_image(), output, peak=4000, samples_format='tif')
    assert written == os.path.join(tmp, '00013-castle-hdr.tiff'), written
    assert os.path.getsize(written) > 0


def test_output_need_not_exist():
    written = convert_hdr(sample_image(), os.path.join(tmp, 'sub', 'unsaved.png'), samples_format='png')
    assert written == os.path.join(tmp, 'sub', 'unsaved-hdr.png') and os.path.isfile(written), written


def test_unsupported_format_writes_nothing():
    output, _before = saved_output('plain.webp')
    error = unavailable(lambda: convert_hdr(sample_image(), output, samples_format='webp'))
    assert error.code == 'format-unsupported' and 'webp' in error.reason and str(error) == error.reason, (error.code, error.reason)
    assert sorted(name for name in os.listdir(tmp) if name.startswith('plain')) == ['plain.webp']


def test_missing_executable_is_reported():
    output, _before = saved_output('gain.jpg')
    with patched(hdr_process.shutil, 'which', lambda name: None):
        error = unavailable(lambda: convert_hdr(sample_image(), output, samples_format='jpg'))
    assert (error.code, error.reason) == ('encoder-missing', 'ultrahdr_app is not installed'), (error.code, error.reason)
    assert not os.path.exists(os.path.join(tmp, 'gain-hdr.jpg'))


def test_failed_install_is_reported():
    def fail():
        raise ImportError('No module named imagecodecs')
    output, _before = saved_output('install.png')
    with without_modules('imagecodecs'), patched(hdr_process, '_get_imagecodecs', fail):
        error = unavailable(lambda: convert_hdr(sample_image(), output, samples_format='avif'))
    assert error.code == 'encoder-install-failed' and 'imagecodecs' in error.reason, (error.code, error.reason)


def test_bad_arguments_raise_value_errors():
    output = os.path.join(tmp, 'args.png')
    for kwargs in ({'method': 'None'}, {'method': 'Learned'}, {'peak': 50}, {'gamut': 'Rec.709'}):
        try:
            convert_hdr(sample_image(), output, samples_format='png', **kwargs)
        except ValueError:
            continue
        raise AssertionError(f'{kwargs} accepted')
    try:
        convert_hdr(None, output, samples_format='png')
    except ValueError:
        pass
    else:
        raise AssertionError('missing image accepted')
    assert not os.path.exists(os.path.join(tmp, 'args-hdr.png'))


def run_all():
    log.warning('=== HDR step ===')
    cat = category('support')
    for fn in [
        test_png_is_written_by_opencv,
        test_format_spelling_is_normalized,
        test_unsupported_format_names_itself,
        test_jpg_needs_the_executable,
        test_missing_package_is_installed_on_first_use,
        test_missing_tifffile_is_not_installed,
        test_only_exr_replaces_the_output,
        test_default_format_is_the_setting,
    ]:
        run_test(cat, fn)
    cat = category('convert')
    for fn in [
        test_png_lands_beside_the_output,
        test_gamut_reaches_the_file,
        test_extension_follows_the_hdr_format,
        test_output_need_not_exist,
        test_unsupported_format_writes_nothing,
        test_missing_executable_is_reported,
        test_failed_install_is_reported,
        test_bad_arguments_raise_value_errors,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed')
    return total_failed == 0


if __name__ == '__main__':
    try:
        success = run_all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if success else 1)
