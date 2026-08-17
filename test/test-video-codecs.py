#!/usr/bin/env python
"""
Offline unit tests for the video encoder option strings in modules.video_models.

``parse_options`` treats ``:`` and ``,`` as separators and ``=`` as the only assignment, so a
segment without ``=`` becomes a valueless flag set to ``'1'``. That makes ffmpeg command line
spellings parse into something legal but wrong: ``crf:16`` yields ``{'crf': '1', '16': '1'}`` and
``crf=23:b:v=0`` pins the generic bitrate option to 1 bit per second. Neither raises, so the only
way to catch it is to assert the shape of the strings themselves.

Covers:

- ``parse_options`` semantics for both separators, the flag form, and the dict passthrough
- every preset in ``codecs_config`` reaching the encoder as the keys it was written with
- the shipped ``mp4_opt`` default parsing to the value it names

No running server required. Nothing is encoded.

Usage:
    python test/test-video-codecs.py
"""

import os
import sys

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

from modules.errors import log                                       # pylint: disable=wrong-import-position
from modules.video_models.video_codecs import codecs_config          # pylint: disable=wrong-import-position
from modules.video_models.video_save import parse_options            # pylint: disable=wrong-import-position


results: dict[str, dict] = {}


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
        record(cat, False, name, f'exception: {e}')


# ============================================================
# parse_options semantics
# ============================================================

def test_assignment_uses_equals():
    assert parse_options('crf=16') == {'crf': '16'}


def test_colon_separates_pairs():
    assert parse_options('crf=18:preset=medium') == {'crf': '18', 'preset': 'medium'}


def test_comma_separates_pairs():
    assert parse_options('crf=18,preset=medium') == {'crf': '18', 'preset': 'medium'}


def test_bare_item_becomes_a_flag():
    assert parse_options('fastseek') == {'fastseek': '1'}


def test_whitespace_is_stripped():
    assert parse_options(' crf = 18 : preset = medium ') == {'crf': '18', 'preset': 'medium'}


def test_empty_and_dict_inputs():
    assert parse_options('') == {}
    assert parse_options('   ') == {}
    assert parse_options(None) == {}
    assert parse_options({'crf': '16'}) == {'crf': '16'}


def test_ffmpeg_cli_spelling_does_not_assign():
    # the failure this suite exists for: legal parse, wrong meaning, no exception
    assert parse_options('crf:16') != {'crf': '16'}


# ============================================================
# shipped presets
# ============================================================

def test_every_preset_segment_assigns():
    broken = []
    for name, cfg in codecs_config.items():
        options = cfg.get('options', '')
        if not options:
            continue
        for segment in options.replace(',', ':').split(':'):
            if segment.strip() and '=' not in segment:
                broken.append(f'{name}="{options}" segment="{segment}"')
    assert not broken, f'segments parse as flags instead of assignments: {broken}'


def test_no_preset_key_is_numeric():
    numeric = []
    for name, cfg in codecs_config.items():
        for key in parse_options(cfg.get('options', '')):
            if key.isdigit():
                numeric.append(f'{name}: {key}')
    assert not numeric, f'a value was parsed as a key: {numeric}'


def test_no_preset_sets_a_degenerate_bitrate():
    # b is a generic AVCodecContext option, so a stray b=1 caps the encoder at 1 bit per second
    bad = []
    for name, cfg in codecs_config.items():
        bitrate = parse_options(cfg.get('options', '')).get('b', None)
        if bitrate is not None and bitrate != '0':
            bad.append(f'{name}: b={bitrate}')
    assert not bad, f'bitrate pinned to a nonzero value: {bad}'


def test_every_preset_keeps_its_key_count():
    for name, cfg in codecs_config.items():
        options = cfg.get('options', '')
        if not options:
            continue
        segments = [s for s in options.replace(',', ':').split(':') if s.strip()]
        parsed = parse_options(options)
        assert len(parsed) == len(segments), f'{name}="{options}" parsed to {parsed}'


# ============================================================
# shipped defaults
# ============================================================

def test_run_default_names_its_own_value():
    import inspect
    from modules.video_models import video_run
    default = inspect.signature(video_run.run).parameters['mp4_opt'].default
    assert parse_options(default) == {'crf': '16'}, f'default "{default}" parsed to {parse_options(default)}'


def run_all():
    log.warning('=== Video codec options ===')
    cat = category('parser')
    for fn in [
        test_assignment_uses_equals,
        test_colon_separates_pairs,
        test_comma_separates_pairs,
        test_bare_item_becomes_a_flag,
        test_whitespace_is_stripped,
        test_empty_and_dict_inputs,
        test_ffmpeg_cli_spelling_does_not_assign,
    ]:
        run_test(cat, fn)
    cat = category('presets')
    for fn in [
        test_every_preset_segment_assigns,
        test_no_preset_key_is_numeric,
        test_no_preset_sets_a_degenerate_bitrate,
        test_every_preset_keeps_its_key_count,
    ]:
        run_test(cat, fn)
    cat = category('defaults')
    for fn in [
        test_run_default_names_its_own_value,
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
    import time
    t0 = time.time()
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
