#!/usr/bin/env python
"""
Offline unit tests for output path sanitizing in modules.image.namegen.

``FilenameGenerator.sanitize`` translates characters Windows rejects and strips trailing
``. , _`` and spaces from each path component. A configured output folder passed as ``base``
is kept as given, so a folder named ``out_`` or ``#1`` receives the saves instead of a sibling.

Covers:

- the base folder kept byte for byte, the generated path below it still sanitized
- ``..``, trailing dots, ``:`` and reserved names in generated components
- calls without a base, and paths outside the base, sanitized whole
- ``save_image`` writing into a configured folder whose name ends in ``_``
- generated names carrying no ``:``, so a pattern starting with a letter and a colon cannot become a Windows drive

No running server required.

Usage:
    python test/test-namegen.py
"""

import os
import sys
import shutil
import tempfile
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

from PIL import Image                                                # pylint: disable=wrong-import-position
from modules import shared                                           # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position
from modules.image.namegen import FilenameGenerator                  # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-namegen-')
namegen = FilenameGenerator()


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


def stub_p():
    """What FilenameGenerator reads from a processing object for the tokens these tests use."""
    return SimpleNamespace(all_seeds=[1234], seed=1234, all_prompts=['a cat'], prompt='a cat', width=64, height=64)


def components(path: str) -> list[str]:
    return path.replace('\\', '/').split('/')


# ============================================================
# sanitize
# ============================================================

def test_base_kept_as_given():
    for folder in ['out_', 'out.', 'run #1', 'cost $5', '\uff4f\uff55\uff54']: # trailing strip, translated chars, NFKC
        base = os.path.join(work_dir, folder)
        result = namegen.sanitize(os.path.join(base, 'a#b', 'c.png'), base=base)
        assert result == os.path.join(base, 'a_b', 'c.png'), f'base="{base}" result="{result}"'


def test_generated_part_sanitized():
    base = os.path.join(work_dir, 'out_')
    generated = os.sep.join(['cat.', '..', 'a:b', 'CON', 'x.png']) # os.path.join reads 'a:' as a drive on Windows and drops the base
    result = namegen.sanitize(base + os.sep + generated, base=base)
    parts = components(result[len(base) + 1:])
    assert parts == ['cat', 'a_b', '_', 'x.png'], f'generated parts {parts}'


def test_without_base_unchanged():
    assert namegen.sanitize(os.path.join('dir_', 'a#b.png')) == os.path.join('dir', 'a_b.png')


def test_path_outside_base_sanitized_whole():
    base = os.path.join(work_dir, 'out_')
    other = os.path.join(work_dir, 'other_', 'a.png')
    assert namegen.sanitize(other, base=base) == namegen.sanitize(other)


# ============================================================
# save_image
# ============================================================

def test_save_image_into_folder_ending_in_underscore():
    from modules.image.save import save_image
    folder = os.path.join(work_dir, 'save_')
    previous = shared.opts.data.get('save_to_dirs', None)
    shared.opts.data['save_to_dirs'] = False
    try:
        filename, _txt, _exif = save_image(Image.new('RGB', (8, 8)), path=folder, forced_filename='probe', extension='png')
    finally:
        shared.opts.data['save_to_dirs'] = previous
    assert os.path.dirname(filename) == folder, f'saved to "{filename}", configured "{folder}"'
    assert os.path.isfile(filename), f'missing "{filename}"'


def test_save_image_sanitizes_generated_dir():
    from modules.image.save import save_image
    folder = os.path.join(work_dir, 'dirs_')
    filename, _txt, _exif = save_image(Image.new('RGB', (8, 8)), path=folder, forced_filename='probe', extension='png', save_to_dirs='cat.')
    assert os.path.dirname(filename) == os.path.join(folder, 'cat'), f'saved to "{filename}"'
    assert os.path.isfile(filename), f'missing "{filename}"'


# ============================================================
# apply
# ============================================================

def test_generated_names_carry_no_colon():
    """Literal pattern text and token values both reach the name; a letter and a colon at the start of a component is a drive on Windows."""
    generator = FilenameGenerator(stub_p(), seed=1234, prompt='a cat', image=None)
    names = {pattern: generator.apply(pattern) for pattern in ('a: [seed]', 'x:[seed]/b:y', '[seed]-c:')}
    colons = {pattern: name for pattern, name in names.items() if ':' in name}
    assert not colons, f'names with a colon: {colons}'
    assert names['a: [seed]'] == 'a_ 1234', f"literal text changed beyond the colon: {names['a: [seed]']!r}"


def test_save_image_pattern_with_drive_letter():
    """On Windows a folder pattern starting "a:" dropped the output folder and saved relative to drive A; the file must land below the configured folder."""
    from modules.image.save import save_image
    folder = os.path.join(work_dir, 'drive')
    keys = ('save_to_dirs', 'directories_filename_pattern', 'samples_filename_pattern')
    previous = {k: shared.opts.data.get(k, None) for k in keys}
    shared.opts.data.update({'save_to_dirs': True, 'directories_filename_pattern': 'a: [seed]', 'samples_filename_pattern': 'b: [seed]'})
    try:
        filename, _txt, _exif = save_image(Image.new('RGB', (8, 8)), path=folder, seed=1234, prompt='a cat', extension='png')
    finally:
        shared.opts.data.update(previous)
    assert os.path.commonpath([os.path.abspath(filename), os.path.abspath(folder)]) == os.path.abspath(folder), f'saved to "{filename}", configured "{folder}"'
    assert os.path.isfile(filename), f'missing "{filename}"'


def run_all():
    log.warning('=== Output path sanitize ===')
    cat = category('sanitize')
    for fn in [
        test_base_kept_as_given,
        test_generated_part_sanitized,
        test_without_base_unchanged,
        test_path_outside_base_sanitized_whole,
    ]:
        run_test(cat, fn)
    cat = category('apply')
    run_test(cat, test_generated_names_carry_no_colon)
    cat = category('save')
    for fn in [
        test_save_image_into_folder_ending_in_underscore,
        test_save_image_sanitizes_generated_dir,
        test_save_image_pattern_with_drive_letter,
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
    try:
        ok = run_all()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
