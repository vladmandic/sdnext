#!/usr/bin/env python
"""
Offline tests for the stored settings check (``Options.load`` and ``Options.validate_pending``).

Covers:

- a stored key whose option an extension adds after load is validated then, not reported as unknown
- a late option's stored value reset when it is out of range or of the wrong type, as a built-in option's is at load
- keys no option claims reported once, after extensions have added theirs
- built-in options still validated at load, compatibility and ``uiux_`` keys never reported
- every setting the Tag Autocomplete script stores registered as an option and sent to the page

No running server required.

Usage:
    python test/test-settings-validation.py
"""

import os
import sys
import json
import time
import shutil
import logging
import tempfile
import contextlib

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
from modules.options import OptionInfo                               # pylint: disable=wrong-import-position
from modules.options_handler import Options                          # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-settings-validation-', suffix='-work')


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

    def warnings(self) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]


@contextlib.contextmanager
def capture_log():
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def builtin_templates() -> dict:
    return {
        'steps': OptionInfo(20, 'Steps', None, {'minimum': 1, 'maximum': 200}),
        'name': OptionInfo('', 'Name'),
    }


def make_options(stored: dict, name: str) -> Options:
    """Options over a config file holding exactly the stored keys."""
    folder = os.path.join(work_dir, name)
    os.makedirs(folder, exist_ok=True)
    config = os.path.join(folder, 'config.json')
    with open(config, 'w', encoding='utf8') as f:
        json.dump(stored, f)
    return Options(builtin_templates(), filename=config, secrets=os.path.join(folder, 'secrets.json'))


def unknown_lines(messages: list[str]) -> list[str]:
    return [m for m in messages if 'unknown=' in m]


# --- options added after load ---

def test_extension_option_not_reported():
    """a stored key claimed by an option added after load is not reported; an unclaimed key is, after the extensions"""
    with capture_log() as cap:
        opts = make_options({'steps': 30, 'ext_folder': '/data/ext', 'stale_key': True}, 'extension')
        at_load = unknown_lines(cap.warnings())
        assert at_load == [], f'unknown keys reported before extensions add their options: {at_load}'
        opts.add_option('ext_folder', OptionInfo('', 'Extension folder', folder=True))
        opts.validate_pending()
        after = unknown_lines(cap.warnings())
    assert after == ["Setting validation: unknown=['stale_key']"], f'reported after extensions: {after}'
    assert opts.data['ext_folder'] == '/data/ext', f'extension value changed: {opts.data["ext_folder"]!r}'
    assert opts.data['stale_key'] is True, 'an unknown key is kept for the next save'


def test_late_option_value_validated():
    """a late option's stored value outside its range or of the wrong type is reset to its default"""
    with capture_log() as cap:
        opts = make_options({'ext_strength': 5.0, 'ext_count': 'many', 'ext_ok': 0.7}, 'late-values')
        opts.add_option('ext_strength', OptionInfo(0.5, 'Extension strength', None, {'minimum': 0, 'maximum': 1}))
        opts.add_option('ext_count', OptionInfo(1, 'Extension count'))
        opts.add_option('ext_ok', OptionInfo(0.5, 'Extension ratio', None, {'minimum': 0, 'maximum': 1}))
        opts.validate_pending()
        messages = cap.warnings()
    assert opts.data['ext_strength'] == 0.5, f'out of range value kept: {opts.data["ext_strength"]!r}'
    assert opts.data['ext_count'] == 1, f'wrong type kept: {opts.data["ext_count"]!r}'
    assert opts.data['ext_ok'] == 0.7, f'valid value changed: {opts.data["ext_ok"]!r}'
    assert any('ext_strength' in m and '=5.0 ' in m for m in messages), f'no range message: {messages}'
    assert any('ext_count=many (str expected=int)' in m for m in messages), f'no type warning: {messages}'
    assert unknown_lines(messages) == [], f'late options reported unknown: {messages}'


def test_pending_checked_once():
    """a second pass (a UI rebuild in the same process) reports nothing again"""
    with capture_log() as cap:
        opts = make_options({'stale_key': 1}, 'once')
        opts.validate_pending()
        first = len(unknown_lines(cap.warnings()))
        opts.validate_pending()
        second = len(unknown_lines(cap.warnings()))
    assert (first, second) == (1, 1), f'unknown warnings after first and second pass: {first}, {second}'


# --- unchanged at load ---

def test_builtin_validated_at_load():
    """a built-in option's stored value is checked at load, before any extension runs"""
    with capture_log() as cap:
        opts = make_options({'steps': 500, 'name': 5}, 'builtin')
        messages = cap.warnings()
    assert opts.data['steps'] == 20, f'out of range value kept: {opts.data["steps"]!r}'
    assert opts.data['name'] == '', f'wrong type kept: {opts.data["name"]!r}'
    assert any('steps' in m and '=500 ' in m for m in messages), f'no range message: {messages}'
    assert any('name=5 (int expected=str)' in m for m in messages), f'no type warning: {messages}'


def test_compatibility_keys_not_reported():
    """compatibility keys and uiux_ keys have no option and are never reported"""
    with capture_log() as cap:
        opts = make_options({'clip_skip': 2, 'uiux_theme': 'dark'}, 'compat')
        opts.validate_pending()
        messages = unknown_lines(cap.warnings())
    assert messages == [], f'reported: {messages}'
    assert opts.data['clip_skip'] == 2 and opts.data['uiux_theme'] == 'dark', 'kept keys changed'


# --- settings stored outside an option ---

def test_autocomplete_settings_registered():
    """every setting the Tag Autocomplete script stores is an option, so the page's settings carry its stored value"""
    import re
    from modules import shared
    with open(os.path.join(script_dir, 'scripts', 'autocomplete.py'), encoding='utf8') as f:
        keys = sorted(set(re.findall(r"opts\.data(?:\.get\(|\[)'(autocomplete_\w+)'", f.read())))
    assert len(keys) >= 7, f'script keys not found: {keys}'
    templates = dict(shared.options_templates)
    missing = [k for k in keys if k not in templates]
    assert missing == [], f'stored without an option: {missing}'
    folder = os.path.join(work_dir, 'autocomplete')
    os.makedirs(folder, exist_ok=True)
    config = os.path.join(folder, 'config.json')
    stored = {k: not templates[k].default for k in keys if isinstance(templates[k].default, bool)}
    with open(config, 'w', encoding='utf8') as f:
        json.dump(stored, f)
    with capture_log() as cap:
        opts = Options(templates, filename=config, secrets=os.path.join(folder, 'secrets.json'))
        opts.validate_pending()
        messages = unknown_lines(cap.warnings())
    assert messages == [], f'reported: {messages}'
    values = json.loads(opts.dumpjson())['values']
    wrong = {k: values.get(k) for k, v in stored.items() if values.get(k) != v}
    assert wrong == {}, f'page settings differ from the stored values: {wrong}'


def run_all():
    log.warning('=== Settings validation ===')
    cat = category('late options')
    for fn in [
        test_extension_option_not_reported,
        test_late_option_value_validated,
        test_pending_checked_once,
    ]:
        run_test(cat, fn)
    cat = category('load')
    for fn in [
        test_builtin_validated_at_load,
        test_compatibility_keys_not_reported,
    ]:
        run_test(cat, fn)
    cat = category('registered')
    run_test(cat, test_autocomplete_settings_registered)

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
