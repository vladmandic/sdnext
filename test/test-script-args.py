#!/usr/bin/env python
"""
Offline unit tests for script argument resolution in modules.scripts_manager.

A hook is driven by a slice of the processing object's script args, taken from the range the
script recorded when its UI was built. The vector is only as long as the caller made it, so
the range and the vector can disagree:

- a truncated slice would splat fewer positionals than the hook signature takes
- a script that declares no arguments has a complete slice even when the vector is empty
- a caller-supplied override replaces the slice outright

Covers ``resolve_script_args`` over those cases, and asserts no hook runner slices the vector
on its own.

No running server required.

Usage:
    python test/test-script-args.py
"""

import os
import sys
import inspect

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

from modules.errors import log  # pylint: disable=wrong-import-position


results: dict[str, dict] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
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


def skip(cat: str, name: str, reason: str):
    results[cat]['skipped'] += 1
    results[cat]['tests'].append(('SKIP', name))
    log.warning(f'  SKIP: {name} ({reason})')


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        elif isinstance(ok, str):
            skip(cat, name, ok)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e: # pylint: disable=broad-except
        record(cat, False, name, f'exception: {type(e).__name__}: {e}')


class StubScript:
    """What resolve_script_args reads off a script: a title and the range its ui claimed."""
    def __init__(self, args_from=None, args_to=None, name='stub'):
        self.name = name
        if args_from is not None:
            self.args_from = args_from
        if args_to is not None:
            self.args_to = args_to

    def title(self):
        return self.name


def resolver():
    from modules import scripts_manager
    return scripts_manager.resolve_script_args


# ============================================================
# The declared range against the vector
# ============================================================

def test_full_vector_gives_the_declared_slice():
    resolve = resolver()
    assert resolve(StubScript(0, 3), ['a', 'b', 'c']) == ['a', 'b', 'c']
    assert resolve(StubScript(1, 3), ['a', 'b', 'c']) == ['b', 'c']


def test_longer_vector_still_gives_the_declared_slice():
    resolve = resolver()
    assert resolve(StubScript(2, 4), ['a', 'b', 'c', 'd', 'e']) == ['c', 'd']


def test_truncated_slice_is_refused():
    resolve = resolver()
    assert resolve(StubScript(2, 5), ['a', 'b', 'c', 'd']) is None


def test_empty_vector_against_declared_args_is_refused():
    resolve = resolver()
    assert resolve(StubScript(0, 3), []) is None
    assert resolve(StubScript(0, 3), ()) is None


def test_script_declaring_no_args_still_runs():
    """The case a plain empty-slice rule would break: nothing to fill, so nothing is missing."""
    resolve = resolver()
    assert resolve(StubScript(5, 5), []) == []
    assert resolve(StubScript(5, 5), ['a', 'b']) == []


def test_uninitialized_range_is_refused():
    resolve = resolver()
    assert resolve(StubScript(0, 0), ['a', 'b']) is None


def test_inverted_range_is_refused():
    resolve = resolver()
    assert resolve(StubScript(5, 3), ['a', 'b', 'c', 'd', 'e', 'f']) is None


def test_missing_range_attributes_are_refused():
    resolve = resolver()
    assert resolve(StubScript(), ['a', 'b']) is None
    assert resolve(StubScript(args_from=0), ['a', 'b']) is None
    assert resolve(StubScript(args_to=2), ['a', 'b']) is None


# ============================================================
# Caller-supplied overrides
# ============================================================

def test_override_wins_over_the_slice():
    resolve = resolver()
    script = StubScript(0, 3, name='over')
    assert resolve(script, ['a', 'b', 'c'], {'over': ['x']}) == ['x']


def test_override_wins_even_when_the_vector_is_short():
    """An override is the whole point when the caller never built a vector."""
    resolve = resolver()
    script = StubScript(0, 3, name='over')
    assert resolve(script, [], {'over': ['x', 'y', 'z']}) == ['x', 'y', 'z']


def test_override_holding_none_falls_through():
    resolve = resolver()
    script = StubScript(0, 2, name='over')
    assert resolve(script, ['a', 'b'], {'over': None}) == ['a', 'b']


def test_override_holding_an_empty_list_is_honored():
    resolve = resolver()
    script = StubScript(0, 2, name='over')
    assert resolve(script, ['a', 'b'], {'over': []}) == []


def test_override_for_another_script_is_ignored():
    resolve = resolver()
    script = StubScript(0, 2, name='mine')
    assert resolve(script, ['a', 'b'], {'other': ['x']}) == ['a', 'b']


def test_no_override_table_falls_through():
    resolve = resolver()
    script = StubScript(0, 2)
    assert resolve(script, ['a', 'b'], None) == ['a', 'b']
    assert resolve(script, ['a', 'b'], {}) == ['a', 'b']


# ============================================================
# The runners route through the resolver
# ============================================================

def test_no_hook_runner_slices_the_vector_itself():
    from modules import scripts_manager
    source = inspect.getsource(scripts_manager.ScriptRunner)
    assert 'p.script_args[' not in source, 'a hook runner slices the vector instead of resolving it'
    assert 'args[script.args_from:script.args_to]' not in source, 'a runner slices a vector instead of resolving it'


def test_every_alwayson_runner_resolves():
    from modules import scripts_manager
    hooks = ['before_process', 'process', 'process_images', 'before_process_batch', 'process_batch',
             'postprocess', 'postprocess_batch', 'postprocess_batch_list', 'postprocess_image']
    for hook in hooks:
        fn = getattr(scripts_manager.ScriptRunner, hook, None)
        assert fn is not None, f'{hook} is missing'
        source = inspect.getsource(fn)
        assert 'resolve_script_args' in source, f'{hook} does not resolve its args'


def test_selectable_paths_resolve():
    from modules import scripts_manager
    for hook in ['run', 'after']:
        source = inspect.getsource(getattr(scripts_manager.ScriptRunner, hook))
        assert 'resolve_script_args' in source, f'{hook} does not resolve its args'


def run_all():
    log.warning('=== declared range ===')
    cat = category('range')
    for fn in [
        test_full_vector_gives_the_declared_slice,
        test_longer_vector_still_gives_the_declared_slice,
        test_truncated_slice_is_refused,
        test_empty_vector_against_declared_args_is_refused,
        test_script_declaring_no_args_still_runs,
        test_uninitialized_range_is_refused,
        test_inverted_range_is_refused,
        test_missing_range_attributes_are_refused,
    ]:
        run_test(cat, fn)

    log.warning('=== overrides ===')
    cat = category('override')
    for fn in [
        test_override_wins_over_the_slice,
        test_override_wins_even_when_the_vector_is_short,
        test_override_holding_none_falls_through,
        test_override_holding_an_empty_list_is_honored,
        test_override_for_another_script_is_ignored,
        test_no_override_table_falls_through,
    ]:
        run_test(cat, fn)

    log.warning('=== runners ===')
    cat = category('runners')
    for fn in [
        test_no_hook_runner_slices_the_vector_itself,
        test_every_alwayson_runner_resolves,
        test_selectable_paths_resolve,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
        total_skipped += info['skipped']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    success = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if success else 1)
