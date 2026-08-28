#!/usr/bin/env python
"""
Offline unit tests for the attention and sparse axes of the xyz grid.

Covers:

- every [Attention] and [Sparse] axis resolves its choices and targets a registered option
- applying an axis and then leaving the grid restores shared.opts.data exactly, keys the axis
  introduced included, so a grid never leaks its last cell into the session
- the axes backed by boolean options take the string the dropdown hands them
- the sdp override axis turns a label, a plus joined pair or None into the option's list
- an override axis apply reaches the router: the rebuilt chain contains the backend it named
- the restore set covers every setting the sparse stage reads

No running server required. Nothing is moved to the accelerator, and no axis value that would
install a package is used.

Usage:
    python test/test-xyz-attention.py
"""

import os
import sys

import torch

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

stock_sdpa = torch.nn.functional.scaled_dot_product_attention # importing shared installs the configured hijacks in-process

from modules.errors import log                                    # pylint: disable=wrong-import-position
from modules import attention, shared                             # pylint: disable=wrong-import-position
from modules.attention.sparse import stage as sparse_stage        # pylint: disable=wrong-import-position
from scripts.xyz import xyz_grid_shared as xyz                    # pylint: disable=wrong-import-position
from scripts.xyz.xyz_grid_classes import axis_options             # pylint: disable=wrong-import-position


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
    try:
        ok = fn()
        record(cat, ok is not False, fn.__name__)
    except AssertionError as e:
        record(cat, False, fn.__name__, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, fn.__name__, f'exception: {e}')
        import traceback
        traceback.print_exc()


def attention_axes():
    return [axis for axis in axis_options if axis.label.startswith('[Attention]') or axis.label.startswith('[Sparse]')]


def sample_value(axis):
    """A value the axis accepts that is not the current one, avoiding backends whose prepare installs a package."""
    if axis.choices is None:
        return int(shared.opts.get(option_of(axis) or 'sparse_attention_budget') or 0) + 5
    choices = [choice for choice in axis.choices() if choice not in ['Sage attention', 'Flash attention', 'Triton Flash attention']]
    current = str(shared.opts.get(option_of(axis)) if option_of(axis) else '')
    return next((choice for choice in choices if str(choice) != current), choices[0])


def option_of(axis):
    """The option an axis writes, read back from the closure the axis factory built."""
    closure = getattr(axis.apply, '__closure__', None) or ()
    for cell in closure:
        if isinstance(cell.cell_contents, str) and cell.cell_contents in shared.opts.data_labels:
            return cell.cell_contents
    return {'[Attention] SDP override': 'sdp_overrides', '[Attention] Dispatcher': 'hf_attention'}.get(axis.label, None)


def bool_axes():
    return [axis for axis in attention_axes() if isinstance(shared.opts.get(option_of(axis) or ''), bool)]


# ============================================================
# Tests
# ============================================================

def test_axes_are_registered():
    axes = attention_axes()
    assert len(axes) >= 10, f'only {len(axes)} attention axes'
    for axis in axes:
        option = option_of(axis)
        assert option is not None, f'{axis.label} names no option'
        assert option in shared.opts.data_labels, f'{axis.label} targets unknown option {option}'
    return True


def test_axis_choices_resolve():
    for axis in attention_axes():
        if axis.choices is None:
            assert axis.type is int, f'{axis.label} has no choices and is not numeric'
            continue
        choices = axis.choices()
        assert isinstance(choices, list) and len(choices) > 0, f'{axis.label} resolved {choices}'
    return True


def test_axes_write_and_restore_exactly():
    for axis in attention_axes():
        before = dict(shared.opts.data)
        saved = xyz.save_attention()
        axis.apply(None, sample_value(axis), [])
        assert dict(shared.opts.data) != before, f'{axis.label} wrote nothing'
        xyz.restore_attention(saved)
        after = dict(shared.opts.data)
        leaked = {key for key in set(before) | set(after) if before.get(key, '<absent>') != after.get(key, '<absent>')}
        assert not leaked, f'{axis.label} leaked {sorted(leaked)}'
    return True


def test_bool_axes_coerce_the_string_the_dropdown_sends():
    axes = bool_axes()
    assert len(axes) >= 3, f'only {len(axes)} boolean axes'
    for axis in axes:
        option = option_of(axis)
        saved = xyz.save_attention()
        axis.apply(None, 'True', [])
        assert shared.opts.data[option] is True, f'{axis.label} took "True" as {shared.opts.data[option]!r}'
        axis.apply(None, 'False', [])
        assert shared.opts.data[option] is False, f'{axis.label} took "False" as {shared.opts.data[option]!r}'
        xyz.restore_attention(saved)
    return True


def test_override_axis_parses_labels():
    saved = xyz.save_attention()
    try:
        xyz.apply_attention_overrides(None, 'None', [])
        assert shared.opts.data['sdp_overrides'] == [], shared.opts.data['sdp_overrides']
        xyz.apply_attention_overrides(None, 'Flex attention', [])
        assert shared.opts.data['sdp_overrides'] == ['Flex attention'], shared.opts.data['sdp_overrides']
        xyz.apply_attention_overrides(None, 'Flex attention+SDNQ attention', [])
        assert shared.opts.data['sdp_overrides'] == ['Flex attention', 'SDNQ attention'], shared.opts.data['sdp_overrides']
    finally:
        xyz.restore_attention(saved)
    return True


def test_override_axis_rebuilds_the_chain():
    saved = xyz.save_attention()
    try:
        xyz.apply_attention_overrides(None, 'Flex attention', [])
        assert 'flex' in attention.get_plan().chain(), attention.get_plan().chain()
        xyz.apply_attention_overrides(None, 'None', [])
        assert 'flex' not in attention.get_plan().chain(), attention.get_plan().chain()
    finally:
        xyz.restore_attention(saved)
    return True


def test_dispatcher_axis_clears_on_none():
    saved = xyz.save_attention()
    try:
        xyz.apply_attention_dispatcher(None, 'native', [])
        assert shared.opts.data['hf_attention'] == 'native', shared.opts.data['hf_attention']
        xyz.apply_attention_dispatcher(None, 'None', [])
        assert shared.opts.data['hf_attention'] == '', repr(shared.opts.data['hf_attention'])
    finally:
        xyz.restore_attention(saved)
    return True


def test_restore_set_covers_every_attention_setting():
    covered = set(xyz.attention_options())
    missing = set(sparse_stage.OPTION_NAMES) - covered
    assert not missing, f'sparse settings outside the restore set: {sorted(missing)}'
    for backend in attention.registry.backends.values():
        outside = set(backend.options) - covered
        assert not outside, f'{backend.label} settings outside the restore set: {sorted(outside)}'
    return True


def run_all():
    log.warning('=== xyz attention axes ===')
    cat = category('axes')
    for fn in [
        test_axes_are_registered,
        test_axis_choices_resolve,
    ]:
        run_test(cat, fn)

    log.warning('=== apply and restore ===')
    cat = category('apply')
    for fn in [
        test_axes_write_and_restore_exactly,
        test_bool_axes_coerce_the_string_the_dropdown_sends,
        test_override_axis_parses_labels,
        test_override_axis_rebuilds_the_chain,
        test_dispatcher_axis_clears_on_none,
        test_restore_set_covers_every_attention_setting,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat_name, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = run_all()
    torch.nn.functional.scaled_dot_product_attention = stock_sdpa
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
