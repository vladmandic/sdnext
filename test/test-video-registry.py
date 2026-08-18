#!/usr/bin/env python
"""
Offline unit tests for the video model registry in modules.video_models.models_def.

The registry answers two questions about a row: whether it names a loadable model, and how a
runner should wire its inputs. Both were previously recovered from display-name substrings at
each call site, which drifted.

Covers:

- the sentinel contract: the None placeholder and the dropdown separators name no model, and
  every accessor excludes them
- row uniqueness, so a duplicated entry cannot reach the dropdown twice
- ``dispatch_mode`` totality: every registered row classifies, so a new row that declares
  neither a name marker nor a mapped pipeline class fails here rather than generating as t2v
- ``dispatch_mode`` equivalence against the ladder it replaced, for every row the ladder
  classified
- the eight condition rows the ladder did not classify, and the six condition-class rows whose
  names do declare a mode

No running server required.

Usage:
    python test/test-video-registry.py
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

from modules.errors import log                       # pylint: disable=wrong-import-position
from modules.video_models import models_def          # pylint: disable=wrong-import-position


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


def loadable_rows():
    for engine, rows in models_def.models.items():
        for row in rows:
            if models_def.is_model(row):
                yield engine, row


def sentinel_rows():
    for engine, rows in models_def.models.items():
        for row in rows:
            if not models_def.is_model(row):
                yield engine, row


def old_ladder(row):
    """The name-marker ladder dispatch_mode replaced, as the oracle for equivalence.

    'unknown' stands for the branch in run() that warned and wired nothing; the api ladder
    reported those same rows as t2v.
    """
    if row.workflow is not None:
        return 'workflow'
    if 'T2V' in row.name:
        return 't2v'
    if 'I2V' in row.name:
        return 'i2v'
    if 'FLF2V' in row.name:
        return 'flf2v'
    if 'VACE' in row.name:
        return 'vace'
    if 'Animate' in row.name:
        return 'animate'
    return 'unknown'


# ============================================================
# Sentinel contract and row identity
# ============================================================

def test_registry_is_populated():
    assert len(models_def.models) > 0, 'registry failed to build'
    assert sum(1 for _ in loadable_rows()) > 50, 'registry lost most of its rows'


def test_sentinels_name_no_model():
    for engine, row in sentinel_rows():
        assert row.name == 'None' or row.name.startswith('─'), f'[{engine}] unexpected sentinel {row.name}'
        assert row.repo is None, f'[{engine}] sentinel "{row.name}" carries a repo'


def test_accessors_exclude_sentinels():
    for engine in models_def.models:
        names = models_def.model_names(engine)
        for name in names:
            assert name != 'None' and not name.startswith('─'), f'[{engine}] sentinel "{name}" listed as a model'
    for engine in models_def.engines():
        assert models_def.model_names(engine), f'[{engine}] listed as an engine with no models'


def test_find_rejects_sentinels():
    for engine, row in sentinel_rows():
        assert models_def.find(engine, row.name) is None, f'[{engine}] find resolved sentinel "{row.name}"'


def test_find_is_case_insensitive():
    engine, row = next(iter(loadable_rows()))
    assert models_def.find(engine, row.name) is row
    assert models_def.find(engine.lower(), row.name.lower()) is row
    assert models_def.find(engine.upper(), row.name.upper()) is row


def test_find_rejects_unknown_names():
    engine, _row = next(iter(loadable_rows()))
    assert models_def.find(engine, 'no such model') is None
    assert models_def.find('no such engine', 'no such model') is None
    assert models_def.find(engine, None) is None


def test_rows_are_unique_within_an_engine():
    seen = {}
    for engine, row in loadable_rows():
        key = (engine, row.name.lower())
        assert key not in seen, f'[{engine}] duplicate row "{row.name}"'
        seen[key] = row


# ============================================================
# Mode derivation
# ============================================================

def test_every_row_classifies():
    """The extensibility gate: a row declaring neither a marker nor a mapped class fails here."""
    unknown = [f'[{engine}] {row.name}' for engine, row in loadable_rows() if models_def.dispatch_mode(row) == 'unknown']
    assert not unknown, f'rows with no mode: {unknown}'


def test_mode_matches_the_ladder_it_replaced():
    """Every row the old ladder classified keeps its answer; only its blind spot changes."""
    changed = []
    for engine, row in loadable_rows():
        old = old_ladder(row)
        new = models_def.dispatch_mode(row)
        if old != 'unknown' and old != new:
            changed.append(f'[{engine}] {row.name}: {old} -> {new}')
    assert not changed, f'mode changed on rows the ladder already handled: {changed}'


def test_the_ladder_blind_spot_is_the_condition_rows():
    unclassified = [row for _engine, row in loadable_rows() if old_ladder(row) == 'unknown']
    assert len(unclassified) == 8, f'expected 8 rows the ladder missed, found {len(unclassified)}'
    for row in unclassified:
        assert models_def.dispatch_mode(row) == 'condition', f'"{row.name}" resolved as {models_def.dispatch_mode(row)}'


def test_named_modes_win_over_the_pipeline_class():
    """Six LTXConditionPipeline rows are named T2V or I2V and generate as such."""
    checked = 0
    for _engine, row in loadable_rows():
        cls = row.repo_cls if isinstance(row.repo_cls, str) else getattr(row.repo_cls, '__name__', None)
        if cls not in ('LTXConditionPipeline', 'LTX2ConditionPipeline'):
            continue
        if 'T2V' in row.name:
            assert models_def.dispatch_mode(row) == 't2v', f'"{row.name}" lost its declared mode'
            checked += 1
        elif 'I2V' in row.name:
            assert models_def.dispatch_mode(row) == 'i2v', f'"{row.name}" lost its declared mode'
            checked += 1
    assert checked == 6, f'expected 6 condition-class rows declaring a mode, found {checked}'


def test_workflow_rows_report_workflow():
    rows = [row for _engine, row in loadable_rows() if row.workflow is not None]
    assert rows, 'registry carries no workflow rows'
    for row in rows:
        assert models_def.dispatch_mode(row) == 'workflow', f'"{row.name}" resolved as {models_def.dispatch_mode(row)}'


def test_flf2v_row_reports_flf2v():
    rows = [row for _engine, row in loadable_rows() if 'FLF2V' in row.name]
    assert rows, 'registry carries no flf2v row'
    for row in rows:
        assert models_def.dispatch_mode(row) == 'flf2v', f'"{row.name}" resolved as {models_def.dispatch_mode(row)}'


def test_mode_resolves_from_a_class_object():
    """resolve_model synthesizes rows whose repo_cls is a class, not the registry's string."""
    class WanPipeline: # pylint: disable=too-few-public-methods
        pass
    row = models_def.Model(name='local folder with no markers', repo_cls=WanPipeline)
    assert models_def.dispatch_mode(row) == 't2v'


def test_mode_resolves_a_custom_pipeline():
    row = models_def.Model(name='local folder with no markers', custom='GoogleVeoVideoPipeline')
    assert models_def.dispatch_mode(row) == 't2v'


def test_missing_row_is_unknown():
    assert models_def.dispatch_mode(None) == 'unknown'
    assert models_def.dispatch_mode(models_def.Model(name='unregistered')) == 'unknown'


def test_class_table_has_no_stale_entries():
    """Every mapped class is one the registry actually uses, so the table cannot rot unnoticed."""
    registered = models_def.pipeline_classes()
    stale = [cls for cls in models_def.CLASS_MODES if cls not in registered]
    assert not stale, f'class table names pipelines the registry does not carry: {stale}'


def run_all():
    log.warning('=== sentinels and identity ===')
    cat = category('registry')
    for fn in [
        test_registry_is_populated,
        test_sentinels_name_no_model,
        test_accessors_exclude_sentinels,
        test_find_rejects_sentinels,
        test_find_is_case_insensitive,
        test_find_rejects_unknown_names,
        test_rows_are_unique_within_an_engine,
    ]:
        run_test(cat, fn)

    log.warning('=== mode derivation ===')
    cat = category('mode')
    for fn in [
        test_every_row_classifies,
        test_mode_matches_the_ladder_it_replaced,
        test_the_ladder_blind_spot_is_the_condition_rows,
        test_named_modes_win_over_the_pipeline_class,
        test_workflow_rows_report_workflow,
        test_flf2v_row_reports_flf2v,
        test_mode_resolves_from_a_class_object,
        test_mode_resolves_a_custom_pipeline,
        test_missing_row_is_unknown,
        test_class_table_has_no_stale_entries,
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
