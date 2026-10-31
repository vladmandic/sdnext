#!/usr/bin/env python
"""
Offline tests for the component check in ``modular_load.load_modular_pipe``.

``load_components`` builds each component in its own try/except and leaves a failed one unset, so the loader compares
the result against the components the workflow declares.

Covers:

- a component the workflow declares that did not load: the load returns None and logs the missing list at ERROR
- every declared component loaded: the load returns the pipe
- a component only another workflow declares: unset by design, the load returns the pipe

The pipeline class is a stub whose ``load_components`` sets the components a case lists. No downloads, no running server.

Usage:
    python test/test-modular-load.py
"""

import os
import sys
import time
import logging
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

from modules import modular_load        # pylint: disable=wrong-import-position
from modules.errors import log          # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
workflows = {'t2v': ['text_encoder', 'transformer', 'vae'], 'ref2v': ['text_encoder', 'transformer_ref', 'vae']}


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
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


class StubBlocks:
    def __init__(self, workflow: str | None = None):
        names = workflows[workflow] if workflow else sorted({n for v in workflows.values() for n in v})
        self.expected_components = [SimpleNamespace(name=n) for n in names]

    def get_workflow(self, workflow: str):
        return StubBlocks(workflow)


def stub_pipeline(loads: list[str]):
    """A modular pipeline class whose ``load_components`` sets only ``loads``, as one that failed to build the rest."""
    all_names = sorted({n for v in workflows.values() for n in v})

    class StubModularPipeline:
        def __init__(self, blocks=None, **kwargs):  # pylint: disable=unused-argument
            self._blocks = blocks
            self._component_specs = {}
            for name in all_names:
                setattr(self, name, None)

        def update_components(self, **kwargs):
            for name, component in kwargs.items():
                setattr(self, name, component)

        def load_components(self, **kwargs):  # pylint: disable=unused-argument
            for name in loads:
                setattr(self, name, SimpleNamespace(name=name))

        @property
        def components(self):
            return {name: getattr(self, name) for name in all_names}

    return StubModularPipeline


def load(loads: list[str], workflow: str):
    return modular_load.load_modular_pipe(stub_pipeline(loads), 'owner/repo', workflow=workflow, blocks=StubBlocks())


def test_missing_component_refuses_pipe():
    """Red before the fix: the missing list was logged and the pipe returned, so the video tab cached it by name."""
    with capture_log() as cap:
        pipe = load(['text_encoder', 'transformer'], 't2v')
    errors = [m for m in cap.messages(logging.ERROR) if m.startswith('Load modular:') and "missing=['vae']" in m]
    assert len(errors) == 1, f'expected one missing-component error, got {cap.messages(logging.ERROR)}'
    assert pipe is None, f'pipe returned with vae={getattr(pipe, "vae", "?")}'
    return True


def test_complete_pipe_loads():
    with capture_log() as cap:
        pipe = load(['text_encoder', 'transformer', 'vae'], 't2v')
    assert pipe is not None, f'complete pipe refused: {cap.messages(logging.ERROR)}'
    assert not cap.messages(logging.ERROR), f'errors on a complete load: {cap.messages(logging.ERROR)}'
    return True


def test_other_workflow_component_stays_unset():
    """transformer_ref belongs to ref2v only, so a t2v load without it is complete."""
    with capture_log() as cap:
        pipe = load(['text_encoder', 'transformer', 'vae'], 't2v')
        assert pipe is not None and pipe.transformer_ref is None, 'fixture: t2v load should leave transformer_ref unset'
        pipe_ref = load(['text_encoder', 'transformer_ref', 'vae'], 'ref2v')
    assert pipe_ref is not None and pipe_ref.transformer is None, f'ref2v load refused: {cap.messages(logging.ERROR)}'
    assert not cap.messages(logging.ERROR), f'errors: {cap.messages(logging.ERROR)}'
    return True


def run_all():
    log.warning('=== Modular load component check ===')
    cat = category('components')
    for fn in [
        test_missing_component_refuses_pipe,
        test_complete_pipe_loads,
        test_other_workflow_component_stays_unset,
    ]:
        run_test(cat, fn)

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
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
