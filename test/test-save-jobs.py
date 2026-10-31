#!/usr/bin/env python
"""
Offline unit tests for saves running inside a job.

Image, video and model saves record History entries under their own id through ``State.record``
and leave the running job as it is: its id, name, counters, flags and results, which the progress
endpoints and API clients read while the job runs.

Covers:

- ``State.record`` entries: begin, output and end under one id, with the written paths
- an image save, a Video tab save and a legacy video save inside a running job
- a threaded legacy video save that finishes after the next job began
- ``save_model`` called inside a caller's job
- ``State.begin`` resetting ``interrupted`` for every title
- an image save that fails leaving the save thread running for the next save

No running server required.

Usage:
    python test/test-save-jobs.py
"""

import os
import sys
import time
import shutil
import tempfile
import threading

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
from modules import shared, devices, paths                           # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position

devices.torch_gc = lambda *args, **kwargs: None  # begin/end call it; nothing here holds device memory

results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-save-jobs-')
state = shared.state


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
        record(cat, 'FAIL', name, f'exception: {e}')
    finally:
        if state.job != '':
            state.end()


def running_job(task_id: str):
    """A job as a generation leaves it mid-run: counters advanced, an output registered, flags set."""
    state.begin('Generate', task_id=task_id, api=True)
    state.job_count = 4
    state.job_no = 1
    state.interrupted = True
    state.outputs(os.path.join(work_dir, f'{task_id}-earlier.png'))


def snapshot() -> dict:
    return {
        'id': state.id,
        'job': state.job,
        'job_count': state.job_count,
        'job_no': state.job_no,
        'job_timestamp': state.job_timestamp,
        'api': state.api,
        'interrupted': state.interrupted,
        'results': list(state.results),
    }


def entries_since(mark: int, title: str) -> list[dict]:
    return [h for h in state.state_history[mark:] if h['job'] == title.lower()]


def frames(count: int = 4, size: int = 64) -> list[Image.Image]:
    return [Image.new('RGB', (size, size), (40 * i, 80, 160)) for i in range(count)]


def av_reason():
    from modules.video_models.video_utils import check_av
    return None if check_av() is not None else 'av not installed'


# ============================================================
# state
# ============================================================

def test_record_writes_its_own_entries():
    running_job('record')
    before = snapshot()
    mark = len(state.state_history)
    saving = state.record('Save thing')
    saving.outputs.append('/out/a.png')
    saving.end()
    assert snapshot() == before, f'running job changed: {before} -> {snapshot()}'
    entries = entries_since(mark, 'Save thing')
    assert [e['op'] for e in entries] == ['begin', 'output', 'end'], f'ops {[e["op"] for e in entries]}'
    assert len({e['id'] for e in entries}) == 1 and entries[0]['id'] != 'record', f'ids {[e["id"] for e in entries]}'
    assert entries[1]['outputs'] == ['/out/a.png'], f'outputs {entries[1]["outputs"]}'
    assert entries[2]['duration'] is not None and entries[2]['timestamp'] >= entries[0]['timestamp']


def test_begin_resets_interrupted_for_every_title():
    state.interrupted = True
    state.begin('Save')
    assert state.interrupted is False, 'interrupted kept by a job titled Save'


# ============================================================
# saves inside a running job
# ============================================================

def test_image_save_leaves_running_job():
    from modules.image.save import save_image
    previous = shared.opts.data.get('save_to_dirs', None)
    shared.opts.data['save_to_dirs'] = False
    running_job('image')
    before = snapshot()
    mark = len(state.state_history)
    try:
        filename, _txt, _exif = save_image(Image.new('RGB', (8, 8)), path=work_dir, forced_filename='image', extension='png')
    finally:
        shared.opts.data['save_to_dirs'] = previous
    assert snapshot() == before, f'running job changed: {before} -> {snapshot()}'
    outputs = [o for e in entries_since(mark, 'Save image') for o in e['outputs']]
    assert outputs == [filename], f'recorded {outputs}, saved "{filename}"'
    assert os.path.isfile(outputs[0]), f'recorded path is not the file: "{outputs[0]}"'


def test_video_tab_save_leaves_running_job():
    if av_reason():
        return av_reason()
    import torch
    from modules.video_models.video_save import atomic_save_video
    fn = os.path.join(work_dir, 'tab.mp4')
    running_job('tab')
    before = snapshot()
    mark = len(state.state_history)
    ok = atomic_save_video(fn, torch.zeros((4, 64, 64, 3), dtype=torch.uint8), fps=8)
    assert ok, 'save failed'
    assert snapshot() == before, f'running job changed: {before} -> {snapshot()}'
    outputs = [o for e in entries_since(mark, 'Save video') for o in e['outputs']]
    assert outputs == [fn], f'recorded {outputs}'


def test_legacy_save_registers_with_running_job():
    if av_reason():
        return av_reason()
    from modules import video
    fn = os.path.join(work_dir, 'legacy.mp4')
    running_job('legacy')
    before = snapshot()
    result = video.save_video(None, frames(), filename=fn, video_type='MP4/AVC1', duration=0.5, sync=True) # named by the output pattern
    assert result is not None and os.path.isfile(result), f'save returned {result}'
    after = snapshot()
    assert after['results'] == before['results'] + [result], f'results {after["results"]}'
    after['results'] = before['results']
    assert after == before, f'running job changed: {before} -> {after}'


def test_threaded_legacy_save_leaves_next_job():
    if av_reason():
        return av_reason()
    from modules import video
    from modules.video_models import video_utils
    fn = os.path.join(work_dir, 'threaded.mp4')
    entered, release = threading.Event(), threading.Event()
    check_av = video_utils.check_av

    def held_check_av():
        entered.set()
        release.wait(30)
        return check_av()

    running_job('first')
    mark = len(state.state_history)
    video_utils.check_av = held_check_av
    try:
        fn = video.save_video(None, frames(), filename=fn, video_type='MP4/AVC1', duration=0.5)
        assert entered.wait(30), 'threaded save did not start'
        state.end('first')
        running_job('second')
        before = snapshot()
        release.set()
        deadline = time.time() + 60
        while not any(e['op'] == 'end' for e in entries_since(mark, 'Save video')) and time.time() < deadline:
            time.sleep(0.05)
        assert any(e['op'] == 'end' for e in entries_since(mark, 'Save video')), 'threaded save did not finish'
        assert snapshot() == before, f'next job changed: {before} -> {snapshot()}'
        assert os.path.isfile(fn), 'video not written'
    finally:
        video_utils.check_av = check_av
        release.set()


def test_model_save_leaves_callers_job():
    import sdnq
    from modules import modeldata, sd_models
    calls = []
    loaded = modeldata.model_data.sd_model
    save_sdnq_model = sdnq.save_sdnq_model
    modeldata.model_data.sd_model = type('Pipeline', (), {})()
    sdnq.save_sdnq_model = lambda **kwargs: calls.append(kwargs)
    running_job('model')
    before = snapshot()
    mark = len(state.state_history)
    try:
        result = sd_models.save_model('saved', path=work_dir, shard='1GB')
    finally:
        modeldata.model_data.sd_model = loaded
        sdnq.save_sdnq_model = save_sdnq_model
    assert result == f'Saved: {os.path.join(work_dir, "saved")}', f'result "{result}"'
    assert len(calls) == 1, f'save_sdnq_model calls {len(calls)}'
    assert snapshot() == before, f'running job changed: {before} -> {snapshot()}'
    assert state.state_history[mark:] == [], f'entries {state.state_history[mark:]}'


def test_failed_save_keeps_the_save_thread():
    """save_image waits on the save thread; a save that raises there must still be marked done."""
    from modules.image.save import save_image

    def save(name: str) -> tuple[bool, str | None]:
        out = {}
        worker = threading.Thread(target=lambda: out.setdefault('fn', save_image(Image.new('RGB', (8, 8)), path=work_dir, forced_filename=name, extension='png', info='probe parameters')[0]), daemon=True)
        worker.start()
        worker.join(30)
        return worker.is_alive(), out.get('fn')

    blocked = os.path.join(work_dir, 'params-blocked')
    os.makedirs(blocked)
    previous = (paths.params_path, shared.opts.data.get('save_to_dirs', None))
    paths.params_path = blocked # every save with metadata writes it; a folder makes the open fail
    shared.opts.data['save_to_dirs'] = False
    try:
        hung, _fn = save('failed')
        assert not hung, 'save_image did not return after a save that failed on the save thread'
    finally:
        paths.params_path = previous[0]
    try:
        hung, fn = save('after-failed')
    finally:
        shared.opts.data['save_to_dirs'] = previous[1]
    assert not hung, 'the save after a failed one did not return'
    assert fn is not None and os.path.isfile(fn), f'next save wrote no file: "{fn}"'


def run_all():
    log.warning('=== Save jobs ===')
    cat = category('state')
    for fn in [
        test_record_writes_its_own_entries,
        test_begin_resets_interrupted_for_every_title,
    ]:
        run_test(cat, fn)
    cat = category('saves')
    for fn in [
        test_image_save_leaves_running_job,
        test_video_tab_save_leaves_running_job,
        test_legacy_save_registers_with_running_job,
        test_threaded_legacy_save_leaves_next_job,
        test_model_save_leaves_callers_job,
        test_failed_save_keeps_the_save_thread,
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
    try:
        ok = run_all()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
