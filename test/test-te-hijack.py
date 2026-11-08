#!/usr/bin/env python
"""
Offline tests for the text-encoder hijack (``sd_hijack_te.hijack_encode_prompt``).

The hijack wraps a pipeline's ``encode_prompt`` with the prompt cache, the encode hooks and the state job. An encode
failure has no embeddings to hand back, so it reaches the caller.

Covers:

- an encode that raises: the caller gets the exception, the "TE Encode" job is ended
- an encode that succeeds: the caller gets its embeddings, the job is ended
- ``pad_to_same_length`` on a pipeline with the SD 1.5 ``encode_prompt`` signature: its ``TypeError`` probe reaches it and
  the positional retry pads the shorter embedding
- the prompt cache: a size argument beside the prompt is part of the key, images under any argument name leave the call
  uncached, an identical call is served from the cache

The pipeline is a stub installed as ``shared.sd_model``. No model, no running server.

Usage:
    python test/test-te-hijack.py
"""

import os
import sys
import time

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

import torch                                                    # pylint: disable=wrong-import-position
from modules import shared, sd_hijack_te, prompt_parser_diffusers  # pylint: disable=wrong-import-position
from modules.modeldata import model_data                        # pylint: disable=wrong-import-position
from modules.errors import log                                  # pylint: disable=wrong-import-position

results: dict[str, dict] = {}


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


class FailingPipe:
    def encode_prompt(self, prompt, **kwargs):  # pylint: disable=unused-argument
        raise RuntimeError('CUDA driver error: device not ready')


class EncodingPipe:
    def __init__(self):
        self.embeds = (torch.ones(1, 77, 8), torch.zeros(1, 77, 8))

    def encode_prompt(self, prompt, **kwargs):  # pylint: disable=unused-argument
        return self.embeds


class GridPipe:
    """Ideogram's signature: the packed sequence length follows the grid passed beside the prompt."""

    def __init__(self):
        self.calls = 0

    def encode_prompt(self, prompt, grid_h, grid_w, device=None):  # pylint: disable=unused-argument
        self.calls += 1
        return (torch.zeros(1, grid_h * grid_w, 4),)


class ImagePipe:
    """Boogu's signature: the instruction is encoded together with input images under their own name."""

    def encode_prompt(self, prompt, input_images=None):  # pylint: disable=unused-argument
        return (torch.full((1, 4), float(input_images[0].getpixel((0, 0))[0])),)


class PositionalPipe:
    """The SD 1.5 signature: everything but the prompt is positional and required."""

    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):  # pylint: disable=unused-argument
        return (torch.full((1, 77, 8), 2.0), None)


def run_hijacked(pipe, call):
    """Install ``pipe`` hijacked as ``shared.sd_model`` and run ``call``; returns (result or exception, job before, job after)."""
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_te.prompt_cache.cache.clear()
    saved_model = model_data.sd_model
    saved_offload = shared.opts.data.get('diffusers_offload_mode', None)
    shared.opts.data['diffusers_offload_mode'] = 'none'
    model_data.sd_model = pipe
    job_before = shared.state.job
    try:
        try:
            outcome = call()
        except Exception as e:  # pylint: disable=broad-except
            outcome = e
        return outcome, job_before, shared.state.job
    finally:
        model_data.sd_model = saved_model
        shared.opts.data['diffusers_offload_mode'] = saved_offload
        sd_hijack_te.prompt_cache.cache.clear()


def test_encode_failure_reaches_caller():
    """Red before the fix: the hijack logged the error and returned the prompt list as the embeddings."""
    pipe = FailingPipe()
    outcome, job_before, job_after = run_hijacked(pipe, lambda: pipe.encode_prompt(prompt=['a cat']))
    assert isinstance(outcome, RuntimeError) and 'device not ready' in str(outcome), f'caller got {outcome!r}'
    assert job_after == job_before, f'TE Encode job left open: job={job_after!r}, before {job_before!r}'
    return True


def test_encode_success_returns_embeddings():
    pipe = EncodingPipe()
    outcome, job_before, job_after = run_hijacked(pipe, lambda: pipe.encode_prompt(prompt=['a cat']))
    assert outcome is pipe.embeds, f'caller got {outcome!r}'
    assert job_after == job_before, f'TE Encode job left open: job={job_after!r}, before {job_before!r}'
    return True


def test_cache_keys_on_size_arguments():
    """Red before the fix: the key was the prompt alone, so a new size got the previous size's sequence."""
    pipe = GridPipe()

    def call():
        pipe.encode_prompt(prompt=['a cat'], grid_h=64, grid_w=64)
        return pipe.encode_prompt(prompt=['a cat'], grid_h=48, grid_w=80)

    outcome, _, _ = run_hijacked(pipe, call)
    assert isinstance(outcome, tuple), f'caller got {outcome!r}'
    assert tuple(outcome[0].shape) == (1, 3840, 4), f'shape {tuple(outcome[0].shape)} for a 48x80 grid'
    return True


def test_cache_skips_image_arguments():
    """Red before the fix: only a kwarg named image disabled the cache, so a second picture got the first one's encoding."""
    from PIL import Image
    pipe = ImagePipe()

    def call():
        pipe.encode_prompt(prompt=['make it night'], input_images=[Image.new('RGB', (8, 8), (10, 0, 0))])
        return pipe.encode_prompt(prompt=['make it night'], input_images=[Image.new('RGB', (8, 8), (200, 0, 0))])

    outcome, _, _ = run_hijacked(pipe, call)
    assert isinstance(outcome, tuple), f'caller got {outcome!r}'
    assert float(outcome[0][0, 0]) == 200.0, f'second picture encoded as {float(outcome[0][0, 0])}'
    return True


def test_cache_hits_identical_call():
    pipe = GridPipe()

    def call():
        first = pipe.encode_prompt(prompt=['a cat'], grid_h=64, grid_w=64)
        second = pipe.encode_prompt(prompt=['a cat'], grid_h=64, grid_w=64)
        return first, second

    outcome, _, _ = run_hijacked(pipe, call)
    assert isinstance(outcome, tuple), f'caller got {outcome!r}'
    assert pipe.calls == 1, f'encoded {pipe.calls} times'
    assert torch.equal(outcome[0][0], outcome[1][0]), 'cached result differs'
    return True


def test_pad_signature_probe_reaches_caller():
    """Red before the fix: the hijack swallowed the TypeError, so pad_to_same_length indexed the empty prompt it got back."""
    pipe = PositionalPipe()
    embeds = [torch.ones(1, 77, 8), torch.ones(1, 154, 8)]
    outcome, _, _ = run_hijacked(pipe, lambda: prompt_parser_diffusers.pad_to_same_length(pipe, embeds, diffusers_zeros_prompt_pad=False))
    assert isinstance(outcome, list), f'pad failed: {outcome!r}'
    assert [tuple(e.shape) for e in outcome] == [(1, 154, 8), (1, 154, 8)], f'shapes {[tuple(e.shape) for e in outcome]}'
    assert torch.equal(outcome[0][:, 77:], torch.full((1, 77, 8), 2.0)), 'padding is not the empty-prompt embedding'
    return True


def run_all():
    log.warning('=== Text encoder hijack ===')
    cat = category('encode')
    for fn in [
        test_encode_failure_reaches_caller,
        test_encode_success_returns_embeddings,
        test_pad_signature_probe_reaches_caller,
    ]:
        run_test(cat, fn)
    cat = category('cache')
    for fn in [
        test_cache_keys_on_size_arguments,
        test_cache_skips_image_arguments,
        test_cache_hits_identical_call,
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
