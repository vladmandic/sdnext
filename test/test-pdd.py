#!/usr/bin/env python
"""
Offline unit tests for modules.lora.network_pdd.

Checks the parallel decoding head mechanics against the reference formulas
shipped with alibaba-pai/MiniMax-H3-Acc-LoRAs (minimax_h3_pdd.py):

- ``detect`` / ``load`` metadata and head-tensor discovery
- ``grid_intervals`` against ``pdd_time_grid`` for the video and audio shifts
- ``ParallelHead`` against ``MiniMaxH3ParallelHead`` on every block, plus the strength blend
- ``install`` / ``restore`` / ``reconcile`` round trips on a stub pipeline
- ``pin`` step and shift override

No running server required.

Usage:
    python test/test-pdd.py
"""

import os
import sys
import json
import types

import torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
_orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = _orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

import diffusers  # pylint: disable=wrong-import-position
from modules import shared  # pylint: disable=wrong-import-position,unused-import # shared must initialize before sd_models, which imports back into it
from modules.errors import log  # pylint: disable=wrong-import-position
from modules.lora import network_pdd  # pylint: disable=wrong-import-position
from pipelines.minimax import minimax_lora  # pylint: disable=wrong-import-position
from pipelines.qwen import qwen21_lora  # pylint: disable=wrong-import-position


NUM_STEPS = 32
BLOCK = 4
HIDDEN = 16
VIDEO_OUT = 8
AUDIO_OUT = 6
METADATA = {'pdd_num_steps': '32', 'pdd_block_size': '4', 'lora_rank': '64', 'lora_alpha': '64.0'}


# ============================================================
# Reference implementation (minimax_h3_pdd.py)
# ============================================================

def reference_time_grid(shift, num_steps):
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shift * sigma / (1 + (shift - 1) * sigma)


def reference_plan(step_sizes, start, block_size):
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype)
    span = step_sizes[start:start + block_size].sum()
    plan[0, start:start + block_size] = step_sizes[start:start + block_size] / span
    return plan


class ReferenceHead(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.num_steps = weight.shape[0]
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = torch.nn.Parameter(bias.clone())
        self.plan = torch.zeros(1, self.num_steps)

    def forward(self, hidden_states):
        plan = self.plan.to(device=self.weight.device, dtype=self.weight.dtype)
        weight = torch.einsum('pn,noi->poi', plan, self.weight).flatten(0, 1)
        bias = torch.einsum('pn,no->po', plan, self.bias).flatten()
        return torch.nn.functional.linear(hidden_states, weight, bias)


# ============================================================
# Test infrastructure
# ============================================================

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
        record(cat, ok is not False, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'{type(e).__name__}: {e}')


# ============================================================
# Fixtures
# ============================================================

class StubPipe:
    """The parts of a modular pipeline the engine touches: components and two schedulers."""

    def __init__(self):
        self.transformer = torch.nn.Module()
        self.transformer.proj_out = torch.nn.Linear(HIDDEN, VIDEO_OUT)
        self.transformer.audio_proj_out = torch.nn.Linear(HIDDEN, AUDIO_OUT)
        self.scheduler = diffusers.MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = diffusers.MiniMaxH3Scheduler(shift=3.0)
        self.num_timesteps = 29


def make_heads():
    torch.manual_seed(0)
    return network_pdd.ParallelHeads(NUM_STEPS, BLOCK, {
        'proj_out': (torch.randn(NUM_STEPS, VIDEO_OUT, HIDDEN), torch.randn(NUM_STEPS, VIDEO_OUT)),
        'audio_proj_out': (torch.randn(NUM_STEPS, AUDIO_OUT, HIDDEN), torch.randn(NUM_STEPS, AUDIO_OUT)),
    })


def make_net(heads, strength=1.0):
    return types.SimpleNamespace(name='pdd-test', te_multiplier=strength, extras={network_pdd.EXTRAS_KEY: heads})


# Qwen-Image-2.1-Fun-Acc-4Step: an inference-only export with one head per step, its own sigma grid and replaced norm weights.
QWEN_SIGMAS = [1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0]
QWEN_SCHEDULER = dict(use_dynamic_shifting=True, base_shift=0.5, max_shift=0.9, base_image_seq_len=256, max_image_seq_len=8192, shift_terminal=0.02, time_shift_type='exponential')


class StubQwenPipe:
    """A single-stream model: one bias-free output projection, a norm the export replaces, a dynamic-shift flow scheduler,
    and the read-only num_timesteps property diffusers pipelines expose."""

    def __init__(self):
        self.transformer = torch.nn.Module()
        self.transformer.proj_out = torch.nn.Linear(HIDDEN, VIDEO_OUT, bias=False)
        self.transformer.block = torch.nn.Module()
        self.transformer.block.norm_q = torch.nn.RMSNorm(8)
        self.scheduler = diffusers.FlowMatchEulerDiscreteScheduler(**QWEN_SCHEDULER)
        self._num_timesteps = 40

    @property
    def num_timesteps(self):
        return self._num_timesteps


def qwen_state_dict():
    torch.manual_seed(1)
    return {
        'proj_out.weight': torch.randn(4, VIDEO_OUT, HIDDEN),
        'block.norm_q.weight': torch.full((8,), 2.0),
        'block.attn.to_q.lora_down': torch.zeros(4, HIDDEN),
        'block.attn.to_q.lora_up': torch.zeros(HIDDEN, 4),
        'block.attn.to_k.lora_A.weight': torch.zeros(4, HIDDEN),
        'block.attn.to_k.lora_B.weight': torch.zeros(HIDDEN, 4),
        'block.attn.to_k.alpha': torch.tensor(4.0),
    }


def qwen_heads():
    return network_pdd.load('acc', {'pdd_sigmas': json.dumps(QWEN_SIGMAS)}, qwen_state_dict())


# ============================================================
# Tests: detection and loading
# ============================================================

def test_detect_grid():
    assert network_pdd.detect(METADATA) == (32, 4)
    assert network_pdd.detect({'pdd_num_steps': '16'}) == (16, 1)
    assert network_pdd.detect({}) is None
    assert network_pdd.detect(None) is None


def test_detect_rejects_bad_block():
    try:
        network_pdd.detect({'pdd_num_steps': '32', 'pdd_block_size': '5'})
    except ValueError:
        return True
    raise AssertionError('block size 5 accepted for a 32 grid')


def test_load_collects_heads():
    state_dict = {
        'proj_out.weight': torch.zeros(NUM_STEPS, VIDEO_OUT, HIDDEN),
        'proj_out.bias': torch.zeros(NUM_STEPS, VIDEO_OUT),
        'audio_proj_out.weight': torch.zeros(NUM_STEPS, AUDIO_OUT, HIDDEN),
        'audio_proj_out.bias': torch.zeros(NUM_STEPS, AUDIO_OUT),
        'transformer_blocks.0.attn.to_q.lora_down': torch.zeros(64, HIDDEN),
        'transformer_blocks.0.attn.to_q.lora_up': torch.zeros(HIDDEN, 64),
    }
    heads = network_pdd.load('x', METADATA, state_dict)
    assert heads is not None and set(heads.heads) == {'proj_out', 'audio_proj_out'}, f'heads={None if heads is None else list(heads.heads)}'
    assert heads.nfe == 8 and heads.block_size == 4
    assert heads.heads['proj_out'][1] is not None, 'bias not paired'


def test_load_without_metadata_is_none():
    assert network_pdd.load('x', {'alpha': '1'}, {'proj_out.weight': torch.zeros(NUM_STEPS, VIDEO_OUT, HIDDEN)}) is None


def test_load_without_heads_is_none():
    assert network_pdd.load('x', METADATA, {'transformer_blocks.0.attn.to_q.lora_down': torch.zeros(64, HIDDEN)}) is None


def write_pdd_file(folder, metadata):
    from safetensors.torch import save_file
    path = os.path.join(folder, 'pdd.safetensors')
    save_file({'proj_out.weight': torch.zeros(NUM_STEPS, VIDEO_OUT, HIDDEN), 'proj_out.bias': torch.zeros(NUM_STEPS, VIDEO_OUT)}, path, metadata=metadata)
    return path


def test_try_load_reads_the_grid_from_the_file_header():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        disk = types.SimpleNamespace(name='pdd', filename=write_pdd_file(folder, METADATA), metadata={}) # the cached metadata lacks the grid
        net = network_pdd.try_load('pdd', disk, 1.0)
        assert net is not None and network_pdd.EXTRAS_KEY in net.extras, 'the grid in the file header must be enough'
        assert net.extras[network_pdd.EXTRAS_KEY].nfe == NUM_STEPS // BLOCK


def test_try_load_without_grid_metadata_is_none():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        disk = types.SimpleNamespace(name='pdd', filename=write_pdd_file(folder, {'alpha': '1'}), metadata={})
        assert network_pdd.try_load('pdd', disk, 1.0) is None


# ============================================================
# Tests: grid and fusion math
# ============================================================

def test_grid_intervals_match_reference():
    for shift in (12.0, 3.0):
        scheduler = diffusers.MiniMaxH3Scheduler(shift=shift)
        scheduler.set_shift(4.0) # a live override must not leak into the training grid
        intervals = network_pdd.grid_intervals(scheduler, NUM_STEPS, minimax_lora.PDD)
        reference = reference_time_grid(shift, NUM_STEPS).diff()
        assert intervals is not None and intervals.shape == reference.shape, f'shift={shift} shape={None if intervals is None else intervals.shape}'
        assert torch.allclose(intervals, reference, atol=1e-6), f'shift={shift} maxdiff={(intervals - reference).abs().max().item()}'


def test_steps_for_counts_terminal_sigma():
    assert minimax_lora.PDD.steps_for(8) == 9
    assert minimax_lora.PDD.scheduler_name('audio_proj_out') == 'audio_scheduler'
    assert minimax_lora.PDD.scheduler_name('proj_out') == 'scheduler'


def test_parallel_head_matches_reference_per_block():
    heads = make_heads()
    weight, bias = heads.heads['proj_out']
    base = torch.nn.Linear(HIDDEN, VIDEO_OUT)
    scheduler = diffusers.MiniMaxH3Scheduler(shift=12.0)
    intervals = network_pdd.grid_intervals(scheduler, NUM_STEPS, minimax_lora.PDD)
    stub = types.SimpleNamespace(step_index=None)
    head = network_pdd.ParallelHead(base, weight, bias, 1.0, lambda: stub, BLOCK, intervals)
    reference = ReferenceHead(weight, bias)
    step_sizes = reference_time_grid(12.0, NUM_STEPS).diff()
    x = torch.randn(2, 5, HIDDEN)
    for index in range(NUM_STEPS // BLOCK):
        stub.step_index = None if index == 0 else index
        reference.plan = reference_plan(step_sizes, index * BLOCK, BLOCK).float()
        out = head(x)
        ref = reference(x)
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4), f'block={index} maxdiff={(out - ref).abs().max().item()}' # float32 reduction order differs between tensordot and einsum
        assert head.fused_index == index


def test_parallel_head_strength_blend():
    heads = make_heads()
    weight, bias = heads.heads['proj_out']
    base = torch.nn.Linear(HIDDEN, VIDEO_OUT)
    intervals = network_pdd.grid_intervals(diffusers.MiniMaxH3Scheduler(shift=12.0), NUM_STEPS, minimax_lora.PDD)
    stub = types.SimpleNamespace(step_index=3)
    x = torch.randn(3, HIDDEN)
    full = network_pdd.ParallelHead(base, weight, bias, 1.0, lambda: stub, BLOCK, intervals)(x)
    off = network_pdd.ParallelHead(base, weight, bias, 0.0, lambda: stub, BLOCK, intervals)(x)
    half = network_pdd.ParallelHead(base, weight, bias, 0.5, lambda: stub, BLOCK, intervals)(x)
    assert torch.allclose(off, base(x), atol=1e-6), 'strength 0 is not the base projection'
    assert torch.allclose(half, 0.5 * (full + base(x)), atol=1e-5), 'strength 0.5 is not the midpoint'


def test_parallel_head_clamps_overflow():
    heads = make_heads()
    weight, bias = heads.heads['proj_out']
    intervals = network_pdd.grid_intervals(diffusers.MiniMaxH3Scheduler(shift=12.0), NUM_STEPS, minimax_lora.PDD)
    stub = types.SimpleNamespace(step_index=11)
    head = network_pdd.ParallelHead(torch.nn.Linear(HIDDEN, VIDEO_OUT), weight, bias, 1.0, lambda: stub, BLOCK, intervals)
    head(torch.randn(1, HIDDEN))
    assert head.fused_index == 7 and head.overflow_warned


def test_parallel_head_keeps_base_out_of_tree():
    heads = make_heads()
    weight, bias = heads.heads['proj_out']
    intervals = network_pdd.grid_intervals(diffusers.MiniMaxH3Scheduler(shift=12.0), NUM_STEPS, minimax_lora.PDD)
    head = network_pdd.ParallelHead(torch.nn.Linear(HIDDEN, VIDEO_OUT), weight, bias, 1.0, lambda: None, BLOCK, intervals)
    assert set(dict(head.named_parameters())) == {'weight', 'bias'}, list(dict(head.named_parameters()))
    assert len(list(head.children())) == 0
    assert next(head.parameters()).dtype == torch.float32


# ============================================================
# Tests: install, restore, reconcile, pin
# ============================================================

def test_install_and_restore_round_trip():
    pipe = StubPipe()
    original_video, original_audio = pipe.transformer.proj_out, pipe.transformer.audio_proj_out
    before = original_video.weight.detach().clone()
    before_ptr = original_video.weight.data_ptr()
    heads = make_heads()
    assert network_pdd.install(pipe, make_net(heads), heads, minimax_lora.PDD, ['unet', 'transformer']) is True
    assert isinstance(pipe.transformer.proj_out, network_pdd.ParallelHead)
    assert isinstance(pipe.transformer.audio_proj_out, network_pdd.ParallelHead)
    assert original_video.weight.data_ptr() != before_ptr and torch.equal(original_video.weight, before), 'the stashed projection must own its memory and keep its values'
    assert pipe.sdnext_pdd.steps == 9 and pipe.sdnext_pdd.component == 'transformer'
    pipe.audio_scheduler.set_shift(3.0)
    pipe.scheduler.set_timesteps(9)
    pipe.audio_scheduler.set_timesteps(9)
    out = pipe.transformer.audio_proj_out(torch.randn(2, HIDDEN))
    assert out.shape == (2, AUDIO_OUT)
    assert network_pdd.restore(pipe) is True
    assert pipe.transformer.proj_out is original_video and pipe.transformer.audio_proj_out is original_audio
    assert not hasattr(pipe, 'sdnext_pdd')
    assert network_pdd.restore(pipe) is False


def test_install_refuses_shape_mismatch():
    pipe = StubPipe()
    original = pipe.transformer.proj_out
    heads = make_heads()
    heads.heads['audio_proj_out'] = (torch.randn(NUM_STEPS, AUDIO_OUT + 1, HIDDEN), None)
    assert network_pdd.install(pipe, make_net(heads), heads, minimax_lora.PDD, ['transformer']) is False
    assert pipe.transformer.proj_out is original, 'partial install left a head behind'
    assert not hasattr(pipe, 'sdnext_pdd')


def test_install_needs_an_owner():
    pipe = StubPipe()
    heads = make_heads()
    heads.heads['norm_out.linear'] = (torch.randn(NUM_STEPS, VIDEO_OUT, HIDDEN), None)
    assert network_pdd.install(pipe, make_net(heads), heads, minimax_lora.PDD, ['transformer']) is False


def test_reconcile_follows_loaded_set():
    pipe = StubPipe()
    heads = make_heads()
    net = make_net(heads)
    saved = network_pdd.arch_spec
    network_pdd.arch_spec = lambda: minimax_lora.PDD
    try:
        assert network_pdd.reconcile(pipe, [net], ['transformer']) is True
        assert network_pdd.reconcile(pipe, [net], ['transformer']) is False, 'unchanged set reinstalled'
        stronger = make_net(heads, strength=0.5)
        assert network_pdd.reconcile(pipe, [stronger], ['transformer']) is True, 'strength change not applied'
        assert pipe.sdnext_pdd.strength == 0.5
        assert network_pdd.reconcile(pipe, [types.SimpleNamespace(name='plain', te_multiplier=1.0, extras={})], ['transformer']) is True
        assert not hasattr(pipe, 'sdnext_pdd')
        assert network_pdd.reconcile(pipe, [], ['transformer']) is False
    finally:
        network_pdd.arch_spec = saved


def test_pin_overrides_steps_and_shift():
    pipe = StubPipe()
    heads = make_heads()
    assert network_pdd.install(pipe, make_net(heads), heads, minimax_lora.PDD, ['transformer']) is True
    pipe.scheduler.set_shift(4.0)
    pipe.audio_scheduler.set_shift(2.0)
    p = types.SimpleNamespace(steps=30, task_args={'num_inference_steps': 30}, extra_generation_params={'Video shift': 4.0, 'Audio shift': 2.0})
    assert network_pdd.pin(p, pipe) == 9
    assert p.steps == 8 and p.task_args['num_inference_steps'] == 9, f'steps={p.steps} grid_steps={p.task_args["num_inference_steps"]}'
    assert pipe.num_timesteps == 8
    assert pipe.scheduler.shift == 12.0 and pipe.audio_scheduler.shift == 3.0
    assert p.extra_generation_params == {'Video shift': 12.0, 'Audio shift': 3.0}, f'recorded={p.extra_generation_params}'
    network_pdd.restore(pipe)
    assert network_pdd.pin(p, pipe) is None


# ============================================================
# Tests: explicit grids and replaced parameters
# ============================================================

def test_detect_grid_from_sigmas():
    metadata = {'pdd_sigmas': json.dumps(QWEN_SIGMAS)}
    assert network_pdd.detect(metadata) == (4, 1)
    sigmas = network_pdd.grid_sigmas(metadata)
    assert sigmas.dtype == torch.float64 and sigmas.tolist() == QWEN_SIGMAS
    assert network_pdd.detect({'pdd_num_steps': '4', 'pdd_sigmas': json.dumps(QWEN_SIGMAS)}) == (4, 1)


def test_detect_rejects_bad_sigmas():
    for sigmas in ([1.0, 0.5], [0.9, 0.5, 0.0], [1.0, 0.5, 0.6, 0.0]):
        try:
            network_pdd.detect({'pdd_sigmas': json.dumps(sigmas)})
        except ValueError:
            continue
        raise AssertionError(f'sigmas {sigmas} accepted')
    try:
        network_pdd.detect({'pdd_num_steps': '8', 'pdd_sigmas': json.dumps(QWEN_SIGMAS)})
    except ValueError:
        return True
    raise AssertionError('an 8-step grid accepted 5 sigmas')


def test_load_collects_replaced_parameters():
    heads = qwen_heads()
    assert heads is not None and set(heads.heads) == {'proj_out'} and heads.heads['proj_out'][1] is None
    assert set(heads.replaced) == {'block.norm_q.weight'}, f'replaced={list(heads.replaced)}'
    assert heads.nfe == 4 and heads.sigmas.tolist() == QWEN_SIGMAS


def test_try_load_reads_the_sidecar_config():
    import tempfile
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'acc.safetensors')
        save_file(qwen_state_dict(), path, metadata={'format': 'qwenimage21_extracted_prefused_v1'})
        disk = types.SimpleNamespace(name='acc', filename=path, metadata={})
        assert network_pdd.try_load('acc', disk, 1.0) is None, 'no grid anywhere must leave the heads out'
        with open(os.path.join(folder, network_pdd.SIDECAR), 'w', encoding='utf8') as f:
            json.dump({'pdd_num_steps': 4, 'pdd_block_size': 1, 'pdd_sigmas': QWEN_SIGMAS, 'lora_rank': 64}, f)
        net = network_pdd.try_load('acc', disk, 1.0)
        assert net is not None, 'the sidecar grid must be enough'
        heads = net.extras[network_pdd.EXTRAS_KEY]
        assert heads.nfe == 4 and heads.sigmas.tolist() == QWEN_SIGMAS


def test_install_replaces_and_restores_parameters():
    pipe = StubQwenPipe()
    original = pipe.transformer.block.norm_q.weight.detach().clone()
    heads = qwen_heads()
    assert network_pdd.install(pipe, make_net(heads), heads, qwen21_lora.PDD, ['transformer']) is True
    assert torch.equal(pipe.transformer.block.norm_q.weight, torch.full((8,), 2.0))
    network_pdd.restore(pipe)
    assert torch.equal(pipe.transformer.block.norm_q.weight, original)
    assert network_pdd.install(pipe, make_net(heads, strength=0.5), heads, qwen21_lora.PDD, ['transformer']) is True
    assert torch.allclose(pipe.transformer.block.norm_q.weight, original + 0.5 * (2.0 - original))
    network_pdd.restore(pipe)
    assert torch.equal(pipe.transformer.block.norm_q.weight, original)


def test_install_refuses_misshapen_replacement():
    pipe = StubQwenPipe()
    original_head = pipe.transformer.proj_out
    heads = qwen_heads()
    heads.replaced['block.norm_q.weight'] = torch.ones(16)
    assert network_pdd.install(pipe, make_net(heads), heads, qwen21_lora.PDD, ['transformer']) is False
    assert pipe.transformer.proj_out is original_head and not hasattr(pipe, 'sdnext_pdd')


def test_one_head_per_step_on_a_block1_grid():
    pipe = StubQwenPipe()
    heads = qwen_heads()
    assert network_pdd.install(pipe, make_net(heads), heads, qwen21_lora.PDD, ['transformer']) is True
    x = torch.randn(3, HIDDEN)
    for index in range(4):
        pipe.scheduler._step_index = index # pylint: disable=protected-access
        expected = torch.nn.functional.linear(x, heads.heads['proj_out'][0][index])
        assert torch.allclose(pipe.transformer.proj_out(x), expected, atol=1e-6), f'step {index} did not select its own head'
    network_pdd.restore(pipe)


def test_pin_hands_the_exact_grid():
    pipe = StubQwenPipe()
    try:
        pipe.num_timesteps = 4
    except AttributeError:
        pass
    else:
        raise AssertionError('the stub accepts num_timesteps assignment; it does not model the diffusers property')
    heads = qwen_heads()
    assert network_pdd.install(pipe, make_net(heads), heads, qwen21_lora.PDD, ['transformer']) is True
    p = types.SimpleNamespace(steps=40, task_args={}, extra_generation_params={})
    assert network_pdd.pin(p, pipe) == 4 and p.steps == 4 and p.task_args['num_inference_steps'] == 4
    assert pipe.num_timesteps == 40, 'the pipeline counts its own timesteps'
    assert p.task_args['sigmas'] == QWEN_SIGMAS[:-1]
    pipe.scheduler.set_timesteps(sigmas=p.task_args['sigmas'], mu=1.31) # the pipeline passes a resolution-dependent mu
    grid = torch.tensor(QWEN_SIGMAS, dtype=torch.float64)
    assert torch.allclose(pipe.scheduler.sigmas.double(), grid, atol=1e-6), f'sigmas={pipe.scheduler.sigmas.tolist()}'
    unpinned = diffusers.FlowMatchEulerDiscreteScheduler(**QWEN_SCHEDULER)
    unpinned.set_timesteps(sigmas=p.task_args['sigmas'], mu=1.31)
    assert not torch.allclose(unpinned.sigmas.double(), grid, atol=1e-3), 'the unpinned scheduler also lands on the grid; the check does not discriminate'
    network_pdd.restore(pipe)


# ============================================================
# Main
# ============================================================

def main():
    cat = category('detect')
    for fn in (test_detect_grid, test_detect_rejects_bad_block, test_load_collects_heads, test_load_without_metadata_is_none, test_load_without_heads_is_none, test_try_load_reads_the_grid_from_the_file_header, test_try_load_without_grid_metadata_is_none):
        run_test(cat, fn)
    cat = category('math')
    for fn in (test_grid_intervals_match_reference, test_steps_for_counts_terminal_sigma, test_parallel_head_matches_reference_per_block, test_parallel_head_strength_blend, test_parallel_head_clamps_overflow, test_parallel_head_keeps_base_out_of_tree):
        run_test(cat, fn)
    cat = category('lifecycle')
    for fn in (test_install_and_restore_round_trip, test_install_refuses_shape_mismatch, test_install_needs_an_owner, test_reconcile_follows_loaded_set, test_pin_overrides_steps_and_shift):
        run_test(cat, fn)
    cat = category('grid')
    for fn in (test_detect_grid_from_sigmas, test_detect_rejects_bad_sigmas, test_load_collects_replaced_parameters, test_try_load_reads_the_sidecar_config,
               test_install_replaces_and_restores_parameters, test_install_refuses_misshapen_replacement, test_one_head_per_step_on_a_block1_grid, test_pin_hands_the_exact_grid):
        run_test(cat, fn)
    failed = sum(r['failed'] for r in results.values())
    passed = sum(r['passed'] for r in results.values())
    log.info(f'PDD tests: passed={passed} failed={failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
