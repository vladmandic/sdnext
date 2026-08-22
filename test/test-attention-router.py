#!/usr/bin/env python
"""
Offline unit tests for the attention router in modules.attention.

Covers:

- plan construction over every subset of the sdp_overrides choices on cuda, rocm, zluda and cpu
  against an oracle of the stacking order the closure hijacks used: priority, terminal selection,
  platform gating
- gate parity: every backend's declared constraints against a literal transcription of the
  predicate its closure carried, over a grid of shapes, dtypes, devices and masks
- every sdp_overrides choice maps to a registered backend and every backend to a choice
- router dispatch: the first accepting entry wins, the terminal receives declined calls, the
  original sdpa only receives enable_gqa when it is set
- a backend whose prepare raises is skipped without disturbing the rest
- install_router leaves the original sdpa in place for an empty plan
- the dynamic backend pins the pre-dynamic sdpa the sliced path reads
- the generation context: step normalized to the forward about to run on both the classic
  callback and the modular pre-hook, per-pass resets, the in-place step buffer, role scopes

No running server required. Nothing is moved to the accelerator.

Usage:
    python test/test-attention-router.py
"""

import itertools
import logging
import os
import sys
from dataclasses import replace

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

from modules.errors import log                                   # pylint: disable=wrong-import-position
from modules import attention                                    # pylint: disable=wrong-import-position
from modules.attention import router as attention_router         # pylint: disable=wrong-import-position


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
        if ok is False:
            record(cat, False, name)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


# ============================================================
# The closure hijacks this router replaces, transcribed
# ============================================================

# devices.set_sdpa_params applied the hijacks in this order; each wrapped the previous, so the
# last applied was tried first. Dynamic replaced the chain end instead of wrapping it; flex did
# too, which left everything stacked before it unreachable, so it is an ordinary entry now.
OLD_ORDER = ['Dynamic attention', 'Flex attention', 'Triton Flash attention', 'Flash attention', 'Sage attention', 'SDNQ attention']
OLD_TERMINALS = {'Dynamic attention'}
OLD_NAMES = {
    'Dynamic attention': 'dynamic',
    'Flex attention': 'flex',
    'Triton Flash attention': 'triton',
    'Flash attention': 'flash',
    'Sage attention': 'sage',
    'SDNQ attention': 'sdnq',
}
# mirrors shared_defaults.get_default_modes: five choices everywhere, Triton Flash attention added on rocm and zluda
CHOICES = OLD_ORDER
TRITON_PLATFORMS = {'rocm', 'zluda'}

# the four closure predicates transcribed literally, plus the contract flex_attention itself enforces
GATES = {
    'sdnq': lambda q, k, v, m: q.device.type != "cpu" and (q.shape[-2] >= 32 and k.shape[-2] >= 32) and (q.shape[-2] > 512 or k.shape[-2] > 512) and q.shape[-3] > 1,
    'triton': lambda q, k, v, m: q.shape[-1] <= 128 and m is None and q.device.type != "cpu" and k.device == q.device and v.device == q.device,
    'flash': lambda q, k, v, m: q.shape[-1] <= 128 and m is None and q.dtype != torch.float32 and q.device.type != "cpu" and k.device == q.device and v.device == q.device,
    'sage': lambda q, k, v, m: q.shape[-1] in {128, 96, 64} and m is None and q.device.type != "cpu" and k.device == q.device and v.device == q.device,
    'flex': lambda q, k, v, m: q.ndim == 4 and q.device.type != "cpu" and k.device == q.device and v.device == q.device,
}


def oracle_chain(labels, platform_backend):
    enabled = [label for label in OLD_ORDER if label in labels]
    if platform_backend not in TRITON_PLATFORMS:
        enabled = [label for label in enabled if label != 'Triton Flash attention']
    terminal = None
    entries = []
    for label in enabled:
        if label in OLD_TERMINALS:
            terminal = label
        else:
            entries.append(label)
    entries.reverse()
    return [OLD_NAMES[label] for label in entries], (OLD_NAMES[terminal] if terminal else None)


def stub_registry(failing=()):
    """The registered backends with prepares that return a tagged call instead of importing anything."""
    reg = attention.Registry()
    for backend in attention.registry.ordered():
        def prepare(platform, original, name=backend.name): # pylint: disable=unused-argument
            if name in failing:
                raise RuntimeError(f'{name} unavailable')
            def call(*args, **kwargs): # pylint: disable=unused-argument
                return name
            return call
        reg.register(replace(backend, prepare=prepare))
    return reg


def shaped(shape, dtype=torch.float16, device='meta'):
    """A tensor of the given shape without allocating it."""
    return torch.empty(1, dtype=dtype, device=device).expand(*shape)


def sdpa_stub(**kwargs): # pylint: disable=unused-argument
    return 'sdpa'


# ============================================================
# Tests
# ============================================================

def test_plan_matches_stacking_oracle():
    level = log.level
    log.setLevel(logging.ERROR) # platform gating warns per plan
    try:
        plans = 0
        for platform_backend in ('cuda', 'rocm', 'zluda', 'cpu'):
            reg = stub_registry()
            platform = attention.Platform(backend=platform_backend)
            for count in range(len(OLD_ORDER) + 1):
                for labels in itertools.combinations(OLD_ORDER, count):
                    plan = attention.build_plan(list(labels), platform, sdpa_stub, reg)
                    expected_entries, expected_terminal = oracle_chain(labels, platform_backend)
                    got_entries = [entry.backend.name for entry in plan.entries]
                    got_terminal = plan.terminal.backend.name if plan.terminal is not None else None
                    assert got_entries == expected_entries, f'{platform_backend} {labels}: entries {got_entries} != {expected_entries}'
                    assert got_terminal == expected_terminal, f'{platform_backend} {labels}: terminal {got_terminal} != {expected_terminal}'
                    assert plan.chain() == got_entries + [got_terminal or 'sdpa']
                    plans += 1
    finally:
        log.setLevel(level)
    log.info(f'  {plans} plans match the stacking oracle')
    return True


def test_gates_match_transcribed_predicates():
    cases = 0
    lengths = (16, 32, 512, 513, 4096)
    for q_device, kv_device, dtype, heads, q_len, k_len, head_dim, masked, batched in itertools.product(('cpu', 'meta'), ('cpu', 'meta'), (torch.float16, torch.float32), (1, 8), lengths, lengths, (40, 64, 96, 128, 256), (False, True), (False, True)):
        lead = (1,) if batched else ()
        q = shaped((*lead, heads, q_len, head_dim), dtype, q_device)
        k = shaped((*lead, heads, k_len, head_dim), dtype, kv_device)
        v = shaped((*lead, heads, k_len, head_dim), dtype, kv_device)
        m = shaped((*lead, 1, q_len, k_len), torch.bool, q_device) if masked else None
        for name, gate in GATES.items():
            expected = bool(gate(q, k, v, m))
            got = attention.registry.backends[name].constraints.accepts(q, k, v, m)
            assert got == expected, f'{name}: q={tuple(q.shape)} k={tuple(k.shape)} dtype={dtype} devices={q_device}/{kv_device} mask={masked} got={got} expected={expected}'
            cases += 1
    log.info(f'  {cases} gate cases match the transcribed predicates')
    return True


def test_only_dynamic_is_terminal():
    for name, backend in attention.registry.backends.items():
        assert backend.terminal == (name == 'dynamic'), name
    assert attention.registry.backends['dynamic'].constraints == attention.Constraints()
    return True


def test_choices_match_backends():
    labels = attention.registry.labels()
    assert sorted(labels) == sorted(CHOICES), f'registered={labels} choices={CHOICES}'
    for label in CHOICES:
        assert attention.registry.by_label(label) is not None, label
    triton = attention.registry.backends['triton']
    assert triton.platforms == frozenset(TRITON_PLATFORMS), triton.platforms
    for name, backend in attention.registry.backends.items():
        if name != 'triton':
            assert backend.platforms is None, name
    return True


def test_router_dispatch_prefers_priority_then_terminal_then_original():
    calls = []

    def original(**kwargs):
        calls.append(('sdpa', kwargs))
        return 'sdpa'

    reg = attention.Registry()

    def add(name, constraints, priority, terminal=False):
        def prepare(platform, orig): # pylint: disable=unused-argument
            def call(*args, **kwargs): # pylint: disable=unused-argument
                calls.append((name, kwargs))
                return name
            return call
        reg.register(attention.AttentionBackend(name=name, label=f'{name} attention', priority=priority, prepare=prepare, constraints=constraints, terminal=terminal))

    add('narrow', attention.Constraints(head_dims=frozenset({64})), priority=20)
    add('wide', attention.Constraints(), priority=10)
    platform = attention.Platform(backend='cuda')
    router = attention_router.make_router(attention.build_plan(['narrow attention', 'wide attention'], platform, original, reg))
    q64 = shaped((1, 8, 128, 64))
    q128 = shaped((1, 8, 128, 128))
    cpu = shaped((1, 8, 128, 64), device='cpu')
    assert router(q64, q64, q64) == 'narrow'
    assert router(q128, q128, q128) == 'wide'
    assert router(cpu, cpu, cpu) == 'sdpa'
    assert 'enable_gqa' not in calls[-1][1], calls[-1]
    assert router(cpu, cpu, cpu, enable_gqa=True) == 'sdpa'
    assert calls[-1][1].get('enable_gqa') is True, calls[-1]

    add('term', attention.Constraints(), priority=5, terminal=True)
    router = attention_router.make_router(attention.build_plan(['narrow attention', 'term attention'], platform, original, reg))
    assert router(q64, q64, q64) == 'narrow'
    assert router(cpu, cpu, cpu, extra=1) == 'term'
    assert calls[-1][1].get('extra') == 1 and calls[-1][1].get('enable_gqa') is False, calls[-1]
    return True


def test_prepare_failure_skips_backend():
    reg = stub_registry(failing=('sage',))
    plan = attention.build_plan(['Sage attention', 'SDNQ attention', 'Flash attention'], attention.Platform(backend='cuda'), sdpa_stub, reg)
    assert [entry.backend.name for entry in plan.entries] == ['sdnq', 'flash'], plan.chain()
    return True


def test_install_router_keeps_original_for_empty_plan():
    saved = torch.nn.functional.scaled_dot_product_attention
    saved_plan = attention_router.current_plan
    try:
        platform = attention.Platform(backend='cuda')
        plan = attention.install_router([], platform, sdpa_stub, stub_registry())
        assert torch.nn.functional.scaled_dot_product_attention is sdpa_stub
        assert plan.chain() == ['sdpa'], plan.chain()
        plan = attention.install_router(['SDNQ attention', 'Sage attention'], platform, sdpa_stub, stub_registry())
        assert torch.nn.functional.scaled_dot_product_attention is not sdpa_stub
        assert plan.chain() == ['sdnq', 'sage', 'sdpa'], plan.chain()
        assert attention.get_plan() is plan
    finally:
        torch.nn.functional.scaled_dot_product_attention = saved
        attention_router.current_plan = saved_plan
    return True


def test_dynamic_backend_pins_pre_dynamic_sdpa():
    from modules import devices
    saved = devices.sdpa_pre_dyanmic_atten
    try:
        call = attention.registry.backends['dynamic'].prepare(attention.Platform(backend='cuda'), sdpa_stub)
        from modules.sd_hijack_dynamic_atten import dynamic_scaled_dot_product_attention
        assert call is dynamic_scaled_dot_product_attention
        assert devices.sdpa_pre_dyanmic_atten is sdpa_stub
    finally:
        devices.sdpa_pre_dyanmic_atten = saved
    return True


def test_context_classic_ticks_follow_the_callback():
    ctx = attention.context

    class Pipe:
        transformer = object()

    ctx.begin(Pipe(), steps=4)
    assert ctx.current.active and ctx.current.role == 'transformer' and ctx.current.step == 0 and ctx.current.steps == 4
    assert ctx.current.model_key == ('Pipe', 'object'), ctx.current.model_key
    buffer = ctx.current.step_buffer
    for completed in range(4):
        ctx.tick(completed + 1) # the diffusers callback reports the step just completed
        assert ctx.current.step == completed + 1
        assert ctx.current.step_buffer is buffer and int(buffer.item()) == completed + 1
    ctx.new_pass(2) # hires or refiner pass
    assert ctx.current.step == 0 and ctx.current.steps == 2 and int(buffer.item()) == 0
    ctx.end()
    assert not ctx.current.active and ctx.current.role is None and ctx.current.model_key is None and ctx.current.step == 0
    return True


def test_context_modular_ticks_count_forwards():
    ctx = attention.context
    ctx.begin(None, steps=3)
    assert ctx.current.model_key is None
    for expected in range(3):
        ctx.tick() # the modular pre-hook fires before each forward
        assert ctx.current.step == expected, ctx.current.step
    ctx.end()
    return True


def test_context_roles_nest_and_stick():
    ctx = attention.context
    ctx.begin(None)
    with ctx.role('te'):
        assert ctx.current.role == 'te'
        with ctx.role('vae'):
            assert ctx.current.role == 'vae'
        assert ctx.current.role == 'te'
    assert ctx.current.role == 'transformer'
    ctx.set_role('vae')
    assert ctx.current.role == 'vae'
    ctx.end()
    assert ctx.current.role is None
    with ctx.role('te'): # outside a generation the scope still restores what it found
        assert ctx.current.role == 'te'
    assert ctx.current.role is None
    return True


def run_all():
    log.warning('=== attention router ===')
    cat = category('router')
    for fn in [
        test_plan_matches_stacking_oracle,
        test_gates_match_transcribed_predicates,
        test_only_dynamic_is_terminal,
        test_choices_match_backends,
        test_router_dispatch_prefers_priority_then_terminal_then_original,
        test_prepare_failure_skips_backend,
        test_install_router_keeps_original_for_empty_plan,
        test_dynamic_backend_pins_pre_dynamic_sdpa,
    ]:
        run_test(cat, fn)

    log.warning('=== generation context ===')
    cat = category('context')
    for fn in [
        test_context_classic_ticks_follow_the_callback,
        test_context_modular_ticks_count_forwards,
        test_context_roles_nest_and_stick,
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
    torch.nn.functional.scaled_dot_product_attention = stock_sdpa
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
