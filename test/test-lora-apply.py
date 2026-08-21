#!/usr/bin/env python
"""
Offline unit tests for the native LoRA apply paths.

Two paths write a delta onto a module: fuse mode (``network_apply_direct``)
adds it straight onto the live tensors, backup mode (``network_apply_weights``)
adds it onto a cloned copy. Both funnel through ``network_add_weights``, and
only backup mode names the base tensor a bias delta targets, so the fuse-side
bias case is the one with nothing pinning it.

Bias deltas ride the ``diff_b`` key that trainers pair with the weight LoRA on
projection layers. A Linear's weight is ``[out, in]`` and its bias is ``[out]``,
so reading the wrong base tensor for a bias delta throws whenever ``in != out``
and broadcasts silently when they match: both shapes are covered here.

No running server required.

Usage:
    python test/test-lora-apply.py
"""

import os
import sys
import time

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

from modules.errors import log  # pylint: disable=wrong-import-position
from modules import devices  # pylint: disable=wrong-import-position
from modules.lora.lora_apply import network_apply_direct, network_apply_weights  # pylint: disable=wrong-import-position

devices.device = torch.device('cpu')  # apply moves operands to devices.device; keep the suite off the gpu


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}

CAT_FUSE = 'fuse mode'
CAT_BACKUP = 'backup mode'


def record(cat: str, passed: bool, name: str, detail: str = ''):
    if cat not in results:
        results[cat] = {'passed': 0, 'failed': 0}
    status = 'PASS' if passed else 'FAIL'
    results[cat]['passed' if passed else 'failed'] += 1
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
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


CPU = torch.device('cpu')


def make_linear(out_features: int, in_features: int, seed: int = 0):
    """Linear with deterministic weight and bias, plus copies of both as they started."""
    torch.manual_seed(seed)
    module = torch.nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        module.weight.copy_(torch.randn(out_features, in_features) * 0.02)
        module.bias.copy_(torch.randn(out_features) * 0.02)
    return module, module.weight.detach().clone(), module.bias.detach().clone()


def stamp_fuse(module):
    """Mark the module as network_backup_weights leaves it in fuse mode: no tensor backup.

    The bias flag is only set when the module has one, same as the loader does.
    """
    module.network_weights_backup = True
    if getattr(module, 'bias', None) is not None:
        module.network_bias_backup = True


def stamp_backup(module, weight, bias):
    """Mark the module as network_backup_weights leaves it in backup mode: cloned tensors."""
    module.network_weights_backup = weight.clone().to(CPU)
    module.network_bias_backup = bias.clone().to(CPU)


def assert_close(actual, expected, label):
    assert actual.shape == expected.shape, f'{label} shape {tuple(actual.shape)} != {tuple(expected.shape)}'
    assert torch.allclose(actual, expected, atol=1e-6), f'{label} values drifted'


# ============================================================
# Fuse mode
# ============================================================


def test_fuse_bias_delta_asymmetric():
    """A diff_b delta on a Linear whose in and out differ lands on the bias, not the weight."""
    module, w0, b0 = make_linear(32, 8)
    stamp_fuse(module)
    updown = torch.full_like(w0, 0.5)
    ex_bias = torch.full_like(b0, 0.25)
    written = network_apply_direct(module, updown, ex_bias, device=CPU)
    assert written == (True, True), f'reported {written}'
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    assert_close(module.bias.detach(), b0 + 0.25, 'bias')
    return True


def test_fuse_bias_delta_square():
    """The same delta on a square Linear, where a wrong base tensor broadcasts instead of throwing."""
    module, w0, b0 = make_linear(16, 16)
    stamp_fuse(module)
    updown = torch.full_like(w0, 0.5)
    ex_bias = torch.full_like(b0, 0.25)
    network_apply_direct(module, updown, ex_bias, device=CPU)
    assert module.bias.dim() == 1, f'bias became {module.bias.dim()}d'
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    assert_close(module.bias.detach(), b0 + 0.25, 'bias')
    return True


def test_fuse_weight_only_leaves_bias():
    """A LoRA with no bias delta leaves the bias untouched."""
    module, w0, b0 = make_linear(32, 8)
    stamp_fuse(module)
    written = network_apply_direct(module, torch.full_like(w0, 0.5), None, device=CPU)
    assert written == (True, False), f'reported {written}'  # nothing to write is reported the same as refused; the caller knows which by whether it passed a delta
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    assert_close(module.bias.detach(), b0, 'bias')
    return True


def test_fuse_deactivate_restores():
    """Deactivate subtracts the same deltas, returning both tensors to their loaded values."""
    module, w0, b0 = make_linear(32, 8)
    stamp_fuse(module)
    updown = torch.full_like(w0, 0.5)
    ex_bias = torch.full_like(b0, 0.25)
    network_apply_direct(module, updown.clone(), ex_bias.clone(), device=CPU)
    network_apply_direct(module, updown.clone(), ex_bias.clone(), device=CPU, deactivate=True)
    assert_close(module.weight.detach(), w0, 'weight')
    assert_close(module.bias.detach(), b0, 'bias')
    return True


def test_fuse_mismatched_bias_delta_is_refused():
    """A bias delta that genuinely does not fit is dropped, leaving the bias intact.

    The refusal has to reach the caller: network_activate counts the layer as
    refused rather than applied, which is what keeps the summary line honest.
    """
    module, w0, b0 = make_linear(32, 8)
    stamp_fuse(module)
    written = network_apply_direct(module, torch.full_like(w0, 0.5), torch.full((7,), 0.25), device=CPU)
    assert written == (True, False), f'reported {written}'
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    assert_close(module.bias.detach(), b0, 'bias')
    return True


def test_fuse_bias_delta_without_a_bias_is_refused():
    """A delta aimed at a bias the module does not have is counted, not silently dropped.

    The loader lets this through on purpose: whole architectures are built
    bias=False, so a stray diff_b is one unappliable key rather than the wrong
    file. The apply pass is where it has to become visible.
    """
    module = torch.nn.Linear(8, 32, bias=False)
    w0 = module.weight.detach().clone()
    stamp_fuse(module)
    written = network_apply_direct(module, torch.full_like(w0, 0.5), torch.full((32,), 0.25), device=CPU)
    assert written == (True, False), f'reported {written}'
    assert module.bias is None, 'a bias appeared on a module built without one'
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    return True


def test_fuse_mismatched_weight_delta_is_refused():
    """A weight delta that does not fit is dropped while the bias delta still lands."""
    module, w0, b0 = make_linear(32, 8)
    stamp_fuse(module)
    written = network_apply_direct(module, torch.full((32, 5), 0.5), torch.full_like(b0, 0.25), device=CPU)
    assert written == (False, True), f'reported {written}'
    assert_close(module.weight.detach(), w0, 'weight')
    assert_close(module.bias.detach(), b0 + 0.25, 'bias')
    return True


# ============================================================
# Backup mode
# ============================================================


def test_backup_bias_delta_asymmetric():
    """Backup mode adds the deltas onto the cloned base rather than the live tensors."""
    module, w0, b0 = make_linear(32, 8)
    stamp_backup(module, w0, b0)
    updown = torch.full_like(w0, 0.5)
    ex_bias = torch.full_like(b0, 0.25)
    written = network_apply_weights(module, updown, ex_bias, device=CPU)
    assert written == (True, True), f'reported {written}'
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    assert_close(module.bias.detach(), b0 + 0.25, 'bias')
    return True


def test_backup_reapply_is_not_cumulative():
    """Applying twice from the same backup yields one delta, not two."""
    module, w0, b0 = make_linear(32, 8)
    stamp_backup(module, w0, b0)
    for _ in range(2):
        network_apply_weights(module, torch.full_like(w0, 0.5), torch.full_like(b0, 0.25), device=CPU)
    assert_close(module.weight.detach(), w0 + 0.5, 'weight')
    assert_close(module.bias.detach(), b0 + 0.25, 'bias')
    return True


def test_backup_restore_without_delta():
    """Applying with no deltas restores the module to its backup."""
    module, w0, b0 = make_linear(32, 8)
    stamp_backup(module, w0, b0)
    network_apply_weights(module, torch.full_like(w0, 0.5), torch.full_like(b0, 0.25), device=CPU)
    network_apply_weights(module, None, None, device=CPU)
    assert_close(module.weight.detach(), w0, 'weight')
    assert_close(module.bias.detach(), b0, 'bias')
    return True


def run_tests():
    t0 = time.time()
    log.warning('=== fuse mode ===')
    for fn in [
        test_fuse_bias_delta_asymmetric,
        test_fuse_bias_delta_square,
        test_fuse_weight_only_leaves_bias,
        test_fuse_deactivate_restores,
        test_fuse_mismatched_bias_delta_is_refused,
        test_fuse_mismatched_weight_delta_is_refused,
        test_fuse_bias_delta_without_a_bias_is_refused,
    ]:
        run_test(CAT_FUSE, fn)

    log.warning('=== backup mode ===')
    for fn in [
        test_backup_bias_delta_asymmetric,
        test_backup_reapply_is_not_cumulative,
        test_backup_restore_without_delta,
    ]:
        run_test(CAT_BACKUP, fn)

    elapsed = time.time() - t0
    log.warning('=== Results ===')
    total_pass = 0
    total_fail = 0
    for cat, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f'  {cat}: {info["passed"]} passed, {info["failed"]} failed [{status}]')
        total_pass += info['passed']
        total_fail += info['failed']
    log.warning(f'Total: {total_pass} passed, {total_fail} failed in {elapsed:.2f}s')
    return total_fail == 0


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
