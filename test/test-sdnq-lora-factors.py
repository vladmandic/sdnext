#!/usr/bin/env python
"""
Offline unit tests for LoRA application on SDNQ-quantized layers.

Pins two facts established on real checkpoints (see cli/lora-quant-fidelity.py
for the per-model analyzer):

- The requantize path (dequantize + add + requantize) erases sub-step deltas
  on low-bit formats: retention collapses to the ~2/group_size grid-extrema
  floor on uint4, while int8 retains most of the delta. Guards against the
  erasure law silently changing.
- The factor path (modules/lora/lora_sdnq.py) applies plain LoRA deltas
  through the svd side-channel exactly, in both svd layouts, with exact
  stacking, multiplier scaling and bit-exact removal, wired through the real
  networks.network_activate / network_deactivate control flow.
- Multi-LoRA set transitions keep the base pristine: a layer that fell back
  to requantize (mixed factorable/non-factorable set) restores from backup
  before re-entering the factor path, layers targeted by only some of the
  loaded networks stay independent, and untargeted quantized layers are not
  flagged as requantized.

All tensors are synthetic; no model files or running server required.

Usage:
    python test/test-sdnq-lora-factors.py
"""

import os
import sys
import time
from contextlib import contextmanager

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

from modules.errors import log   # pylint: disable=wrong-import-position
from modules import shared, sd_models        # pylint: disable=wrong-import-position
from modules.lora import network, network_lora, lora_sdnq, networks  # pylint: disable=wrong-import-position
from modules.lora import lora_common as l_common   # pylint: disable=wrong-import-position
from modules.sdnq.quantizer import sdnq_quantize_layer, SDNQConfig  # pylint: disable=wrong-import-position

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_F, IN_F, RANK = 512, 512, 8

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
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


def build_layer(weights_dtype='uint4', use_quantized_matmul=False, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04)
    cfg = SDNQConfig(weights_dtype=weights_dtype, group_size=0, hadamard_group_size=256, use_hadamard=True,
                     use_svd=False, use_quantized_matmul=use_quantized_matmul, dequantize_fp32=False,
                     quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=torch.bfloat16, param_name='test.weight')
    layer.network_layer_name = 'lora_transformer_test'
    layer.network_current_names = ()
    return layer


def dq(layer):
    return layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=layer.zero_point,
                                  svd_up=layer.svd_up, svd_down=layer.svd_down,
                                  skip_quantized_matmul=layer.sdnq_dequantizer.use_quantized_matmul,
                                  dtype=torch.float32, skip_compile=True)


def make_delta(seed=1, sigma=3e-4):
    torch.manual_seed(seed)
    A = torch.randn(RANK, IN_F, device=DEVICE) * (sigma ** 0.5)
    B = torch.randn(OUT_F, RANK, device=DEVICE) * (sigma ** 0.5)
    return A, B, B @ A


class MockNOD:
    def __init__(self, name):
        self.filename = f'/tmp/{name}.safetensors'
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'


def make_net(name, layer, A, B, te_mult=1.0, alpha=None, dora=False):
    net = network.Network(name, MockNOD(name))
    net.te_multiplier = te_mult
    net.unet_multiplier = [te_mult] * 3
    w = {'lora_up.weight': B.cpu(), 'lora_down.weight': A.cpu()}
    if alpha is not None:
        w['alpha'] = torch.tensor(float(alpha))
    if dora:
        w['dora_scale'] = torch.ones(B.shape[0], 1)
    nw = network.NetworkWeights(network_key=layer.network_layer_name, sd_key=layer.network_layer_name, w=w, sd_module=layer)
    mod = network_lora.NetworkModuleLora(net, nw)
    net.modules[layer.network_layer_name] = mod
    return net


def rho_of(E, D):
    return float(E.flatten() @ D.flatten() / D.flatten().square().sum())


def requant_effective(layer, D):
    """The lossy fallback path: quantize(W_dq + D) fresh with the layer's own params."""
    from modules.sdnq.quantizer import sdnq_quantize_layer_weight
    deq = layer.sdnq_dequantizer
    Wdq = dq(layer)
    deq2, data2 = sdnq_quantize_layer_weight(Wdq + D, layer_class_name='Linear', weights_dtype=deq.weights_dtype,
                                             group_size=deq.group_size, hadamard_group_size=deq.hadamard_group_size,
                                             use_hadamard=deq.use_hadamard, use_svd=False, use_quantized_matmul=False,
                                             dequantize_fp32=False, torch_dtype=torch.bfloat16)
    W2 = deq2(data2['weight'], data2['scale'], zero_point=data2['zero_point'], svd_up=None, svd_down=None, dtype=torch.float32, skip_compile=True)
    return W2 - Wdq


# ============================================================
# Tests - the erasure law (why the factor path exists)
# ============================================================

CAT_LAW = category('erasure-law')


def test_uint4_erases_substep_delta():
    layer = build_layer('uint4')
    _A, _B, D = make_delta(sigma=2e-4)
    rho = rho_of(requant_effective(layer, D), D)
    group = layer.sdnq_dequantizer.group_size
    floor = 2.0 / group
    assert rho < 4 * floor, f'rho={rho:.4f} expected near extrema floor {floor:.4f}'
    return True


def test_int8_retains_delta():
    layer = build_layer('int8')
    _A, _B, D = make_delta(sigma=2e-4)
    rho = rho_of(requant_effective(layer, D), D)
    assert rho > 0.5, f'rho={rho:.4f} expected int8 to retain most of the delta'
    return True


# ============================================================
# Tests - factor path exactness
# ============================================================

CAT_FACTOR = category('factor-path')


def test_apply_exact_and_remove_bitexact():
    layer = build_layer('uint4')
    A, B, D = make_delta()
    net = make_net('one', layer, A, B)
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(net)
    wanted = (('one', 1.0, 1.0, None),)
    assert lora_sdnq.factor_candidate(layer, layer.network_layer_name, wanted) is True
    Wdq0 = dq(layer)
    assert lora_sdnq.apply_factors(layer, layer.network_layer_name, wanted) is True
    rho = rho_of(dq(layer) - Wdq0, D)
    assert rho > 0.99, f'rho={rho:.4f}'
    assert lora_sdnq.remove_factors(layer) is True
    assert torch.equal(dq(layer), Wdq0), 'remove must be bit-exact'
    assert layer.svd_up is None and not hasattr(layer, 'sdnq_lora_svd_stash')
    l_common.loaded_networks.clear()
    return True


def test_multiplier_and_alpha_scaling():
    layer = build_layer('uint4')
    A, B, D = make_delta()
    net = make_net('one', layer, A, B, te_mult=0.5, alpha=RANK // 2)  # alpha/rank = 0.5
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(net)
    Wdq0 = dq(layer)
    lora_sdnq.apply_factors(layer, layer.network_layer_name, (('one', 0.5, 0.5, None),))
    rho = rho_of(dq(layer) - Wdq0, D)
    assert abs(rho - 0.25) < 0.01, f'expected 0.5*0.5 scaling, rho={rho:.4f}'
    lora_sdnq.remove_factors(layer)
    l_common.loaded_networks.clear()
    return True


def test_stacking_two_networks():
    layer = build_layer('uint4')
    A1, B1, D1 = make_delta(seed=1)
    A2, B2, D2 = make_delta(seed=2)
    l_common.loaded_networks.clear()
    l_common.loaded_networks.extend([make_net('a', layer, A1, B1), make_net('b', layer, A2, B2)])
    Wdq0 = dq(layer)
    lora_sdnq.apply_factors(layer, layer.network_layer_name, (('a', 1.0, 1.0, None), ('b', 1.0, 1.0, None)))
    rho = rho_of(dq(layer) - Wdq0, D1 + D2)
    assert rho > 0.99, f'rho={rho:.4f}'
    lora_sdnq.remove_factors(layer)
    l_common.loaded_networks.clear()
    return True


def test_matmul_layout_transposed():
    layer = build_layer('uint4', use_quantized_matmul=True)
    A, B, D = make_delta()
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('one', layer, A, B))
    Wdq0 = dq(layer)
    res = lora_sdnq.apply_factors(layer, layer.network_layer_name, (('one', 1.0, 1.0, None),))
    rho = rho_of(dq(layer) - Wdq0, D)
    assert res is True and rho > 0.99, f'rho={rho:.4f}'
    lora_sdnq.remove_factors(layer)
    assert torch.equal(dq(layer), Wdq0)
    l_common.loaded_networks.clear()
    return True


def test_dora_falls_back():
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('dora', layer, A, B, dora=True))
    assert lora_sdnq.factor_candidate(layer, layer.network_layer_name, (('dora', 1.0, 1.0, None),)) is False
    l_common.loaded_networks.clear()
    return True


# ============================================================
# Tests - memory accounting across apply modes
# ============================================================

CAT_MEM = category('memory')


def tensor_bytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def test_factor_path_memory_is_factors_only():
    """Factor path: no weight/quant-state backups; added memory = the factor tensors."""
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('one', layer, A, B))
    lora_sdnq.apply_factors(layer, layer.network_layer_name, (('one', 1.0, 1.0, None),))
    assert getattr(layer, 'network_weights_backup', None) is None
    assert not hasattr(layer, 'sdnq_dequantizer_backup') and not hasattr(layer, 'sdnq_scale_backup')
    added = tensor_bytes(layer.svd_up) + tensor_bytes(layer.svd_down)
    expected = RANK * (OUT_F + IN_F) * 2  # bf16 factors
    assert added == expected, f'factor bytes {added} != expected {expected}'
    would_be_backup = tensor_bytes(layer.weight) + tensor_bytes(layer.scale) + tensor_bytes(layer.zero_point)
    assert added < would_be_backup / 4, f'factors {added}B should undercut the {would_be_backup}B backup this layer would otherwise clone'
    lora_sdnq.remove_factors(layer)
    assert layer.svd_up is None and layer.svd_down is None
    l_common.loaded_networks.clear()
    return True


def test_backup_mode_clones_full_quant_state():
    """Fallback in backup mode: packed weight + scale + zero_point are cloned to cpu."""
    from modules.lora.lora_apply import network_backup_weights
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('dora', layer, A, B, dora=True))  # non-factorable
    reported = network_backup_weights(layer, layer.network_layer_name, (('dora', 1.0, 1.0, None),), fuse=False)
    assert isinstance(layer.network_weights_backup, torch.Tensor) and layer.network_weights_backup.device.type == 'cpu'
    assert hasattr(layer, 'sdnq_dequantizer_backup') and isinstance(layer.sdnq_scale_backup, torch.Tensor)
    assert reported == tensor_bytes(layer.weight), f'reported {reported} != packed weight bytes {tensor_bytes(layer.weight)}'
    total = reported + tensor_bytes(layer.sdnq_scale_backup) + tensor_bytes(layer.sdnq_zero_point_backup)
    expected_min = OUT_F * IN_F // 2  # uint4 packs two weights per byte
    assert total >= expected_min, f'backup {total}B below packed-weight floor {expected_min}B'
    l_common.loaded_networks.clear()
    return True


def test_fuse_mode_marker_takes_no_memory():
    """Fuse mode stores a boolean marker instead of tensors; guard forces backup on quantized models."""
    from modules.lora.lora_apply import network_backup_weights
    from modules.lora import lora_overrides
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('dora', layer, A, B, dora=True))
    reported = network_backup_weights(layer, layer.network_layer_name, (('dora', 1.0, 1.0, None),), fuse=True)
    assert layer.network_weights_backup is True and reported == 0
    assert not hasattr(layer, 'sdnq_dequantizer_backup')

    # the guard: a quantized component forces fuse off model-wide regardless of the option
    class MockCfg:
        quantization_config = {'quant_method': 'sdnq'}
    class MockSd:
        pass
    sd = MockSd()
    sd.transformer = torch.nn.Linear(4, 4)
    sd.transformer.config = MockCfg()
    from modules.modeldata import model_data
    prev_model = model_data.sd_model
    old_fuse = shared.opts.lora_fuse_native
    try:
        model_data.sd_model = sd
        shared.opts.lora_fuse_native = True
        assert lora_overrides.disable_fuse() is True
        assert lora_overrides.fuse_native() is False
    finally:
        shared.opts.lora_fuse_native = old_fuse
        model_data.sd_model = prev_model
    l_common.loaded_networks.clear()
    return True


# ============================================================
# Tests - integration through networks.network_activate
# ============================================================

CAT_E2E = category('activate-e2e')


class MockHolder(torch.nn.Module):
    @property
    def device(self):
        return DEVICE


@contextmanager
def mock_model(**layers):
    """Install a one-component mock pipeline holding the given layers as shared.sd_model."""
    class MockPipe:
        pass
    class MockSd:
        pass
    holder = MockHolder()
    for attr, lyr in layers.items():
        setattr(holder, attr, lyr)
    pipe = MockPipe()
    pipe.transformer = holder
    sd = MockSd()
    sd.pipe = pipe
    from modules.modeldata import model_data
    model_data.sd_model = sd
    real_offload = sd_models.set_diffuser_offload
    sd_models.set_diffuser_offload = lambda *a, **k: None
    old_fuse = shared.opts.lora_fuse_native
    try:
        yield
    finally:
        shared.opts.lora_fuse_native = old_fuse
        sd_models.set_diffuser_offload = real_offload
        l_common.loaded_networks.clear()
        l_common.previously_loaded_networks.clear()


def activate(*nets):
    l_common.loaded_networks.clear()
    l_common.loaded_networks.extend(nets)
    networks.network_activate()


def test_network_activate_roundtrip():
    layer = build_layer('uint4')
    A, B, D = make_delta()
    net = make_net('one', layer, A, B)

    with mock_model(lin=layer):
        Wdq0 = dq(layer)
        activate(net)
        rho = rho_of(dq(layer) - Wdq0, D)
        assert rho > 0.99, f'rho={rho:.4f}'
        assert getattr(layer, 'network_weights_backup', None) is None, 'factor path must not take weight backups'

        activate()  # restore pass
        assert torch.equal(dq(layer), Wdq0), 'unload must restore bit-exact'

        # fuse-mode deactivate route
        layer.network_current_names = ()
        activate(net)
        l_common.previously_loaded_networks[:] = l_common.loaded_networks
        shared.opts.lora_fuse_native = True
        networks.network_deactivate()
        assert torch.equal(dq(layer), Wdq0), 'fuse-mode deactivate must restore bit-exact'
    return True


# ============================================================
# Tests - multi-LoRA set transitions between the two paths
# ============================================================

CAT_TRANS = category('transitions')


def test_mixed_family_transition_restores_base():
    layer = build_layer('uint4')
    bystander = build_layer('uint4', seed=7)
    bystander.network_layer_name = 'lora_transformer_bystander'
    A, B, D = make_delta()
    net_plain = make_net('plain', layer, A, B)
    A2, B2, _ = make_delta(seed=5, sigma=3e-3)
    net_dora = make_net('doranet', layer, A2, B2, dora=True)

    noted = []
    real_report = lora_sdnq.report_fallbacks
    def capture_report():
        noted.append(len(lora_sdnq.fallback_layers))
        real_report()
    lora_sdnq.report_fallbacks = capture_report
    try:
        with mock_model(lin=layer, bystander=bystander):
            Wdq0 = dq(layer)
            activate(net_plain)
            assert hasattr(layer, 'sdnq_lora_svd_stash'), 'plain set must take the factor path'
            assert noted[-1] == 0, f'untargeted quantized layers must not be flagged as requantized: noted={noted[-1]}'

            activate(net_plain, net_dora)
            assert not hasattr(layer, 'sdnq_lora_svd_stash') and isinstance(layer.network_weights_backup, torch.Tensor), 'mixed set must fall back with a tensor backup'
            assert not torch.equal(dq(layer), Wdq0), 'fallback must have requantized the weights'
            assert noted[-1] == 1, f'exactly the requantized layer must be flagged: noted={noted[-1]}'

            activate(net_plain)
            assert hasattr(layer, 'sdnq_lora_svd_stash'), 'plain-only set must re-enter the factor path'
            rho = rho_of(dq(layer) - Wdq0, D)
            assert rho > 0.99, f'rho={rho:.4f}'
            stash, up, down = layer.sdnq_lora_svd_stash, layer.svd_up, layer.svd_down
            lora_sdnq.remove_factors(layer)
            base_clean = torch.equal(dq(layer), Wdq0)
            layer.sdnq_lora_svd_stash, layer.svd_up, layer.svd_down = stash, up, down
            assert base_clean, 'base under factors must be restored from backup on mixed-set exit'

            activate()
            assert torch.equal(dq(layer), Wdq0), 'unload must return bit-exact pristine'
    finally:
        lora_sdnq.report_fallbacks = real_report
    return True


def test_partial_coverage_layers_stay_independent():
    layer_plain = build_layer('uint4')
    layer_dora = build_layer('uint4', seed=7)
    layer_dora.network_layer_name = 'lora_transformer_other'
    A, B, D = make_delta()
    net_plain = make_net('plain', layer_plain, A, B)
    A2, B2, _ = make_delta(seed=5, sigma=3e-3)
    net_dora = make_net('dorafar', layer_dora, A2, B2, dora=True)

    with mock_model(lin=layer_plain, other=layer_dora):
        Wdq0, Wdq0_dora = dq(layer_plain), dq(layer_dora)
        activate(net_plain, net_dora)
        assert hasattr(layer_plain, 'sdnq_lora_svd_stash') and getattr(layer_plain, 'network_weights_backup', None) is None, 'plain layer must stay on the factor path'
        assert isinstance(getattr(layer_dora, 'network_weights_backup', None), torch.Tensor), 'dora layer must take the backup fallback'
        rho = rho_of(dq(layer_plain) - Wdq0, D)
        assert rho > 0.99, f'rho={rho:.4f}'
        activate()
        assert torch.equal(dq(layer_plain), Wdq0), 'factor layer must restore bit-exact'
        assert torch.equal(dq(layer_dora), Wdq0_dora), 'fallback layer must restore bit-exact'
    return True


def run_tests():
    t0 = time.time()
    log.warning('=== Erasure law ===')
    for fn in [test_uint4_erases_substep_delta, test_int8_retains_delta]:
        run_test(CAT_LAW, fn)
    log.warning('=== Factor path ===')
    for fn in [test_apply_exact_and_remove_bitexact, test_multiplier_and_alpha_scaling, test_stacking_two_networks, test_matmul_layout_transposed, test_dora_falls_back]:
        run_test(CAT_FACTOR, fn)
    log.warning('=== Memory accounting ===')
    for fn in [test_factor_path_memory_is_factors_only, test_backup_mode_clones_full_quant_state, test_fuse_mode_marker_takes_no_memory]:
        run_test(CAT_MEM, fn)
    log.warning('=== Activate integration ===')
    for fn in [test_network_activate_roundtrip]:
        run_test(CAT_E2E, fn)
    log.warning('=== Set transitions ===')
    for fn in [test_mixed_family_transition_restores_base, test_partial_coverage_layers_stay_independent]:
        run_test(CAT_TRANS, fn)

    elapsed = time.time() - t0
    log.warning('=== Results ===')
    total_pass = total_fail = 0
    for cat, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f'  {cat}: {info["passed"]} passed, {info["failed"]} failed [{status}]')
        total_pass += info['passed']
        total_fail += info['failed']
    log.warning(f'Total: {total_pass} passed, {total_fail} failed in {elapsed:.2f}s')
    return total_fail == 0


if __name__ == '__main__':
    with torch.inference_mode():
        ok = run_tests()
    sys.exit(0 if ok else 1)
