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
  through the svd side-channel exactly, in both svd layouts and across
  quantization configs (hadamard on/off, checkpoint svd correction present
  or absent), with exact stacking, multiplier scaling and bit-exact
  removal, wired through the real networks.network_activate /
  network_deactivate control flow.
- Multi-LoRA set transitions keep the base pristine: a layer that fell back
  to requantize (mixed factorable/non-factorable set) restores from backup
  before re-entering the factor path, layers targeted by only some of the
  loaded networks stay independent, and untargeted quantized layers are not
  flagged as requantized.
- Robustness: factor removal restores onto the layer's current device after
  an offload-style move, and a shape-mismatched network stacked onto a
  factor-mode layer downgrades to the legacy path instead of raising.
- Hosting: on sub-8-bit layers, non-factorable sets ride the side-channel as
  a truncated svd of their calc_updown delta: low-rank content survives
  whole, dense content beats the requantize floor by a wide margin, int8
  and rank 0 keep the requantize path, removal stays bit-exact, and the
  svd's random projections never touch the generation rng stream.
- Calibration: per-channel activation statistics weight the hosted
  truncation toward loud input channels for better output-space retention;
  low-rank content still survives whole, disabling the option reproduces
  plain truncation bit-exact, and the capture hooks accumulate, persist
  and reload statistics correctly, gated by option, format width and
  model compile.
- Factor cache: hosted factors replay bit-identically from the disk cache
  without re-running the svd, a configuration change (multiplier) misses
  and writes a separate entry, and budget 0 writes nothing.

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
from sdnq.quantizer import sdnq_quantize_layer, SDNQConfig  # pylint: disable=wrong-import-position

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


def build_layer(weights_dtype='uint4', use_quantized_matmul=False, seed=0, use_hadamard=True, use_svd=False):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04)
    cfg = SDNQConfig(weights_dtype=weights_dtype, group_size=0, hadamard_group_size=256, use_hadamard=use_hadamard,
                     use_svd=use_svd, svd_rank=32, use_quantized_matmul=use_quantized_matmul, dequantize_fp32=False,
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
    from sdnq.quantizer import sdnq_quantize_layer_weight
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


def assert_factor_roundtrip(layer, tag):
    """Apply-exact plus bit-exact removal on the given layer, whatever its quantization config."""
    A, B, D = make_delta()
    Wdq0 = dq(layer)
    orig_up = layer.svd_up
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('one', layer, A, B))
    wanted = (('one', 1.0, 1.0, None),)
    assert lora_sdnq.factor_candidate(layer, layer.network_layer_name, wanted) is True, f'{tag}: not a factor candidate'
    assert lora_sdnq.apply_factors(layer, layer.network_layer_name, wanted) is True, f'{tag}: apply failed'
    E = dq(layer) - Wdq0
    rho = rho_of(E, D)
    resid = float((E - D).norm() / D.norm())
    assert rho > 0.99 and resid < 0.2, f'{tag}: rho={rho:.4f} resid={resid:.4f}'
    assert lora_sdnq.remove_factors(layer) and torch.equal(dq(layer), Wdq0), f'{tag}: remove not bit-exact'
    assert layer.svd_up is orig_up, f'{tag}: original svd factors not restored'
    l_common.loaded_networks.clear()


def test_no_hadamard_checkpoint():
    """Checkpoints quantized without hadamard: factors attach unrotated."""
    assert_factor_roundtrip(build_layer('uint4', use_hadamard=False), 'plain')
    assert_factor_roundtrip(build_layer('uint4', use_hadamard=False, use_quantized_matmul=True), 'matmul')
    return True


def test_checkpoint_svd_factors_preserved():
    """Checkpoints quantized with their own svd correction keep it under apply/remove."""
    layer = build_layer('uint4', use_svd=True)
    assert layer.svd_up is not None, 'quantizer produced no svd correction'
    assert_factor_roundtrip(layer, 'plain')
    assert_factor_roundtrip(build_layer('uint4', use_svd=True, use_quantized_matmul=True), 'matmul')
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
    shared.opts.lora_fuse_native = False # a real quantized model forces backup mode; the mock carries no quantization config, so pin it instead of inheriting the running config
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
        with host_rank(0), mock_model(lin=layer, bystander=bystander): # pins the requantize fallback; hosted transitions are covered in the hosting category
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

    with host_rank(0), mock_model(lin=layer_plain, other=layer_dora): # pins the requantize fallback for the non-factorable layer
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


@contextmanager
def apply_method(value):
    old = getattr(shared.opts, 'lora_sdnq_apply', 'exact')
    shared.opts.lora_sdnq_apply = value
    try:
        yield
    finally:
        shared.opts.lora_sdnq_apply = old


def test_mechanism_gate_declines_candidates():
    """The requantize option must gate every svd-channel entry point and flip the apply-stamp token."""
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    l_common.loaded_networks.clear()
    l_common.loaded_networks.append(make_net('one', layer, A, B))
    wanted = (('one', 1.0, 1.0, None),)
    try:
        assert lora_sdnq.factor_candidate(layer, layer.network_layer_name, wanted)
        with host_rank(64):
            assert lora_sdnq.host_candidate(layer, layer.network_layer_name, wanted)
        assert lora_sdnq.signature() == ''
        with apply_method('requantize'):
            assert not lora_sdnq.factor_candidate(layer, layer.network_layer_name, wanted)
            with host_rank(64):
                assert not lora_sdnq.host_candidate(layer, layer.network_layer_name, wanted)
            assert lora_sdnq.signature() == '|quant=requantize'
    finally:
        l_common.loaded_networks.clear()
    return True


def test_requantize_option_routes_to_legacy_path():
    """With the option set, a factorable set must take the classic backup-and-requantize path end to end."""
    layer = build_layer('uint4')
    A, B, _D = make_delta(sigma=3e-3)
    net = make_net('one', layer, A, B)
    with apply_method('requantize'), mock_model(lin=layer):
        shared.opts.lora_fuse_native = False # a real quantized model forces backup mode; the mock carries no quantization config
        Wdq0 = dq(layer)
        activate(net)
        assert not hasattr(layer, 'sdnq_lora_svd_stash'), 'legacy path must not touch the svd channel'
        assert layer.svd_up is None, 'legacy path must leave the channel empty'
        assert isinstance(layer.network_weights_backup, torch.Tensor), 'legacy path must take a tensor backup'
        assert not torch.equal(dq(layer), Wdq0), 'legacy path must requantize the weights'
        activate()
        assert torch.equal(dq(layer), Wdq0), 'legacy restore must be bit-exact from backup'
    return True


def test_mechanism_flip_strips_attached_factors():
    """Flipping to requantize with factors attached must strip them before the weight path takes the layer; flipping back must re-enter the factor path."""
    layer = build_layer('uint4')
    A, B, _D = make_delta(sigma=3e-3)
    net = make_net('one', layer, A, B)
    with mock_model(lin=layer):
        shared.opts.lora_fuse_native = False # a real quantized model forces backup mode; the mock carries no quantization config
        Wdq0 = dq(layer)
        activate(net)
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'default mechanism must take the factor path'
        E_exact = dq(layer) - Wdq0
        with apply_method('requantize'):
            activate(net) # same set; the mechanism token in the apply stamp must force re-processing
            assert not hasattr(layer, 'sdnq_lora_svd_stash'), 'flip must strip the attached factors'
            assert layer.svd_up is None, 'stripped channel must be empty, or the requantized delta double-applies'
            assert isinstance(layer.network_weights_backup, torch.Tensor), 'flipped layer must continue on the backup path'
        activate(net) # flip back within the same loaded set
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'flip back must re-enter the factor path'
        assert torch.equal(dq(layer) - Wdq0, E_exact), 'exact re-apply must restore the base from backup before attaching'
        activate()
        assert torch.equal(dq(layer), Wdq0), 'unload must return bit-exact pristine'
    return True


def test_mechanism_flip_restore_pass_strips():
    """A restore-only pass under the requantize option must still drop attached factors."""
    layer = build_layer('uint4')
    A, B, _D = make_delta(sigma=3e-3)
    net = make_net('one', layer, A, B)
    with mock_model(lin=layer):
        Wdq0 = dq(layer)
        activate(net)
        assert hasattr(layer, 'sdnq_lora_svd_stash')
        with apply_method('requantize'):
            activate() # unload with the gate closed: the fallthrough strip is the only removal route
            assert not hasattr(layer, 'sdnq_lora_svd_stash'), 'restore pass must strip the factors'
            assert torch.equal(dq(layer), Wdq0), 'strip must restore bit-exact'
            assert layer.network_current_names == (), 'stripped layer must be stamped restored'
    return True


CAT_HOST = category('hosting')


@contextmanager
def host_rank(rank):
    old = getattr(shared.opts, 'lora_sdnq_host_rank', 0)
    shared.opts.lora_sdnq_host_rank = rank
    try:
        yield
    finally:
        shared.opts.lora_sdnq_host_rank = old


def make_dense_net(name, layer, D):
    """A full-family (dense diff) network module: non-factorable by construction."""
    from modules.lora import network_full
    net = network.Network(name, MockNOD(name))
    net.te_multiplier = 1.0
    net.unet_multiplier = [1.0] * 3
    nw = network.NetworkWeights(network_key=layer.network_layer_name, sd_key=layer.network_layer_name,
                                w={'diff': D.cpu()}, sd_module=layer)
    net.modules[layer.network_layer_name] = network_full.NetworkModuleFull(net, nw)
    return net


def test_hosted_low_rank_delta_is_kept():
    layer = build_layer('uint4')
    _A, _B, D = make_delta(sigma=3e-3)
    net = make_dense_net('densenet', layer, D) # low-rank content in a non-factorable container
    with host_rank(64), mock_model(lin=layer):
        Wdq0 = dq(layer)
        activate(net)
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'hosted set must ride the side-channel'
        assert getattr(layer, 'network_weights_backup', None) is None, 'hosted layers must not take a weight backup'
        rho = rho_of(dq(layer) - Wdq0, D)
        assert rho > 0.95, f'rank-8 delta under cap 64 must be kept nearly whole: rho={rho:.4f}'
        activate()
        assert torch.equal(dq(layer), Wdq0), 'unload must restore bit-exact'
    return True


def test_hosted_dense_delta_beats_requant():
    layer = build_layer('uint4')
    torch.manual_seed(3)
    D = torch.randn(OUT_F, IN_F, device=DEVICE) * 3e-4 # full-rank, sub-step: requant erases it
    requant_rho = rho_of(requant_effective(layer, D), D)
    net = make_dense_net('densefull', layer, D)
    with host_rank(256), mock_model(lin=layer):
        Wdq0 = dq(layer)
        activate(net)
        hosted_rho = rho_of(dq(layer) - Wdq0, D)
        assert hosted_rho > 0.4, f'hosted rho={hosted_rho:.3f}'
        assert hosted_rho > requant_rho + 0.3, f'hosting must beat requant by a wide margin: {hosted_rho:.3f} vs {requant_rho:.3f}'
        activate()
        assert torch.equal(dq(layer), Wdq0)
    return True


def test_hosted_skips_int8():
    layer = build_layer('int8')
    _A, _B, D = make_delta(sigma=3e-3)
    net = make_dense_net('int8net', layer, D)
    with host_rank(256), mock_model(lin=layer):
        activate(net)
        assert not hasattr(layer, 'sdnq_lora_svd_stash'), 'int8 must keep the requantize path'
        assert isinstance(getattr(layer, 'network_weights_backup', None), torch.Tensor), 'int8 fallback must take the backup'
        activate()
    return True


def test_hosted_disabled_by_option():
    layer = build_layer('uint4')
    _A, _B, D = make_delta(sigma=3e-3)
    net = make_dense_net('offnet', layer, D)
    with host_rank(0), mock_model(lin=layer):
        activate(net)
        assert not hasattr(layer, 'sdnq_lora_svd_stash'), 'rank 0 must disable hosting'
        activate()
    return True


def test_hosted_transitions_and_rng_isolation():
    layer = build_layer('uint4')
    A, B, D = make_delta()
    net_plain = make_net('plainh', layer, A, B)
    _A2, _B2, D2 = make_delta(seed=9, sigma=3e-3)
    net_dense = make_dense_net('denseh', layer, D2)
    with host_rank(256), mock_model(lin=layer):
        Wdq0 = dq(layer)
        rng0 = torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else torch.get_rng_state()
        activate(net_dense) # hosted
        rng1 = torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else torch.get_rng_state()
        assert torch.equal(rng0, rng1), 'hosting must not consume the generation rng stream'
        assert hasattr(layer, 'sdnq_lora_svd_stash')
        activate(net_plain) # exact replaces hosted
        rho = rho_of(dq(layer) - Wdq0, D)
        assert rho > 0.99, f'exact set after hosted set: rho={rho:.4f}'
        activate(net_plain, net_dense) # mixed set hosts the combined delta
        rho_mix = rho_of(dq(layer) - Wdq0, D + D2)
        assert rho_mix > 0.9, f'mixed hosted rho={rho_mix:.4f}'
        activate()
        assert torch.equal(dq(layer), Wdq0), 'unload must restore bit-exact'
    return True


CAT_CALIB = category('calibration')


@contextmanager
def host_calib(value):
    old = getattr(shared.opts, 'lora_sdnq_host_calib', False)
    shared.opts.lora_sdnq_host_calib = value
    try:
        yield
    finally:
        shared.opts.lora_sdnq_host_calib = old


def test_calibrated_hosting_beats_plain():
    layer = build_layer('uint4')
    torch.manual_seed(11)
    scale = torch.ones(IN_F, device=DEVICE)
    scale[:32] = 40.0 # a few loud input channels, the shape real activations have
    D = torch.randn(OUT_F, IN_F, device=DEVICE) * 3e-4
    X = torch.randn(1024, IN_F, device=DEVICE) * scale
    Y = X @ D.t()
    net = make_dense_net('calnet', layer, D)

    def out_rho(E):
        return float((X @ E.t()).flatten() @ Y.flatten() / Y.square().sum())

    with host_rank(32), host_calib(True), mock_model(lin=layer):
        Wdq0 = dq(layer)
        activate(net)
        plain = out_rho(dq(layer) - Wdq0)
        activate()
        layer.sdnq_calib_rms = scale.cpu() # statistics as the capture leaves them
        activate(net)
        weighted = out_rho(dq(layer) - Wdq0)
        activate()
        del layer.sdnq_calib_rms
        assert torch.equal(dq(layer), Wdq0), 'unload must restore bit-exact'
        assert weighted > plain + 0.2, f'calibrated hosting must beat plain in output space: {weighted:.3f} vs {plain:.3f}'
    return True


def test_calibrated_low_rank_delta_survives():
    layer = build_layer('uint4')
    _A, _B, D = make_delta(sigma=3e-3)
    net = make_dense_net('calfull', layer, D)
    with host_rank(64), host_calib(True), mock_model(lin=layer):
        Wdq0 = dq(layer)
        torch.manual_seed(21)
        layer.sdnq_calib_rms = torch.rand(IN_F) * 10 + 0.1 # arbitrary positive statistics: unscale must round-trip
        activate(net)
        rho = rho_of(dq(layer) - Wdq0, D)
        activate()
        del layer.sdnq_calib_rms
        assert rho > 0.95, f'rank-8 delta under weighted cap 64 must be kept nearly whole: rho={rho:.4f}'
        assert torch.equal(dq(layer), Wdq0)
    return True


def test_calib_option_off_matches_plain():
    layer = build_layer('uint4')
    torch.manual_seed(31)
    D = torch.randn(OUT_F, IN_F, device=DEVICE) * 3e-4
    net = make_dense_net('caloff', layer, D)
    with host_rank(64), mock_model(lin=layer):
        Wdq0 = dq(layer)
        with host_calib(False):
            layer.sdnq_calib_rms = torch.rand(IN_F) + 0.5
            activate(net)
            off = dq(layer)
            activate()
        del layer.sdnq_calib_rms
        with host_calib(True):
            activate(net) # no statistics attribute: plain truncation
            plain = dq(layer)
            activate()
        assert torch.equal(off, plain), 'option off must reproduce the uncalibrated truncation bit-exact'
        assert torch.equal(dq(layer), Wdq0)
    return True


class MockCheckpointInfo:
    def __init__(self, name):
        self.name = name


class MockCalibSd:
    def __init__(self, name, **layers):
        self.transformer = MockHolder()
        for attr, lyr in layers.items():
            setattr(self.transformer, attr, lyr)
        self.sd_checkpoint_info = MockCheckpointInfo(name)


def test_calib_capture_persist_roundtrip():
    import tempfile
    from modules.lora import lora_calib
    layer_a = build_layer('uint4', seed=41)
    layer_b = build_layer('uint4', seed=42)
    sd = MockCalibSd('test/calib-model', la=layer_a, lb=layer_b)
    old_root, old_tokens = lora_calib.calib_root, lora_calib.TOKENS_DONE
    with tempfile.TemporaryDirectory() as tmp, host_calib(True):
        try:
            lora_calib.calib_root = tmp
            lora_calib.TOKENS_DONE = 2048
            lora_calib.on_model_loaded(sd)
            assert len(lora_calib.capture['handles']) == 2, 'both sub-8-bit linears must hook'
            torch.manual_seed(51)
            scale = torch.linspace(0.1, 4.0, IN_F, device=DEVICE)
            xs = []
            for _ in range(2): # exactly the completion threshold, so statistics cover every forward
                x = (torch.randn(1024, IN_F, device=DEVICE) * scale).to(torch.bfloat16)
                xs.append(x.float())
                layer_a(x)
                layer_b(x)
            assert lora_calib.capture['complete'], 'capture must complete once enough tokens are seen'
            path = lora_calib.calib_file('test/calib-model')
            assert os.path.isfile(path), f'statistics must persist to {path}'
            expected = torch.cat(xs).square().mean(dim=0).sqrt().cpu()
            assert torch.allclose(layer_a.sdnq_calib_rms, expected, rtol=1e-3, atol=1e-5), 'streamed rms must match the seen activations'
            del layer_a.sdnq_calib_rms, layer_b.sdnq_calib_rms
            lora_calib.on_model_loaded(sd) # second load takes the cached path
            assert len(lora_calib.capture['handles']) == 0, 'cached statistics must not re-attach capture hooks'
            assert torch.allclose(layer_a.sdnq_calib_rms, expected, rtol=1e-3, atol=1e-5), 'reload must restore the persisted rms'
            del layer_a.sdnq_calib_rms, layer_b.sdnq_calib_rms
        finally:
            lora_calib.calib_root, lora_calib.TOKENS_DONE = old_root, old_tokens
            lora_calib.detach_capture()
    return True


def test_calib_capture_gates():
    from modules.lora import lora_calib
    sd_int8 = MockCalibSd('test/calib-int8', lin=build_layer('int8', seed=43))
    with host_calib(True):
        lora_calib.on_model_loaded(sd_int8)
        assert len(lora_calib.capture['handles']) == 0, 'int8-only models have nothing to calibrate'
    sd_u4 = MockCalibSd('test/calib-gates', lin=build_layer('uint4', seed=44))
    with host_calib(False):
        lora_calib.on_model_loaded(sd_u4)
        assert len(lora_calib.capture['handles']) == 0, 'option off must disable capture'
    old_compile = getattr(shared.opts, 'cuda_compile', None)
    with host_calib(True):
        shared.opts.cuda_compile = ['Model']
        try:
            lora_calib.on_model_loaded(sd_u4)
            assert len(lora_calib.capture['handles']) == 0, 'model compile must disable capture'
        finally:
            shared.opts.cuda_compile = old_compile
    lora_calib.detach_capture()
    return True


CAT_FCACHE = category('factor-cache')


@contextmanager
def host_cache(gb, root):
    from modules.lora import lora_factor_cache
    old_gb = getattr(shared.opts, 'lora_sdnq_host_cache', 0)
    old_root = lora_factor_cache.cache_root
    shared.opts.lora_sdnq_host_cache = gb
    lora_factor_cache.cache_root = root
    lora_factor_cache.state.update(wn=None, sig=None, path=None, dirty=False, hits=0, misses=0)
    lora_factor_cache.state['store'] = {}
    try:
        yield lora_factor_cache
    finally:
        shared.opts.lora_sdnq_host_cache = old_gb
        lora_factor_cache.cache_root = old_root
        lora_factor_cache.state.update(wn=None, sig=None, path=None, dirty=False, hits=0, misses=0)
        lora_factor_cache.state['store'] = {}


def cache_fixture(tmp, layer, name='cachenet', sigma=3e-4, seed=61):
    """Dense net whose on-disk file exists (signature needs a stat-able path) plus a mock checkpoint identity."""
    torch.manual_seed(seed)
    D = torch.randn(OUT_F, IN_F, device=DEVICE) * sigma
    net = make_dense_net(name, layer, D)
    lora_file = os.path.join(tmp, f'{name}.safetensors')
    with open(lora_file, 'wb') as f:
        f.write(b'0' * 64)
    net.network_on_disk.filename = lora_file
    from modules.modeldata import model_data
    model_data.sd_model.sd_checkpoint_info = MockCheckpointInfo('test/cache-model')
    return net, D


def raise_no_svd(*_args, **_kwargs):
    raise AssertionError('svd must not run on a cache hit')


def test_factor_cache_roundtrip_bitexact():
    import tempfile
    layer = build_layer('uint4')
    with tempfile.TemporaryDirectory() as tmp:
        with host_rank(64), host_cache(10, os.path.join(tmp, 'cache')), mock_model(lin=layer):
            net, _D = cache_fixture(tmp, layer)
            Wdq0 = dq(layer)
            activate(net)
            first_up = layer.svd_up.detach().clone()
            first_down = layer.svd_down.detach().clone()
            activate() # pass end flushed the entry; unload restores the base
            files = os.listdir(os.path.join(tmp, 'cache'))
            assert len(files) == 1, f'one cache entry expected, got {files}'
            bf16_bytes = (first_up.numel() + first_down.numel()) * 2
            entry_bytes = os.path.getsize(os.path.join(tmp, 'cache', files[0]))
            assert entry_bytes < bf16_bytes * 0.62 + 8192, f'int8 entry must be about half the bf16 factor bytes: {entry_bytes} vs {bf16_bytes}'
            real_svd = torch.svd_lowrank
            torch.svd_lowrank = raise_no_svd
            try:
                activate(net) # same configuration: must replay from disk without touching the svd
            finally:
                torch.svd_lowrank = real_svd
            assert torch.equal(layer.svd_up, first_up), 'cache hit must replay bit-identical up factors'
            assert torch.equal(layer.svd_down, first_down), 'cache hit must replay bit-identical down factors'
            activate()
            assert torch.equal(dq(layer), Wdq0), 'unload must restore bit-exact'
    return True


def test_factor_cache_invalidates_on_multiplier():
    import tempfile
    layer = build_layer('uint4')
    with tempfile.TemporaryDirectory() as tmp:
        with host_rank(64), host_cache(10, os.path.join(tmp, 'cache')), mock_model(lin=layer):
            net, _D = cache_fixture(tmp, layer)
            activate(net)
            up_full = layer.svd_up.detach().clone()
            activate()
            net.te_multiplier = 0.7
            net.unet_multiplier = [0.7] * 3
            activate(net) # different multiplier: different signature, fresh svd, second entry
            assert not torch.equal(layer.svd_up, up_full), 'multiplier change must produce different factors'
            activate()
            files = os.listdir(os.path.join(tmp, 'cache'))
            assert len(files) == 2, f'two cache entries expected, got {files}'
    return True


def test_factor_cache_int8_quantization():
    from modules.lora import lora_factor_cache as fc
    torch.manual_seed(71)
    t = torch.randn(64, 128, device=DEVICE) * torch.logspace(-3, 0, 64, device=DEVICE)[:, None] # rows spanning magnitudes
    q, s = fc.quantize_rowwise(t)
    assert q.dtype == torch.int8
    dq = fc.dequantize_rowwise(q, s)
    err = (dq - t).abs().max(dim=1).values
    assert bool((err <= s.squeeze(1) * 0.51).all()), 'rowwise int8 error must stay within half a step'
    cos = torch.nn.functional.cosine_similarity(dq.flatten(), t.flatten(), dim=0)
    assert float(cos) > 0.99995, f'int8 roundtrip cosine {float(cos):.6f}'
    return True


def test_factor_cache_disabled_at_zero():
    import tempfile
    layer = build_layer('uint4')
    with tempfile.TemporaryDirectory() as tmp:
        with host_rank(64), host_cache(0, os.path.join(tmp, 'cache')), mock_model(lin=layer):
            net, _D = cache_fixture(tmp, layer)
            activate(net)
            activate()
            assert not os.path.isdir(os.path.join(tmp, 'cache')), 'budget 0 must write nothing'
    return True


CAT_ROBUST = category('robustness')


def test_remove_factors_after_device_move():
    layer = build_layer('uint4', use_svd=True) # checkpoint svd correction so the stash holds real tensors
    A, B, _D = make_delta()
    net = make_net('mover', layer, A, B)
    with mock_model(lin=layer):
        Wdq0 = dq(layer)
        orig_up = layer.svd_up.detach().clone()
        activate(net)
        assert hasattr(layer, 'sdnq_lora_svd_stash')
        layer.to('cpu') # offload moves registered params, never the stash tuple
        activate()
        assert layer.svd_up.device == layer.scale.device, f'restored svd must live on the layer device, got {layer.svd_up.device} vs {layer.scale.device}'
        assert torch.equal(layer.svd_up, orig_up.to('cpu')), 'restored svd values must match the original factors'
        layer.to(DEVICE)
        assert torch.equal(dq(layer), Wdq0), 'round trip must restore bit-exact'
    return True


def test_stacked_shape_mismatch_falls_back():
    from types import SimpleNamespace
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    net_good = make_net('good', layer, A, B)
    torch.manual_seed(9)
    A_bad = torch.randn(RANK, IN_F, device=DEVICE) * 0.01
    B_bad = torch.randn(OUT_F // 2, RANK, device=DEVICE) * 0.01 # wrong out_features for this layer
    net_bad = make_net('badshape', layer, A_bad, B_bad)
    prev_enl = l_common.extra_network_lora
    l_common.extra_network_lora = SimpleNamespace(errors={}) # the error path reports through the extra-networks registry
    try:
        with host_rank(0), mock_model(lin=layer):
            Wdq0 = dq(layer)
            activate(net_good)
            assert hasattr(layer, 'sdnq_lora_svd_stash')
            activate(net_good, net_bad) # must not raise: a malformed stack downgrades the layer to the legacy path
            assert not hasattr(layer, 'sdnq_lora_svd_stash'), 'shape-mismatched stack must leave factor mode'
            activate()
            assert torch.equal(dq(layer), Wdq0), 'unload must restore bit-exact pristine'
    finally:
        l_common.extra_network_lora = prev_enl
    return True


def run_tests():
    t0 = time.time()
    log.warning('=== Erasure law ===')
    for fn in [test_uint4_erases_substep_delta, test_int8_retains_delta]:
        run_test(CAT_LAW, fn)
    log.warning('=== Factor path ===')
    for fn in [test_apply_exact_and_remove_bitexact, test_multiplier_and_alpha_scaling, test_stacking_two_networks, test_matmul_layout_transposed, test_dora_falls_back, test_no_hadamard_checkpoint, test_checkpoint_svd_factors_preserved]:
        run_test(CAT_FACTOR, fn)
    log.warning('=== Memory accounting ===')
    for fn in [test_factor_path_memory_is_factors_only, test_backup_mode_clones_full_quant_state, test_fuse_mode_marker_takes_no_memory]:
        run_test(CAT_MEM, fn)
    log.warning('=== Activate integration ===')
    for fn in [test_network_activate_roundtrip]:
        run_test(CAT_E2E, fn)
    log.warning('=== Set transitions ===')
    for fn in [test_mixed_family_transition_restores_base, test_partial_coverage_layers_stay_independent,
               test_mechanism_gate_declines_candidates, test_requantize_option_routes_to_legacy_path,
               test_mechanism_flip_strips_attached_factors, test_mechanism_flip_restore_pass_strips]:
        run_test(CAT_TRANS, fn)
    log.warning('=== Hosting ===')
    for fn in [test_hosted_low_rank_delta_is_kept, test_hosted_dense_delta_beats_requant, test_hosted_skips_int8,
               test_hosted_disabled_by_option, test_hosted_transitions_and_rng_isolation]:
        run_test(CAT_HOST, fn)
    log.warning('=== Calibration ===')
    for fn in [test_calibrated_hosting_beats_plain, test_calibrated_low_rank_delta_survives, test_calib_option_off_matches_plain,
               test_calib_capture_persist_roundtrip, test_calib_capture_gates]:
        run_test(CAT_CALIB, fn)
    log.warning('=== Factor cache ===')
    for fn in [test_factor_cache_roundtrip_bitexact, test_factor_cache_invalidates_on_multiplier,
               test_factor_cache_int8_quantization, test_factor_cache_disabled_at_zero]:
        run_test(CAT_FCACHE, fn)
    log.warning('=== Robustness ===')
    for fn in [test_remove_factors_after_device_move, test_stacked_shape_mismatch_falls_back]:
        run_test(CAT_ROBUST, fn)

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
