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
- Compile: the factor add runs inside the single compiled dequant graph
  (fullgraph, no breaks) and matches the eager result; factor ranks pad to
  a fixed bucket ladder so set switches inside a bucket reuse the compiled
  graph while a novel bucket compiles exactly once, and padding changes
  the dequantized weight by nothing beyond reduction-order ulp.

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
from modules.lora import network, network_lora, lora_sdnq, lora_stack, networks  # pylint: disable=wrong-import-position
from modules.lora import lora_common as l_common   # pylint: disable=wrong-import-position
from modules.sdnq.quantizer import sdnq_quantize_layer, SDNQConfig  # pylint: disable=wrong-import-position

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_F, IN_F, RANK = 512, 512, 8
shared.opts.lora_stack_mode = 'sum' # suite baseline regardless of user config; stack tests set modes via their own context managers

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
    real_balanced = sd_models.apply_balanced_offload
    sd_models.apply_balanced_offload = lambda sd_model=None, **k: sd_model # mock pipes carry no offload state
    old_fuse = shared.opts.lora_fuse_native
    try:
        yield
    finally:
        shared.opts.lora_fuse_native = old_fuse
        sd_models.set_diffuser_offload = real_offload
        sd_models.apply_balanced_offload = real_balanced
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


CAT_STACK = category('stack-dense')


@contextmanager
def stack_mode(name, dens=None):
    old_m = getattr(shared.opts, 'lora_stack_mode', 'sum')
    old_d = getattr(shared.opts, 'lora_stack_density', 0.5)
    shared.opts.lora_stack_mode = name
    if dens is not None:
        shared.opts.lora_stack_density = dens
    try:
        yield
    finally:
        shared.opts.lora_stack_mode = old_m
        shared.opts.lora_stack_density = old_d


def test_ties_sign_consensus_drops_conflicts():
    with stack_mode('ties', dens=1.0): # density 1 disables the trim, isolating sign election
        d1 = torch.tensor([[1.0, 1.0, -1.0]], device=DEVICE)
        d2 = torch.tensor([[2.0, -0.5, -2.0]], device=DEVICE)
        out = lora_stack.combine([('a', d1), ('b', d2)], 'lora_transformer_test')
    expected = torch.tensor([[1.5, 1.0, -1.5]], device=DEVICE) # agree: mean; conflict: majority-mass side only
    assert torch.allclose(out, expected), f'{out.tolist()}'
    return True


def test_dare_mask_is_deterministic_across_calls():
    torch.manual_seed(21)
    d1 = torch.randn(64, 96, device=DEVICE) * 1e-2
    d2 = torch.randn(64, 96, device=DEVICE) * 1e-2
    with stack_mode('dare_linear', dens=0.5):
        out1 = lora_stack.combine([('a', d1), ('b', d2)], 'lora_transformer_test')
        out2 = lora_stack.combine([('a', d1), ('b', d2)], 'lora_transformer_test')
        other = lora_stack.combine([('a', d1), ('b', d2)], 'lora_transformer_other')
    assert torch.equal(out1, out2), 'same layer and nets must draw the same masks'
    assert not torch.equal(out1, other), 'a different layer must draw different masks'
    return True


def test_dare_rescales_by_inverse_density():
    torch.manual_seed(22)
    d1 = torch.randn(64, 96, device=DEVICE)
    d2 = torch.randn(64, 96, device=DEVICE)
    with stack_mode('dare_linear', dens=0.5):
        out = lora_stack.combine([('a', d1), ('b', d2)], 'lora_transformer_test')
    cands = torch.stack([torch.zeros_like(d1), 2 * d1, 2 * d2, 2 * d1 + 2 * d2])
    nearest = (cands - out.unsqueeze(0)).abs().min(dim=0).values
    assert float(nearest.max()) < 1e-5, 'every element must be a 1/density-rescaled subset sum'
    zero_frac = float((out == 0).float().mean())
    assert 0.1 < zero_frac < 0.45, f'both-dropped fraction {zero_frac} should sit near 0.25'
    return True


def test_magnitude_prune_keeps_top_density():
    torch.manual_seed(23)
    d1 = torch.randn(128, 64, device=DEVICE)
    d2 = torch.zeros_like(d1) # inert second delta isolates the trim
    with stack_mode('magnitude_prune', dens=0.25):
        out = lora_stack.combine([('a', d1), ('b', d2)], 'lora_transformer_test')
    kept = out != 0
    frac = float(kept.float().mean())
    assert 0.2 < frac < 0.3, f'kept fraction {frac}'
    assert torch.equal(out[kept], d1[kept]), 'kept elements must pass through unchanged'
    assert float(d1.abs()[~kept].max()) <= float(d1.abs()[kept].min()) + 1e-6, 'kept set must be the top magnitudes'
    return True


def test_dense_two_plain_loras_hosted_not_summed():
    layer = build_layer('uint4')
    A1, B1, D1 = make_delta(seed=31, sigma=1e-2)
    A2, B2, D2 = make_delta(seed=32, sigma=1e-2)
    n1 = make_net('td1', layer, A1, B1)
    n2 = make_net('td2', layer, A2, B2)
    with host_rank(64), mock_model(lin=layer):
        Wdq0 = dq(layer)
        with stack_mode('ties', dens=0.5):
            activate(n1, n2)
            # hosted at rank 64 leaves a rank-64 factor bucket; the exact concat of two rank-8 nets would leave 16
            assert layer.svd_up.shape[1] == 64, f'dense mode must route a factorable pair to hosting, rank={layer.svd_up.shape[1]}'
            eff = dq(layer) - Wdq0
            activate()
        assert torch.equal(dq(layer), Wdq0), 'removal must restore bit-exact'
        with stack_mode('ties', dens=0.5):
            ref = lora_stack.combine([('td1', D1), ('td2', D2)], 'lora_transformer_test')
    s = D1 + D2
    assert float((eff - s).norm() / s.norm()) > 0.05, 'ties result must differ from the plain sum'
    assert rho_of(eff, ref) > 0.8, f'hosted ties delta must track the ties reference, rho={rho_of(eff, ref):.3f}' # rank-64 truncation of the densified delta keeps ~0.89
    assert float((eff - ref).norm()) < float((eff - s).norm()), 'hosted result must sit closer to the ties reference than to the plain sum'
    return True


def test_single_net_ignores_dense_mode():
    layer = build_layer('uint4')
    A, B, D = make_delta(seed=33)
    net = make_net('solo', layer, A, B)
    with mock_model(lin=layer), stack_mode('ties', dens=0.5):
        Wdq0 = dq(layer)
        activate(net)
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'single net must stay on the exact factor path'
        assert rho_of(dq(layer) - Wdq0, D) > 0.99
        activate()
        assert torch.equal(dq(layer), Wdq0)
    return True


def test_te_layer_stays_plain_sum():
    layer = build_layer('uint4')
    layer.network_layer_name = 'lora_te_test'
    A1, B1, D1 = make_delta(seed=34)
    A2, B2, D2 = make_delta(seed=35)
    n1 = make_net('te1', layer, A1, B1)
    n2 = make_net('te2', layer, A2, B2)
    with mock_model(lin=layer), stack_mode('ties', dens=0.5):
        Wdq0 = dq(layer)
        activate(n1, n2)
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'te layers must stay on the exact concat path'
        assert rho_of(dq(layer) - Wdq0, D1 + D2) > 0.99
        activate()
        assert torch.equal(dq(layer), Wdq0)
    return True


def test_sum_mode_keeps_exact_stacking():
    layer = build_layer('uint4')
    A1, B1, D1 = make_delta(seed=36)
    A2, B2, D2 = make_delta(seed=37)
    n1 = make_net('s1', layer, A1, B1)
    n2 = make_net('s2', layer, A2, B2)
    with mock_model(lin=layer), stack_mode('sum'):
        Wdq0 = dq(layer)
        activate(n1, n2)
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'sum mode must keep the exact concat path'
        assert layer.svd_up.shape[1] == 16, f'sum mode must concat exactly, rank={layer.svd_up.shape[1]}'
        assert rho_of(dq(layer) - Wdq0, D1 + D2) > 0.99
        activate()
        assert torch.equal(dq(layer), Wdq0)
    return True


CAT_SELECT = category('stack-select')


@contextmanager
def select_mode(name, alpha=None, disc=None):
    old = {k: getattr(shared.opts, k, None) for k in ('lora_stack_mode', 'lora_stack_alpha', 'lora_stack_discrepancy')}
    shared.opts.lora_stack_mode = name
    if alpha is not None:
        shared.opts.lora_stack_alpha = alpha
    if disc is not None:
        shared.opts.lora_stack_discrepancy = disc
    lora_stack.clear()
    lora_stack.warned.clear()
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(shared.opts, k, v)
        lora_stack.clear()


def select_pair(layer, seed0=41, seed1=42, scale1=1.0):
    A1, B1, D1 = make_delta(seed=seed0, sigma=1e-2)
    A2, B2, D2 = make_delta(seed=seed1, sigma=1e-2)
    if scale1 != 1.0:
        A2, D2 = A2 * scale1, D2 * scale1
    n1 = make_net('subject', layer, A1, B1)
    n2 = make_net('style', layer, A2, B2)
    return n1, n2, D1, D2


def test_select_flip_schedule_end_to_end():
    layer = build_layer('uint4')
    n1, n2, D1, D2 = select_pair(layer)
    with mock_model(lin=layer), select_mode('klora', alpha=1.5):
        Wdq0 = dq(layer)
        activate(n1, n2)
        entry = lora_stack.state['entries'].get('lora_transformer_test')
        assert entry is not None and entry['kind'] == 'factor', 'a factorable pair must register factor segments'
        assert entry['segments'][0] == (0, 8) and entry['segments'][1] == (8, 16), f'segments {entry["segments"]}'
        total = 20
        lora_stack.reset(total)
        flips = [s for s, layers in lora_stack.state['flips'].items() for _ in layers]
        assert len(flips) <= 1, 'a monotone ramp allows at most one flip per layer'
        eff0 = dq(layer) - Wdq0
        winner0 = 0 if rho_of(eff0, D1) > rho_of(eff0, D2) else 1
        for s in range(total):
            lora_stack.on_step(s)
        eff1 = dq(layer) - Wdq0
        if flips:
            assert rho_of(eff1, D2) > 0.99, 'after the flip the style delta must be selected'
            assert rho_of(eff0, D1) > 0.99, 'before the flip the subject delta must be selected'
        else:
            assert rho_of(eff1, [D1, D2][winner0]) > 0.99
        activate()
        assert torch.equal(dq(layer), Wdq0), 'removal from an end-of-schedule state must restore bit-exact'
    return True


def test_select_initial_style_when_ramp_starts_won():
    layer = build_layer('uint4')
    n1, n2, _D1, D2 = select_pair(seed0=43, seed1=44, scale1=8.0, layer=layer) # style delta dominates
    with mock_model(lin=layer), select_mode('estlora', alpha=1.0, disc=0.5):
        Wdq0 = dq(layer)
        activate(n1, n2)
        lora_stack.reset(20)
        eff = dq(layer) - Wdq0
        assert rho_of(eff, D2) > 0.99, 'a layer whose style side wins at step 0 must start style-selected'
    return True


def test_select_flip_is_inplace_and_shape_stable():
    layer = build_layer('uint4')
    n1, n2, _D1, _D2 = select_pair(layer, seed0=45, seed1=46)
    with mock_model(lin=layer), select_mode('klora'):
        activate(n1, n2)
        param_id = id(layer.svd_up)
        shape = tuple(layer.svd_up.shape)
        lora_stack.reset(20)
        entry = lora_stack.state['entries']['lora_transformer_test']
        (s0, s1), (t0, t1), transposed = entry['segments']
        zeroed = lora_stack.segment_view(layer.svd_up.data, t0, t1, transposed)
        kept = lora_stack.segment_view(layer.svd_up.data, s0, s1, transposed)
        assert float(zeroed.abs().sum()) == 0.0 or float(kept.abs().sum()) == 0.0, 'exactly one segment must be zeroed initially'
        for s in range(20):
            lora_stack.on_step(s)
        assert id(layer.svd_up) == param_id and tuple(layer.svd_up.shape) == shape, 'flips must mutate in place, never reassign'
    return True


def test_select_matmul_transposed_layout():
    layer = build_layer('uint4', use_quantized_matmul=True)
    n1, n2, D1, D2 = select_pair(layer, seed0=47, seed1=48)
    with mock_model(lin=layer), select_mode('klora'):
        Wdq0 = dq(layer)
        activate(n1, n2)
        entry = lora_stack.state['entries']['lora_transformer_test']
        assert entry['segments'][2] is True, 'quantized-matmul layout must register as transposed'
        lora_stack.reset(20)
        eff = dq(layer) - Wdq0
        assert max(rho_of(eff, D1), rho_of(eff, D2)) > 0.99, 'initial selection must realize one delta exactly'
        activate()
        assert torch.equal(dq(layer), Wdq0)
    return True


def test_select_per_net_hosted_pair():
    layer = build_layer('uint4')
    torch.manual_seed(49)
    Dd1 = (torch.randn(OUT_F, 24, device=DEVICE) @ torch.randn(24, IN_F, device=DEVICE)) * 1e-3 # rank inside the host cap so truncation is near-lossless
    Dd2 = (torch.randn(OUT_F, 24, device=DEVICE) @ torch.randn(24, IN_F, device=DEVICE)) * 1e-3
    n1 = make_dense_net('lk1', layer, Dd1)
    n2 = make_dense_net('lk2', layer, Dd2)
    with host_rank(32), mock_model(lin=layer), select_mode('klora'):
        Wdq0 = dq(layer)
        activate(n1, n2)
        entry = lora_stack.state['entries'].get('lora_transformer_test')
        assert entry is not None, 'non-factorable pairs must register through per-net hosting'
        assert entry['segments'][0] == (0, 32) and entry['segments'][1] == (32, 64), f'segments {entry["segments"]}'
        lora_stack.reset(20)
        eff = dq(layer) - Wdq0
        best = max(rho_of(eff, Dd1), rho_of(eff, Dd2))
        assert best > 0.9, f'initial selection must realize one hosted delta, rho={best:.3f}'
        activate()
        assert torch.equal(dq(layer), Wdq0)
    return True


def test_select_reset_restores_initial_state():
    layer = build_layer('uint4')
    n1, n2, _D1, _D2 = select_pair(layer, seed0=51, seed1=52)
    with mock_model(lin=layer), select_mode('klora'):
        activate(n1, n2)
        lora_stack.reset(20)
        initial = dq(layer)
        for s in range(20):
            lora_stack.on_step(s)
        lora_stack.reset(20)
        assert torch.equal(dq(layer), initial), 'a fresh pass must restore the initial selection without re-activation'
    return True


def test_select_deactivate_from_midflip():
    layer = build_layer('uint4')
    n1, n2, _D1, _D2 = select_pair(layer, seed0=53, seed1=54)
    with mock_model(lin=layer), select_mode('klora'):
        Wdq0 = dq(layer)
        activate(n1, n2)
        lora_stack.reset(20)
        for s in range(10):
            lora_stack.on_step(s)
        activate()
        assert torch.equal(dq(layer), Wdq0), 'removal mid-schedule must restore bit-exact'
        assert not lora_stack.state['entries'], 'removal must drop the selection entry'
    return True


def test_select_requires_exactly_two_nets():
    layer = build_layer('uint4')
    A3, B3, _D3 = make_delta(seed=55)
    n1, n2, _D1, _D2 = select_pair(layer, seed0=56, seed1=57)
    n3 = make_net('third', layer, A3, B3)
    with mock_model(lin=layer), select_mode('klora'):
        activate(n1, n2, n3)
        assert hasattr(layer, 'sdnq_lora_svd_stash'), 'three nets must fall back to the exact concat path'
        assert not lora_stack.state['entries'], 'no selection entries outside the two-net case'
        activate()
    return True


def test_select_gated_off_when_compiled():
    layer = build_layer('uint4')
    n1, n2, _D1, _D2 = select_pair(layer, seed0=58, seed1=59)
    old_compile = getattr(shared.opts, 'cuda_compile', None)
    try:
        shared.opts.cuda_compile = ['Model']
        with mock_model(lin=layer), select_mode('klora'):
            activate(n1, n2)
            assert not lora_stack.state['entries'], 'select must gate off under model compile'
            assert hasattr(layer, 'sdnq_lora_svd_stash'), 'gated select behaves as sum'
            activate()
    finally:
        shared.opts.cuda_compile = old_compile
    return True


def test_est_energy_matches_full_frobenius():
    torch.manual_seed(60)
    up = torch.randn(64, 8, device=DEVICE)
    down = torch.randn(8, 96, device=DEVICE)
    gram = lora_stack.score_energy(up, down)
    full = float((up @ down).square().sum())
    assert abs(gram - full) / full < 1e-5, f'{gram} vs {full}'
    return True


def test_select_weight_kind_plain_layer():
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(OUT_F, IN_F, device=DEVICE) * 0.02)
    lin.network_layer_name = 'lora_transformer_plain'
    lin.network_current_names = ()
    A1, B1, D1 = make_delta(seed=61, sigma=1e-2)
    A2, B2, D2 = make_delta(seed=62, sigma=1e-2)
    n1 = make_net('w1', lin, A1, B1)
    n2 = make_net('w2', lin, A2, B2)
    W0 = lin.weight.detach().float().clone()
    with mock_model(lin=lin), select_mode('klora'):
        activate(n1, n2)
        entry = lora_stack.state['entries'].get('lora_transformer_plain')
        assert entry is not None and entry['kind'] == 'weight', 'plain layers must register weight-kind selection'
        assert torch.equal(lin.weight.detach().float(), W0), 'weights stay pristine until the schedule applies a winner'
        lora_stack.reset(20)
        eff = lin.weight.detach().float() - W0
        assert max(rho_of(eff, D1), rho_of(eff, D2)) > 0.95, 'initial selection must apply one delta from backup'
        for s in range(20):
            lora_stack.on_step(s)
        activate()
        assert torch.equal(lin.weight.detach().float(), W0), 'restore-only pass must return the pristine weight'
    return True


CAT_COMPILE = category('compile')


def dq_compiled(layer):
    # the production entry: skip_compile left at its default so the shared compiled dequant runs
    return layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=layer.zero_point,
                                  svd_up=layer.svd_up, svd_down=layer.svd_down,
                                  skip_quantized_matmul=layer.sdnq_dequantizer.use_quantized_matmul,
                                  dtype=torch.float32)


def graph_stats():
    from torch._dynamo.utils import counters
    return int(counters['stats']['unique_graphs']), sum(counters['graph_break'].values())


def test_factor_add_inside_compiled_graph():
    from modules.sdnq.common import use_torch_compile
    if not use_torch_compile:
        return True # compile disabled at sdnq import (no triton); nothing to pin
    import torch._dynamo
    from torch._dynamo.utils import counters
    layer = build_layer('uint4')
    A, B, _D = make_delta()
    dtype = layer.sdnq_dequantizer.result_dtype
    torch._dynamo.reset()
    counters.clear()
    lora_sdnq.append_factors(layer, [B.to(dtype)], [A.to(dtype)])
    W_c = dq_compiled(layer)
    graphs, breaks = graph_stats()
    assert breaks == 0, f'graph breaks in the compiled dequant: {breaks}'
    assert graphs == 1, f'factor-bearing dequant must be one compiled region, got {graphs} graphs'
    W_e = dq(layer)
    assert torch.allclose(W_c, W_e, rtol=1e-3, atol=1e-4), f'compiled vs eager dequant diverged, max {float((W_c - W_e).abs().max()):.3e}'
    lora_sdnq.remove_factors(layer)
    return True


def test_rank_bucket_graph_reuse():
    from modules.sdnq.common import use_torch_compile
    if not use_torch_compile:
        return True
    import torch._dynamo
    from torch._dynamo.utils import counters
    import modules.sdnq.common as sdnq_common
    layer = build_layer('uint4', use_hadamard=False)
    dtype = layer.sdnq_dequantizer.result_dtype
    torch.manual_seed(13)
    mk = lambda r: (torch.randn(OUT_F, r, device=DEVICE, dtype=dtype) * 0.01, torch.randn(r, IN_F, device=DEVICE, dtype=dtype) * 0.01)
    B8, A8 = mk(8)
    B6, A6 = mk(6)
    B24, A24 = mk(24)

    torch._dynamo.reset()
    counters.clear()
    lora_sdnq.append_factors(layer, [B8], [A8])
    assert layer.svd_up.shape[1] == 8, f'rank 8 must bucket to 8, got {layer.svd_up.shape[1]}'
    dq_compiled(layer)
    g_first, _ = graph_stats()

    lora_sdnq.remove_factors(layer)
    lora_sdnq.append_factors(layer, [B6], [A6])
    assert layer.svd_up.shape[1] == 8, f'rank 6 must pad to bucket 8, got {layer.svd_up.shape[1]}'
    assert float(layer.svd_up[:, 6:].abs().sum()) == 0.0, 'pad columns must be exact zeros'
    dq_compiled(layer)
    g_same, _ = graph_stats()
    assert g_same == g_first, f'same bucket must reuse the graph: {g_first} -> {g_same}'

    lora_sdnq.remove_factors(layer)
    lora_sdnq.append_factors(layer, [B24], [A24])
    assert layer.svd_up.shape[1] == 32, f'rank 24 must pad to bucket 32, got {layer.svd_up.shape[1]}'
    dq_compiled(layer)
    g_novel, _ = graph_stats()
    assert g_novel == g_first + 1, f'novel bucket must compile exactly one new graph: {g_first} -> {g_novel}'

    lora_sdnq.remove_factors(layer)
    lora_sdnq.append_factors(layer, [B8], [A8])
    dq_compiled(layer)
    g_back, _ = graph_stats()
    assert g_back == g_novel, f'returning to a seen bucket must be free: {g_novel} -> {g_back}'

    W_padded = dq(layer)
    lora_sdnq.remove_factors(layer)
    old_flag = sdnq_common.use_torch_compile
    sdnq_common.use_torch_compile = False
    try:
        lora_sdnq.append_factors(layer, [B8], [A8])
        assert layer.svd_up.shape[1] == 8
        W_unpadded = dq(layer)
    finally:
        sdnq_common.use_torch_compile = old_flag
        lora_sdnq.remove_factors(layer)
    assert torch.allclose(W_padded, W_unpadded, rtol=0.0, atol=1e-6), f'padding must be inert beyond reduction-order ulp, max {float((W_padded - W_unpadded).abs().max()):.3e}'
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
    for fn in [test_mixed_family_transition_restores_base, test_partial_coverage_layers_stay_independent]:
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
    log.warning('=== Stack modes: dense ===')
    for fn in [test_ties_sign_consensus_drops_conflicts, test_dare_mask_is_deterministic_across_calls, test_dare_rescales_by_inverse_density,
               test_magnitude_prune_keeps_top_density, test_dense_two_plain_loras_hosted_not_summed, test_single_net_ignores_dense_mode,
               test_te_layer_stays_plain_sum, test_sum_mode_keeps_exact_stacking]:
        run_test(CAT_STACK, fn)
    log.warning('=== Stack modes: select ===')
    for fn in [test_select_flip_schedule_end_to_end, test_select_initial_style_when_ramp_starts_won, test_select_flip_is_inplace_and_shape_stable,
               test_select_matmul_transposed_layout, test_select_per_net_hosted_pair, test_select_reset_restores_initial_state,
               test_select_deactivate_from_midflip, test_select_requires_exactly_two_nets, test_select_gated_off_when_compiled,
               test_est_energy_matches_full_frobenius, test_select_weight_kind_plain_layer]:
        run_test(CAT_SELECT, fn)
    log.warning('=== Compile ===')
    for fn in [test_factor_add_inside_compiled_graph, test_rank_bucket_graph_reuse]:
        run_test(CAT_COMPILE, fn)
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
