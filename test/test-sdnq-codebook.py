#!/usr/bin/env python
"""
Offline unit tests for SDNQ codebook quantization (cb2-cb6).

Pins the properties the codebook dtypes are built on:

- The Lloyd-Max fit never loses to the uniform affine grid it is initialized
  from: per-iteration MSE is monotone, so the fitted levels beat or match the
  matching uintN grid on the same tensor, across weight distributions. The
  zero-iteration fit reproduces the affine grid exactly.
- Packed index storage roundtrips bit-exact through the uintN packers at
  every registered width, including grouped 3-D layouts.
- Every dequant path agrees: the weight-only path equals the hand-computed
  codebook[idx] * scale reference bit-exact, the small-batch matmul bypass
  equals the dequantized F.linear, and the int8-GEMM path stays within the
  activation-quantization error band measured on int8/uint4 controls.
- A zero output channel cannot poison the shared codebook.
- The dynamic-quantization ladder walks codebook rungs only when the
  requested dtype is itself a codebook type.
- Layer state serializes the codebook and the pre-quantized load path keeps
  its stored dtype (int8 fresh, fp32 ingested) under dequantize_fp32.
- The asym_w4a8_int8 ingest adoption reproduces the container's dequant
  exactly: nibble order, folded per-channel x per-group scales, verbatim
  fp32 codebook, and the convrot (Hadamard) round trip.

All tensors are synthetic; no model files or running server required.

Usage:
    python test/test-sdnq-codebook.py
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
_orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = _orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from modules.errors import log  # pylint: disable=wrong-import-position
from modules.sdnq.common import dtype_dict, weights_dtype_order, sdnq_keys  # pylint: disable=wrong-import-position
from modules.sdnq.quant_utils import fit_codebook, quantize_weight_codebook, quantize_weight, rotate_hadamard, get_hadamard  # pylint: disable=wrong-import-position
from modules.sdnq.dequantizer import dequantize_weight, dequantize_asymmetric  # pylint: disable=wrong-import-position
from modules.sdnq.packed_int import pack_int, unpack_int  # pylint: disable=wrong-import-position
from modules.sdnq.quantizer import sdnq_quantize_layer, sdnq_quantize_layer_weight, sdnq_quantize_layer_weight_dynamic, SDNQConfig, SDNQQuantizer  # pylint: disable=wrong-import-position
from modules.lora.lora_sdnq import effective_grid_step  # pylint: disable=wrong-import-position
from pipelines.native_transformer import adopt_asym_w4a8_layer, OverrideArchMismatch  # pylint: disable=wrong-import-position

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_F, IN_F = 256, 512
CB_DTYPES = ('cb2', 'cb3', 'cb4', 'cb5', 'cb6')

results: dict[str, dict] = {}

CAT_FIT = 'fit'
CAT_PACK = 'pack'
CAT_DEQUANT = 'dequant'
CAT_FORWARD = 'forward'
CAT_LADDER = 'ladder'
CAT_STATE = 'state'
CAT_LORA = 'lora'
CAT_INGEST = 'ingest'


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


def weight_distributions(seed: int = 0) -> dict[str, torch.Tensor]:
    # spread of shapes the fit must hold on: symmetric, skewed, heavy-tailed, bimodal
    g = torch.Generator(device='cpu').manual_seed(seed)
    base = torch.randn(OUT_F, IN_F, generator=g) * 0.04
    return {
        'gaussian': base.clone(),
        'skewed': (base.abs() ** 1.5) * torch.sign(base + 0.3),
        'heavy_tailed': base * (1.0 + base.abs() * 25.0),
        'bimodal': base + torch.sign(base) * 0.05,
    }


def normalized_values(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp_(min=torch.finfo(torch.float32).tiny)
    return (weight.float() / scale)


def assignment_mse(values: torch.Tensor, levels: torch.Tensor) -> float:
    midpoints = levels[1:].add(levels[:-1]).mul(0.5)
    picked = levels[torch.bucketize(values, midpoints)]
    return float(torch.mean((values - picked) ** 2))


def measure_fit_floor() -> float:
    # noise floor of the fit itself: scatter_add on CUDA is non-deterministic, so the
    # same input can produce slightly different levels; every fitted-MSE comparison
    # below must carry this measured floor, not an assumed one
    values = normalized_values(weight_distributions(seed=7)['gaussian']).to(DEVICE)
    mses = [assignment_mse(values, fit_codebook(values, 16)) for _ in range(2)]
    return abs(mses[0] - mses[1])


FIT_FLOOR = None


def test_fit_deterministic_on_cpu():
    values = normalized_values(weight_distributions(seed=1)['gaussian'])
    a = fit_codebook(values, 16)
    b = fit_codebook(values, 16)
    assert torch.equal(a, b), 'cpu fit is not deterministic'
    return True


def test_fit_zero_iterations_is_affine_grid():
    values = normalized_values(weight_distributions(seed=2)['skewed'])
    levels = fit_codebook(values, 16, iterations=0)
    w_min, w_max = values.min(), values.max()
    grid = torch.arange(16, dtype=values.dtype).div(15).mul(w_max - w_min).add(w_min)
    assert torch.allclose(levels, grid, atol=0, rtol=0), 'zero-iteration fit must equal the uniform min/max grid'
    return True


def test_fit_beats_affine_grid():
    # the invariant the dtype is built on: init == affine grid + monotone iterations
    for name, weight in weight_distributions(seed=3).items():
        values = normalized_values(weight).to(DEVICE)
        for num_levels in (4, 16, 64):
            init = fit_codebook(values, num_levels, iterations=0)
            fitted = fit_codebook(values, num_levels)
            mse_init = assignment_mse(values, init)
            mse_fit = assignment_mse(values, fitted)
            assert mse_fit <= mse_init + FIT_FLOOR, f'{name} n={num_levels}: fitted {mse_fit:.3e} > affine {mse_init:.3e} + floor {FIT_FLOOR:.1e}'
    return True


def test_quantize_beats_uint4_end_to_end():
    # full path incl. the int8 level snap; the snap moves each level by at most
    # 0.5/127 of the level range, far below 4-bit quantization noise, so a 1.05
    # margin over the exact pre-snap invariant covers it with slack to spare
    ratios = []
    for name, weight in weight_distributions(seed=4).items():
        w = weight.to(DEVICE)
        q, scale, codebook = quantize_weight_codebook(w.float(), -1, 'cb4')
        cb_dq = codebook.to(torch.float32)[q.to(torch.int32)] * scale
        cb_mse = float(torch.mean((w.float() - cb_dq) ** 2))
        aq, ascale, azp = quantize_weight(w.float(), -1, 'uint4')
        affine_dq = aq.to(torch.float32) * ascale + azp
        affine_mse = float(torch.mean((w.float() - affine_dq) ** 2))
        ratios.append((name, cb_mse / affine_mse))
        assert cb_mse <= affine_mse * 1.05, f'{name}: cb4 mse {cb_mse:.3e} vs uint4 {affine_mse:.3e}'
    log.info('  cb4/uint4 mse ratios: ' + ', '.join(f'{n}={r:.3f}' for n, r in ratios))
    return True


def test_zero_row_does_not_poison_codebook():
    w = weight_distributions(seed=5)['gaussian'].to(DEVICE)
    w[0].zero_()
    q, scale, codebook = quantize_weight_codebook(w.float(), -1, 'cb4')
    assert torch.isfinite(codebook.to(torch.float32)).all(), 'codebook has non-finite levels'
    assert torch.isfinite(scale).all(), 'scale has non-finite values'
    dq = codebook.to(torch.float32)[q.to(torch.int32)] * scale
    # the clamped scale bounds the row at |level| * tiny * level_absmax / 127 <= tiny,
    # which flushes to zero in any compute dtype; exact zero is not the invariant
    assert float(dq[0].abs().max()) <= torch.finfo(torch.float32).tiny, f'zero row dequant {float(dq[0].abs().max()):.3e} above the clamp bound'
    err = float(torch.mean((w[1:].float() - dq[1:]) ** 2)) / float(w[1:].float().square().mean())
    assert err < 0.05, f'other rows degraded: relative mse {err:.3e}'
    return True


def test_fit_survives_clusters_past_fp32_count_limit():
    # a float32 count accumulator stops incrementing at 2^24 (16777216 + 1 == 16777216
    # in fp32); per-channel normalization of outlier-heavy weights concentrates a
    # 100M-element tensor into a few central clusters that stay past that cliff for
    # every iteration, so the skewed means never heal; 16384x6144 with boosted input
    # channels is the smallest real geometry that crosses it, and the affine invariant
    # must hold there like everywhere else
    g = torch.Generator(device='cpu').manual_seed(11)
    w = (torch.randn(16384, 6144, generator=g) * 0.02)
    w[:, [1, 3072, 6142]] *= 8.0
    w = w.to(DEVICE)
    q, scale, codebook = quantize_weight_codebook(w.float(), -1, 'cb4')
    cb_dq = codebook.to(torch.float32)[q.to(torch.int32)] * scale
    cb_mse = float(torch.mean((w.float() - cb_dq) ** 2))
    del q, scale, codebook, cb_dq
    aq, ascale, azp = quantize_weight(w.float(), -1, 'uint4')
    affine_dq = aq.to(torch.float32) * ascale + azp
    affine_mse = float(torch.mean((w.float() - affine_dq) ** 2))
    assert cb_mse <= affine_mse * 1.05, f'cb4 mse {cb_mse:.3e} vs uint4 {affine_mse:.3e} at 16384x6144'
    return True


def test_pack_roundtrip_all_widths():
    g = torch.Generator(device='cpu').manual_seed(6)
    for wdt in CB_DTYPES:
        num_levels = dtype_dict[wdt]['max'] + 1
        for shape in ((32, 64), (16, 8, 24)):  # 2-D per-channel and 3-D grouped layouts
            idx = torch.randint(0, num_levels, shape, generator=g, dtype=torch.uint8)
            packed = pack_int(idx.clone(), wdt)
            unpacked = unpack_int(packed, wdt, torch.Size(shape))
            assert torch.equal(unpacked.to(torch.uint8), idx), f'{wdt} {shape}: pack/unpack mismatch'
    return True


def test_dequant_matches_reference():
    w = weight_distributions(seed=8)['gaussian'].to(DEVICE)
    q, scale, codebook = quantize_weight_codebook(w.float(), -1, 'cb4')
    packed = pack_int(q.clone(), 'cb4')
    ref = (codebook.to(torch.float32)[q.to(torch.int32)] * scale)
    out = dequantize_weight('cb4', packed, scale, codebook=codebook, quantized_weight_shape=q.shape, dtype=torch.float32)
    assert torch.equal(out, ref), f'weight-only dequant differs from reference by {float((out - ref).abs().max()):.3e}'
    return True


def test_dequant_grouped_matches_reference():
    groups, group_size = 8, IN_F // 8
    w = weight_distributions(seed=9)['heavy_tailed'].to(DEVICE).float()
    wg = w.unflatten(-1, (groups, group_size))
    q, scale, codebook = quantize_weight_codebook(wg, -1, 'cb4')
    packed = pack_int(q.clone(), 'cb4')
    ref = (codebook.to(torch.float32)[q.to(torch.int32)] * scale).view(OUT_F, IN_F)
    out = dequantize_weight('cb4', packed, scale, codebook=codebook, quantized_weight_shape=q.shape, result_shape=torch.Size((OUT_F, IN_F)), dtype=torch.float32)
    assert torch.equal(out, ref), 'grouped dequant differs from reference'
    return True


def build_layer(weights_dtype='cb4', use_quantized_matmul=False, use_hadamard=False, use_svd=False, seed=0, group_size=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04)
    reference = lin.weight.detach().clone()
    cfg = SDNQConfig(weights_dtype=weights_dtype, group_size=group_size, hadamard_group_size=256, use_hadamard=use_hadamard,
                     use_svd=use_svd, svd_rank=32, use_quantized_matmul=use_quantized_matmul, dequantize_fp32=False,
                     quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=torch.bfloat16, param_name='test.weight')
    return layer, reference


def dequantized_weight(layer) -> torch.Tensor:
    return layer.sdnq_dequantizer(
        layer.weight, layer.scale, zero_point=layer.zero_point,
        svd_up=layer.svd_up, svd_down=layer.svd_down, codebook=layer.codebook,
        skip_quantized_matmul=layer.sdnq_dequantizer.use_quantized_matmul, skip_compile=True,
    )


def test_layer_forward_matches_dequant_linear():
    layer, _ = build_layer('cb4')
    x = torch.randn(4, IN_F, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x)
    ref = torch.nn.functional.linear(x, dequantized_weight(layer))  # pylint: disable=not-callable
    assert torch.equal(out, ref), 'weight-only forward differs from dequantized linear'
    return True


def test_matmul_bypass_matches_dequant_linear():
    # auto resolves cb4 to half the affine group (32): storage parity with uint4's
    # scale+zero_point pair at measurably better fidelity, on the requant matmul path
    default_layer, _ = build_layer('cb4', use_quantized_matmul=True)
    assert default_layer.sdnq_dequantizer.group_size == 32, f'cb4 auto group {default_layer.sdnq_dequantizer.group_size}, expected 32'
    assert default_layer.sdnq_dequantizer.re_quantize_for_matmul, 'grouped cb4 must take the requant matmul path'
    layer, _ = build_layer('cb4', use_quantized_matmul=True, group_size=-1)
    assert layer.sdnq_dequantizer.use_quantized_matmul, 'matmul not enabled on the layer'
    assert not layer.sdnq_dequantizer.re_quantize_for_matmul, 'row-wise cb4 must take the direct matmul path'
    x = torch.randn(4, IN_F, device=DEVICE, dtype=torch.bfloat16)  # < 32 rows: the small-batch bypass
    out = layer(x)
    ref = torch.nn.functional.linear(x, dequantized_weight(layer))  # pylint: disable=not-callable
    assert torch.equal(out, ref), 'small-batch bypass differs from dequantized linear'
    return True


def gemm_vs_dequant_relerr(weights_dtype: str) -> float:
    layer, _ = build_layer(weights_dtype, use_quantized_matmul=True)
    x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x).float()
    ref = torch.nn.functional.linear(x, dequantized_weight(layer)).float()  # pylint: disable=not-callable
    return float((out - ref).norm() / ref.norm().clamp(min=1e-12))


def test_gemm_path_error_within_control_band():
    # the GEMM-vs-dequant gap is dominated by dynamic int8 activation quantization,
    # identical machinery for every integer dtype; int8 and uint4 are the controls
    # and 2x their band guards the cb-specific step (unpack + LUT gather) - a wrong
    # nibble order or LUT produces ~100% relative error, far outside any band
    controls = {d: gemm_vs_dequant_relerr(d) for d in ('int8', 'uint4')}
    cb = gemm_vs_dequant_relerr('cb4')
    bound = max(controls.values()) * 2.0
    log.info(f'  gemm relerr: cb4={cb:.4f} controls={ {k: round(v, 4) for k, v in controls.items()} } bound={bound:.4f}')
    assert cb <= bound, f'cb4 gemm relerr {cb:.4f} above control band {bound:.4f}'
    return True


def test_matmul_with_hadamard_and_svd():
    for kwargs in (dict(use_hadamard=True), dict(use_svd=True), dict(use_hadamard=True, use_svd=True)):
        layer, reference = build_layer('cb4', use_quantized_matmul=True, **kwargs)
        x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
        out = layer(x).float()
        ref = torch.nn.functional.linear(x, reference).float()  # pylint: disable=not-callable
        relerr = float((out - ref).norm() / ref.norm())
        assert relerr < 0.2, f'{kwargs}: relative error vs bf16 reference {relerr:.3f}'
    return True


def test_cb4_beats_int4_against_reference():
    # control comparison on identical weights and inputs: the codebook grid must not
    # lose to the uniform int4 grid it replaces
    errs = {}
    for wdt in ('cb4', 'int4'):
        layer, reference = build_layer(wdt, seed=11)
        x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
        out = layer(x).float()
        ref = torch.nn.functional.linear(x, reference).float()  # pylint: disable=not-callable
        errs[wdt] = float((out - ref).norm() / ref.norm())
    log.info(f'  output relerr vs bf16: cb4={errs["cb4"]:.4f} int4={errs["int4"]:.4f}')
    assert errs['cb4'] <= errs['int4'], f'cb4 {errs["cb4"]:.4f} worse than int4 {errs["int4"]:.4f}'
    return True


def test_conv_forward_matches_dequant():
    torch.manual_seed(12)
    conv = torch.nn.Conv2d(16, 32, 3, padding=1, bias=False, dtype=torch.bfloat16, device=DEVICE)
    cfg = SDNQConfig(weights_dtype='cb4', quant_conv=True, dequantize_fp32=False,
                     quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(conv, cfg, torch_dtype=torch.bfloat16, param_name='test.weight')
    assert layer.codebook is not None, 'conv layer has no codebook'
    x = torch.randn(1, 16, 8, 8, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x)
    dq = layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=layer.svd_up, svd_down=layer.svd_down, codebook=layer.codebook, skip_compile=True)
    ref = torch.nn.functional.conv2d(x, dq, padding=1)  # pylint: disable=not-callable
    assert torch.equal(out, ref), 'conv forward differs from dequantized conv'
    return True


def test_embedding_forward_matches_dequant():
    torch.manual_seed(13)
    emb = torch.nn.Embedding(64, 128, dtype=torch.bfloat16, device=DEVICE)
    cfg = SDNQConfig(weights_dtype='cb4', quant_embedding=True, dequantize_fp32=False,
                     quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(emb, cfg, torch_dtype=torch.bfloat16, param_name='test.weight')
    assert layer.codebook is not None, 'embedding layer has no codebook'
    ids = torch.tensor([[0, 5, 63], [7, 7, 1]], device=DEVICE)
    out = layer(ids)
    dq = layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=layer.svd_up, svd_down=layer.svd_down, codebook=layer.codebook, skip_compile=True)
    ref = torch.nn.functional.embedding(ids, dq)  # pylint: disable=not-callable
    assert torch.equal(out, ref), 'embedding forward differs from dequantized lookup'
    return True


def test_ladder_contains_cb_rungs_in_order():
    for wdt in CB_DTYPES:
        assert weights_dtype_order.count(wdt) == 1, f'{wdt} not exactly once in weights_dtype_order'
        nbits = dtype_dict[wdt]['num_bits']
        next_int = f'int{nbits + 1}'
        assert weights_dtype_order.index(wdt) < weights_dtype_order.index(next_int), f'{wdt} must precede {next_int}'
        assert weights_dtype_order.index(wdt) > weights_dtype_order.index(f'uint{nbits}'), f'{wdt} must follow uint{nbits}'
    return True


def test_dynamic_from_int4_never_lands_on_cb():
    torch.manual_seed(14)
    weight = torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04
    result = sdnq_quantize_layer_weight_dynamic(weight, layer_class_name='Linear', weights_dtype='int4', dynamic_loss_threshold=1e-9, torch_dtype=torch.bfloat16)
    if result is not None:
        deq, _ = result
        assert not dtype_dict[deq.weights_dtype].get('is_codebook', False), f'int4 dynamic landed on {deq.weights_dtype}'
    return True


def test_dynamic_from_cb4_walks_cb():
    torch.manual_seed(15)
    weight = torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04
    result = sdnq_quantize_layer_weight_dynamic(weight, layer_class_name='Linear', weights_dtype='cb4', dynamic_loss_threshold=1.0, torch_dtype=torch.bfloat16)
    assert result is not None, 'cb4 dynamic returned nothing'
    deq, weight_data = result
    assert deq.weights_dtype == 'cb4', f'loose threshold must accept the first rung, got {deq.weights_dtype}'
    assert weight_data.get('codebook') is not None, 'dynamic result carries no codebook'
    return True


def test_state_dict_carries_codebook():
    layer, _ = build_layer('cb4')
    sd = layer.state_dict()
    assert 'codebook' in sd, f'codebook missing from state_dict keys {sorted(sd.keys())}'
    assert sd['codebook'].dtype == torch.int8, f'fresh codebook stored as {sd["codebook"].dtype}'
    assert 'codebook' in sdnq_keys, 'codebook missing from sdnq_keys'
    return True


def test_prequantized_load_keeps_codebook_dtype():
    # the HF pre-quantized path fp32-upcasts float side tensors under dequantize_fp32;
    # the codebook must keep its stored dtype in both the fresh (int8) and ingested (fp32) forms
    layer, _ = build_layer('cb4')
    model = torch.nn.Module()
    model.lin = layer
    cfg = SDNQConfig(weights_dtype='cb4', dequantize_fp32=True)
    quantizer = SDNQQuantizer(cfg, pre_quantized=True)
    quantizer.torch_dtype = torch.bfloat16
    assert quantizer.check_if_quantized_param(model, layer.codebook.data, 'lin.codebook'), 'codebook not claimed as a quantized param'
    quantizer.create_quantized_param(model, torch.arange(16, dtype=torch.int8), 'lin.codebook', DEVICE)
    assert model.lin.codebook.dtype == torch.int8, f'int8 codebook became {model.lin.codebook.dtype}'
    quantizer.create_quantized_param(model, torch.linspace(-1.0, 1.0, 16, dtype=torch.float32), 'lin.codebook', DEVICE)
    assert model.lin.codebook.dtype == torch.float32, f'fp32 codebook became {model.lin.codebook.dtype}'
    quantizer.create_quantized_param(model, layer.scale.data.to(torch.bfloat16), 'lin.scale', DEVICE)
    assert model.lin.scale.dtype == torch.float32, 'scale must still upcast under dequantize_fp32'
    return True


def test_effective_grid_step_scales_with_level_gap():
    cb_layer, _ = build_layer('cb4')
    int8_layer, _ = build_layer('int8')
    int8_step = effective_grid_step(int8_layer)
    assert abs(int8_step - float(int8_layer.scale.detach().float().mean())) < 1e-12, 'int8 step must equal mean scale'
    gaps = cb_layer.codebook.detach().float().sort().values.diff().mean()
    expected = float(cb_layer.scale.detach().float().mean()) * float(gaps)
    assert abs(effective_grid_step(cb_layer) - expected) < 1e-9, 'cb step must be scale times mean level gap'
    assert effective_grid_step(cb_layer) > float(cb_layer.scale.detach().float().mean()), 'cb step must exceed the raw scale unit'
    return True


def pack_kijai(idx: torch.Tensor) -> torch.Tensor:
    # the container packs the even flat index into the HIGH nibble (hi_first)
    pairs = idx.reshape(idx.shape[0], -1, 2)
    return (pairs[..., 0] << 4 | pairs[..., 1]).to(torch.uint8).view(torch.int8)


def make_w4a8_sidecars(out_f: int, in_f: int, group_size: int, seed: int):
    g = torch.Generator(device='cpu').manual_seed(seed)
    idx = torch.randint(0, 16, (out_f, in_f), generator=g, dtype=torch.uint8)
    codebook = torch.sort(torch.randn(16, generator=g)).values
    s_channel = torch.rand(out_f, generator=g) * 0.05 + 0.01
    s_rel = (torch.rand(out_f, in_f // group_size, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
    return idx, codebook, s_channel, s_rel


def test_ingest_adopt_matches_container_dequant():
    out_f, in_f, group_size = 32, 64, 16
    idx, codebook, s_channel, s_rel = make_w4a8_sidecars(out_f, in_f, group_size, seed=16)
    name = 'blocks.0.mlp.fc1'
    sd = {
        f'{name}.weight': pack_kijai(idx),
        f'{name}.weight_codebook': codebook.clone(),
        f'{name}.weight_s_channel': s_channel.clone(),
        f'{name}.weight_s_rel': s_rel.clone(),
    }
    lin = torch.nn.Linear(in_f, out_f, bias=False, device='meta')
    adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': group_size})
    assert f'{name}.weight_codebook' not in sd and f'{name}.weight_s_rel' not in sd and f'{name}.weight_s_channel' not in sd, 'sidecars must be consumed'
    assert sd[f'{name}.codebook'].dtype == torch.float32, 'ingested codebook must stay fp32'
    assert torch.equal(sd[f'{name}.codebook'], codebook.float()), 'ingested codebook must be adopted verbatim'
    # container dequant reference: codebook[idx] * s_channel * s_rel per group of 16
    scales_full = (s_rel.float() * s_channel.reshape(-1, 1)).repeat_interleave(group_size, dim=1)
    ref = codebook.float()[idx.int()] * scales_full
    groups = in_f // group_size
    out = dequantize_weight(
        'cb4', sd[f'{name}.weight'], sd[f'{name}.scale'], codebook=sd[f'{name}.codebook'],
        quantized_weight_shape=torch.Size((out_f, groups, group_size)), result_shape=torch.Size((out_f, in_f)), dtype=torch.float32,
    )
    assert torch.equal(out, ref), f'ingested dequant differs from container reference by {float((out - ref).abs().max()):.3e}'
    return True


def test_ingest_nibble_order_roundtrip():
    idx = torch.arange(256, dtype=torch.uint8).reshape(16, 16) % 16
    name = 'l'
    sd = {
        f'{name}.weight': pack_kijai(idx),
        f'{name}.weight_codebook': torch.linspace(-1.0, 1.0, 16),
        f'{name}.weight_s_channel': torch.ones(16),
        f'{name}.weight_s_rel': torch.ones(16, 1).to(torch.float8_e4m3fn),
    }
    lin = torch.nn.Linear(16, 16, bias=False, device='meta')
    adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': 16})
    unpacked = unpack_int(sd[f'{name}.weight'], 'cb4', torch.Size((16, 1, 16))).reshape(16, 16)
    assert torch.equal(unpacked.to(torch.uint8), idx), 'nibble swap does not recover the original index order'
    return True


def test_ingest_convrot_roundtrip():
    # convrot weights are stored rotated; the dequantizer re-applies the same regular
    # Hadamard, so rotating the reference identically must reproduce it bit-exact
    out_f, in_f, group_size = 16, 256, 16
    idx, codebook, s_channel, s_rel = make_w4a8_sidecars(out_f, in_f, group_size, seed=17)
    name = 'l'
    sd = {
        f'{name}.weight': pack_kijai(idx),
        f'{name}.weight_codebook': codebook.clone(),
        f'{name}.weight_s_channel': s_channel.clone(),
        f'{name}.weight_s_rel': s_rel.clone(),
    }
    lin = torch.nn.Linear(in_f, out_f, bias=False, device='meta')
    adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': group_size, 'convrot': True, 'convrot_groupsize': 256})
    scales_full = (s_rel.float() * s_channel.reshape(-1, 1)).repeat_interleave(group_size, dim=1)
    stored = codebook.float()[idx.int()] * scales_full
    hadamard = get_hadamard(256, dtype=torch.float32, device=torch.device('cpu'))
    ref = rotate_hadamard(stored, hadamard=hadamard)
    out = dequantize_weight(
        'cb4', sd[f'{name}.weight'], sd[f'{name}.scale'], codebook=sd[f'{name}.codebook'], hadamard=hadamard,
        quantized_weight_shape=torch.Size((out_f, in_f // group_size, group_size)), result_shape=torch.Size((out_f, in_f)), dtype=torch.float32,
    )
    assert torch.equal(out, ref), 'convrot dequant does not match the rotated reference'
    return True


def test_ingest_rejects_missing_sidecars():
    name = 'l'
    lin = torch.nn.Linear(64, 32, bias=False, device='meta')
    sd = {f'{name}.weight': torch.zeros(32, 32, dtype=torch.int8)}
    try:
        adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': 16})
        raise AssertionError('expected OverrideArchMismatch for missing sidecars')
    except OverrideArchMismatch as e:
        assert 'missing' in str(e), f'unexpected message: {e}'
    return True


def run_all() -> bool:
    global FIT_FLOOR  # pylint: disable=global-statement
    started = time.time()
    FIT_FLOOR = measure_fit_floor()
    log.warning(f'device={DEVICE} fit noise floor={FIT_FLOOR:.3e}')

    suites = [
        (CAT_FIT, [test_fit_deterministic_on_cpu, test_fit_zero_iterations_is_affine_grid, test_fit_beats_affine_grid, test_quantize_beats_uint4_end_to_end, test_zero_row_does_not_poison_codebook, test_fit_survives_clusters_past_fp32_count_limit]),
        (CAT_PACK, [test_pack_roundtrip_all_widths]),
        (CAT_DEQUANT, [test_dequant_matches_reference, test_dequant_grouped_matches_reference]),
        (CAT_FORWARD, [test_layer_forward_matches_dequant_linear, test_matmul_bypass_matches_dequant_linear, test_gemm_path_error_within_control_band, test_matmul_with_hadamard_and_svd, test_cb4_beats_int4_against_reference, test_conv_forward_matches_dequant, test_embedding_forward_matches_dequant]),
        (CAT_LADDER, [test_ladder_contains_cb_rungs_in_order, test_dynamic_from_int4_never_lands_on_cb, test_dynamic_from_cb4_walks_cb]),
        (CAT_STATE, [test_state_dict_carries_codebook, test_prequantized_load_keeps_codebook_dtype]),
        (CAT_LORA, [test_effective_grid_step_scales_with_level_gap]),
        (CAT_INGEST, [test_ingest_adopt_matches_container_dequant, test_ingest_nibble_order_roundtrip, test_ingest_convrot_roundtrip, test_ingest_rejects_missing_sidecars]),
    ]
    with torch.inference_mode():
        for cat, tests in suites:
            log.warning(f'=== {cat} ===')
            category(cat)
            for fn in tests:
                run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = total_failed = 0
    for cat, data in results.items():
        status = 'PASS' if data['failed'] == 0 else 'FAIL'
        log.warning(f'{cat}: {data["passed"]} passed, {data["failed"]} failed [{status}]')
        total_passed += data['passed']
        total_failed += data['failed']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed in {time.time() - started:.1f}s')
    return total_failed == 0


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
