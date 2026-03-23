#!/usr/bin/env python
"""
Offline unit tests for color grading and latent corrections.

Tests two systems:
- Pixel-space color grading (modules/processing_grading.py)
- Latent-space corrections (modules/processing_correction.py)

No running server required. Tests core logic with synthetic inputs.

Usage:
    python test/test-grading.py
"""

import os
import sys
import time
import types
import torch
import numpy as np
from types import SimpleNamespace

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Initialize cmd_args before any module imports (required by shared.py)
import modules.cmd_args
import installer
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

# Mock sd_vae_taesd to break circular import:
# processing_correction -> sd_vae_taesd -> shared -> shared_items -> sd_vae_taesd (circle)
_mock_taesd = types.ModuleType('modules.vae.sd_vae_taesd')
_mock_taesd.TAESD_MODELS = {'taesd': None}
_mock_taesd.CQYAN_MODELS = {}
_mock_taesd.encode = lambda x: torch.zeros(1, 4, 8, 8)
sys.modules['modules.vae.sd_vae_taesd'] = _mock_taesd

from modules.errors import log

# Results tracking
results = {
    'grading_params': {'passed': 0, 'failed': 0, 'tests': []},
    'grading_functions': {'passed': 0, 'failed': 0, 'tests': []},
    'correction_primitives': {'passed': 0, 'failed': 0, 'tests': []},
    'correction_pipeline': {'passed': 0, 'failed': 0, 'tests': []},
}
current_category = 'grading_params'


def record(passed, name, detail=''):
    status = 'PASS' if passed else 'FAIL'
    results[current_category]['passed' if passed else 'failed'] += 1
    results[current_category]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if passed:
        log.info(msg)
    else:
        log.error(msg)


def set_category(cat):
    global current_category  # pylint: disable=global-statement
    current_category = cat


# ============================================================
# Color Grading: GradingParams and utility functions
# ============================================================

def test_grading_params_defaults():
    """GradingParams has correct defaults."""
    from modules.processing_grading import GradingParams
    p = GradingParams()
    assert p.brightness == 0.0
    assert p.contrast == 0.0
    assert p.saturation == 0.0
    assert p.hue == 0.0
    assert p.gamma == 1.0
    assert p.sharpness == 0.0
    assert p.color_temp == 6500
    assert p.shadows == 0.0
    assert p.midtones == 0.0
    assert p.highlights == 0.0
    assert p.clahe_clip == 0.0
    assert p.clahe_grid == 8
    assert p.shadows_tint == "#000000"
    assert p.highlights_tint == "#ffffff"
    assert p.split_tone_balance == 0.5
    assert p.vignette == 0.0
    assert p.grain == 0.0
    assert p.lut_cube_file == ""
    assert p.lut_strength == 1.0
    return True


def test_grading_is_active():
    """is_active() returns False for defaults, True when any param differs."""
    from modules.processing_grading import GradingParams, is_active
    assert not is_active(GradingParams()), "defaults should be inactive"
    assert is_active(GradingParams(brightness=0.1)), "non-default brightness should be active"
    assert is_active(GradingParams(gamma=0.9)), "non-default gamma should be active"
    assert is_active(GradingParams(shadows_tint="#ff0000")), "non-default tint should be active"
    assert is_active(GradingParams(vignette=0.5)), "non-default vignette should be active"
    assert not is_active(GradingParams(brightness=0.0, gamma=1.0, color_temp=6500)), "all-default should be inactive"
    return True


def test_grading_float_coercion():
    """__post_init__ coerces int inputs to float (Gradio sends int for float sliders)."""
    from modules.processing_grading import GradingParams
    p = GradingParams(brightness=1, contrast=2, gamma=1, color_temp=6500)
    assert isinstance(p.brightness, float), f"expected float, got {type(p.brightness)}"
    assert isinstance(p.contrast, float), f"expected float, got {type(p.contrast)}"
    assert isinstance(p.gamma, float), f"expected float, got {type(p.gamma)}"
    assert isinstance(p.color_temp, float), f"expected float, got {type(p.color_temp)}"
    return True


def test_hex_to_rgb():
    """_hex_to_rgb converts hex color strings correctly."""
    from modules.processing_grading import _hex_to_rgb
    assert _hex_to_rgb("#000000") == (0.0, 0.0, 0.0), "black"
    assert _hex_to_rgb("#ffffff") == (1.0, 1.0, 1.0), "white"
    r, g, b = _hex_to_rgb("#ff0000")
    assert abs(r - 1.0) < 1e-6 and abs(g) < 1e-6 and abs(b) < 1e-6, "red"
    r, g, b = _hex_to_rgb("#00ff00")
    assert abs(r) < 1e-6 and abs(g - 1.0) < 1e-6 and abs(b) < 1e-6, "green"
    r, g, b = _hex_to_rgb("#0000ff")
    assert abs(r) < 1e-6 and abs(g) < 1e-6 and abs(b - 1.0) < 1e-6, "blue"
    # without hash
    r, g, b = _hex_to_rgb("ff8040")
    assert r > g > b, "orange-ish ordering"
    # invalid length returns black
    assert _hex_to_rgb("#fff") == (0.0, 0.0, 0.0), "short hex returns black"
    return True


def test_kelvin_to_rgb():
    """_kelvin_to_rgb_scale returns sensible values at known temperatures."""
    from modules.processing_grading import _kelvin_to_rgb_scale
    # 6500K (reference) should be approximately (1, 1, 1) - tolerance is wide because
    # the Planckian formula approximation normalizes to a hardcoded ref point
    r, g, b = _kelvin_to_rgb_scale(6500)
    assert abs(r - 1.0) < 0.15 and abs(g - 1.0) < 0.15 and abs(b - 1.0) < 0.15, f"6500K: ({r:.3f}, {g:.3f}, {b:.3f})"
    # warm (3000K) should have r > b
    r, g, b = _kelvin_to_rgb_scale(3000)
    assert r > b, f"3000K should be warm: r={r:.3f} b={b:.3f}"
    # cool (10000K) should have b > r
    r, g, b = _kelvin_to_rgb_scale(10000)
    assert b > r, f"10000K should be cool: r={r:.3f} b={b:.3f}"
    # all values should be non-negative; very low temps have zero blue (physically correct)
    for temp in [1000, 2000, 4000, 8000, 15000, 40000]:
        r, g, b = _kelvin_to_rgb_scale(temp)
        assert r >= 0 and g >= 0 and b >= 0, f"{temp}K has negative channel: ({r:.3f}, {g:.3f}, {b:.3f})"
    # moderate temps should have all positive channels
    for temp in [3000, 5000, 6500, 10000]:
        r, g, b = _kelvin_to_rgb_scale(temp)
        assert r > 0 and g > 0 and b > 0, f"{temp}K has non-positive channel: ({r:.3f}, {g:.3f}, {b:.3f})"
    return True


# ============================================================
# Color Grading: torch-based functions (need kornia for some)
# ============================================================

def _make_test_image_tensor(h=64, w=64):
    """Create a synthetic RGB test image tensor [1, 3, H, W] in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 3, h, w, dtype=torch.float32)


def _make_test_pil_image(h=64, w=64):
    """Create a synthetic RGB PIL image."""
    from PIL import Image
    arr = np.random.RandomState(42).randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, 'RGB')


def test_apply_vignette():
    """_apply_vignette darkens edges more than center."""
    from modules.processing_grading import _apply_vignette
    img = torch.ones(1, 3, 64, 64, dtype=torch.float32)
    result = _apply_vignette(img, strength=1.0)
    center_val = result[0, 0, 32, 32].item()
    corner_val = result[0, 0, 0, 0].item()
    assert center_val > corner_val, f"center ({center_val:.3f}) should be brighter than corner ({corner_val:.3f})"
    assert result.shape == img.shape, "shape preserved"
    assert not torch.isnan(result).any(), "no NaN"
    # zero strength should be identity
    result_zero = _apply_vignette(img, strength=0.0)
    assert torch.allclose(result_zero, img), "zero strength is identity"
    return True


def test_apply_grain():
    """_apply_grain adds noise (output differs from input, stays in valid range)."""
    from modules.processing_grading import _apply_grain
    img = torch.ones(1, 3, 64, 64, dtype=torch.float32) * 0.5
    result = _apply_grain(img, strength=0.5)
    assert not torch.equal(result, img), "grain should modify the image"
    assert result.shape == img.shape, "shape preserved"
    assert result.min() >= 0.0 and result.max() <= 1.0, "output clamped to [0, 1]"
    assert not torch.isnan(result).any(), "no NaN"
    return True


def test_apply_color_temp():
    """_apply_color_temp shifts R/B channels for warm/cool temperatures."""
    from modules.processing_grading import _apply_color_temp
    img = torch.ones(1, 3, 64, 64, dtype=torch.float32) * 0.5
    # warm
    warm = _apply_color_temp(img, 3000)
    assert warm[0, 0].mean() > warm[0, 2].mean(), "warm: red > blue"
    # cool
    cool = _apply_color_temp(img, 10000)
    assert cool[0, 2].mean() > cool[0, 0].mean(), "cool: blue > red"
    # neutral
    neutral = _apply_color_temp(img, 6500)
    assert torch.allclose(neutral, img, atol=0.05), "6500K is near-neutral"
    assert warm.shape == img.shape, "shape preserved"
    return True


def test_apply_shadows_midtones_highlights():
    """_apply_shadows_midtones_highlights modifies tone without NaN/shape issues."""
    try:
        from modules.processing_grading import _apply_shadows_midtones_highlights
    except ImportError:
        return None  # kornia not available
    img = _make_test_image_tensor()
    # shadows boost
    result = _apply_shadows_midtones_highlights(img, shadows=0.5, midtones=0.0, highlights=0.0)
    assert result.shape == img.shape, "shape preserved"
    assert not torch.isnan(result).any(), "no NaN"
    assert result.min() >= 0.0 and result.max() <= 1.0, "output in [0, 1]"
    # all zero should be near-identity (kornia conversions may introduce tiny diffs)
    result_zero = _apply_shadows_midtones_highlights(img, shadows=0.0, midtones=0.0, highlights=0.0)
    assert torch.allclose(result_zero, img, atol=1e-3), "zero params is near-identity"
    return True


def test_grade_image_pipeline():
    """Full grade_image pipeline runs without errors for various param combos."""
    try:
        import modules.devices as devices_mod
        devices_mod.device = torch.device('cpu')
        devices_mod.dtype = torch.float32
        from modules.processing_grading import GradingParams, grade_image, is_active
    except ImportError:
        return None  # kornia not available
    img = _make_test_pil_image()
    # basic adjustments
    params = GradingParams(brightness=0.1, contrast=0.2, saturation=-0.1)
    assert is_active(params)
    result = grade_image(img, params)
    assert result.size == img.size, "output size matches input"
    assert result.mode == 'RGB', "output is RGB"
    # tone adjustments
    params = GradingParams(shadows=0.3, midtones=-0.2, highlights=0.1)
    result = grade_image(img, params)
    assert result.size == img.size
    # effects
    params = GradingParams(vignette=0.5, grain=0.3)
    result = grade_image(img, params)
    assert result.size == img.size
    # hue and gamma
    params = GradingParams(hue=0.1, gamma=0.8, sharpness=0.5)
    result = grade_image(img, params)
    assert result.size == img.size
    # color temp
    params = GradingParams(color_temp=3000)
    result = grade_image(img, params)
    assert result.size == img.size
    # split toning
    params = GradingParams(shadows_tint="#003366", highlights_tint="#ffcc00", split_tone_balance=0.7)
    result = grade_image(img, params)
    assert result.size == img.size
    return True


def test_grade_image_edge_cases():
    """grade_image handles edge cases: all-black, all-white, tiny images."""
    try:
        import modules.devices as devices_mod
        devices_mod.device = torch.device('cpu')
        devices_mod.dtype = torch.float32
        from modules.processing_grading import GradingParams, grade_image
    except ImportError:
        return None
    from PIL import Image
    params = GradingParams(brightness=0.2, contrast=0.3, vignette=0.5, grain=0.2)
    # all black
    black = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), 'RGB')
    result = grade_image(black, params)
    assert result.size == black.size, "black image handled"
    # all white
    white = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8), 'RGB')
    result = grade_image(white, params)
    assert result.size == white.size, "white image handled"
    # tiny image
    tiny = Image.fromarray(np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8), 'RGB')
    result = grade_image(tiny, params)
    assert result.size == tiny.size, "tiny image handled"
    return True


# ============================================================
# Latent Corrections: primitive tensor operations
# ============================================================

def test_soft_clamp_tensor():
    """soft_clamp_tensor shrinks outliers toward mean, preserves values within bounds."""
    from modules.processing_correction import soft_clamp_tensor
    # within bounds: no change
    tensor = torch.randn(4, 64, 64) * 0.5
    result = soft_clamp_tensor(tensor, threshold=0.8, boundary=4)
    assert torch.allclose(result, tensor, atol=1e-5), "within-bounds tensor unchanged"
    # with outliers: should clamp
    tensor_outliers = torch.randn(4, 64, 64)
    tensor_outliers[0, 0, 0] = 10.0
    tensor_outliers[1, 0, 0] = -10.0
    result = soft_clamp_tensor(tensor_outliers.clone(), threshold=0.8, boundary=4)
    assert result[0, 0, 0] < tensor_outliers[0, 0, 0], "positive outlier reduced"
    assert result[1, 0, 0] > tensor_outliers[1, 0, 0], "negative outlier raised"
    assert result.shape == tensor_outliers.shape, "shape preserved"
    assert not torch.isnan(result).any(), "no NaN"
    # zero threshold: identity
    result_zero = soft_clamp_tensor(tensor_outliers.clone(), threshold=0, boundary=4)
    assert torch.allclose(result_zero, tensor_outliers), "zero threshold is identity"
    return True


def test_center_tensor():
    """center_tensor adjusts mean of tensor channels."""
    from modules.processing_correction import center_tensor
    tensor = torch.randn(4, 64, 64) + 2.0  # offset mean
    original_mean = tensor.mean().item()
    # full shift should reduce mean toward offset
    result = center_tensor(tensor.clone(), channel_shift=0.0, full_shift=1.0, offset=0.0)
    assert abs(result.mean().item()) < abs(original_mean), "full shift centers toward zero"
    # channel shift
    result_ch = center_tensor(tensor.clone(), channel_shift=1.0, full_shift=0.0, offset=0.0)
    for c in range(4):
        assert abs(result_ch[c].mean().item()) < abs(tensor[c].mean().item()), f"channel {c} centered"
    # no-op
    result_noop = center_tensor(tensor.clone(), channel_shift=0.0, full_shift=0.0, offset=0.0)
    assert torch.allclose(result_noop, tensor), "zero params is identity"
    # with offset
    result_offset = center_tensor(tensor.clone(), channel_shift=0.0, full_shift=1.0, offset=5.0)
    assert result_offset.mean().item() > 0, "offset shifts mean positive"
    return True


def test_sharpen_tensor():
    """sharpen_tensor applies sharpening convolution, preserves shape."""
    from modules.processing_correction import sharpen_tensor
    tensor = torch.randn(4, 64, 64)
    # zero ratio: identity
    result_zero = sharpen_tensor(tensor.clone(), ratio=0)
    assert torch.allclose(result_zero, tensor), "zero ratio is identity"
    # positive ratio: should modify
    result = sharpen_tensor(tensor.clone(), ratio=0.5)
    assert result.shape == tensor.shape, "shape preserved"
    assert not torch.isnan(result).any(), "no NaN"
    assert not torch.isinf(result).any(), "no Inf"
    assert not torch.equal(result, tensor), "sharpening modifies tensor"
    return True


def test_maximize_tensor():
    """maximize_tensor normalizes tensor range."""
    from modules.processing_correction import maximize_tensor
    tensor = torch.randn(4, 64, 64) * 0.5
    # boundary 1.0: identity
    result_id = maximize_tensor(tensor.clone(), boundary=1.0)
    assert torch.allclose(result_id, tensor), "boundary 1.0 is identity"
    # boundary 2.0: should expand range
    result = maximize_tensor(tensor.clone(), boundary=2.0)
    assert result.abs().max() > tensor.abs().max(), "boundary 2.0 expands range"
    assert result.shape == tensor.shape, "shape preserved"
    assert not torch.isnan(result).any(), "no NaN"
    # boundary 0.5: should compress range
    result_small = maximize_tensor(tensor.clone(), boundary=0.5)
    assert result_small.abs().max() < tensor.abs().max() + 0.1, "boundary 0.5 compresses"
    return True


# ============================================================
# Latent Corrections: correction() pipeline with mock p object
# ============================================================

def _make_mock_p(**overrides):
    """Create a mock processing object with default hdr params."""
    defaults = {
        'hdr_mode': 0,
        'hdr_brightness': 0.0,
        'hdr_color': 0.0,
        'hdr_sharpen': 0.0,
        'hdr_clamp': False,
        'hdr_boundary': 4.0,
        'hdr_threshold': 0.95,
        'hdr_maximize': False,
        'hdr_max_center': 0.6,
        'hdr_max_boundary': 1.0,
        'hdr_color_picker': '#000000',
        'hdr_tint_ratio': 0.0,
        'correction_total_steps': 20,
        'correction_steps_mid': 10,
        'correction_steps_late': 4,
        'extra_generation_params': {},
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_correction_noop():
    """correction() with all-zero params is near-identity."""
    from modules.processing_correction import correction
    p = _make_mock_p()
    latent = torch.randn(4, 64, 64)
    for step in [0, 5, 10, 15, 19]:
        result = correction(p, 500, latent.clone(), step=step)
        assert torch.allclose(result, latent, atol=1e-5), f"step {step}: no-op correction should be identity"
    return True


def test_correction_early_clamp():
    """correction() applies soft_clamp in early steps when hdr_clamp=True."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_clamp=True, hdr_threshold=0.8, hdr_boundary=4.0)
    latent = torch.randn(4, 64, 64)
    latent[0, 0, 0] = 10.0  # outlier
    # step 0 of 20 = progress 0.0 (early)
    result = correction(p, 999, latent.clone(), step=0)
    assert result[0, 0, 0] < 10.0, "outlier should be clamped"
    assert "Latent clamp" in p.extra_generation_params, "clamp recorded in params"
    return True


def test_correction_mid_color():
    """correction() applies color centering in mid steps."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_color=0.5)
    latent = torch.randn(4, 64, 64) + 1.0  # offset channels
    original_ch_means = [latent[c].mean().item() for c in range(1, 4)]
    # step 6 of 20 = progress 0.3 (mid range)
    result = correction(p, 700, latent.clone(), step=6)
    new_ch_means = [result[c].mean().item() for c in range(1, 4)]
    # at least some channels should have their mean reduced (centered)
    centered_count = sum(1 for o, n in zip(original_ch_means, new_ch_means) if abs(n) < abs(o))
    assert centered_count > 0, "at least one color channel should be more centered"
    assert "Latent color" in p.extra_generation_params, "color recorded in params"
    return True


def test_correction_late_brightness():
    """correction() applies brightness offset in late steps."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_brightness=2.0)
    latent = torch.randn(4, 64, 64)
    original_mean = latent[0].mean().item()
    # step 17 of 20 = progress 0.85 (late)
    result = correction(p, 100, latent.clone(), step=17)
    new_mean = result[0].mean().item()
    assert new_mean != original_mean, "brightness should shift channel 0 mean"
    assert "Latent brightness" in p.extra_generation_params, "brightness recorded in params"
    return True


def test_correction_sharpen():
    """correction() applies sharpening in sharpen range."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_sharpen=1.0)
    latent = torch.randn(4, 64, 64)
    # step 15 of 20 = progress 0.75 (sharpen range)
    result = correction(p, 200, latent.clone(), step=15)
    assert not torch.equal(result, latent), "sharpening should modify latent"
    assert "Latent sharpen" in p.extra_generation_params, "sharpen recorded in params"
    return True


def test_correction_maximize():
    """correction() applies maximize/normalize in very late steps."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_maximize=True, hdr_max_center=0.6, hdr_max_boundary=2.0)
    latent = torch.randn(4, 64, 64) * 0.5
    # step 19 of 20 = progress 0.95 (very late)
    result = correction(p, 10, latent.clone(), step=19)
    assert result.abs().max() > latent.abs().max(), "maximize should expand range"
    assert "Latent max" in p.extra_generation_params, "maximize recorded in params"
    return True


def test_correction_multichannel():
    """correction() uses multi-channel path for >4 channel latents."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_brightness=2.0, hdr_color=0.5)
    # 16-channel latent (e.g. Flux 2)
    latent = torch.randn(16, 64, 64)
    # mid step: color centering on all channels
    p.extra_generation_params = {}
    result_mid = correction(p, 700, latent.clone(), step=6)
    assert result_mid.shape == latent.shape, "multi-channel shape preserved"
    assert not torch.isnan(result_mid).any(), "no NaN"
    # late step: brightness via multiplicative scaling
    p.extra_generation_params = {}
    result_late = correction(p, 100, latent.clone(), step=17)
    assert result_late.shape == latent.shape, "multi-channel shape preserved"
    assert not torch.isnan(result_late).any(), "no NaN"
    assert "Latent brightness" in p.extra_generation_params, "brightness recorded"
    return True


def test_correction_shape_preservation():
    """correction() preserves shape and dtype for various latent sizes."""
    from modules.processing_correction import correction
    p = _make_mock_p(hdr_clamp=True, hdr_color=0.3, hdr_brightness=1.0, hdr_sharpen=0.5)
    shapes = [(4, 64, 64), (4, 32, 32), (4, 128, 128), (8, 64, 64), (16, 32, 32)]
    for shape in shapes:
        for step in [0, 6, 15, 19]:
            p.extra_generation_params = {}
            latent = torch.randn(shape)
            result = correction(p, 500, latent.clone(), step=step)
            assert result.shape == latent.shape, f"shape {shape} step {step}: shape mismatch"
            assert result.dtype == latent.dtype, f"shape {shape} step {step}: dtype mismatch"
            assert not torch.isnan(result).any(), f"shape {shape} step {step}: NaN"
            assert not torch.isinf(result).any(), f"shape {shape} step {step}: Inf"
    return True


def test_correction_step_ranges():
    """correction() applies different operations at different progress points."""
    from modules.processing_correction import correction
    p = _make_mock_p(
        hdr_clamp=True, hdr_color=0.5, hdr_brightness=1.0,
        hdr_sharpen=0.5, hdr_maximize=True, hdr_max_boundary=2.0,
    )
    latent = torch.randn(4, 64, 64)
    latent[0, 0, 0] = 10.0  # outlier for clamp testing
    expected_params_per_range = {
        0: ["Latent clamp"],          # early: progress 0.0
        6: ["Latent color"],           # mid: progress 0.3
        15: ["Latent sharpen"],        # sharpen: progress 0.75
        17: ["Latent brightness"],     # late: progress 0.85
        19: ["Latent max"],            # very late: progress 0.95
    }
    for step, expected_keys in expected_params_per_range.items():
        p.extra_generation_params = {}
        correction(p, 500, latent.clone(), step=step)
        for key in expected_keys:
            assert key in p.extra_generation_params, f"step {step}: expected '{key}' in params, got {list(p.extra_generation_params.keys())}"
    return True


# ============================================================
# Test runner
# ============================================================

def run_test(fn):
    name = fn.__name__
    try:
        result = fn()
        if result is None:
            log.warning(f'  SKIP: {name} (dependency not available)')
            return
        record(True, name)
    except AssertionError as e:
        record(False, name, str(e))
    except Exception as e:
        record(False, name, f"exception: {e}")
        import traceback
        traceback.print_exc()


def run_tests():
    t0 = time.time()

    # Grading params (pure Python, no GPU deps)
    set_category('grading_params')
    log.warning('=== Color Grading: Params & Utilities ===')
    for fn in [test_grading_params_defaults, test_grading_is_active, test_grading_float_coercion,
               test_hex_to_rgb, test_kelvin_to_rgb]:
        run_test(fn)

    # Grading functions (need torch, some need kornia)
    set_category('grading_functions')
    log.warning('=== Color Grading: Tensor Operations ===')
    for fn in [test_apply_vignette, test_apply_grain, test_apply_color_temp,
               test_apply_shadows_midtones_highlights, test_grade_image_pipeline,
               test_grade_image_edge_cases]:
        run_test(fn)

    # Correction primitives (pure torch)
    set_category('correction_primitives')
    log.warning('=== Latent Corrections: Primitives ===')
    for fn in [test_soft_clamp_tensor, test_center_tensor, test_sharpen_tensor,
               test_maximize_tensor]:
        run_test(fn)

    # Correction pipeline (mock p object)
    set_category('correction_pipeline')
    log.warning('=== Latent Corrections: Pipeline ===')
    for fn in [test_correction_noop, test_correction_early_clamp, test_correction_mid_color,
               test_correction_late_brightness, test_correction_sharpen, test_correction_maximize,
               test_correction_multichannel, test_correction_shape_preservation,
               test_correction_step_ranges]:
        run_test(fn)

    t1 = time.time()

    # Summary
    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat, data in results.items():
        total_passed += data['passed']
        total_failed += data['failed']
        status = 'PASS' if data['failed'] == 0 else 'FAIL'
        log.info(f'  {cat}: {data["passed"]} passed, {data["failed"]} failed [{status}]')
    log.warning(f'Total: {total_passed} passed, {total_failed} failed in {t1 - t0:.2f}s')
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
