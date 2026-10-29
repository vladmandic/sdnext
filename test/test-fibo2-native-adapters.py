#!/usr/bin/env python
"""
Offline unit tests for Bria FIBO 2.0 native adapter loaders.

Covers ``pipelines.bria.fibo2_lora`` (LoRA plus DoRA, LoKR, Norm, Full) on the
layouts an adapter can be trained on:

- diffusers / PEFT (``transformer.layers.0.attention.to_q.lora_A.weight``):
  what the release trainer ``fine_tuning/train_lora.py`` saves
- the original checkpoint names that diffusers' convert script renames
  (``cross_q``, ``self_out``, ``all_x_embedder.2-1``)
- the ComfyUI single-file names: fused ``attention.qkv`` and ``to_kv``,
  fused ``w13`` with each half zero-padded to a multiple of 256 rows,
  ``attention.out``, ``q_norm``; and the kohya flat-underscore form

Tests build a mock FIBO 2-shaped transformer after
``diffusers.BriaFibo2Transformer2DModel``, write synthetic safetensors files
and exercise the full loader path from state dict to ``NetworkModule*``.

No running server required.

Usage:
    python test/test-fibo2-native-adapters.py
"""

import os
import sys
import tempfile
import time
from types import SimpleNamespace

import torch
import safetensors.torch

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

from modules.errors import log   # pylint: disable=wrong-import-position
from modules.lora import (         # pylint: disable=wrong-import-position
    network, network_lora, network_lokr, network_norm, network_full,
)
from modules.lora.native_adapter import ChunkSpec  # pylint: disable=wrong-import-position
from pipelines.bria import fibo2_lora as F  # pylint: disable=wrong-import-position


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
# Mock FIBO 2 transformer
# ============================================================
# Proportions follow BriaFibo2Transformer2DModel: the FFN width is
# ``(dim * 8 // 3) // 64 * 64`` and adaLN_modulation outputs ``4 * dim``.
# HIDDEN is chosen so the FFN width is not a multiple of 256, which is what
# the ComfyUI fused ``w13`` pads it to.

HIDDEN = 224                                   # dim
HEAD_DIM = 32                                  # per-head width of the qk norms
CAP = 128                                      # cap_feat_dim, the cross-attention key/value width
MLP_HIDDEN = (HIDDEN * 8 // 3) // 64 * 64      # 576
MLP_PADDED = -(-MLP_HIDDEN // 256) * 256       # 768
ADALN_OUT = 4 * HIDDEN
PATCH_IN = 128                                 # patch_size^2 * in_channels
N_LAYERS = 2
INJECTION_LAYERS = (1,)


# pylint: disable=attribute-defined-outside-init
class Holder(torch.nn.Module):
    """Empty container module - children are attached dynamically."""


def build_attention(kv_dim=None):
    """Mirror ``BriaFibo2Attention``: split projections, a one-element ``to_out`` ModuleList, qk RMSNorms."""
    attn = Holder()
    attn.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    attn.to_k = torch.nn.Linear(kv_dim or HIDDEN, HIDDEN, bias=False)
    attn.to_v = torch.nn.Linear(kv_dim or HIDDEN, HIDDEN, bias=False)
    attn.to_out = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN, bias=False)])
    attn.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    attn.norm_k = torch.nn.RMSNorm(HEAD_DIM)
    return attn


def build_ffn():
    ffn = Holder()
    ffn.w1 = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=False)
    ffn.w2 = torch.nn.Linear(MLP_HIDDEN, HIDDEN, bias=False)
    ffn.w3 = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=False)
    return ffn


def build_block(injection: bool):
    """Mirror ``BriaFibo2TransformerBlock``; an injection block also reads the text encoder."""
    block = Holder()
    block.attention = build_attention()
    block.feed_forward = build_ffn()
    block.attention_norm1 = torch.nn.RMSNorm(HIDDEN)
    block.ffn_norm1 = torch.nn.RMSNorm(HIDDEN)
    block.attention_norm2 = torch.nn.RMSNorm(HIDDEN)
    block.ffn_norm2 = torch.nn.RMSNorm(HIDDEN)
    block.adaLN_modulation = torch.nn.Sequential(torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True))
    if injection:
        block.cross_attn_norm1 = torch.nn.RMSNorm(HIDDEN)
        block.cross_attention = build_attention(kv_dim=CAP)
        block.cross_attn_norm2 = torch.nn.RMSNorm(HIDDEN)
        block.cross_attn_gate = torch.nn.Parameter(torch.zeros(1))
    return block


def build_perceiver_layer():
    """Mirror ``BriaFibo2PerceiverLayer``."""
    layer = Holder()
    layer.cross_attn_norm = torch.nn.RMSNorm(HIDDEN)
    layer.cross_attention = build_attention(kv_dim=CAP)
    layer.self_attn_norm = torch.nn.RMSNorm(HIDDEN)
    layer.self_attention = build_attention()
    layer.ffn_norm = torch.nn.RMSNorm(HIDDEN)
    layer.ffn = build_ffn()
    return layer


def build_mock_transformer():
    """Build a torch.nn.Module mimicking ``BriaFibo2Transformer2DModel``."""
    transformer = Holder()
    transformer.x_embedder = torch.nn.Linear(PATCH_IN, HIDDEN, bias=True)
    transformer.t_embedder = Holder()
    transformer.t_embedder.mlp = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=True),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64, bias=True),
    )
    transformer.noise_refiner = torch.nn.ModuleList([build_block(injection=False)])
    transformer.perceiver = Holder()
    transformer.perceiver.latent_queries = torch.nn.Parameter(torch.randn(8, HIDDEN))
    transformer.perceiver.layers = torch.nn.ModuleList([build_perceiver_layer()])
    transformer.perceiver.norm_out = torch.nn.RMSNorm(HIDDEN)
    transformer.layers = torch.nn.ModuleList([build_block(injection=i in INJECTION_LAYERS) for i in range(N_LAYERS)])
    transformer.final_layer = Holder()
    transformer.final_layer.linear = torch.nn.Linear(HIDDEN, PATCH_IN, bias=True)
    transformer.final_layer.adaLN_modulation = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, HIDDEN, bias=True),
    )
    transformer.config = SimpleNamespace(dim=HIDDEN)
    return transformer


class MockFibo2Pipeline:
    """Class name carries 'BriaFibo2' so name-based model-type dispatch routes correctly."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class MockFibo2SdModel:
    """Outer wrapper exposing pipe + network_layer_mapping for
    ``lora_convert.assign_network_names_to_compvis_modules`` to write onto."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'BriaFibo2Pipeline'


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing a FIBO 2-shaped transformer.

    Each test calls this fresh so any prior network_layer_name stamps don't
    leak across tests. Writes directly to ``model_data.sd_model`` to bypass
    the ModelData lock that no-ops the public setter outside webui startup.
    """
    transformer = build_mock_transformer()
    pipe = MockFibo2Pipeline(transformer)
    sd_model = MockFibo2SdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# State-dict synthesizers (one per family/format)
# ============================================================

RANK_LORA = 8
LOKR_W1_DIM = 8


def sd_lora_trainer_format():
    """What ``fine_tuning/train_lora.py`` saves: PEFT keys under ``transformer.`` on every nn.Linear."""
    return {
        'transformer.layers.0.attention.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.0.attention.to_q.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'transformer.layers.0.feed_forward.w1.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.0.feed_forward.w1.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
        'transformer.perceiver.layers.0.cross_attention.to_k.lora_A.weight': torch.randn(RANK_LORA, CAP),
        'transformer.perceiver.layers.0.cross_attention.to_k.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_bare_injection_cross_attention():
    """Bare diffusers path on an injection block's cross-attention value projection."""
    return {
        'layers.1.cross_attention.to_v.lora_A.weight': torch.randn(RANK_LORA, CAP),
        'layers.1.cross_attention.to_v.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_kohya_to_q():
    """Kohya-format LoRA on attention.to_q (flat underscore path)."""
    return {
        'lora_unet_layers_0_attention_to_q.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_layers_0_attention_to_q.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'lora_unet_layers_0_attention_to_q.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_original_names():
    """Original checkpoint names: cross_q, self_out and the patch-size keyed embedder."""
    return {
        'diffusion_model.layers.1.cross_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.1.cross_q.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.perceiver.layers.0.self_out.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.perceiver.layers.0.self_out.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.all_x_embedder.2-1.lora_A.weight': torch.randn(RANK_LORA, PATCH_IN),
        'diffusion_model.all_x_embedder.2-1.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_comfy_fused_qkv():
    """ComfyUI fused ``attention.qkv``: the up-weight stacks Q over K over V, the down-weight is shared."""
    return {
        'diffusion_model.layers.0.attention.qkv.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.qkv.lora_B.weight': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.qkv.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_comfy_fused_kv():
    """ComfyUI fused cross-attention ``to_kv``: K over V."""
    return {
        'diffusion_model.layers.1.cross_attention.to_kv.lora_A.weight': torch.randn(RANK_LORA, CAP),
        'diffusion_model.layers.1.cross_attention.to_kv.lora_B.weight': torch.randn(2 * HIDDEN, RANK_LORA),
    }


def sd_lora_comfy_fused_w13():
    """ComfyUI fused ``w13``: w1 then w3, each half zero-padded to MLP_PADDED rows. Rows carry 1.0 and 2.0 for the slice check."""
    up = torch.zeros(2 * MLP_PADDED, RANK_LORA)
    up[:MLP_HIDDEN] = 1.0
    up[MLP_PADDED:MLP_PADDED + MLP_HIDDEN] = 2.0
    return {
        'diffusion_model.layers.0.feed_forward.w13.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w13.lora_B.weight': up,
    }


def sd_lora_comfy_w2_padded():
    """ComfyUI ``w2`` reads the padded w13 output, so its down factor spans MLP_PADDED columns with a zero tail."""
    down = torch.randn(RANK_LORA, MLP_PADDED)
    down[:, MLP_HIDDEN:] = 0
    return {
        'diffusion_model.layers.0.feed_forward.w2.lora_A.weight': down,
        'diffusion_model.layers.0.feed_forward.w2.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w2.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_comfy_attention_out():
    """ComfyUI ``attention.out``, the diffusers ``attention.to_out.0``."""
    return {
        'diffusion_model.noise_refiner.0.attention.out.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.noise_refiner.0.attention.out.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_with_dora_scale():
    """LoRA carrying a dora_scale companion vector to exercise DoRA threading."""
    return {
        'transformer.layers.0.attention.to_v.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.0.attention.to_v.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'transformer.layers.0.attention.to_v.dora_scale': torch.randn(HIDDEN),
    }


def sd_lokr_adaln():
    """LoKR on the block modulation linear."""
    return {
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.lokr_w2': torch.randn(ADALN_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_norm_comfy_q_norm():
    """Norm delta on the ComfyUI ``q_norm`` name, the diffusers ``norm_q``."""
    return {
        'diffusion_model.layers.0.attention.q_norm.w_norm': torch.randn(HEAD_DIM),
    }


def sd_full_final_layer_original_name():
    """Full weight delta on the original ``all_final_layer.2-1.linear`` name."""
    return {
        'diffusion_model.all_final_layer.2-1.linear.diff': torch.randn(PATCH_IN, HIDDEN),
        'diffusion_model.all_final_layer.2-1.linear.diff_b': torch.randn(PATCH_IN),
    }


# ============================================================
# Helpers: write state dict to disk, mock NetworkOnDisk
# ============================================================


class TempLora:
    """Context manager: writes a state dict to a temp safetensors file and yields
    a ``MockNetworkOnDisk`` pointing at it. Cleans up on exit."""

    def __init__(self, state_dict, name='test'):
        self.state_dict = state_dict
        self.name = name
        self.path = None

    def __enter__(self):
        sd = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in self.state_dict.items()}
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        safetensors.torch.save_file(sd, self.path)
        return MockNetworkOnDisk(self.path, self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class MockNetworkOnDisk:
    """Stand-in for ``network.NetworkOnDisk`` exposing only the attributes the native loaders read."""

    def __init__(self, filename, name):
        self.filename = filename
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'


def assert_shape(t: torch.Tensor, expected_shape, label=''):
    actual = tuple(t.shape)
    assert actual == tuple(expected_shape), f'{label}: shape {actual}, expected {tuple(expected_shape)}'


def make_network_for_module(net_module: network.NetworkModule, te_mul: float = 1.0, unet_mul: float = 1.0):
    """Wire a NetworkModule's parent ``Network`` with given multipliers so calc_updown has a multiplier context."""
    net_module.network.te_multiplier = te_mul
    net_module.network.unet_multiplier = unet_mul
    return net_module


def load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


# ============================================================
# Tests - parsing primitives
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_all_prefixes():
    """parse_key recognizes BFL, PEFT, kohya and bare-diffusers keys."""
    bd = F.BARE_DIFFUSERS_PREFIX_USED
    cases = [
        ('diffusion_model.layers.0.attention.to_q.lora_A.weight', F.LORA_SUFFIXES, ('diffusion_model.', 'layers.0.attention.to_q', 'lora_down.weight')),
        ('transformer.layers.1.cross_attention.to_v.lora_B.weight', F.LORA_SUFFIXES, ('transformer.', 'layers.1.cross_attention.to_v', 'lora_up.weight')),
        ('lora_unet_perceiver_layers_0_ffn_w2.lora_down.weight', F.LORA_SUFFIXES, ('lora_unet_', 'perceiver_layers_0_ffn_w2', 'lora_down.weight')),
        ('noise_refiner.0.attention.to_out.0.lora_A.weight', F.LORA_SUFFIXES, (bd, 'noise_refiner.0.attention.to_out.0', 'lora_down.weight')),
        ('random.unrelated.key', F.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = F.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_resolve_targets_aliases():
    """Original checkpoint names and ComfyUI names resolve to the diffusers module paths; diffusers paths pass through."""
    cases = [
        ('diffusion_model.', 'layers.1.cross_q', 'layers.1.cross_attention.to_q'),
        ('diffusion_model.', 'layers.1.cross_out', 'layers.1.cross_attention.to_out.0'),
        ('diffusion_model.', 'layers.1.cross_norm_k', 'layers.1.cross_attention.norm_k'),
        ('diffusion_model.', 'perceiver.layers.0.self_out', 'perceiver.layers.0.self_attention.to_out.0'),
        ('diffusion_model.', 'perceiver.layers.0.self_v', 'perceiver.layers.0.self_attention.to_v'),
        ('diffusion_model.', 'all_x_embedder.2-1', 'x_embedder'),
        ('diffusion_model.', 'all_final_layer.2-1.adaLN_modulation.1', 'final_layer.adaLN_modulation.1'),
        ('diffusion_model.', 'layers.0.attention.out', 'layers.0.attention.to_out.0'),
        ('diffusion_model.', 'layers.0.attention.q_norm', 'layers.0.attention.norm_q'),
        ('diffusion_model.', 'layers.1.cross_attention.to_out', 'layers.1.cross_attention.to_out.0'),
        ('diffusion_model.', 'perceiver.layers.0.self_attention.to_out', 'perceiver.layers.0.self_attention.to_out.0'),
        ('transformer.', 'layers.0.attention.to_out.0', 'layers.0.attention.to_out.0'),
        ('transformer.', 'layers.0.feed_forward.w3', 'layers.0.feed_forward.w3'),
        ('lora_unet_', 'layers_1_cross_q', 'layers_1_cross_attention_to_q'),
        ('lora_unet_', 'layers_0_attention_out', 'layers_0_attention_to_out_0'),
        ('lora_unet_', 'all_x_embedder_2-1', 'x_embedder'),
    ]
    for prefix, base, expected in cases:
        got = F.resolve_targets(prefix, base)
        assert got == [(expected, None)], f'resolve_targets({prefix!r}, {base!r}) = {got}, expected {expected}'
    return True


def test_resolve_targets_fused():
    """Fused ComfyUI projections resolve to chunked targets; the w13 slice follows the live model's width."""
    install_mock_pipe()
    assert F.ffn_rows() == (MLP_HIDDEN, MLP_PADDED), f'ffn_rows = {F.ffn_rows()}'
    got = F.resolve_targets('diffusion_model.', 'layers.0.attention.qkv')
    assert got == [
        ('layers.0.attention.to_q', ChunkSpec(idx=0, total=3)),
        ('layers.0.attention.to_k', ChunkSpec(idx=1, total=3)),
        ('layers.0.attention.to_v', ChunkSpec(idx=2, total=3)),
    ], f'qkv: {got}'
    got = F.resolve_targets('diffusion_model.', 'perceiver.layers.0.self_attention.to_kv')
    assert got == [
        ('perceiver.layers.0.self_attention.to_k', ChunkSpec(idx=0, total=2)),
        ('perceiver.layers.0.self_attention.to_v', ChunkSpec(idx=1, total=2)),
    ], f'to_kv: {got}'
    got = F.resolve_targets('diffusion_model.', 'perceiver.layers.0.ffn.w13')
    assert got == [
        ('perceiver.layers.0.ffn.w1', ChunkSpec(start=0, end=MLP_HIDDEN)),
        ('perceiver.layers.0.ffn.w3', ChunkSpec(start=MLP_PADDED, end=MLP_PADDED + MLP_HIDDEN)),
    ], f'w13: {got}'
    got = F.resolve_targets('lora_unet_', 'layers_0_feed_forward_w13')
    assert [path for path, _ in got] == ['layers_0_feed_forward_w1', 'layers_0_feed_forward_w3'], f'kohya w13: {got}'
    return True


def test_ffn_rows_release_width():
    """The release transformer width gives the 9536-wide FFN that ComfyUI pads to 9728."""
    sd_model = install_mock_pipe()
    sd_model.pipe.transformer.config.dim = F.DIM_DEFAULT
    assert F.ffn_rows() == (9536, 9728), f'ffn_rows = {F.ffn_rows()}'
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def test_lora_trainer_format():
    """The release trainer's PEFT keys bind on attention, FFN and Perceiver cross-attention."""
    net = load_via(F.try_load_lora, sd_lora_trainer_format())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    assert set(net.modules) == {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_feed_forward_w1',
        'lora_transformer_perceiver_layers_0_cross_attention_to_k',
    }, f'got {set(net.modules)}'
    assert all(isinstance(mod, network_lora.NetworkModuleLora) for mod in net.modules.values())
    return True


def test_lora_bare_injection_cross_attention():
    """A bare diffusers path on an injection block's cross-attention binds."""
    net = load_via(F.try_load_lora, sd_lora_bare_injection_cross_attention())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_layers_1_cross_attention_to_v' in net.modules
    return True


def test_lora_kohya_to_q():
    """Kohya flat-underscore LoRA on attention.to_q binds."""
    net = load_via(F.try_load_lora, sd_lora_kohya_to_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_attention_to_q' in net.modules
    return True


def test_lora_original_names():
    """Original checkpoint names are rewritten to the diffusers tree."""
    net = load_via(F.try_load_lora, sd_lora_original_names())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    assert set(net.modules) == {
        'lora_transformer_layers_1_cross_attention_to_q',
        'lora_transformer_perceiver_layers_0_self_attention_to_out_0',
        'lora_transformer_x_embedder',
    }, f'got {set(net.modules)}'
    return True


def test_lora_comfy_fused_qkv_chunked():
    """ComfyUI fused attention.qkv is split into three chunks targeting to_q/to_k/to_v."""
    net = load_via(F.try_load_lora, sd_lora_comfy_fused_qkv())
    assert net is not None and len(net.modules) == 3, f'expected 3 chunked modules, got {net.modules if net else None}'
    assert set(net.modules) == {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
    }, f'got {set(net.modules)}'
    for nk, mod in net.modules.items():
        assert_shape(mod.up_model.weight, (HIDDEN, RANK_LORA), label=nk)
    return True


def test_lora_comfy_fused_kv_chunked():
    """ComfyUI fused cross-attention to_kv is split into to_k and to_v."""
    net = load_via(F.try_load_lora, sd_lora_comfy_fused_kv())
    assert net is not None and len(net.modules) == 2, f'got {net.modules if net else None}'
    assert set(net.modules) == {
        'lora_transformer_layers_1_cross_attention_to_k',
        'lora_transformer_layers_1_cross_attention_to_v',
    }, f'got {set(net.modules)}'
    for nk, mod in net.modules.items():
        assert_shape(mod.up_model.weight, (HIDDEN, RANK_LORA), label=nk)
    return True


def test_lora_comfy_fused_w13_sliced():
    """ComfyUI fused w13 is sliced to w1 and w3 by row range, dropping each half's zero padding."""
    net = load_via(F.try_load_lora, sd_lora_comfy_fused_w13())
    assert net is not None and len(net.modules) == 2, f'got {net.modules if net else None}'
    w1 = net.modules.get('lora_transformer_layers_0_feed_forward_w1')
    w3 = net.modules.get('lora_transformer_layers_0_feed_forward_w3')
    assert w1 is not None and w3 is not None, f'got {set(net.modules)}'
    assert_shape(w1.up_model.weight, (MLP_HIDDEN, RANK_LORA), label='w1')
    assert_shape(w3.up_model.weight, (MLP_HIDDEN, RANK_LORA), label='w3')
    assert torch.all(w1.up_model.weight == 1.0), 'w1 rows are not the first half'
    assert torch.all(w3.up_model.weight == 2.0), 'w3 rows are not the second half'
    return True


def test_lora_comfy_w2_padded_input_trimmed():
    """ComfyUI w2's zero-padded input columns are dropped so the delta fits the unpadded module."""
    sd = sd_lora_comfy_w2_padded()
    net = load_via(F.try_load_lora, sd)
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    w2 = net.modules.get('lora_transformer_layers_0_feed_forward_w2')
    assert w2 is not None, f'got {set(net.modules)}'
    assert_shape(w2.down_model.weight, (RANK_LORA, MLP_HIDDEN), label='w2 down')
    assert torch.equal(w2.down_model.weight, sd['diffusion_model.layers.0.feed_forward.w2.lora_A.weight'][:, :MLP_HIDDEN]), 'the live columns must be kept as saved'
    return True


def test_lora_comfy_attention_out_alias():
    """ComfyUI attention.out is renamed to attention.to_out.0."""
    net = load_via(F.try_load_lora, sd_lora_comfy_attention_out())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_noise_refiner_0_attention_to_out_0' in net.modules
    return True


def test_lora_dora_threading():
    """dora_scale flows into NetworkModuleLora.dora_scale."""
    net = load_via(F.try_load_lora, sd_lora_with_dora_scale())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None, 'dora_scale not threaded into NetworkModule'
    return True


def test_lokr_adaln():
    """LoKR on adaLN_modulation.0 binds via NetworkModuleLokr."""
    net = load_via(F.try_load_lokr, sd_lokr_adaln())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_noise_refiner_0_adaLN_modulation_0' in net.modules
    assert isinstance(next(iter(net.modules.values())), network_lokr.NetworkModuleLokr)
    return True


def test_norm_comfy_q_norm_alias():
    """A norm delta on the ComfyUI q_norm name binds to norm_q."""
    net = load_via(F.try_load_norm, sd_norm_comfy_q_norm())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_layers_0_attention_norm_q' in net.modules
    assert isinstance(next(iter(net.modules.values())), network_norm.NetworkModuleNorm)
    return True


def test_full_final_layer_original_name():
    """A full delta on the original all_final_layer name binds to final_layer.linear."""
    net = load_via(F.try_load_full, sd_full_final_layer_original_name())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_final_layer_linear' in net.modules
    assert isinstance(next(iter(net.modules.values())), network_full.NetworkModuleFull)
    return True


def test_try_load_chain_merges_families():
    """try_load merges a LoRA and a LoKR from one file."""
    sd = {**sd_lora_trainer_format(), **sd_lokr_adaln()}
    net = load_via(F.try_load, sd)
    assert net is not None and len(net.modules) == 4, f'got {net.modules if net else None}'
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_shape():
    """NetworkModuleLora.calc_updown emits the target's shape."""
    net = load_via(F.try_load_lora, sd_lora_bare_injection_cross_attention())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, CAP)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA calc_updown')
    return True


def test_lora_w13_slice_calc_updown_shape():
    """A w13 row-range slice produces a delta of the w1 projection's shape."""
    net = load_via(F.try_load_lora, sd_lora_comfy_fused_w13())
    mod = make_network_for_module(net.modules['lora_transformer_layers_0_feed_forward_w1'])
    target = torch.randn(MLP_HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='w13 slice calc_updown')
    return True


def test_lokr_calc_updown_shape():
    """NetworkModuleLokr.calc_updown produces the target's shape."""
    net = load_via(F.try_load_lokr, sd_lokr_adaln())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(ADALN_OUT, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoKR calc_updown')
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing primitives ===')
    for fn in [test_parse_key_all_prefixes, test_resolve_targets_aliases, test_resolve_targets_fused, test_ffn_rows_release_width]:
        run_test(CAT_PARSE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_trainer_format,
        test_lora_bare_injection_cross_attention,
        test_lora_kohya_to_q,
        test_lora_original_names,
        test_lora_comfy_fused_qkv_chunked,
        test_lora_comfy_fused_kv_chunked,
        test_lora_comfy_fused_w13_sliced,
        test_lora_comfy_w2_padded_input_trimmed,
        test_lora_comfy_attention_out_alias,
        test_lora_dora_threading,
        test_lokr_adaln,
        test_norm_comfy_q_norm_alias,
        test_full_final_layer_original_name,
        test_try_load_chain_merges_families,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [test_lora_calc_updown_shape, test_lora_w13_slice_calc_updown_shape, test_lokr_calc_updown_shape]:
        run_test(CAT_MATH, fn)

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
