#!/usr/bin/env python
"""
Offline unit tests for Ideogram 4 native adapter loaders.

Covers the four native families bound by ``pipelines.ideogram.ideogram_lora``
(LoRA, LoKR, LoHA, OFT) plus DoRA threading via the universal
``NetworkModule.finalize_updown`` hook.

Tests build a mock Ideogram-4-shaped transformer, write synthetic safetensors
files for each adapter format observed in the wild, and exercise the full
loader path from state dict to ``NetworkModule*`` instantiation.

Save formats are cross-referenced against real Ideogram 4 LoRAs in the wild.
Every released file inspected uses the ``diffusion_model.`` prefix with fused
attention (``attention.qkv`` + ``attention.o``):

- LoRA (``.lora_A`` / ``.lora_B``): e.g. ``vintage-anime-ideogram_v3``,
  ``zjourneyv2``, ``Dorian Cleavenger style_ideogram4``
- LoKR (``.lokr_w1`` / ``.lokr_w2``): e.g. ``Realism_Engine_Ideogram_V4``,
  ``Big_Breast_ID4_v1``

The diffusers ``Ideogram4Transformer2DModel`` exposes split attention
(``layers.{i}.attention.{to_q,to_k,to_v,to_out.0}``), so the loader splits the
fused qkv up-weight into equal Q/K/V chunks and renames ``o`` to ``to_out.0``.
``feed_forward.{w1,w2,w3}`` and ``adaln_modulation`` match the diffusers names
and pass through. PEFT (``transformer.``), OneTrainer (``lora_transformer_``)
and kohya (``lora_unet_``) layouts are covered as defensive variants.

No running server required.

Usage:
    python test/test-ideogram-native-adapters.py
"""

import os
import sys
import tempfile
import time

import torch
import safetensors.torch

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
from modules.lora import (         # pylint: disable=wrong-import-position
    network, network_lora, network_lokr, network_hada, network_oft,
)
from pipelines.ideogram import ideogram_lora as I    # pylint: disable=wrong-import-position


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
# Mock Ideogram 4 transformer
# ============================================================
# Shape constants mirror Ideogram-4 proportions at toy scale. Real model:
# inner_dim = num_attention_heads * attention_head_dim = 18 * 256 = 4608,
# intermediate_size = 12288, adaln_dim = 512, adaln_modulation out = 4 * 4608.

HIDDEN = 96            # inner_dim
HEAD_DIM = 32          # attention_head_dim; n_heads = HIDDEN / HEAD_DIM = 3
MLP_HIDDEN = 256       # intermediate_size (feed_forward.w1/w3 out, w2 in)
ADALN_IN = 48          # adaln conditioning dim (Linear in-features; distinct from HIDDEN)
ADALN_OUT = 4 * HIDDEN # 384, matches Ideogram adaln_modulation out-features
IN_CH = 16             # packed latent channels (input_proj in / final_layer.linear out)
N_LAYERS = 2


# pylint: disable=attribute-defined-outside-init
class _Holder(torch.nn.Module):
    """Empty container module - we attach children dynamically."""


def build_ideogram_block():
    """Mirror ``Ideogram4Transformer2DModel``'s per-block diffusers module layout."""
    block = _Holder()
    # Attention (split projections + per-head q/k RMSNorm)
    block.attention = _Holder()
    block.attention.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attention.to_k = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attention.to_v = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    # to_out is a ModuleList in diffusers; index 0 is the Linear, index 1 is Dropout
    block.attention.to_out = torch.nn.ModuleList([
        torch.nn.Linear(HIDDEN, HIDDEN, bias=False),
        torch.nn.Dropout(0.0),
    ])
    block.attention.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    block.attention.norm_k = torch.nn.RMSNorm(HEAD_DIM)

    # SwiGLU feed-forward: w1/w3 are gate/up at MLP_HIDDEN, w2 is down
    block.feed_forward = _Holder()
    block.feed_forward.w1 = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=False)
    block.feed_forward.w2 = torch.nn.Linear(MLP_HIDDEN, HIDDEN, bias=False)
    block.feed_forward.w3 = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=False)

    # Per-block RMSNorms
    block.attention_norm1 = torch.nn.RMSNorm(HIDDEN)
    block.attention_norm2 = torch.nn.RMSNorm(HIDDEN)
    block.ffn_norm1 = torch.nn.RMSNorm(HIDDEN)
    block.ffn_norm2 = torch.nn.RMSNorm(HIDDEN)

    # Per-block modulation generator (Linear from adaln conditioning dim)
    block.adaln_modulation = torch.nn.Linear(ADALN_IN, ADALN_OUT, bias=True)
    return block


def build_mock_transformer():
    """Build a torch.nn.Module mimicking ``Ideogram4Transformer2DModel``.

    Mirrors the paths real Ideogram LoRAs target:
    ``layers.X.attention.{to_q,to_k,to_v,to_out.0}``,
    ``layers.X.feed_forward.{w1,w2,w3}``, ``layers.X.adaln_modulation``, plus a
    couple of top-level modules for passthrough coverage.
    """
    transformer = _Holder()
    transformer.layers = torch.nn.ModuleList([build_ideogram_block() for _ in range(N_LAYERS)])
    transformer.input_proj = torch.nn.Linear(IN_CH, HIDDEN, bias=True)
    transformer.adaln_proj = torch.nn.Linear(HIDDEN, ADALN_IN, bias=True)
    transformer.final_layer = _Holder()
    transformer.final_layer.linear = torch.nn.Linear(HIDDEN, IN_CH, bias=True)
    return transformer


class _MockIdeogramPipeline:
    """Class name carries 'Ideogram4' so name-based model-type dispatch routes correctly."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockIdeogramSdModel:
    """Outer wrapper exposing pipe + network_layer_mapping for
    ``lora_convert.assign_network_names_to_compvis_modules`` to write onto."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'Ideogram4Pipeline'  # belt-and-suspenders


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing an Ideogram-4-shaped transformer.

    Each test calls this fresh so any prior network_layer_name stamps don't
    leak across tests. Writes directly to ``model_data.sd_model`` to bypass
    the ModelData lock that no-ops the public setter outside webui startup.
    """
    transformer = build_mock_transformer()
    pipe = _MockIdeogramPipeline(transformer)
    sd_model = _MockIdeogramSdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# State-dict synthesizers (one per family/format)
# ============================================================

RANK_LORA = 8
LOKR_W1_DIM = 8


def sd_lora_bfl_fused_qkv():
    """Real Ideogram LoRA: ``diffusion_model.`` fused ``attention.qkv`` (.lora_A/.lora_B).

    Mirrors vintage-anime-ideogram_v3 / zjourneyv2. The up-weight has 3*HIDDEN
    rows (Q over K over V); the loader splits it into equal thirds.
    """
    return {
        'diffusion_model.layers.0.attention.qkv.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.qkv.lora_B.weight': torch.randn(3 * HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_o_alias():
    """Real Ideogram LoRA: ``attention.o`` output proj, renamed to ``attention.to_out.0``."""
    return {
        'diffusion_model.layers.0.attention.o.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.o.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_feed_forward():
    """Real Ideogram LoRA: ``feed_forward.w1`` (SwiGLU gate) passes through unchanged."""
    return {
        'diffusion_model.layers.0.feed_forward.w1.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w1.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_adaln():
    """Real Ideogram LoRA: ``adaln_modulation`` passes through unchanged."""
    return {
        'diffusion_model.layers.0.adaln_modulation.lora_A.weight': torch.randn(RANK_LORA, ADALN_IN),
        'diffusion_model.layers.0.adaln_modulation.lora_B.weight': torch.randn(ADALN_OUT, RANK_LORA),
    }


def sd_lora_full_block():
    """A complete block worth of LoRA: the six saved modules per layer.

    Resolves to eight diffusers targets (qkv -> to_q/to_k/to_v, o -> to_out.0,
    w1, w2, w3, adaln_modulation), mirroring the real 204-module / 34-layer files.
    """
    return {
        'diffusion_model.layers.0.attention.qkv.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.qkv.lora_B.weight': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.o.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.o.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w1.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w1.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w2.lora_A.weight': torch.randn(RANK_LORA, MLP_HIDDEN),
        'diffusion_model.layers.0.feed_forward.w2.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w3.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w3.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.adaln_modulation.lora_A.weight': torch.randn(RANK_LORA, ADALN_IN),
        'diffusion_model.layers.0.adaln_modulation.lora_B.weight': torch.randn(ADALN_OUT, RANK_LORA),
    }


def sd_lora_peft_split_to_q():
    """PEFT / diffusers export: ``transformer.`` prefix, already-split ``to_q``. Passthrough."""
    return {
        'transformer.layers.1.attention.to_q.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.1.attention.to_q.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'transformer.layers.1.attention.to_q.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_onetrainer_flat():
    """OneTrainer ``lora_transformer_`` diffusers-flat keys. Passthrough via shared resolver."""
    return {
        'lora_transformer_layers_0_attention_to_q.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_transformer_layers_0_attention_to_q.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'lora_transformer_layers_0_feed_forward_w2.lora_down.weight': torch.randn(RANK_LORA, MLP_HIDDEN),
        'lora_transformer_layers_0_feed_forward_w2.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_kohya_fused_qkv():
    """Kohya flat-underscore fused qkv (``lora_unet_``). Loader splits into Q/K/V."""
    return {
        'lora_unet_layers_0_attention_qkv.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_layers_0_attention_qkv.lora_up.weight': torch.randn(3 * HIDDEN, RANK_LORA),
        'lora_unet_layers_0_attention_qkv.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_kohya_o():
    """Kohya flat-underscore ``attention_o`` (``lora_unet_``), renamed to ``to_out.0``."""
    return {
        'lora_unet_layers_0_attention_o.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_layers_0_attention_o.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_with_dora_scale():
    """LoRA carrying a dora_scale companion vector to exercise DoRA threading."""
    return {
        'diffusion_model.layers.0.feed_forward.w1.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w1.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w1.dora_scale': torch.randn(MLP_HIDDEN),
    }


def sd_lokr_bfl_fused_qkv():
    """Real Ideogram LoKR: ``diffusion_model.`` fused ``attention.qkv``.

    Mirrors Realism_Engine_Ideogram_V4 / Big_Breast_ID4_v1. kron(w1, w2) has
    3*HIDDEN rows; the loader defers the equal-thirds split to apply time via
    ``NetworkModuleLokrChunk``.
    """
    return {
        'diffusion_model.layers.0.attention.qkv.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.layers.0.attention.qkv.lokr_w2': torch.randn((3 * HIDDEN) // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.layers.0.attention.qkv.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_lokr_bfl_o():
    """Real Ideogram LoKR on ``attention.o`` (non-fused, renamed to ``to_out.0``)."""
    return {
        'diffusion_model.layers.0.attention.o.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.layers.0.attention.o.lokr_w2': torch.randn(HIDDEN // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.layers.0.attention.o.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_loha_bfl_feed_forward():
    """LoHA on ``feed_forward.w1`` (non-fused passthrough). Binds via NetworkModuleHada."""
    return {
        'diffusion_model.layers.0.feed_forward.w1.hada_w1_a': torch.randn(MLP_HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w1.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w1.hada_w2_a': torch.randn(MLP_HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.feed_forward.w1.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.feed_forward.w1.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_loha_fused_qkv():
    """LoHA on fused ``attention.qkv`` -> three NetworkModuleHadaChunk (equal chunks)."""
    return {
        'diffusion_model.layers.0.attention.qkv.hada_w1_a': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.qkv.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.qkv.hada_w2_a': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.qkv.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
    }


def sd_oft_lycoris_feed_forward():
    """OFT LyCORIS-style oft_diag on ``feed_forward.w2`` (non-fused)."""
    num_blocks = 4
    block_size = HIDDEN // num_blocks  # w2 out-features == HIDDEN
    return {
        'diffusion_model.layers.0.feed_forward.w2.oft_blocks': torch.randn(num_blocks, block_size, block_size) * 0.01,
        'diffusion_model.layers.0.feed_forward.w2.oft_diag': torch.ones(num_blocks, block_size),
        'diffusion_model.layers.0.feed_forward.w2.alpha': torch.tensor(0.001),
    }


def sd_oft_fused_qkv_skipped():
    """OFT on fused ``attention.qkv`` - skipped (no row-sliceable OFT structure)."""
    num_blocks = 4
    block_size = (3 * HIDDEN) // num_blocks
    return {
        'diffusion_model.layers.0.attention.qkv.oft_blocks': torch.randn(num_blocks, block_size, block_size) * 0.01,
    }


# ============================================================
# Helpers: write state dict to disk, mock NetworkOnDisk
# ============================================================


class TempLora:
    """Context manager: writes a state dict to a temp safetensors file and yields
    a ``_MockNetworkOnDisk`` pointing at it. Cleans up on exit."""

    def __init__(self, state_dict, name='test'):
        self.state_dict = state_dict
        self.name = name
        self.path = None

    def __enter__(self):
        # safetensors requires contiguous tensors
        sd = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in self.state_dict.items()}
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        safetensors.torch.save_file(sd, self.path)
        return _MockNetworkOnDisk(self.path, self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class _MockNetworkOnDisk:
    """Stand-in for ``network.NetworkOnDisk`` exposing only the attributes
    the native loaders read."""

    def __init__(self, filename, name):
        self.filename = filename
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'


def assert_shape(t: torch.Tensor, expected_shape, label=''):
    actual = tuple(t.shape)
    assert actual == tuple(expected_shape), f'{label}: shape {actual}, expected {tuple(expected_shape)}'


def make_network_for_module(net_module: network.NetworkModule, te_mul: float = 1.0, unet_mul: float = 1.0):
    """Wire a NetworkModule's parent ``Network`` with given multipliers so
    calc_updown has a meaningful multiplier()/calc_scale() context."""
    net_module.network.te_multiplier = te_mul
    net_module.network.unet_multiplier = unet_mul
    return net_module


# ============================================================
# Tests - parsing primitives
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_all_prefixes():
    """parse_key recognizes BFL, PEFT, OneTrainer, kohya, and bare-BFL keys.

    Bare ``layers.`` paths return prefix_used=None (routed through resolve_targets
    rather than the generic passthrough) so a bare fused qkv still splits.
    """
    cases = [
        ('diffusion_model.layers.0.attention.qkv.lora_A.weight',
         I.LORA_SUFFIXES,
         ('diffusion_model.', 'layers.0.attention.qkv', 'lora_down.weight')),
        ('transformer.layers.0.attention.to_v.lora_B.weight',
         I.LORA_SUFFIXES,
         ('transformer.', 'layers.0.attention.to_v', 'lora_up.weight')),
        ('lora_transformer_layers_0_feed_forward_w2.lora_down.weight',
         I.LORA_SUFFIXES,
         ('lora_transformer_', 'layers_0_feed_forward_w2', 'lora_down.weight')),
        ('lora_unet_layers_0_attention_qkv.lora_down.weight',
         I.LORA_SUFFIXES,
         ('lora_unet_', 'layers_0_attention_qkv', 'lora_down.weight')),
        # Bare BFL path -> prefix_used None
        ('layers.0.attention.o.lora_A.weight',
         I.LORA_SUFFIXES,
         (None, 'layers.0.attention.o', 'lora_down.weight')),
        ('random.unrelated.key', I.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = I.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_marker_disambiguation():
    """Each family's markers reject other families' files."""
    pure_lora = {
        'diffusion_model.layers.0.attention.qkv.lora_A.weight': torch.zeros(1, 1),
        'diffusion_model.layers.0.attention.qkv.lora_B.weight': torch.zeros(1, 1),
    }
    assert I.has_marker(pure_lora, I.LORA_MARKERS)
    assert not I.has_marker(pure_lora, I.LOKR_MARKERS)
    assert not I.has_marker(pure_lora, I.LOHA_MARKERS)
    assert not I.has_marker(pure_lora, I.OFT_MARKERS)

    pure_lokr = {
        'diffusion_model.layers.0.attention.qkv.lokr_w1': torch.zeros(1, 1),
        'diffusion_model.layers.0.attention.qkv.lokr_w2': torch.zeros(1, 1),
    }
    assert I.has_marker(pure_lokr, I.LOKR_MARKERS)
    assert not I.has_marker(pure_lokr, I.LORA_MARKERS)
    assert not I.has_marker(pure_lokr, I.LOHA_MARKERS)
    assert not I.has_marker(pure_lokr, I.OFT_MARKERS)
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def test_lora_bfl_fused_qkv_chunked():
    """BFL fused attention.qkv splits into 3 chunks targeting to_q/to_k/to_v.

    The up-weight is chunked along dim 0 at load time (each chunk HIDDEN rows);
    the down-weight is shared.
    """
    net = _load_via(I.try_load_lora, sd_lora_bfl_fused_qkv())
    assert net is not None and len(net.modules) == 3, f'expected 3 chunked modules, got {net.modules if net else None}'
    expected = {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
    }
    assert set(net.modules) == expected, f'got {set(net.modules)}'
    for nk, mod in net.modules.items():
        assert isinstance(mod, network_lora.NetworkModuleLora), f'{nk}: type={type(mod).__name__}'
        assert_shape(mod.up_model.weight, (HIDDEN, RANK_LORA), label=nk)
    return True


def test_lora_bfl_o_alias():
    """BFL attention.o output proj is renamed to attention.to_out.0."""
    net = _load_via(I.try_load_lora, sd_lora_bfl_o_alias())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_attention_to_out_0' in net.modules
    return True


def test_lora_bfl_feed_forward_passthrough():
    """BFL feed_forward.w1 binds unchanged (diffusers name matches)."""
    net = _load_via(I.try_load_lora, sd_lora_bfl_feed_forward())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_feed_forward_w1' in net.modules
    return True


def test_lora_bfl_adaln_passthrough():
    """BFL adaln_modulation binds unchanged (diffusers name matches)."""
    net = _load_via(I.try_load_lora, sd_lora_bfl_adaln())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_adaln_modulation' in net.modules
    return True


def test_lora_full_block():
    """Six saved modules resolve to eight diffusers targets across one block."""
    net = _load_via(I.try_load_lora, sd_lora_full_block())
    assert net is not None and len(net.modules) == 8, f'expected 8 modules, got {net.modules if net else None}'
    expected = {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
        'lora_transformer_layers_0_attention_to_out_0',
        'lora_transformer_layers_0_feed_forward_w1',
        'lora_transformer_layers_0_feed_forward_w2',
        'lora_transformer_layers_0_feed_forward_w3',
        'lora_transformer_layers_0_adaln_modulation',
    }
    assert set(net.modules) == expected, f'got {set(net.modules)}'
    return True


def test_lora_peft_split_passthrough():
    """PEFT transformer. prefix on an already-split to_q passes through."""
    net = _load_via(I.try_load_lora, sd_lora_peft_split_to_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_1_attention_to_q' in net.modules
    return True


def test_lora_onetrainer_flat_passthrough():
    """OneTrainer lora_transformer_ flat keys bind via the shared passthrough (no rename)."""
    net = _load_via(I.try_load_lora, sd_lora_onetrainer_flat())
    assert net is not None and len(net.modules) == 2, f'got {net.modules if net else None}'
    assert set(net.modules) == {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_feed_forward_w2',
    }, f'got {set(net.modules)}'
    return True


def test_lora_kohya_fused_qkv_chunked():
    """Kohya flat-underscore fused qkv splits into to_q/to_k/to_v."""
    net = _load_via(I.try_load_lora, sd_lora_kohya_fused_qkv())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    assert set(net.modules) == {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
    }, f'got {set(net.modules)}'
    return True


def test_lora_kohya_o_alias():
    """Kohya flat-underscore attention_o is renamed to to_out_0."""
    net = _load_via(I.try_load_lora, sd_lora_kohya_o())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_attention_to_out_0' in net.modules
    return True


def test_lora_dora_threading():
    """dora_scale flows into NetworkModuleLora.dora_scale."""
    net = _load_via(I.try_load_lora, sd_lora_with_dora_scale())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None, 'dora_scale not threaded into NetworkModule'
    return True


def test_lokr_bfl_fused_qkv_chunked():
    """BFL fused LoKR on attention.qkv emits 3 chunked modules via NetworkModuleLokrChunk."""
    net = _load_via(I.try_load_lokr, sd_lokr_bfl_fused_qkv())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    expected = {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
    }
    assert set(net.modules) == expected
    for nk, mod in net.modules.items():
        assert isinstance(mod, network_lokr.NetworkModuleLokrChunk), f'{nk}: type={type(mod).__name__}'
    return True


def test_lokr_bfl_o():
    """BFL LoKR on attention.o binds via NetworkModuleLokr, renamed to to_out.0."""
    net = _load_via(I.try_load_lokr, sd_lokr_bfl_o())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_attention_to_out_0' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lokr.NetworkModuleLokr)
    return True


def test_loha_bfl_feed_forward():
    """BFL LoHA on a non-fused feed_forward target binds via NetworkModuleHada."""
    net = _load_via(I.try_load_loha, sd_loha_bfl_feed_forward())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_hada.NetworkModuleHada)
    return True


def test_loha_fused_qkv_chunked():
    """LoHA on fused attention.qkv emits 3 HadaChunk modules (equal chunks)."""
    net = _load_via(I.try_load_loha, sd_loha_fused_qkv())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    expected = {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
    }
    assert set(net.modules) == expected, f'got {set(net.modules)}'
    for mod in net.modules.values():
        assert isinstance(mod, network_hada.NetworkModuleHadaChunk)
    return True


def test_oft_lycoris_no_npe():
    """OFT loader handles LyCORIS oft_diag files without NoneType-attr errors."""
    net = _load_via(I.try_load_oft, sd_oft_lycoris_feed_forward())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_oft.NetworkModuleOFT)
    return True


def test_oft_fused_qkv_skipped():
    """OFT on fused attention.qkv is skipped (no row-sliceable structure)."""
    net = _load_via(I.try_load_oft, sd_oft_fused_qkv_skipped())
    assert net is None or len(net.modules) == 0, f'expected no modules, got {net.modules if net else None}'
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_shape():
    """NetworkModuleLora.calc_updown emits the right shape against a target weight."""
    net = _load_via(I.try_load_lora, sd_lora_bfl_o_alias())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA calc_updown')
    return True


def test_lora_chunk_calc_updown_shape():
    """A chunked Q/K/V LoRA target produces a single-projection-shaped delta."""
    net = _load_via(I.try_load_lora, sd_lora_bfl_fused_qkv())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)  # one Q/K/V projection's shape
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA chunk calc_updown')
    return True


def test_lokr_calc_updown_shape():
    """NetworkModuleLokr.calc_updown produces a tensor matching the target's shape."""
    net = _load_via(I.try_load_lokr, sd_lokr_bfl_o())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoKR calc_updown')
    return True


def test_lokr_chunk_calc_updown_shape():
    """LokrChunk returns the designated row range; chunk shape matches the split target."""
    net = _load_via(I.try_load_lokr, sd_lokr_bfl_fused_qkv())
    assert net is not None
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)  # one Q/K/V projection's shape
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LokrChunk calc_updown')
    return True


def test_loha_calc_updown_shape():
    """NetworkModuleHada produces shapes matching the target."""
    net = _load_via(I.try_load_loha, sd_loha_bfl_feed_forward())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(MLP_HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoHA calc_updown')
    return True


def test_oft_calc_updown_shape():
    """NetworkModuleOFT calc_updown shape sanity against the target weight."""
    net = _load_via(I.try_load_oft, sd_oft_lycoris_feed_forward())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, MLP_HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='OFT calc_updown')
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing primitives ===')
    for fn in [test_parse_key_all_prefixes, test_marker_disambiguation]:
        run_test(CAT_PARSE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_bfl_fused_qkv_chunked,
        test_lora_bfl_o_alias,
        test_lora_bfl_feed_forward_passthrough,
        test_lora_bfl_adaln_passthrough,
        test_lora_full_block,
        test_lora_peft_split_passthrough,
        test_lora_onetrainer_flat_passthrough,
        test_lora_kohya_fused_qkv_chunked,
        test_lora_kohya_o_alias,
        test_lora_dora_threading,
        test_lokr_bfl_fused_qkv_chunked,
        test_lokr_bfl_o,
        test_loha_bfl_feed_forward,
        test_loha_fused_qkv_chunked,
        test_oft_lycoris_no_npe,
        test_oft_fused_qkv_skipped,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [
        test_lora_calc_updown_shape,
        test_lora_chunk_calc_updown_shape,
        test_lokr_calc_updown_shape,
        test_lokr_chunk_calc_updown_shape,
        test_loha_calc_updown_shape,
        test_oft_calc_updown_shape,
    ]:
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
