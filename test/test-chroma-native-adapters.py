#!/usr/bin/env python
"""
Offline unit tests for Chroma native adapter loaders.

Covers the four native families currently supported by ``pipelines.chroma.chroma_lora``
(LoRA, LoKR, LoHA, OFT) plus DoRA threading via the universal
``NetworkModule.finalize_updown`` hook.

Tests build a mock Chroma-shaped transformer, write synthetic safetensors
files for each adapter format observed in the wild, and exercise the full
loader path including the Flux-to-diffusers rename table and the unique
single-block ``linear1`` unequal-chunk slicing.

Save formats are cross-referenced against real Chroma LoRAs:

- BFL / AI-toolkit (``diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight``):
  e.g. ``Chroma - Lenovo UltraReal``
- kohya (``lora_unet_double_blocks_0_img_attn_proj.lora_down.weight``):
  e.g. ``90s_anime_aesthetic_Chroma``
- PEFT (``transformer.transformer_blocks.0.attn.to_q.lora_down.weight``)

Chroma LoRAs are trained against the Flux block layout (``double_blocks``,
``single_blocks``) regardless of save format. The diffusers
``ChromaTransformer2DModel`` exposes split-attention modules at
``transformer_blocks.X.attn.{to_q,to_k,to_v,...}`` and
``single_transformer_blocks.X.{attn.*, proj_mlp, proj_out}``. The loader
path-rewrites Flux paths to diffusers names and handles two distinct
fused-weight layouts:

- **Equal chunks** (double_blocks img_attn.qkv / txt_attn.qkv at
  ``[HIDDEN, HIDDEN, HIDDEN]``): LoRA chunks at load via ``torch.chunk``;
  LoKR defers via ``NetworkModuleLokrChunk``.
- **Unequal chunks** (single_blocks linear1 at
  ``[HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN]``): LoRA slices row ranges at load;
  LoKR defers via ``NetworkModuleLokrSliceChunk``.

LoHA and OFT on fused targets are skipped with a warning (no slice variant).

The ``distilled_guidance_layer`` (Chroma's central modulation generator that
replaces Flux's per-block ``norm1.linear``) is a real module path that
``assign_network_names_to_compvis_modules`` registers, so LoRAs targeting it
pass through unchanged.

No running server required.

Usage:
    python test/test-chroma-native-adapters.py
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
from modules import shared        # pylint: disable=wrong-import-position
from modules.lora import (         # pylint: disable=wrong-import-position
    network, network_lora, network_lokr, network_hada, network_oft,
)
from modules.lora import lora_common as l_common   # pylint: disable=wrong-import-position
from pipelines.chroma import chroma_lora as C    # pylint: disable=wrong-import-position


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
# Mock Chroma transformer
# ============================================================
# Shape constants chosen to mirror ChromaTransformer2DModel proportions
# while keeping tensors small. Real Chroma1-HD: inner_dim=3072,
# mlp_hidden=12288. We use HIDDEN=96, MLP_HIDDEN=384 (4x), so the unequal
# single-block linear1 partition [HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN] =
# [96, 96, 96, 384] (analogous to real [3072, 3072, 3072, 12288]).

HIDDEN = 96
HEAD_DIM = 32                       # N_HEADS = HIDDEN / HEAD_DIM = 3
MLP_RATIO = 4
MLP_HIDDEN = HIDDEN * MLP_RATIO     # 384
QKV_FUSED_OUT = 3 * HIDDEN          # 288 (img_attn.qkv / txt_attn.qkv output dim)
LINEAR1_OUT = 3 * HIDDEN + MLP_HIDDEN  # 672 (single block linear1 fused output)
LINEAR2_IN = HIDDEN + MLP_HIDDEN    # 480 (single block proj_out input - attn out + mlp out concat)
N_DOUBLE = 2
N_SINGLE = 2


# pylint: disable=attribute-defined-outside-init
class _Holder(torch.nn.Module):
    """Empty container module - we attach children dynamically."""


def build_double_block():
    """Mirror ``ChromaTransformerBlock``'s diffusers-side module layout.

    Uses ``FluxAttention(added_kv_proj_dim=dim)`` so both img-side and
    context-side QKV + output projections are present.
    """
    block = _Holder()
    # FluxAttention sub-modules
    block.attn = _Holder()
    block.attn.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.to_k = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.to_v = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.to_out = torch.nn.ModuleList([
        torch.nn.Linear(HIDDEN, HIDDEN, bias=True),
        torch.nn.Dropout(0.0),
    ])
    block.attn.add_q_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.add_k_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.add_v_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.to_add_out = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    block.attn.norm_k = torch.nn.RMSNorm(HEAD_DIM)
    block.attn.norm_added_q = torch.nn.RMSNorm(HEAD_DIM)
    block.attn.norm_added_k = torch.nn.RMSNorm(HEAD_DIM)

    # FeedForward modules: net = [GELU(proj=Linear), Dropout, Linear]
    block.ff = _Holder()
    block.ff.net = torch.nn.ModuleList()
    proj_act = _Holder()
    proj_act.proj = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=True)
    block.ff.net.append(proj_act)
    block.ff.net.append(torch.nn.Dropout(0.0))
    block.ff.net.append(torch.nn.Linear(MLP_HIDDEN, HIDDEN, bias=True))

    block.ff_context = _Holder()
    block.ff_context.net = torch.nn.ModuleList()
    proj_act_ctx = _Holder()
    proj_act_ctx.proj = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=True)
    block.ff_context.net.append(proj_act_ctx)
    block.ff_context.net.append(torch.nn.Dropout(0.0))
    block.ff_context.net.append(torch.nn.Linear(MLP_HIDDEN, HIDDEN, bias=True))

    # norm1 / norm1_context / norm2 / norm2_context are AdaLayerNormZeroPruned
    # or LayerNorm(elementwise_affine=False) - no learnable weight at the
    # block-norm level, so we don't need LoRA-targetable norm modules here.

    return block


def build_single_block():
    """Mirror ``ChromaSingleTransformerBlock`` - has proj_mlp + attn (pre_only) + proj_out."""
    block = _Holder()
    block.attn = _Holder()
    block.attn.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.to_k = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.to_v = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    block.attn.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    block.attn.norm_k = torch.nn.RMSNorm(HEAD_DIM)
    # pre_only=True so no to_out
    block.proj_mlp = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=True)
    block.proj_out = torch.nn.Linear(LINEAR2_IN, HIDDEN, bias=True)
    return block


def build_mock_transformer():
    """Build a torch.nn.Module mimicking ``ChromaTransformer2DModel``."""
    transformer = _Holder()
    transformer.transformer_blocks = torch.nn.ModuleList([build_double_block() for _ in range(N_DOUBLE)])
    transformer.single_transformer_blocks = torch.nn.ModuleList([build_single_block() for _ in range(N_SINGLE)])
    # distilled_guidance_layer - Chroma's central modulation approximator
    # Minimal stand-in: just one linear submodule the tests can target
    transformer.distilled_guidance_layer = _Holder()
    transformer.distilled_guidance_layer.in_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    return transformer


class _MockChromaPipeline:
    """Class name carries 'Chroma' so name-based model-type dispatch routes correctly."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockChromaSdModel:
    """Outer wrapper holding pipe + network_layer_mapping."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'ChromaPipeline'


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing a Chroma-shaped transformer.

    Each test re-installs so any prior network_layer_name stamps don't leak.
    Writes directly to model_data.sd_model to bypass the ModelData lock.

    Also patches ``chroma_lora.QKV_DIMS`` and ``chroma_lora.LINEAR1_DIMS`` to
    match the test mock's scaled-down ``HIDDEN`` / ``MLP_HIDDEN``. The module
    hardcodes Chroma1-HD's 3072 / 12288, which mismatches small test tensors
    and causes ``split_fused_lora_group``'s
    ``up.shape[0] != sum(dims)`` gate to reject every fused fixture.
    """
    transformer = build_mock_transformer()
    pipe = _MockChromaPipeline(transformer)
    sd_model = _MockChromaSdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model

    # chroma_lora's get_block_counts() reads transformer.config.num_layers /
    # num_single_layers. Stamp that here so build_static_rename gets the right
    # block counts for the test mock (defaults are 19/38 which our 2/2 mock doesn't have).
    transformer.config = _ChromaConfig(num_layers=N_DOUBLE, num_single_layers=N_SINGLE)

    # Patch the hardcoded Chroma1-HD dims to the test scale.
    C.QKV_DIMS = [HIDDEN, HIDDEN, HIDDEN]
    C.LINEAR1_DIMS = [HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN]
    return sd_model


class _ChromaConfig:
    def __init__(self, num_layers, num_single_layers):
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers


# ============================================================
# State-dict synthesizers (one per family/format)
# ============================================================

RANK_LORA = 8
LOKR_W1_DIM = 8


def sd_lora_bfl_img_attn_proj():
    """BFL LoRA on double-block img_attn.proj.

    BFL path maps to diffusers ``transformer_blocks.0.attn.to_out.0`` via
    ``DOUBLE_RENAME_TEMPLATES``.
    """
    return {
        'diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.0.img_attn.proj.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_bfl_img_attn_qkv_fused():
    """BFL LoRA on fused img_attn.qkv. Loader splits up-weight along dim 0 into Q/K/V."""
    return {
        'diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight': torch.randn(QKV_FUSED_OUT, RANK_LORA),
        'diffusion_model.double_blocks.0.img_attn.qkv.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_bfl_txt_attn_qkv_fused():
    """BFL LoRA on fused txt_attn.qkv. Loader emits 3 chunks to add_{q,k,v}_proj."""
    return {
        'diffusion_model.double_blocks.0.txt_attn.qkv.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight': torch.randn(QKV_FUSED_OUT, RANK_LORA),
    }


def sd_lora_bfl_img_mlp():
    """BFL LoRA on double-block img_mlp.0 and img_mlp.2.

    img_mlp.0 -> ff.net.0.proj, img_mlp.2 -> ff.net.2.
    """
    return {
        'diffusion_model.double_blocks.1.img_mlp.0.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.1.img_mlp.0.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.1.img_mlp.2.lora_A.weight': torch.randn(RANK_LORA, MLP_HIDDEN),
        'diffusion_model.double_blocks.1.img_mlp.2.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_txt_mlp():
    """BFL LoRA on double-block txt_mlp.0 and txt_mlp.2 - context side."""
    return {
        'diffusion_model.double_blocks.0.txt_mlp.0.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.0.txt_mlp.0.lora_B.weight': torch.randn(MLP_HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_single_linear1_unequal():
    """BFL LoRA on single-block linear1.

    linear1 fuses Q/K/V/proj_mlp at unequal dims [HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN].
    Loader emits 4 targets with unequal row-range chunks.
    """
    return {
        'diffusion_model.single_blocks.0.linear1.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.single_blocks.0.linear1.lora_B.weight': torch.randn(LINEAR1_OUT, RANK_LORA),
    }


def sd_lora_bfl_single_linear2():
    """BFL LoRA on single-block linear2 (-> single_transformer_blocks.X.proj_out)."""
    return {
        'diffusion_model.single_blocks.0.linear2.lora_A.weight': torch.randn(RANK_LORA, LINEAR2_IN),
        'diffusion_model.single_blocks.0.linear2.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_kohya_img_attn_proj():
    """Kohya flat-underscore LoRA on img_attn.proj. Mirrors 90s_anime_aesthetic_Chroma."""
    return {
        'lora_unet_double_blocks_0_img_attn_proj.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_double_blocks_0_img_attn_proj.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'lora_unet_double_blocks_0_img_attn_proj.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_kohya_img_attn_qkv_fused():
    """Kohya LoRA on fused img_attn.qkv."""
    return {
        'lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight': torch.randn(QKV_FUSED_OUT, RANK_LORA),
        'lora_unet_double_blocks_0_img_attn_qkv.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_peft_to_q():
    """PEFT-format LoRA targeting a split diffusers path (no rename, no chunking)."""
    return {
        'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_distilled_guidance():
    """LoRA targeting Chroma's distilled_guidance_layer (passes through unchanged)."""
    return {
        'diffusion_model.distilled_guidance_layer.in_proj.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.distilled_guidance_layer.in_proj.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_with_dora_scale():
    """LoRA with dora_scale companion to exercise DoRA threading."""
    return {
        'transformer.transformer_blocks.0.attn.to_v.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.transformer_blocks.0.attn.to_v.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'transformer.transformer_blocks.0.attn.to_v.dora_scale': torch.randn(HIDDEN),
    }


def sd_lokr_bfl_img_attn_proj():
    """BFL LoKR on a non-fused proj target. Loader uses NetworkModuleLokr (no chunk)."""
    return {
        'diffusion_model.double_blocks.0.img_attn.proj.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.double_blocks.0.img_attn.proj.lokr_w2': torch.randn(HIDDEN // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.double_blocks.0.img_attn.proj.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_lokr_bfl_img_attn_qkv_equal_chunks():
    """BFL LoKR on fused img_attn.qkv (equal chunks).

    Loader emits 3 NetworkModuleLokrSliceChunk via row ranges [0:HIDDEN], [HIDDEN:2*HIDDEN], [2*HIDDEN:3*HIDDEN].
    (Chroma's implementation slices even equal-chunks via row ranges since the same logic handles both.)
    """
    return {
        'diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2': torch.randn(QKV_FUSED_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.double_blocks.0.img_attn.qkv.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_lokr_bfl_single_linear1_unequal():
    """BFL LoKR on fused single-block linear1 (UNEQUAL chunks).

    Loader emits 4 NetworkModuleLokrSliceChunk with row ranges matching
    [HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN] partitions.
    """
    return {
        'diffusion_model.single_blocks.0.linear1.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.single_blocks.0.linear1.lokr_w2': torch.randn(LINEAR1_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.single_blocks.0.linear1.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_loha_bfl_img_attn_proj():
    """LoHA on a non-fused target binds via NetworkModuleHada."""
    return {
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w1_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w2_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.1.img_attn.proj.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_loha_bfl_img_attn_qkv_skipped():
    """LoHA on fused img_attn.qkv is dropped by the loader (no slice variant for LoHA)."""
    return {
        'diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_a': torch.randn(QKV_FUSED_OUT, RANK_LORA),
        'diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_a': torch.randn(QKV_FUSED_OUT, RANK_LORA),
        'diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
    }


def sd_oft_bfl_img_attn_proj():
    """OFT (LyCORIS oft_diag form) on non-fused target."""
    num_blocks = 4
    block_size = HIDDEN // num_blocks
    return {
        'diffusion_model.double_blocks.0.img_attn.proj.oft_blocks': torch.randn(num_blocks, block_size, block_size) * 0.01,
        'diffusion_model.double_blocks.0.img_attn.proj.oft_diag': torch.ones(num_blocks, block_size),
        'diffusion_model.double_blocks.0.img_attn.proj.alpha': torch.tensor(0.001),
    }


def sd_oft_bfl_img_attn_qkv_skipped():
    """OFT on fused img_attn.qkv - dropped by the loader (OFT structure tied to out_features)."""
    num_blocks = 4
    block_size = QKV_FUSED_OUT // num_blocks
    return {
        'diffusion_model.double_blocks.0.img_attn.qkv.oft_blocks': torch.randn(num_blocks, block_size, block_size) * 0.01,
    }


# ============================================================
# Helpers
# ============================================================


class TempLora:
    """Context manager: writes a state dict to a temp safetensors file."""

    def __init__(self, state_dict, name='test'):
        self.state_dict = state_dict
        self.name = name
        self.path = None

    def __enter__(self):
        sd = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in self.state_dict.items()}
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        safetensors.torch.save_file(sd, self.path)
        return _MockNetworkOnDisk(self.path, self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class _MockNetworkOnDisk:
    def __init__(self, filename, name):
        self.filename = filename
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'


def assert_shape(t: torch.Tensor, expected_shape, label=''):
    actual = tuple(t.shape)
    assert actual == tuple(expected_shape), f'{label}: shape {actual}, expected {tuple(expected_shape)}'


def make_network_for_module(net_module: network.NetworkModule, te_mul: float = 1.0, unet_mul: float = 1.0):
    net_module.network.te_multiplier = te_mul
    net_module.network.unet_multiplier = unet_mul
    return net_module


# ============================================================
# Tests - parsing primitives
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_all_prefixes():
    """parse_key recognizes BFL, PEFT, kohya keys with the right flat_key + suffix.

    Chroma's parse_key returns (flat_key, suffix) where flat_key is the
    underscored Flux-layout path (before static rename).
    """
    cases = [
        # BFL prefix -> dotted base flattened to underscores
        ('diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight',
         C.LORA_SUFFIXES,
         ('double_blocks_0_img_attn_proj', 'lora_down.weight')),
        # PEFT prefix -> stays in diffusers form
        ('transformer.transformer_blocks.0.attn.to_q.lora_B.weight',
         C.LORA_SUFFIXES,
         ('transformer_blocks_0_attn_to_q', 'lora_up.weight')),
        # kohya prefix - already flat underscores
        ('lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight',
         C.LORA_SUFFIXES,
         ('double_blocks_0_img_attn_qkv', 'lora_down.weight')),
        # Unrelated keys reject cleanly
        ('random.unrelated.key', C.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = C.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_marker_disambiguation():
    """Each family's markers reject other families' files."""
    pure_lora = {
        'lora_unet_double_blocks_0_img_attn_proj.lora_down.weight': torch.zeros(1, 1),
        'lora_unet_double_blocks_0_img_attn_proj.lora_up.weight': torch.zeros(1, 1),
    }
    assert C.has_marker(pure_lora, C.LORA_MARKERS)
    assert not C.has_marker(pure_lora, C.LOKR_MARKERS)
    assert not C.has_marker(pure_lora, C.LOHA_MARKERS)
    assert not C.has_marker(pure_lora, C.OFT_MARKERS)

    pure_lokr = {
        'diffusion_model.double_blocks.0.img_attn.proj.lokr_w1': torch.zeros(1, 1),
        'diffusion_model.double_blocks.0.img_attn.proj.lokr_w2': torch.zeros(1, 1),
    }
    assert C.has_marker(pure_lokr, C.LOKR_MARKERS)
    assert not C.has_marker(pure_lokr, C.LORA_MARKERS)
    return True


def test_static_rename_table():
    """build_static_rename produces correct Flux to diffusers path remappings.

    Verifies the rename templates against the documented chroma_lora layout:
    double_blocks_X_img_attn_proj -> transformer_blocks_X_attn_to_out_0, etc.
    """
    rename = C.build_static_rename(N_DOUBLE, N_SINGLE)
    # Spot-check key remappings
    assert rename['double_blocks_0_img_attn_proj'] == 'transformer_blocks_0_attn_to_out_0'
    assert rename['double_blocks_0_txt_attn_proj'] == 'transformer_blocks_0_attn_to_add_out'
    assert rename['double_blocks_1_img_mlp_0'] == 'transformer_blocks_1_ff_net_0_proj'
    assert rename['double_blocks_1_img_mlp_2'] == 'transformer_blocks_1_ff_net_2'
    assert rename['double_blocks_0_txt_mlp_0'] == 'transformer_blocks_0_ff_context_net_0_proj'
    assert rename['single_blocks_0_linear2'] == 'single_transformer_blocks_0_proj_out'
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def test_lora_bfl_img_attn_proj():
    """BFL LoRA on img_attn.proj renames to attn.to_out.0."""
    net = _load_via(C.try_load_lora, sd_lora_bfl_img_attn_proj())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_transformer_blocks_0_attn_to_out_0' in net.modules
    return True


def test_lora_bfl_img_attn_qkv_chunked():
    """BFL LoRA on fused img_attn.qkv emits 3 chunks targeting to_q/to_k/to_v."""
    net = _load_via(C.try_load_lora, sd_lora_bfl_img_attn_qkv_fused())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    expected = {
        'lora_transformer_transformer_blocks_0_attn_to_q',
        'lora_transformer_transformer_blocks_0_attn_to_k',
        'lora_transformer_transformer_blocks_0_attn_to_v',
    }
    assert set(net.modules) == expected
    # Each chunked up tensor has shape (HIDDEN, RANK), not (QKV_FUSED_OUT, RANK)
    for nk, mod in net.modules.items():
        assert_shape(mod.up_model.weight, (HIDDEN, RANK_LORA), label=nk)
    return True


def test_lora_bfl_txt_attn_qkv_chunked():
    """BFL LoRA on fused txt_attn.qkv emits 3 chunks targeting add_q/k/v_proj (context side)."""
    net = _load_via(C.try_load_lora, sd_lora_bfl_txt_attn_qkv_fused())
    assert net is not None and len(net.modules) == 3
    expected = {
        'lora_transformer_transformer_blocks_0_attn_add_q_proj',
        'lora_transformer_transformer_blocks_0_attn_add_k_proj',
        'lora_transformer_transformer_blocks_0_attn_add_v_proj',
    }
    assert set(net.modules) == expected
    return True


def test_lora_bfl_img_mlp():
    """img_mlp.0 -> ff.net.0.proj, img_mlp.2 -> ff.net.2."""
    net = _load_via(C.try_load_lora, sd_lora_bfl_img_mlp())
    assert net is not None and len(net.modules) == 2
    assert 'lora_transformer_transformer_blocks_1_ff_net_0_proj' in net.modules
    assert 'lora_transformer_transformer_blocks_1_ff_net_2' in net.modules
    return True


def test_lora_bfl_txt_mlp():
    """txt_mlp.0 -> ff_context.net.0.proj."""
    net = _load_via(C.try_load_lora, sd_lora_bfl_txt_mlp())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_ff_context_net_0_proj' in net.modules
    return True


def test_lora_bfl_single_linear1_unequal_chunks():
    """BFL LoRA on single linear1 emits 4 targets with UNEQUAL row ranges.

    Partitions: [HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN] -> to_q, to_k, to_v, proj_mlp.
    The first three chunks have (HIDDEN, RANK) up-shape; the fourth has (MLP_HIDDEN, RANK).
    """
    net = _load_via(C.try_load_lora, sd_lora_bfl_single_linear1_unequal())
    assert net is not None and len(net.modules) == 4, f'got {net.modules if net else None}'
    expected = {
        'lora_transformer_single_transformer_blocks_0_attn_to_q',
        'lora_transformer_single_transformer_blocks_0_attn_to_k',
        'lora_transformer_single_transformer_blocks_0_attn_to_v',
        'lora_transformer_single_transformer_blocks_0_proj_mlp',
    }
    assert set(net.modules) == expected
    # proj_mlp has the MLP_HIDDEN chunk; QKV targets have HIDDEN
    for nk, mod in net.modules.items():
        if nk.endswith('proj_mlp'):
            assert_shape(mod.up_model.weight, (MLP_HIDDEN, RANK_LORA), label=nk)
        else:
            assert_shape(mod.up_model.weight, (HIDDEN, RANK_LORA), label=nk)
    return True


def test_lora_bfl_single_linear2():
    """linear2 -> single_transformer_blocks.X.proj_out (no chunking)."""
    net = _load_via(C.try_load_lora, sd_lora_bfl_single_linear2())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_single_transformer_blocks_0_proj_out' in net.modules
    return True


def test_lora_kohya_img_attn_proj():
    """Kohya flat-underscore on non-fused target binds with same diffusers-path key as BFL."""
    net = _load_via(C.try_load_lora, sd_lora_kohya_img_attn_proj())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn_to_out_0' in net.modules
    return True


def test_lora_kohya_img_attn_qkv_chunked():
    """Kohya fused img_attn.qkv splits into 3 chunks same as BFL form."""
    net = _load_via(C.try_load_lora, sd_lora_kohya_img_attn_qkv_fused())
    assert net is not None and len(net.modules) == 3
    expected = {
        'lora_transformer_transformer_blocks_0_attn_to_q',
        'lora_transformer_transformer_blocks_0_attn_to_k',
        'lora_transformer_transformer_blocks_0_attn_to_v',
    }
    assert set(net.modules) == expected
    return True


def test_lora_peft_to_q():
    """PEFT format with diffusers paths passes through unchanged."""
    net = _load_via(C.try_load_lora, sd_lora_peft_to_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn_to_q' in net.modules
    return True


def test_lora_distilled_guidance():
    """LoRA on distilled_guidance_layer passes through unchanged (real module path)."""
    net = _load_via(C.try_load_lora, sd_lora_distilled_guidance())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_distilled_guidance_layer_in_proj' in net.modules
    return True


def test_lora_dora_threading():
    """dora_scale flows into NetworkModuleLora."""
    net = _load_via(C.try_load_lora, sd_lora_with_dora_scale())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None
    return True


def test_lokr_bfl_img_attn_proj():
    """BFL LoKR on non-fused proj binds via NetworkModuleLokr (no chunk class)."""
    net = _load_via(C.try_load_lokr, sd_lokr_bfl_img_attn_proj())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn_to_out_0' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lokr.NetworkModuleLokr) and not isinstance(mod, network_lokr.NetworkModuleLokrChunk)
    return True


def test_lokr_bfl_img_attn_qkv_slice_chunked():
    """BFL LoKR on fused img_attn.qkv emits 3 SliceChunk modules with row ranges.

    Chroma's expand_chroma_fused_lokr uses NetworkModuleLokrSliceChunk for all
    fused targets (even equal-chunks), since the implementation is general
    and start/end ranges express both forms.
    """
    net = _load_via(C.try_load_lokr, sd_lokr_bfl_img_attn_qkv_equal_chunks())
    assert net is not None and len(net.modules) == 3, f'got {net.modules if net else None}'
    expected = {
        'lora_transformer_transformer_blocks_0_attn_to_q',
        'lora_transformer_transformer_blocks_0_attn_to_k',
        'lora_transformer_transformer_blocks_0_attn_to_v',
    }
    assert set(net.modules) == expected
    for nk, mod in net.modules.items():
        assert isinstance(mod, network_lokr.NetworkModuleLokrSliceChunk), f'{nk}: type={type(mod).__name__}'
        # Each chunk row range is HIDDEN rows wide
        assert mod.end_row - mod.start_row == HIDDEN, f'{nk}: range={mod.start_row}:{mod.end_row}'
    return True


def test_lokr_bfl_single_linear1_unequal_chunks():
    """BFL LoKR on fused linear1 emits 4 SliceChunks with UNEQUAL ranges.

    Critical chroma-specific path: HIDDEN/HIDDEN/HIDDEN/MLP_HIDDEN partition.
    """
    net = _load_via(C.try_load_lokr, sd_lokr_bfl_single_linear1_unequal())
    assert net is not None and len(net.modules) == 4, f'got {net.modules if net else None}'
    # Check the proj_mlp chunk has the longer row range (MLP_HIDDEN)
    proj_mlp_key = 'lora_transformer_single_transformer_blocks_0_proj_mlp'
    assert proj_mlp_key in net.modules
    proj_mlp = net.modules[proj_mlp_key]
    assert proj_mlp.end_row - proj_mlp.start_row == MLP_HIDDEN, \
        f'proj_mlp range={proj_mlp.start_row}:{proj_mlp.end_row}, expected width={MLP_HIDDEN}'
    # The three QKV chunks should each be HIDDEN rows wide
    for proj in ('attn_to_q', 'attn_to_k', 'attn_to_v'):
        nk = f'lora_transformer_single_transformer_blocks_0_{proj}'
        mod = net.modules[nk]
        assert mod.end_row - mod.start_row == HIDDEN, f'{nk}: range={mod.start_row}:{mod.end_row}'
    return True


def test_loha_bfl_img_attn_proj():
    """LoHA on non-fused target binds via NetworkModuleHada."""
    net = _load_via(C.try_load_loha, sd_loha_bfl_img_attn_proj())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_hada.NetworkModuleHada)
    return True


def test_loha_bfl_img_attn_qkv_skipped():
    """LoHA on fused img_attn.qkv is dropped (chroma has no LoHA chunk variant)."""
    net = _load_via(C.try_load_loha, sd_loha_bfl_img_attn_qkv_skipped())
    assert net is None or len(net.modules) == 0
    return True


def test_oft_bfl_img_attn_proj():
    """LyCORIS oft_diag form loads on non-fused target without NoneType errors."""
    net = _load_via(C.try_load_oft, sd_oft_bfl_img_attn_proj())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_oft.NetworkModuleOFT)
    return True


def test_oft_bfl_img_attn_qkv_skipped():
    """OFT on fused img_attn.qkv is dropped (no row-sliceable OFT structure)."""
    net = _load_via(C.try_load_oft, sd_oft_bfl_img_attn_qkv_skipped())
    assert net is None or len(net.modules) == 0
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_shape():
    net = _load_via(C.try_load_lora, sd_lora_bfl_img_attn_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA calc_updown')
    return True


def test_lokr_calc_updown_shape():
    net = _load_via(C.try_load_lokr, sd_lokr_bfl_img_attn_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoKR calc_updown')
    return True


def test_lokr_slicechunk_equal_calc_updown_shape():
    """LokrSliceChunk with equal-width range produces (HIDDEN, HIDDEN) output."""
    net = _load_via(C.try_load_lokr, sd_lokr_bfl_img_attn_qkv_equal_chunks())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LokrSliceChunk equal range')
    return True


def test_lokr_slicechunk_unequal_calc_updown_shape():
    """LokrSliceChunk on the proj_mlp chunk produces (MLP_HIDDEN, HIDDEN) output.

    This exercises the path that motivated NetworkModuleLokrSliceChunk's
    existence: unequal partition where torch.chunk would not work.
    """
    net = _load_via(C.try_load_lokr, sd_lokr_bfl_single_linear1_unequal())
    proj_mlp_key = 'lora_transformer_single_transformer_blocks_0_proj_mlp'
    mod = make_network_for_module(net.modules[proj_mlp_key])
    target = torch.randn(MLP_HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LokrSliceChunk unequal proj_mlp')
    return True


def test_loha_calc_updown_shape():
    net = _load_via(C.try_load_loha, sd_loha_bfl_img_attn_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoHA calc_updown')
    return True


def test_oft_calc_updown_shape():
    net = _load_via(C.try_load_oft, sd_oft_bfl_img_attn_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='OFT calc_updown')
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing primitives ===')
    for fn in [test_parse_key_all_prefixes, test_marker_disambiguation, test_static_rename_table]:
        run_test(CAT_PARSE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_bfl_img_attn_proj,
        test_lora_bfl_img_attn_qkv_chunked,
        test_lora_bfl_txt_attn_qkv_chunked,
        test_lora_bfl_img_mlp,
        test_lora_bfl_txt_mlp,
        test_lora_bfl_single_linear1_unequal_chunks,
        test_lora_bfl_single_linear2,
        test_lora_kohya_img_attn_proj,
        test_lora_kohya_img_attn_qkv_chunked,
        test_lora_peft_to_q,
        test_lora_distilled_guidance,
        test_lora_dora_threading,
        test_lokr_bfl_img_attn_proj,
        test_lokr_bfl_img_attn_qkv_slice_chunked,
        test_lokr_bfl_single_linear1_unequal_chunks,
        test_loha_bfl_img_attn_proj,
        test_loha_bfl_img_attn_qkv_skipped,
        test_oft_bfl_img_attn_proj,
        test_oft_bfl_img_attn_qkv_skipped,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [
        test_lora_calc_updown_shape,
        test_lokr_calc_updown_shape,
        test_lokr_slicechunk_equal_calc_updown_shape,
        test_lokr_slicechunk_unequal_calc_updown_shape,
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
