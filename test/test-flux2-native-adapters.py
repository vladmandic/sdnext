#!/usr/bin/env python
"""
Offline unit tests for Flux2/Klein native adapter loaders.

Covers the nine native families (LoRA, LoKR, LoHA, OFT, BOFT, IA3,
GLoRA, Norm, Full) plus DoRA threading via the universal
NetworkModule.finalize_updown hook, and ex_bias accumulation across
stacked Norm adapters.

The tests build a mock Flux2-shaped transformer, write synthetic
safetensors files for each adapter format, and exercise the full loader
path: state-dict -> group_by_suffixes -> resolve_targets ->
NetworkModule* -> calc_updown -> finalize_updown.

Loader correctness is verified against the documented LyCORIS / kohya
save formats. End-to-end inference verification against real adapter
files remains an open gap for the rarer families.

No running server required.

Usage:
    python test/test-flux2-native-adapters.py
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
# parse_args() registers the full main+compat option set; installer.add_args
# adds the installer-specific options on top.
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

from modules.logger import log   # pylint: disable=wrong-import-position
from modules import shared        # pylint: disable=wrong-import-position
from modules.lora import (         # pylint: disable=wrong-import-position
    network, network_lora, network_lokr, network_hada, network_oft, network_boft,
    network_ia3, network_glora, network_norm, network_full,
)
from modules.lora import lora_apply   # pylint: disable=wrong-import-position
from modules.lora import lora_common as l_common   # pylint: disable=wrong-import-position
from pipelines.flux import flux2_lora as F         # pylint: disable=wrong-import-position


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
# Mock Flux2-shaped transformer
# ============================================================

# Small but realistic shapes so tests are fast.
HIDDEN = 128
QKV_OUT = 128
MLP_OUT = 512
HEAD_DIM = 32
SINGLE_FUSED_OUT = 3 * QKV_OUT + MLP_OUT  # to_qkv_mlp_proj
SINGLE_OUT_IN = QKV_OUT + MLP_OUT          # to_out
N_DOUBLE = 2
N_SINGLE = 2


# pylint: disable=attribute-defined-outside-init
# `_Holder` is a generic container we attach diffusers-shaped children to at
# runtime to mirror Flux2's transformer module tree.
class _Holder(torch.nn.Module):
    """Empty container module — we attach Linear/RMSNorm children dynamically."""


def build_mock_transformer():
    """Build a torch.nn.Module mimicking Flux2's diffusers-side module tree.

    Mirrors the paths in F2_SINGLE_MAP / F2_DOUBLE_MAP / F2_QKV_MAP plus the
    RMSNorm targets inside attention so that try_load_norm has something to
    bind. Sizes are scaled-down but proportional so chunking math is realistic.
    """
    transformer = _Holder()
    transformer.transformer_blocks = torch.nn.ModuleList()
    for _ in range(N_DOUBLE):
        block = _Holder()
        block.attn = _Holder()
        block.attn.to_q = torch.nn.Linear(HIDDEN, QKV_OUT, bias=False)
        block.attn.to_k = torch.nn.Linear(HIDDEN, QKV_OUT, bias=False)
        block.attn.to_v = torch.nn.Linear(HIDDEN, QKV_OUT, bias=False)
        block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(QKV_OUT, HIDDEN, bias=False)])
        block.attn.add_q_proj = torch.nn.Linear(HIDDEN, QKV_OUT, bias=False)
        block.attn.add_k_proj = torch.nn.Linear(HIDDEN, QKV_OUT, bias=False)
        block.attn.add_v_proj = torch.nn.Linear(HIDDEN, QKV_OUT, bias=False)
        block.attn.to_add_out = torch.nn.Linear(QKV_OUT, HIDDEN, bias=False)
        block.attn.norm_q = torch.nn.RMSNorm(HEAD_DIM)
        block.attn.norm_k = torch.nn.RMSNorm(HEAD_DIM)
        block.attn.norm_added_q = torch.nn.RMSNorm(HEAD_DIM)
        block.attn.norm_added_k = torch.nn.RMSNorm(HEAD_DIM)
        block.ff = _Holder()
        block.ff.linear_in = torch.nn.Linear(HIDDEN, MLP_OUT, bias=False)
        block.ff.linear_out = torch.nn.Linear(MLP_OUT, HIDDEN, bias=False)
        block.ff_context = _Holder()
        block.ff_context.linear_in = torch.nn.Linear(HIDDEN, MLP_OUT, bias=False)
        block.ff_context.linear_out = torch.nn.Linear(MLP_OUT, HIDDEN, bias=False)
        transformer.transformer_blocks.append(block)
    transformer.single_transformer_blocks = torch.nn.ModuleList()
    for _ in range(N_SINGLE):
        sblock = _Holder()
        sblock.attn = _Holder()
        sblock.attn.to_qkv_mlp_proj = torch.nn.Linear(HIDDEN, SINGLE_FUSED_OUT, bias=False)
        sblock.attn.to_out = torch.nn.Linear(SINGLE_OUT_IN, HIDDEN, bias=False)
        transformer.single_transformer_blocks.append(sblock)
    return transformer


class _MockFlux2Pipeline:
    """Class name carries 'Flux2' so shared.sd_model_type returns 'f2'
    via modeldata.get_model_type's name-based dispatch."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockFlux2SdModel:
    """Outer wrapper holding the pipe + the network_layer_mapping that
    lora_convert.assign_network_names_to_compvis_modules writes onto."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'Flux2Pipeline'  # belt-and-suspenders


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing a Flux2-shaped transformer.

    Each test calls this fresh so module.network_layer_name stamps from prior
    tests don't leak (notably the loader-local stamping in try_load_norm).

    Bypasses the ModelData lock by writing directly to model_data.sd_model;
    shared.sd_model = ... goes through set_sd_model which is a no-op when
    model_data.locked is True (the default outside webui startup).
    """
    transformer = build_mock_transformer()
    pipe = _MockFlux2Pipeline(transformer)
    sd_model = _MockFlux2SdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# State-dict synthesizers (one per family)
# ============================================================

# Realistic ranks for the synthesized factors.
RANK_LORA = 8
RANK_LOKR = 4
LOKR_W1_DIM = 8
LOKR_W2_DIM = HIDDEN // LOKR_W1_DIM


def sd_lora_kohya_qkv():
    """Kohya-format LoRA targeting fused img_attn.qkv (3-way chunked at load)."""
    return {
        # Down weight is shared across Q/K/V; up weight is fused (3*QKV_OUT, RANK).
        'lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight':   torch.randn(3 * QKV_OUT, RANK_LORA),
        'lora_unet_double_blocks_0_img_attn_qkv.alpha':            torch.tensor(float(RANK_LORA)),
    }


def sd_lora_bfl_proj():
    """BFL/AI-toolkit-format LoRA on a non-fused double_blocks proj target."""
    return {
        'diffusion_model.double_blocks.1.img_attn.proj.lora_A.weight': torch.randn(RANK_LORA, QKV_OUT),
        'diffusion_model.double_blocks.1.img_attn.proj.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.1.img_attn.proj.alpha':         torch.tensor(float(RANK_LORA)),
    }


def sd_lora_peft_to_q():
    """Diffusers PEFT-format LoRA targeting a single split-QKV diffusers path."""
    return {
        'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.randn(QKV_OUT, RANK_LORA),
    }


def sd_lora_bare_bfl_mlp():
    """Bare BFL keys (no prefix) on an MLP target — exercises the bare fallback."""
    return {
        'double_blocks.0.img_mlp.0.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'double_blocks.0.img_mlp.0.lora_B.weight': torch.randn(MLP_OUT, RANK_LORA),
    }


def sd_lora_peft_saved_fal_style():
    """PEFT-saved LoRA: base_model.model. wrapper around BFL paths.

    Mirrors the actual key layout of fal/flux-2-klein-4B-outpaint-lora as
    inspected from its safetensors header on HF — first 12 keys all of the
    form ``base_model.model.double_blocks.0.img_attn.proj.lora_A.weight``.
    """
    return {
        'base_model.model.double_blocks.1.img_attn.proj.lora_A.weight': torch.randn(RANK_LORA, QKV_OUT),
        'base_model.model.double_blocks.1.img_attn.proj.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'base_model.model.double_blocks.1.img_attn.proj.alpha':         torch.tensor(float(RANK_LORA)),
    }


def sd_lora_peft_saved_dreambooth_style():
    """PEFT-saved LoRA where the wrapped path is diffusers-form (HF DreamBooth).

    HF DreamBooth scripts target diffusers modules by name (e.g.
    ``transformer.transformer_blocks.0.attn.to_q``); after peft.save_pretrained
    wraps the in-memory keys, the saved form is
    ``base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_A.weight``.
    """
    return {
        'base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.randn(QKV_OUT, RANK_LORA),
    }


def sd_lora_peft_saved_diffusers_style():
    """PEFT-saved LoRA with bare-diffusers paths and a ``.default.`` adapter-name infix.

    Mirrors the actual key layout of community Flux2 LoRAs exported via
    ``Flux2Transformer2DModel.save_lora_adapter()`` after attaching an unnamed
    PEFT adapter (e.g. ``lenovo_flux_klein9b.safetensors``):

    - No ``transformer.`` / ``diffusion_model.`` / ``base_model.model.`` wrapper
    - Keys start with bare ``transformer_blocks.`` or ``single_transformer_blocks.``
    - ``.lora_A.default.weight`` / ``.lora_B.default.weight`` (PEFT adapter slot)
    """
    return {
        # Single-block fused module — exercises the bare-diffusers prefix branch.
        'single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default.weight': torch.randn(RANK_LORA, HIDDEN),
        'single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.default.weight': torch.randn(SINGLE_FUSED_OUT, RANK_LORA),
        # Double-block split target — exercises both the bare-diffusers branch and the .default. strip.
        'transformer_blocks.0.attn.to_q.lora_A.default.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer_blocks.0.attn.to_q.lora_B.default.weight': torch.randn(QKV_OUT, RANK_LORA),
    }


def sd_lora_with_dora_scale():
    """LoRA with a dora_scale companion vector — exercises DoRA threading."""
    return {
        'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.randn(QKV_OUT, RANK_LORA),
        'transformer.transformer_blocks.0.attn.to_q.dora_scale':    torch.randn(QKV_OUT),
    }


def sd_lokr_bfl_proj():
    """BFL-format LoKR on a non-fused proj target."""
    return {
        # w1 = (LOKR_W1_DIM, LOKR_W1_DIM), w2 = (LOKR_W2_DIM, LOKR_W2_DIM); kron is (HIDDEN, HIDDEN).
        # Target shape is (HIDDEN, QKV_OUT) here; we need shape compat for calc_updown.
        # Use w1=(QKV_OUT/LOKR_W2_DIM, HIDDEN/LOKR_W2_DIM) outer factor; LyCORIS picks dims by factorization.
        'diffusion_model.double_blocks.1.img_attn.proj.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.double_blocks.1.img_attn.proj.lokr_w2': torch.randn(QKV_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.double_blocks.1.img_attn.proj.alpha':   torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_lokr_kohya_qkv():
    """Kohya LoKR targeting fused img_attn.qkv — exercises NetworkModuleLokrChunk."""
    # Target after kron is (3*QKV_OUT, HIDDEN); chunk along dim=0 gives Q/K/V slices.
    fused_out = 3 * QKV_OUT
    # Pick w1=(8, ...) so kron of w1 with w2 spans fused_out x HIDDEN.
    w1_dim_out = 8
    w2_dim_out = fused_out // w1_dim_out
    w1_dim_in = 4
    w2_dim_in = HIDDEN // w1_dim_in
    return {
        'lora_unet_double_blocks_0_img_attn_qkv.lokr_w1': torch.randn(w1_dim_out, w1_dim_in),
        'lora_unet_double_blocks_0_img_attn_qkv.lokr_w2': torch.randn(w2_dim_out, w2_dim_in),
        'lora_unet_double_blocks_0_img_attn_qkv.alpha':   torch.tensor(float(w1_dim_in)),
    }


def sd_lokr_simpletuner_lycoris_style():
    """LyCORIS-standalone-format LoKR (SimpleTuner save).

    Mirrors the actual key layout of markury/flux2k9b-simpletuner-lokr-loona
    as inspected from its safetensors header on HF — keys of the form
    ``lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1`` (288 such keys
    in the real file). The path under ``lycoris_`` is a diffusers path with
    dots rendered as underscores; resolve_targets returns it verbatim and
    the caller's ``.replace('.', '_')`` is a no-op.

    Two targets here exercise both an img-side projection (to_q) and a
    txt-side projection (add_k_proj) which uses underscores in the module
    name itself — the latter is the case where naïve underscore-to-dot
    expansion would corrupt the path.
    """
    return {
        'lycoris_transformer_blocks_0_attn_to_q.lokr_w1':       torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'lycoris_transformer_blocks_0_attn_to_q.lokr_w2':       torch.randn(QKV_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'lycoris_transformer_blocks_0_attn_to_q.alpha':         torch.tensor(float(LOKR_W1_DIM)),
        'lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w2': torch.randn(QKV_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'lycoris_transformer_blocks_0_attn_add_k_proj.alpha':   torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_loha_bfl_proj():
    """BFL-format LoHA on a non-fused proj target."""
    return {
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w1_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w1_b': torch.randn(RANK_LORA, QKV_OUT),
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w2_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.double_blocks.1.img_attn.proj.hada_w2_b': torch.randn(RANK_LORA, QKV_OUT),
        'diffusion_model.double_blocks.1.img_attn.proj.alpha':     torch.tensor(float(RANK_LORA)),
    }


def sd_loha_kohya_qkv():
    """Kohya LoHA targeting fused img_attn.qkv — exercises NetworkModuleHadaChunk."""
    fused_out = 3 * QKV_OUT
    return {
        'lora_unet_double_blocks_0_img_attn_qkv.hada_w1_a': torch.randn(fused_out, RANK_LORA),
        'lora_unet_double_blocks_0_img_attn_qkv.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_double_blocks_0_img_attn_qkv.hada_w2_a': torch.randn(fused_out, RANK_LORA),
        'lora_unet_double_blocks_0_img_attn_qkv.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_double_blocks_0_img_attn_qkv.alpha':     torch.tensor(float(RANK_LORA)),
    }


def sd_oft_kohya_proj():
    """Kohya-format OFT (oft_blocks) on a non-fused proj target."""
    # OFT block-diagonal: out_dim = HIDDEN, num_blocks=8, block_size=16.
    return {
        'lora_unet_double_blocks_1_img_attn_proj.oft_blocks': torch.zeros(8, 16, 16),
        'lora_unet_double_blocks_1_img_attn_proj.alpha':      torch.tensor(1e-3),
    }


def sd_oft_lycoris_proj():
    """LyCORIS-format OFT (oft_diag) — exercises the network_oft.py:58 NPE-fix path.

    LyCORIS reads ``self.dim = oft_diag.shape[1]`` (the block_size) and computes
    ``block_size, num_blocks = factorization(out_dim, dim)``. ``factorization``
    always returns ``(smaller, larger)``. For out_dim=HIDDEN=128 with dim=8, it
    returns (8, 16) ⇒ block_size=8, num_blocks=16. The tensor must therefore
    have shape ``(num_blocks, block_size, block_size) = (16, 8, 8)``.
    """
    return {
        'lora_unet_double_blocks_1_img_attn_proj.oft_diag': torch.zeros(16, 8, 8),
    }


def sd_oft_kohya_qkv_skipped():
    """OFT on fused img_attn.qkv — must be skipped with a warning (no chunk class)."""
    return {
        'lora_unet_double_blocks_0_img_attn_qkv.oft_blocks': torch.zeros(8, 48, 48),
        'lora_unet_double_blocks_0_img_attn_qkv.alpha':      torch.tensor(1e-3),
    }


def sd_boft_butterfly():
    """BOFT (butterfly-OFT) — re-uses ``oft_blocks`` key with 4-D shape.

    Mirrors the LyCORIS upstream save layout (boft.py weight_list:
    ``oft_blocks, rescale, alpha``; tensor shape per __init__:
    ``(boft_m, block_num, block_size, block_size)``).

    For HIDDEN=128 with block_size=16, block_num=8:
    - boft_m derives from sum-of-set-bits of (block_num-1) + 1 = bits(7)+1 = 4
    - cascade indices: k = 2^i * r_b for i in 0..3, with r_b = 8
    - all four stages divide cleanly into 128-row tensors
    """
    return {
        'lora_unet_double_blocks_1_img_attn_proj.oft_blocks':
            torch.randn(4, 8, 16, 16) * 0.01,
        'lora_unet_double_blocks_1_img_attn_proj.alpha':
            torch.tensor(1e-3),
    }


def sd_ia3_proj():
    """IA3 on a non-fused target. on_input=False ⇒ output-axis scaling."""
    return {
        'lora_unet_double_blocks_1_img_attn_proj.weight':   torch.randn(HIDDEN),
        'lora_unet_double_blocks_1_img_attn_proj.on_input': torch.tensor(0),
    }


def sd_ia3_qkv_skipped():
    """IA3 on fused QKV — must be skipped with a warning."""
    return {
        'lora_unet_double_blocks_0_img_attn_qkv.weight':   torch.randn(3 * QKV_OUT),
        'lora_unet_double_blocks_0_img_attn_qkv.on_input': torch.tensor(0),
    }


def sd_glora_proj():
    """GLoRA on a non-fused proj target.

    Target host module is ``attn.to_out.0`` with weight shape (out, in)
    = (HIDDEN, QKV_OUT). NetworkModuleGLora computes
    ``updown = (w2b @ w1b) + ((target @ w2a) @ w1a)``, which needs:

    - w2b: (out, rank), w1b: (rank, in)   ⇒ first term shape (out, in)
    - w2a: (in, rank),  w1a: (rank, in)   ⇒ second term ((out,in) @ (in,rank)) @ (rank,in) = (out, in)
    """
    out_dim = HIDDEN
    in_dim = QKV_OUT
    return {
        'lora_unet_double_blocks_1_img_attn_proj.a1.weight': torch.randn(RANK_LORA, in_dim),
        'lora_unet_double_blocks_1_img_attn_proj.a2.weight': torch.randn(in_dim, RANK_LORA),
        'lora_unet_double_blocks_1_img_attn_proj.b1.weight': torch.randn(RANK_LORA, in_dim),
        'lora_unet_double_blocks_1_img_attn_proj.b2.weight': torch.randn(out_dim, RANK_LORA),
        'lora_unet_double_blocks_1_img_attn_proj.alpha':     torch.tensor(float(RANK_LORA)),
    }


def sd_norm_peft_attn_norm_q():
    """Norm targeting a real Flux2 RMSNorm (attn.norm_q)."""
    return {
        'transformer.transformer_blocks.0.attn.norm_q.w_norm': torch.randn(HEAD_DIM) * 0.01,
        'transformer.transformer_blocks.0.attn.norm_q.b_norm': torch.randn(HEAD_DIM) * 0.01,
    }


def sd_full_proj():
    """Full-rank delta on a proj target. diff matches host weight shape (out, in)."""
    return {
        'lora_unet_double_blocks_1_img_attn_proj.diff':   torch.randn(HIDDEN, QKV_OUT) * 0.01,
        'lora_unet_double_blocks_1_img_attn_proj.diff_b': torch.randn(HIDDEN) * 0.01,
    }


# ============================================================
# Test plumbing helpers
# ============================================================

class TempLora:
    """Context-manager that materializes a synthetic state dict as a real
    safetensors file and yields a NetworkOnDisk pointing at it.

    Required because flux2_lora's loaders read via sd_models.read_state_dict,
    which only accepts a filename. Cleans up the temp file on exit.
    """

    def __init__(self, state_dict, name='test'):
        self.state_dict = state_dict
        self.name = name
        self.path = None

    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        # safetensors requires contiguous tensors.
        sd = {k: v.contiguous() for k, v in self.state_dict.items()}
        safetensors.torch.save_file(sd, self.path)
        return _MockNetworkOnDisk(self.name, self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class _MockNetworkOnDisk:
    """Minimal NetworkOnDisk shim — loaders only touch .filename, .name, .shorthash, .sd_version."""

    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.shorthash = ''
        self.sd_version = 'f2'
        self.metadata = {}

    def read_hash(self):
        pass


def assert_shape(t: torch.Tensor, expected_shape, label=''):
    actual = tuple(t.shape)
    expected = tuple(expected_shape)
    assert actual == expected, f'{label} shape {actual} != expected {expected}'


def assert_finite(t: torch.Tensor, label=''):
    assert torch.isfinite(t).all(), f'{label} contains non-finite values'


def make_network_for_module(net_module: network.NetworkModule, te_mul: float = 1.0, unet_mul: float = 1.0):
    """Wire a NetworkModule into a Network with given multipliers so calc_updown
    has a meaningful multiplier()/calc_scale() context."""
    net_module.network.te_multiplier = te_mul
    net_module.network.unet_multiplier = unet_mul
    return net_module


# ============================================================
# Tests — parsing primitives
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_all_prefixes():
    cases = [
        ('lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight',
         F.LORA_SUFFIXES,
         ('lora_unet_', 'double_blocks_0_img_attn_qkv', 'lora_down.weight')),
        ('diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight',
         F.LORA_SUFFIXES,
         ('diffusion_model.', 'double_blocks.0.img_attn.proj', 'lora_down.weight')),
        ('transformer.transformer_blocks.0.attn.to_q.lora_B.weight',
         F.LORA_SUFFIXES,
         ('transformer.', 'transformer_blocks.0.attn.to_q', 'lora_up.weight')),
        ('double_blocks.10.img_mlp.0.lora_A.weight',
         F.LORA_SUFFIXES,
         (None, 'double_blocks.10.img_mlp.0', 'lora_down.weight')),
        ('random.unrelated.key', F.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = F.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_resolve_targets_qkv_chunking():
    from modules.lora.native_adapter import ChunkSpec
    # Kohya double_blocks fused QKV → three chunks targeting Q/K/V.
    targets = F.resolve_targets('lora_unet_', 'double_blocks_0_img_attn_qkv')
    assert targets == [
        ('transformer_blocks.0.attn.to_q', ChunkSpec(idx=0, total=3)),
        ('transformer_blocks.0.attn.to_k', ChunkSpec(idx=1, total=3)),
        ('transformer_blocks.0.attn.to_v', ChunkSpec(idx=2, total=3)),
    ], f'kohya img_attn.qkv → {targets}'

    targets = F.resolve_targets('lora_unet_', 'double_blocks_5_txt_attn_qkv')
    assert targets == [
        ('transformer_blocks.5.attn.add_q_proj', ChunkSpec(idx=0, total=3)),
        ('transformer_blocks.5.attn.add_k_proj', ChunkSpec(idx=1, total=3)),
        ('transformer_blocks.5.attn.add_v_proj', ChunkSpec(idx=2, total=3)),
    ], f'kohya txt_attn.qkv → {targets}'

    targets = F.resolve_targets('diffusion_model.', 'single_blocks.7.linear1')
    assert targets == [('single_transformer_blocks.7.attn.to_qkv_mlp_proj', None)]

    targets = F.resolve_targets('transformer.', 'transformer_blocks.0.attn.to_q')
    assert targets == [('transformer_blocks.0.attn.to_q', None)]

    targets = F.resolve_targets('weird_prefix.', 'whatever')
    assert targets == []
    return True


def test_parse_key_peft_wrapper_unwrap():
    """base_model.model. wrapper is stripped before format detection.

    Verifies the unwrap handles all four content cases:
    - BFL keys under the wrapper (fal style)
    - Diffusers paths under transformer. under the wrapper (HF DreamBooth style)
    - Bare BFL under the wrapper
    - Wrapper not present (passthrough)
    """
    cases = [
        ('base_model.model.double_blocks.1.img_attn.proj.lora_A.weight',
         F.LORA_SUFFIXES,
         (None, 'double_blocks.1.img_attn.proj', 'lora_down.weight')),
        ('base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_A.weight',
         F.LORA_SUFFIXES,
         ('transformer.', 'transformer_blocks.0.attn.to_q', 'lora_down.weight')),
        ('base_model.model.lora_unet_double_blocks_0_img_attn_proj.lora_down.weight',
         F.LORA_SUFFIXES,
         ('lora_unet_', 'double_blocks_0_img_attn_proj', 'lora_down.weight')),
        ('diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight',  # no wrapper
         F.LORA_SUFFIXES,
         ('diffusion_model.', 'double_blocks.0.img_attn.proj', 'lora_down.weight')),
    ]
    for key, suffixes, expected in cases:
        got = F.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_parse_key_lycoris_prefix():
    """lycoris_ prefix yields an underscored diffusers path; no _ -> . conversion."""
    cases = [
        ('lycoris_transformer_blocks_0_attn_to_q.lokr_w1',
         F.LOKR_SUFFIXES,
         ('lycoris_', 'transformer_blocks_0_attn_to_q', 'lokr_w1')),
        ('lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w2',
         F.LOKR_SUFFIXES,
         ('lycoris_', 'transformer_blocks_0_attn_add_k_proj', 'lokr_w2')),
        ('lycoris_single_transformer_blocks_5_attn_to_qkv_mlp_proj.lokr_w1',
         F.LOKR_SUFFIXES,
         ('lycoris_', 'single_transformer_blocks_5_attn_to_qkv_mlp_proj', 'lokr_w1')),
    ]
    for key, suffixes, expected in cases:
        got = F.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'

    # resolve_targets: the underscored path is returned verbatim (no chunk).
    targets = F.resolve_targets('lycoris_', 'transformer_blocks_0_attn_add_k_proj')
    assert targets == [('transformer_blocks_0_attn_add_k_proj', None)], f'targets={targets}'
    return True


def test_parse_key_bare_diffusers_and_peft_default():
    """Bare-diffusers paths + ``.lora_[AB].<adapter_name>.weight`` infix.

    Covers ``Flux2Transformer2DModel.save_lora_adapter()`` output where the file
    has no wrapping prefix and PEFT keeps the adapter slot name in the suffix.
    """
    bd = F.BARE_DIFFUSERS_PREFIX_USED
    cases = [
        ('single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default.weight',
         F.LORA_SUFFIXES,
         (bd, 'single_transformer_blocks.0.attn.to_qkv_mlp_proj', 'lora_down.weight')),
        ('transformer_blocks.0.attn.to_q.lora_B.default.weight',
         F.LORA_SUFFIXES,
         (bd, 'transformer_blocks.0.attn.to_q', 'lora_up.weight')),
        # Non-default adapter name also strips.
        ('transformer_blocks.7.attn.to_v.lora_A.style.weight',
         F.LORA_SUFFIXES,
         (bd, 'transformer_blocks.7.attn.to_v', 'lora_down.weight')),
        # No adapter-name infix passes through unchanged.
        ('transformer_blocks.0.attn.to_q.lora_A.weight',
         F.LORA_SUFFIXES,
         (bd, 'transformer_blocks.0.attn.to_q', 'lora_down.weight')),
        # Wrapped path with adapter-name infix: unwrap + strip both apply.
        ('base_model.model.transformer.transformer_blocks.0.attn.to_q.lora_A.default.weight',
         F.LORA_SUFFIXES,
         ('transformer.', 'transformer_blocks.0.attn.to_q', 'lora_down.weight')),
    ]
    for key, suffixes, expected in cases:
        got = F.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'

    # resolve_targets passes the bare-diffusers path through verbatim.
    targets = F.resolve_targets(bd, 'single_transformer_blocks.5.attn.to_out')
    assert targets == [('single_transformer_blocks.5.attn.to_out', None)], f'targets={targets}'
    return True


def test_marker_disambiguation():
    # Each family's marker must reject other families' files.
    pure_lora = {'lora_unet_x.lora_down.weight': torch.zeros(1, 1),
                 'lora_unet_x.lora_up.weight':   torch.zeros(1, 1)}
    assert F.has_marker(pure_lora, F.LORA_MARKERS)
    assert not F.has_marker(pure_lora, F.LOKR_MARKERS)
    assert not F.has_marker(pure_lora, F.LOHA_MARKERS)
    assert not F.has_marker(pure_lora, F.OFT_MARKERS)
    assert not F.has_marker(pure_lora, F.IA3_MARKERS)
    assert not F.has_marker(pure_lora, F.GLORA_MARKERS)
    assert not F.has_marker(pure_lora, F.NORM_MARKERS)
    assert not F.has_marker(pure_lora, F.FULL_MARKERS)

    ia3 = {'lora_unet_x.weight': torch.zeros(1), 'lora_unet_x.on_input': torch.tensor(1)}
    assert F.has_marker(ia3, F.IA3_MARKERS)
    # IA3 shares ".weight" but only on_input is the disambiguator.
    return True


# ============================================================
# Tests — loaders (one per family)
# ============================================================

CAT_LOADER = category('loader')


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def test_lora_kohya_fused_qkv_chunked():
    net = _load_via(F.try_load_lora, sd_lora_kohya_qkv())
    assert net is not None and len(net.modules) == 3, f'expected 3 chunked modules, got {net.modules if net else None}'
    # Network keys correspond to the three split projections.
    expected_keys = {
        'lora_transformer_transformer_blocks_0_attn_to_q',
        'lora_transformer_transformer_blocks_0_attn_to_k',
        'lora_transformer_transformer_blocks_0_attn_to_v',
    }
    assert set(net.modules) == expected_keys, f'got {set(net.modules)}'
    # Every module is a NetworkModuleLora; up-weight has been chunked along dim 0.
    for nk, mod in net.modules.items():
        assert isinstance(mod, network_lora.NetworkModuleLora), f'{nk}: type={type(mod).__name__}'
        # The chunked up tensor has shape (QKV_OUT, RANK), not (3*QKV_OUT, RANK).
        up_shape = tuple(mod.up_model.weight.shape)
        assert up_shape == (QKV_OUT, RANK_LORA), f'{nk}: up shape {up_shape}'
    return True


def test_lora_bfl_non_fused():
    net = _load_via(F.try_load_lora, sd_lora_bfl_proj())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    nk, mod = next(iter(net.modules.items()))
    assert nk == 'lora_transformer_transformer_blocks_1_attn_to_out_0', f'{nk}'
    assert isinstance(mod, network_lora.NetworkModuleLora)
    return True


def test_lora_peft_format():
    net = _load_via(F.try_load_lora, sd_lora_peft_to_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn_to_q' in net.modules
    return True


def test_lora_bare_bfl_format():
    net = _load_via(F.try_load_lora, sd_lora_bare_bfl_mlp())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_ff_linear_in' in net.modules
    return True


def test_lora_peft_saved_fal_style():
    """Real-world PEFT-saved LoRA with BFL paths under base_model.model. wrapper."""
    net = _load_via(F.try_load_lora, sd_lora_peft_saved_fal_style())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    # base_model.model.double_blocks.1.img_attn.proj → transformer_blocks.1.attn.to_out.0
    assert 'lora_transformer_transformer_blocks_1_attn_to_out_0' in net.modules
    return True


def test_lora_peft_saved_dreambooth_style():
    """PEFT-saved LoRA with diffusers paths under base_model.model. wrapper.

    HF DreamBooth-style: the path under base_model.model. starts with
    transformer., which the unwrap+reparse routes through the standard PEFT
    branch. Verifies the unwrap handles nested format prefixes correctly.
    """
    net = _load_via(F.try_load_lora, sd_lora_peft_saved_dreambooth_style())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn_to_q' in net.modules
    return True


def test_lora_peft_saved_diffusers_style():
    """Bare-diffusers paths + PEFT ``.default.`` adapter-name infix.

    Exercises the bare-diffusers prefix branch (``single_transformer_blocks.``,
    ``transformer_blocks.``) combined with the PEFT-saved ``.lora_A.default.weight``
    suffix shape produced by ``Flux2Transformer2DModel.save_lora_adapter()``.
    """
    net = _load_via(F.try_load_lora, sd_lora_peft_saved_diffusers_style())
    assert net is not None and len(net.modules) == 2, f'expected 2 modules, got {net.modules if net else None}'
    assert 'lora_transformer_single_transformer_blocks_0_attn_to_qkv_mlp_proj' in net.modules
    assert 'lora_transformer_transformer_blocks_0_attn_to_q' in net.modules
    return True


def test_lora_dora_threading():
    net = _load_via(F.try_load_lora, sd_lora_with_dora_scale())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None, 'dora_scale not threaded into NetworkModule'
    # Apply path runs through finalize_updown which calls apply_weight_decompose.
    target = torch.randn(QKV_OUT, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, (QKV_OUT, HIDDEN), 'DoRA updown')
    assert_finite(updown, 'DoRA updown')
    return True


def test_lokr_bfl_non_fused():
    net = _load_via(F.try_load_lokr, sd_lokr_bfl_proj())
    assert net is not None and len(net.modules) == 1
    nk, mod = next(iter(net.modules.items()))
    assert isinstance(mod, network_lokr.NetworkModuleLokr) and not isinstance(mod, network_lokr.NetworkModuleLokrChunk), \
        f'{nk}: type={type(mod).__name__}'
    return True


def test_lokr_kohya_fused_qkv_chunked():
    net = _load_via(F.try_load_lokr, sd_lokr_kohya_qkv())
    assert net is not None and len(net.modules) == 3
    for nk, mod in net.modules.items():
        assert isinstance(mod, network_lokr.NetworkModuleLokrChunk), f'{nk}: type={type(mod).__name__}'
        assert 0 <= mod.chunk_index < 3 and mod.num_chunks == 3
    return True


def test_lokr_simpletuner_lycoris_format():
    """Real-world LyCORIS-standalone LoKR with lycoris_ prefix on diffusers paths.

    Critical case: one of the targets is ``add_k_proj`` whose Linear name has
    embedded underscores. Confirms the loader does NOT do a naive _ -> .
    expansion on the base path; the underscored form must round-trip through
    the network_layer_mapping lookup intact.
    """
    net = _load_via(F.try_load_lokr, sd_lokr_simpletuner_lycoris_style())
    assert net is not None and len(net.modules) == 2, f'got {net.modules if net else None}'
    expected = {
        'lora_transformer_transformer_blocks_0_attn_to_q',
        'lora_transformer_transformer_blocks_0_attn_add_k_proj',
    }
    assert set(net.modules) == expected, f'got {set(net.modules)}'
    for mod in net.modules.values():
        assert isinstance(mod, network_lokr.NetworkModuleLokr)
    return True


def test_loha_bfl_non_fused():
    net = _load_via(F.try_load_loha, sd_loha_bfl_proj())
    assert net is not None and len(net.modules) == 1
    nk, mod = next(iter(net.modules.items()))
    assert isinstance(mod, network_hada.NetworkModuleHada) and not isinstance(mod, network_hada.NetworkModuleHadaChunk), \
        f'{nk}: type={type(mod).__name__}'
    return True


def test_loha_kohya_fused_qkv_chunked():
    net = _load_via(F.try_load_loha, sd_loha_kohya_qkv())
    assert net is not None and len(net.modules) == 3
    for nk, mod in net.modules.items():
        assert isinstance(mod, network_hada.NetworkModuleHadaChunk), f'{nk}: type={type(mod).__name__}'
        assert 0 <= mod.chunk_index < 3 and mod.num_chunks == 3
    return True


def test_oft_kohya_non_fused():
    net = _load_via(F.try_load_oft, sd_oft_kohya_proj())
    assert net is not None and len(net.modules) == 1
    _, mod = next(iter(net.modules.items()))
    assert isinstance(mod, network_oft.NetworkModuleOFT)
    assert mod.is_kohya is True and mod.constraint is not None
    return True


def test_oft_lycoris_no_npe():
    """LyCORIS OFT (oft_diag) loads and calc_updown completes without
    accessing self.constraint, which is None in the LyCORIS path.
    """
    net = _load_via(F.try_load_oft, sd_oft_lycoris_proj())
    assert net is not None and len(net.modules) == 1
    _, mod = next(iter(net.modules.items()))
    assert isinstance(mod, network_oft.NetworkModuleOFT)
    assert mod.is_kohya is False and mod.constraint is None
    target = mod.sd_module.weight  # real (HIDDEN, QKV_OUT) tensor from the mock
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape), 'LyCORIS OFT updown')
    assert_finite(updown, 'LyCORIS OFT updown')
    return True


def test_oft_fused_qkv_skipped():
    """OFT on fused QKV must be dropped (with warning) — no chunk class exists."""
    net = _load_via(F.try_load_oft, sd_oft_kohya_qkv_skipped())
    # Loader returns None when no modules bound (skipped + nothing else).
    assert net is None or len(net.modules) == 0
    return True


def test_boft_loads_and_applies():
    """BOFT files (4-D oft_blocks) bind to NetworkModuleBOFT via the
    try_load_oft dispatch. Verifies shape attrs derive correctly from
    the 4-D tensor and the butterfly-cascade calc_updown returns a
    finite delta of host weight shape.
    """
    net = _load_via(F.try_load_oft, sd_boft_butterfly())
    assert net is not None and len(net.modules) == 1, \
        f'BOFT did not load (got {net.modules if net else None})'
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_boft.NetworkModuleBOFT), \
        f'expected NetworkModuleBOFT, got {type(mod).__name__}'
    assert isinstance(mod, network_boft.NetworkModuleBOFT) and not isinstance(mod, network_oft.NetworkModuleOFT), \
        'BOFT must not be a subclass of NetworkModuleOFT (separate algorithm)'
    # BOFT-specific shape attributes resolved correctly
    assert mod.boft_m == 4 and mod.block_num == 8 and mod.block_size == 16, \
        f'shape attrs: boft_m={mod.boft_m} block_num={mod.block_num} block_size={mod.block_size}'
    # End-to-end calc_updown via the butterfly cascade
    target = mod.sd_module.weight  # (HIDDEN=128, QKV_OUT=128) for to_out.0
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape), 'BOFT updown')
    assert_finite(updown, 'BOFT updown')
    return True


def test_ia3_basic_load():
    net = _load_via(F.try_load_ia3, sd_ia3_proj())
    assert net is not None and len(net.modules) == 1
    _, mod = next(iter(net.modules.items()))
    assert isinstance(mod, network_ia3.NetworkModuleIa3)
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape), 'IA3 updown')
    return True


def test_ia3_fused_qkv_skipped():
    net = _load_via(F.try_load_ia3, sd_ia3_qkv_skipped())
    assert net is None or len(net.modules) == 0
    return True


def test_glora_basic_load_and_alpha():
    """GLoRA loads, self.dim is populated from w1b.shape[0], and
    calc_scale returns alpha/dim rather than the 1.0 fallback.
    """
    state_dict = sd_glora_proj()
    net = _load_via(F.try_load_glora, state_dict)
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_glora.NetworkModuleGLora)
    assert mod.dim is not None, 'GLoRA dim not set (regression)'
    expected_dim = state_dict['lora_unet_double_blocks_1_img_attn_proj.b1.weight'].shape[0]
    assert mod.dim == expected_dim, f'GLoRA dim {mod.dim} != {expected_dim}'
    # alpha=RANK_LORA (8), dim=expected_dim (8) → calc_scale = 1.0; instructive but matches kohya convention.
    expected_alpha = float(RANK_LORA)
    assert mod.alpha == expected_alpha, f'alpha={mod.alpha}'
    expected_scale = expected_alpha / expected_dim
    actual_scale = mod.calc_scale()
    assert abs(actual_scale - expected_scale) < 1e-6, f'calc_scale={actual_scale} expected {expected_scale} (alpha/dim)'
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape), 'GLoRA updown')
    assert_finite(updown, 'GLoRA updown')
    return True


def test_norm_loader_local_stamping():
    """Regression for the loader-local stamping behavior in try_load_norm.

    lora_convert.assign_network_names_to_compvis_modules deliberately skips
    setting module.network_layer_name on transformer norm modules. The Norm
    loader must stamp them itself for any module it actually binds, so the
    activation loop (which checks network_layer_name is not None) can apply
    the delta. Verify both: pre-loader the norm module has no
    network_layer_name; post-loader the bound target does.
    """
    sd_model = install_mock_pipe()
    norm_module = sd_model.pipe.transformer.transformer_blocks[0].attn.norm_q
    assert not getattr(norm_module, 'network_layer_name', None), \
        'mock norm should not have network_layer_name pre-load'
    # First call assign_network_names manually to mirror what resolve_mapping does.
    from modules.lora import lora_convert
    lora_convert.assign_network_names_to_compvis_modules(sd_model.pipe)
    # The carve-out skipped stamping for norm modules.
    assert not getattr(norm_module, 'network_layer_name', None), \
        'norm module should still be unstamped after assign_network_names (lora_convert.py:502 carve-out)'
    # Now run the loader.
    state_dict = sd_norm_peft_attn_norm_q()
    with TempLora(state_dict, name='norm_test') as nod:
        net = F.try_load_norm('norm_test', nod, lora_scale=1.0)
    assert net is not None and len(net.modules) == 1, 'Norm loader produced no modules'
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_norm.NetworkModuleNorm)
    # Loader-local stamping: the bound module now has network_layer_name set.
    assert getattr(norm_module, 'network_layer_name', None) == 'lora_transformer_transformer_blocks_0_attn_norm_q', \
        f'Norm loader did not stamp network_layer_name (got {getattr(norm_module, "network_layer_name", None)!r})'
    # calc_updown returns updown + ex_bias for w_norm + b_norm.
    target = norm_module.weight
    updown, ex_bias = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape), 'Norm updown')
    assert ex_bias is not None, 'Norm ex_bias should be present when b_norm is set'
    assert_shape(ex_bias, tuple(target.shape), 'Norm ex_bias')
    return True


def test_full_basic_load():
    state_dict = sd_full_proj()
    net = _load_via(F.try_load_full, state_dict)
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_full.NetworkModuleFull)
    target = mod.sd_module.weight
    updown, ex_bias = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape), 'Full updown')
    assert ex_bias is not None and tuple(ex_bias.shape) == (HIDDEN,), f'Full ex_bias shape {ex_bias.shape if ex_bias is not None else None}'
    return True


# ============================================================
# Tests — calc_updown sanity per family
# ============================================================

CAT_MATH = category('math')


def _instantiate_first(net):
    """Convenience: pull the only module out of a single-module Network."""
    assert net is not None and len(net.modules) == 1
    return next(iter(net.modules.values()))


def test_lora_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_lora, sd_lora_bfl_proj()))
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape))
    assert_finite(updown)
    return True


def test_lokr_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_lokr, sd_lokr_bfl_proj()))
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape))
    assert_finite(updown)
    return True


def test_lokr_chunk_calc_updown_shape():
    """LokrChunk produces a (QKV_OUT, HIDDEN) chunk from a (3*QKV_OUT, HIDDEN) full kron."""
    net = _load_via(F.try_load_lokr, sd_lokr_kohya_qkv())
    for mod in net.modules.values():
        target = mod.sd_module.weight  # split projection: shape (HIDDEN, QKV_OUT) for to_q etc.
        updown, _ = mod.calc_updown(target)
        # `target.shape` is (out=QKV_OUT, in=HIDDEN) for the diffusers Linear; the chunk class
        # returns the chunk of kron(w1,w2) sliced along dim 0 (the fused-output axis).
        assert_shape(updown, (QKV_OUT, HIDDEN))
        assert_finite(updown)
    return True


def test_loha_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_loha, sd_loha_bfl_proj()))
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape))
    assert_finite(updown)
    return True


def test_loha_chunk_calc_updown_shape():
    """HadaChunk produces a (QKV_OUT, HIDDEN) chunk per Q/K/V."""
    net = _load_via(F.try_load_loha, sd_loha_kohya_qkv())
    for mod in net.modules.values():
        target = mod.sd_module.weight
        updown, _ = mod.calc_updown(target)
        assert_shape(updown, (QKV_OUT, HIDDEN))
        assert_finite(updown)
    return True


def test_oft_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_oft, sd_oft_kohya_proj()))
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape))
    assert_finite(updown)
    return True


def test_ia3_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_ia3, sd_ia3_proj()))
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    # IA3 returns target * w; output shape matches target.
    assert_shape(updown, tuple(target.shape))
    assert_finite(updown)
    return True


def test_glora_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_glora, sd_glora_proj()))
    target = mod.sd_module.weight
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape))
    assert_finite(updown)
    return True


def test_full_calc_updown_shape():
    mod = _instantiate_first(_load_via(F.try_load_full, sd_full_proj()))
    target = mod.sd_module.weight
    updown, ex_bias = mod.calc_updown(target)
    assert_shape(updown, tuple(target.shape))
    assert_shape(ex_bias, (HIDDEN,))
    assert_finite(updown)
    return True


# ============================================================
# Tests — apply path / regressions in shared infra
# ============================================================

CAT_APPLY = category('apply')


def test_ex_bias_accumulation_two_norms():
    """Stacking two Norm adapters that both produce ex_bias accumulates
    correctly: network_calc_weights sums their contributions instead of
    raising on a tensor truthiness check.
    """
    install_mock_pipe()
    norm_module = shared.sd_model.pipe.transformer.transformer_blocks[0].attn.norm_q

    # Build two Norm Networks targeting the same norm module.
    state_dict_a = {
        'transformer.transformer_blocks.0.attn.norm_q.w_norm': torch.full((HEAD_DIM,), 0.01),
        'transformer.transformer_blocks.0.attn.norm_q.b_norm': torch.full((HEAD_DIM,), 0.02),
    }
    state_dict_b = {
        'transformer.transformer_blocks.0.attn.norm_q.w_norm': torch.full((HEAD_DIM,), 0.03),
        'transformer.transformer_blocks.0.attn.norm_q.b_norm': torch.full((HEAD_DIM,), 0.04),
    }
    nets = []
    for label, sd in (('a', state_dict_a), ('b', state_dict_b)):
        with TempLora(sd, name=f'norm_{label}') as nod:
            n = F.try_load_norm(f'norm_{label}', nod, lora_scale=1.0)
        n.te_multiplier = 1.0
        n.unet_multiplier = 1.0
        nets.append(n)

    # Patch loaded_networks for the duration of this test.
    saved_loaded = list(l_common.loaded_networks)
    l_common.loaded_networks.clear()
    l_common.loaded_networks.extend(nets)
    try:
        # network_calc_weights iterates loaded_networks and hits the
        # `if batch_ex_bias:` line on the second adapter.
        layer_name = norm_module.network_layer_name
        # Bind layer_name in case prior test cleared it.
        if not layer_name:
            from modules.lora import lora_convert
            lora_convert.assign_network_names_to_compvis_modules(shared.sd_model.pipe)
            layer_name = 'lora_transformer_transformer_blocks_0_attn_norm_q'
            norm_module.network_layer_name = layer_name
        batch_updown, batch_ex_bias = lora_apply.network_calc_weights(norm_module, layer_name)
        assert batch_updown is not None, 'batch_updown should be present (two w_norm contributions)'
        assert batch_ex_bias is not None, 'batch_ex_bias should be present (two b_norm contributions)'
        assert_shape(batch_ex_bias, (HEAD_DIM,))
        # Two contributions of 0.02 + 0.04 = 0.06 (pre-multiplier).
        # multiplier()=te_multiplier=1.0 for both; finalize_updown applies it to ex_bias.
        expected_b = 0.02 + 0.04
        assert torch.allclose(batch_ex_bias, torch.full_like(batch_ex_bias, expected_b), atol=1e-5), \
            f'ex_bias accumulated to {batch_ex_bias[0].item():.6f}, expected {expected_b:.6f}'
    finally:
        l_common.loaded_networks.clear()
        l_common.loaded_networks.extend(saved_loaded)
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing primitives ===')
    for fn in [test_parse_key_all_prefixes, test_resolve_targets_qkv_chunking,
               test_parse_key_peft_wrapper_unwrap, test_parse_key_lycoris_prefix,
               test_parse_key_bare_diffusers_and_peft_default,
               test_marker_disambiguation]:
        run_test(CAT_PARSE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_kohya_fused_qkv_chunked,
        test_lora_bfl_non_fused,
        test_lora_peft_format,
        test_lora_bare_bfl_format,
        test_lora_peft_saved_fal_style,
        test_lora_peft_saved_dreambooth_style,
        test_lora_peft_saved_diffusers_style,
        test_lora_dora_threading,
        test_lokr_bfl_non_fused,
        test_lokr_kohya_fused_qkv_chunked,
        test_lokr_simpletuner_lycoris_format,
        test_loha_bfl_non_fused,
        test_loha_kohya_fused_qkv_chunked,
        test_oft_kohya_non_fused,
        test_oft_lycoris_no_npe,
        test_oft_fused_qkv_skipped,
        test_boft_loads_and_applies,
        test_ia3_basic_load,
        test_ia3_fused_qkv_skipped,
        test_glora_basic_load_and_alpha,
        test_norm_loader_local_stamping,
        test_full_basic_load,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [
        test_lora_calc_updown_shape,
        test_lokr_calc_updown_shape,
        test_lokr_chunk_calc_updown_shape,
        test_loha_calc_updown_shape,
        test_loha_chunk_calc_updown_shape,
        test_oft_calc_updown_shape,
        test_ia3_calc_updown_shape,
        test_glora_calc_updown_shape,
        test_full_calc_updown_shape,
    ]:
        run_test(CAT_MATH, fn)

    log.warning('=== Apply path / shared-infra regressions ===')
    for fn in [test_ex_bias_accumulation_two_norms]:
        run_test(CAT_APPLY, fn)

    elapsed = time.time() - t0

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat, data in results.items():
        total_passed += data['passed']
        total_failed += data['failed']
        status = 'PASS' if data['failed'] == 0 else 'FAIL'
        log.info(f'  {cat}: {data["passed"]} passed, {data["failed"]} failed [{status}]')
    log.warning(f'Total: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s')
    if total_failed:
        sys.exit(1)


if __name__ == '__main__':
    run_tests()
