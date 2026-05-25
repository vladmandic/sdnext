#!/usr/bin/env python
"""
Offline unit tests for Anima native adapter loaders.

Anima is the only native arch with a multi-component network namespace: keys
route into ``lora_transformer_*`` (Cosmos 2.0 DiT), ``lora_llm_adapter_*`` (a
custom Qwen3-projection MLP), or ``lora_te_*`` (Qwen3 text encoder). Routing
is parameterized in ``modules.lora.native_adapter`` via the ``network_prefix``
callable that ``pipelines.anima.anima_lora`` supplies.

Covers the eight families exposed through native_adapter's generics (LoRA,
LoKR, LoHA, OFT, IA3, GLoRA, Norm, Full), focused on:

- LoRA across all five recognized prefixes (BFL transformer / BFL llm_adapter /
  BFL text_encoder / kohya unet / kohya te)
- LoHA on both kohya and BFL transformer paths (the on-disk format of
  ``scenery-anima-base.safetensors``)
- Cosmos 2.0 path rename direct unit tests (every entry in
  ``COSMOS_2_FLAT_RENAME``)
- network_prefix_for component routing
- DoRA threading via the universal NetworkModule.finalize_updown hook
- calc_updown shape sanity for LoRA / LoHA across components

Save formats are cross-referenced against real Anima LoRAs in the wild:

- kohya transformer (``lora_unet_blocks_0_self_attn_q_proj.lora_down.weight``):
  e.g. ``BlueArcStyle``, every entry under the community ``Anima-Preview*`` set
- kohya text-encoder (``lora_te_layers_0_self_attn_q_proj.lora_down.weight``):
  e.g. ``BlueArcStyle`` (mixed unet+te kohya saves)
- kohya LoHA (``lora_unet_blocks_0_cross_attn_q_proj.hada_w1_a``):
  e.g. ``scenery-anima-base``
- BFL / AI-toolkit transformer (``diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight``):
  community AI-toolkit / sd-scripts saves before kohya-LoCon adoption

The diffusers Cosmos 2.0 transformer layout
(``transformer_blocks[i].attn1.{to_q,to_k,to_v,to_out[0],norm_q,norm_k}``,
``transformer_blocks[i].attn2.{to_q,to_k,to_v,to_out[0]}``,
``transformer_blocks[i].ff.net.{0.proj,2}``, six adaLN linears across
``norm1/norm2/norm3.linear_1/_2``, plus top-level ``time_embed.t_embedder``,
``time_embed.norm``, ``patch_embed.proj``, ``norm_out.linear_1/_2``, and
``proj_out``) is taken straight from
``diffusers.models.transformers.transformer_cosmos.Cosmos2TransformerModel``.
Qwen3 layout (``layers[i].self_attn.{q,k,v,o}_proj``,
``layers[i].mlp.{gate,up,down}_proj``) is taken from
``transformers.models.qwen3.Qwen3Model``.

No running server required.

Usage:
    python test/test-anima-native-adapters.py
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
    network, network_lora, network_hada,
)
from modules.lora import lora_common as l_common   # pylint: disable=wrong-import-position
from pipelines.anima import anima_lora as A   # pylint: disable=wrong-import-position


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
# Mock Anima pipeline (transformer + llm_adapter + text_encoder)
# ============================================================
# Test scale preserves Anima's cross-attention asymmetry (image hidden vs
# text-encoder hidden differ): HIDDEN is the DiT width, CROSS_DIM is the
# text-encoder projection width used as the kv-source for cross-attention.
# Real Anima-Preview uses HIDDEN=2048, CROSS_DIM=1024; this test uses
# proportional small values so safetensors fixtures stay cheap.

HIDDEN = 96            # DiT hidden width
CROSS_DIM = 48         # cross-attention kv-source width (text encoder output)
HEAD_DIM = 32
MLP_HIDDEN = 256       # transformer feed-forward hidden
ADALN_OUT = HIDDEN     # each of norm{1,2,3}.linear_{1,2} outputs HIDDEN
TE_HIDDEN = 64
TE_MLP_HIDDEN = 128
# AnimaLLMAdapter: source_dim=target_dim=model_dim=1024 by default. Tests use
# 64 for size: ADAPTER_DIM is source/target/model_dim, ADAPTER_MLP is
# model_dim * mlp_ratio (4.0 in upstream).
ADAPTER_DIM = 64
ADAPTER_HEAD_DIM = 16
ADAPTER_MLP = ADAPTER_DIM * 4
N_BLOCKS = 2
N_TE_LAYERS = 2
N_ADAPTER_BLOCKS = 2   # AnimaLLMAdapter num_layers (upstream default 6)


# pylint: disable=attribute-defined-outside-init
class _Holder(torch.nn.Module):
    """Empty container module - children attached dynamically."""


def build_cosmos_block():
    """Mirror diffusers' Cosmos2TransformerBlock layout."""
    block = _Holder()

    # Self-attention (attn1)
    block.attn1 = _Holder()
    block.attn1.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attn1.to_k = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attn1.to_v = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attn1.to_out = torch.nn.ModuleList([
        torch.nn.Linear(HIDDEN, HIDDEN, bias=False),
        torch.nn.Dropout(0.0),
    ])
    block.attn1.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    block.attn1.norm_k = torch.nn.RMSNorm(HEAD_DIM)

    # Cross-attention (attn2) - kv source is the text encoder output
    block.attn2 = _Holder()
    block.attn2.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attn2.to_k = torch.nn.Linear(CROSS_DIM, HIDDEN, bias=False)
    block.attn2.to_v = torch.nn.Linear(CROSS_DIM, HIDDEN, bias=False)
    block.attn2.to_out = torch.nn.ModuleList([
        torch.nn.Linear(HIDDEN, HIDDEN, bias=False),
        torch.nn.Dropout(0.0),
    ])

    # Feed-forward (diffusers FeedForward style with GELU(proj))
    block.ff = _Holder()
    block.ff.net = torch.nn.ModuleList()
    proj_act = _Holder()
    proj_act.proj = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=True)
    block.ff.net.append(proj_act)
    block.ff.net.append(torch.nn.Dropout(0.0))
    block.ff.net.append(torch.nn.Linear(MLP_HIDDEN, HIDDEN, bias=True))

    # adaLN modulation - six separate Linear modules across norm1/norm2/norm3
    for norm_name in ('norm1', 'norm2', 'norm3'):
        norm_holder = _Holder()
        norm_holder.linear_1 = torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True)
        norm_holder.linear_2 = torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True)
        setattr(block, norm_name, norm_holder)

    return block


def build_mock_transformer():
    """Mirror diffusers' Cosmos2TransformerModel top-level layout."""
    transformer = _Holder()
    transformer.transformer_blocks = torch.nn.ModuleList([build_cosmos_block() for _ in range(N_BLOCKS)])

    # Time embedding
    transformer.time_embed = _Holder()
    transformer.time_embed.t_embedder = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    transformer.time_embed.norm = torch.nn.RMSNorm(HIDDEN)

    # Patch embedding
    transformer.patch_embed = _Holder()
    transformer.patch_embed.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)

    # Output projection
    transformer.norm_out = _Holder()
    transformer.norm_out.linear_1 = torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True)
    transformer.norm_out.linear_2 = torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True)
    transformer.proj_out = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)

    return transformer


def _build_adapter_attention(query_dim, context_dim):
    """Mirror the ``Attention`` submodule inside AnimaLLMAdapter's TransformerBlock.

    Six learnable submodules per attention: q_proj / q_norm / k_proj / k_norm
    / v_proj / o_proj. q_norm and k_norm are per-head RMSNorms; the projections
    use bias=False per upstream.
    """
    attn = _Holder()
    attn.q_proj = torch.nn.Linear(query_dim, ADAPTER_DIM, bias=False)
    attn.q_norm = torch.nn.RMSNorm(ADAPTER_HEAD_DIM, eps=1e-6)
    attn.k_proj = torch.nn.Linear(context_dim, ADAPTER_DIM, bias=False)
    attn.k_norm = torch.nn.RMSNorm(ADAPTER_HEAD_DIM, eps=1e-6)
    attn.v_proj = torch.nn.Linear(context_dim, ADAPTER_DIM, bias=False)
    attn.o_proj = torch.nn.Linear(ADAPTER_DIM, query_dim, bias=False)
    return attn


def _build_adapter_block():
    """Mirror AnimaLLMAdapter.TransformerBlock with use_self_attn=True.

    Layout (from modeling_llm_adapter.py in the Anima-Preview-3 repo):
      norm_self_attn (RMSNorm), self_attn (Attention),
      norm_cross_attn (RMSNorm), cross_attn (Attention, kv from source),
      norm_mlp (RMSNorm), mlp (Sequential[Linear, GELU, Linear]).

    Cross-attention's k/v take ``source_dim`` (= ADAPTER_DIM in this mock);
    in the real adapter ``source_dim`` is the Qwen3 hidden (1024).
    """
    block = _Holder()
    block.norm_self_attn = torch.nn.RMSNorm(ADAPTER_DIM, eps=1e-6)
    block.self_attn = _build_adapter_attention(query_dim=ADAPTER_DIM, context_dim=ADAPTER_DIM)
    block.norm_cross_attn = torch.nn.RMSNorm(ADAPTER_DIM, eps=1e-6)
    block.cross_attn = _build_adapter_attention(query_dim=ADAPTER_DIM, context_dim=ADAPTER_DIM)
    block.norm_mlp = torch.nn.RMSNorm(ADAPTER_DIM, eps=1e-6)
    # nn.Sequential -> named children are '0', '1', '2'. Real LoRAs target the
    # two Linears at indices 0 and 2.
    block.mlp = torch.nn.Sequential(
        torch.nn.Linear(ADAPTER_DIM, ADAPTER_MLP),
        torch.nn.GELU(),
        torch.nn.Linear(ADAPTER_MLP, ADAPTER_DIM),
    )
    return block


def build_mock_llm_adapter():
    """Mirror AnimaLLMAdapter (modeling_llm_adapter.py).

    Top-level: in_proj (Linear or Identity), blocks[N] (TransformerBlock),
    out_proj (Linear), norm (RMSNorm). Embedding is omitted - no observed
    LoRAs target it and including ``nn.Embedding`` would generate spurious
    network_name entries for every vocab row.
    """
    adapter = _Holder()
    adapter.in_proj = torch.nn.Linear(ADAPTER_DIM, ADAPTER_DIM, bias=True)
    adapter.blocks = torch.nn.ModuleList([_build_adapter_block() for _ in range(N_ADAPTER_BLOCKS)])
    adapter.out_proj = torch.nn.Linear(ADAPTER_DIM, ADAPTER_DIM, bias=True)
    adapter.norm = torch.nn.RMSNorm(ADAPTER_DIM, eps=1e-6)
    return adapter


def build_mock_text_encoder():
    """Mirror the Qwen3 text-encoder layout."""
    te = _Holder()
    te.layers = torch.nn.ModuleList()
    for _ in range(N_TE_LAYERS):
        layer = _Holder()
        layer.self_attn = _Holder()
        layer.self_attn.q_proj = torch.nn.Linear(TE_HIDDEN, TE_HIDDEN, bias=False)
        layer.self_attn.k_proj = torch.nn.Linear(TE_HIDDEN, TE_HIDDEN, bias=False)
        layer.self_attn.v_proj = torch.nn.Linear(TE_HIDDEN, TE_HIDDEN, bias=False)
        layer.self_attn.o_proj = torch.nn.Linear(TE_HIDDEN, TE_HIDDEN, bias=False)
        layer.mlp = _Holder()
        layer.mlp.gate_proj = torch.nn.Linear(TE_HIDDEN, TE_MLP_HIDDEN, bias=False)
        layer.mlp.up_proj = torch.nn.Linear(TE_HIDDEN, TE_MLP_HIDDEN, bias=False)
        layer.mlp.down_proj = torch.nn.Linear(TE_MLP_HIDDEN, TE_HIDDEN, bias=False)
        layer.input_layernorm = torch.nn.RMSNorm(TE_HIDDEN)
        layer.post_attention_layernorm = torch.nn.RMSNorm(TE_HIDDEN)
        te.layers.append(layer)
    te.norm = torch.nn.RMSNorm(TE_HIDDEN)
    return te


class _MockAnimaPipeline:
    """Class name carries 'Anima' so name-based model-type dispatch routes correctly.

    Important: NO ``text_encoder_2`` attribute - its presence would flip the
    TE namespace from ``lora_te_`` to ``lora_te1_`` in
    ``assign_network_names_to_compvis_modules``.
    """

    def __init__(self, transformer, llm_adapter, text_encoder):
        self.transformer = transformer
        self.llm_adapter = llm_adapter
        self.text_encoder = text_encoder


class _MockAnimaSdModel:
    """Outer wrapper exposing ``pipe`` + ``network_layer_mapping`` for
    ``lora_convert.assign_network_names_to_compvis_modules`` to populate."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'AnimaTextToImagePipeline'


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing an Anima-shaped 3-component pipeline.

    Each test re-installs so stamped ``network_layer_name`` attributes from
    prior tests do not leak across runs.
    """
    transformer = build_mock_transformer()
    llm_adapter = build_mock_llm_adapter()
    text_encoder = build_mock_text_encoder()
    pipe = _MockAnimaPipeline(transformer, llm_adapter, text_encoder)
    sd_model = _MockAnimaSdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# State-dict synthesizers (one per family/format/component)
# ============================================================

RANK = 8


# --- LoRA, transformer, kohya prefix ---

def sd_lora_kohya_self_attn_q():
    """Kohya transformer LoRA on self-attention.

    On-disk: lora_unet_blocks_0_self_attn_q_proj
    Cosmos rename -> transformer_blocks_0_attn1_to_q
    Network key   -> lora_transformer_transformer_blocks_0_attn1_to_q
    """
    return {
        'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_self_attn_q_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_self_attn_q_proj.alpha': torch.tensor(float(RANK)),
    }


def sd_lora_kohya_cross_attn_k():
    """Kohya transformer LoRA on cross-attention k_proj (k from text encoder).

    On-disk: lora_unet_blocks_0_cross_attn_k_proj
    Cosmos rename -> transformer_blocks_0_attn2_to_k
    """
    return {
        'lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight': torch.randn(RANK, CROSS_DIM),
        'lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_cross_attn_k_proj.alpha': torch.tensor(float(RANK)),
    }


def sd_lora_kohya_self_attn_output():
    """Kohya transformer LoRA on self_attn.output_proj.

    On-disk: lora_unet_blocks_0_self_attn_output_proj
    Cosmos rename -> transformer_blocks_0_attn1_to_out_0
    """
    return {
        'lora_unet_blocks_0_self_attn_output_proj.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_self_attn_output_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
    }


def sd_lora_kohya_mlp_layer1():
    """Kohya transformer LoRA on mlp_layer1.

    On-disk: lora_unet_blocks_0_mlp_layer1
    Cosmos rename -> transformer_blocks_0_ff_net_0_proj
    """
    return {
        'lora_unet_blocks_0_mlp_layer1.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_mlp_layer1.lora_up.weight': torch.randn(MLP_HIDDEN, RANK),
    }


def sd_lora_kohya_mlp_layer2():
    """Kohya transformer LoRA on mlp_layer2 (the down-projection).

    On-disk: lora_unet_blocks_0_mlp_layer2
    Cosmos rename -> transformer_blocks_0_ff_net_2
    """
    return {
        'lora_unet_blocks_0_mlp_layer2.lora_down.weight': torch.randn(RANK, MLP_HIDDEN),
        'lora_unet_blocks_0_mlp_layer2.lora_up.weight': torch.randn(HIDDEN, RANK),
    }


def sd_lora_kohya_adaln_self_attn_1():
    """Kohya transformer LoRA on adaLN self-attn modulation linear 1.

    On-disk: lora_unet_blocks_0_adaln_modulation_self_attn_1
    Cosmos rename -> transformer_blocks_0_norm1_linear_1
    """
    return {
        'lora_unet_blocks_0_adaln_modulation_self_attn_1.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_adaln_modulation_self_attn_1.lora_up.weight': torch.randn(ADALN_OUT, RANK),
    }


# --- LoRA, text encoder, kohya prefix ---

def sd_lora_kohya_te_q():
    """Kohya TE LoRA - this branch was silently dropped by the legacy resolver
    (no ``lora_te_`` kohya prefix recognized) and is newly handled.

    On-disk: lora_te_layers_0_self_attn_q_proj
    Network key -> lora_te_layers_0_self_attn_q_proj
    """
    return {
        'lora_te_layers_0_self_attn_q_proj.lora_down.weight': torch.randn(RANK, TE_HIDDEN),
        'lora_te_layers_0_self_attn_q_proj.lora_up.weight': torch.randn(TE_HIDDEN, RANK),
        'lora_te_layers_0_self_attn_q_proj.alpha': torch.tensor(float(RANK)),
    }


def sd_lora_kohya_te_mlp_gate():
    """Kohya TE LoRA on mlp.gate_proj.

    On-disk: lora_te_layers_1_mlp_gate_proj
    Network key -> lora_te_layers_1_mlp_gate_proj
    """
    return {
        'lora_te_layers_1_mlp_gate_proj.lora_down.weight': torch.randn(RANK, TE_HIDDEN),
        'lora_te_layers_1_mlp_gate_proj.lora_up.weight': torch.randn(TE_MLP_HIDDEN, RANK),
    }


# --- LoRA, transformer, BFL prefix ---

def sd_lora_bfl_self_attn_q():
    """BFL/AI-toolkit transformer LoRA on self-attention.

    On-disk: diffusion_model.blocks.0.self_attn.q_proj
    Cosmos rename of flattened base -> transformer_blocks_0_attn1_to_q
    """
    return {
        'diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight': torch.randn(RANK, HIDDEN),
        'diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight': torch.randn(HIDDEN, RANK),
    }


def sd_lora_bfl_x_embedder():
    """BFL transformer LoRA on the patch embedder.

    On-disk: diffusion_model.x_embedder_proj_1
    Cosmos rename -> patch_embed_proj
    """
    return {
        'diffusion_model.x_embedder_proj_1.lora_A.weight': torch.randn(RANK, HIDDEN),
        'diffusion_model.x_embedder_proj_1.lora_B.weight': torch.randn(HIDDEN, RANK),
    }


def sd_lora_bfl_final_layer_linear():
    """BFL transformer LoRA on the final projection.

    On-disk: diffusion_model.final_layer_linear
    Cosmos rename -> proj_out
    """
    return {
        'diffusion_model.final_layer_linear.lora_A.weight': torch.randn(RANK, HIDDEN),
        'diffusion_model.final_layer_linear.lora_B.weight': torch.randn(HIDDEN, RANK),
    }


# --- LoRA, llm_adapter, BFL prefix ---
#
# Paths mirror AnimaLLMAdapter's actual module tree (see modeling_llm_adapter.py
# in the Anima-Preview-3 repo). Each block contains two Attention submodules
# (self_attn, cross_attn) each with q/k/v/o projections plus q/k RMSNorms,
# plus a 3-element Sequential MLP.

def sd_lora_bfl_llm_adapter_in_proj():
    """BFL llm_adapter LoRA on the top-level Qwen3->model_dim projection.

    On-disk: diffusion_model.llm_adapter.in_proj
    Network key -> lora_llm_adapter_in_proj
    """
    return {
        'diffusion_model.llm_adapter.in_proj.lora_A.weight': torch.randn(RANK, ADAPTER_DIM),
        'diffusion_model.llm_adapter.in_proj.lora_B.weight': torch.randn(ADAPTER_DIM, RANK),
        'diffusion_model.llm_adapter.in_proj.alpha': torch.tensor(float(RANK)),
    }


def sd_lora_bfl_llm_adapter_self_attn_q():
    """BFL llm_adapter LoRA on a block's self-attention q_proj.

    On-disk: diffusion_model.llm_adapter.blocks.0.self_attn.q_proj
    Network key -> lora_llm_adapter_blocks_0_self_attn_q_proj
    """
    return {
        'diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight': torch.randn(RANK, ADAPTER_DIM),
        'diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight': torch.randn(ADAPTER_DIM, RANK),
    }


def sd_lora_bfl_llm_adapter_cross_attn_v():
    """BFL llm_adapter LoRA on a block's cross-attention v_proj.

    Cross-attention's v consumes the Qwen3 hidden states (the source the
    adapter is built around), so this is a common LoRA target.

    On-disk: diffusion_model.llm_adapter.blocks.1.cross_attn.v_proj
    Network key -> lora_llm_adapter_blocks_1_cross_attn_v_proj
    """
    return {
        'diffusion_model.llm_adapter.blocks.1.cross_attn.v_proj.lora_A.weight': torch.randn(RANK, ADAPTER_DIM),
        'diffusion_model.llm_adapter.blocks.1.cross_attn.v_proj.lora_B.weight': torch.randn(ADAPTER_DIM, RANK),
    }


def sd_lora_bfl_llm_adapter_mlp_in():
    """BFL llm_adapter LoRA on the MLP's input Linear (mlp.0 inside Sequential).

    On-disk: diffusion_model.llm_adapter.blocks.0.mlp.0
    Network key -> lora_llm_adapter_blocks_0_mlp_0
    """
    return {
        'diffusion_model.llm_adapter.blocks.0.mlp.0.lora_A.weight': torch.randn(RANK, ADAPTER_DIM),
        'diffusion_model.llm_adapter.blocks.0.mlp.0.lora_B.weight': torch.randn(ADAPTER_MLP, RANK),
    }


def sd_lora_bfl_llm_adapter_out_proj():
    """BFL llm_adapter LoRA on the top-level output projection.

    On-disk: diffusion_model.llm_adapter.out_proj
    Network key -> lora_llm_adapter_out_proj
    """
    return {
        'diffusion_model.llm_adapter.out_proj.lora_A.weight': torch.randn(RANK, ADAPTER_DIM),
        'diffusion_model.llm_adapter.out_proj.lora_B.weight': torch.randn(ADAPTER_DIM, RANK),
    }


# --- LoRA, text encoder, BFL prefix ---

def sd_lora_bfl_te_q():
    """BFL TE LoRA (the ``text_encoders.qwen3_06b.transformer.model.`` prefix).

    On-disk: text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj
    Network key -> lora_te_layers_0_self_attn_q_proj
    """
    return {
        'text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.lora_A.weight': torch.randn(RANK, TE_HIDDEN),
        'text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.lora_B.weight': torch.randn(TE_HIDDEN, RANK),
    }


# --- LoRA combined: transformer + TE in one file ---

def sd_lora_combined_kohya():
    """Kohya save with both transformer and TE LoRAs (mirrors BlueArcStyle)."""
    sd = sd_lora_kohya_self_attn_q()
    sd.update(sd_lora_kohya_te_q())
    return sd


# --- DoRA ---

def sd_lora_kohya_with_dora():
    """Kohya LoRA carrying a dora_scale companion."""
    return {
        'lora_unet_blocks_0_self_attn_v_proj.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_self_attn_v_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_self_attn_v_proj.dora_scale': torch.randn(HIDDEN),
    }


# --- LoHA ---

def sd_loha_kohya_cross_attn_q():
    """Kohya LoHA on cross-attention q_proj (mirrors scenery-anima-base layout)."""
    return {
        'lora_unet_blocks_0_cross_attn_q_proj.hada_w1_a': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_cross_attn_q_proj.hada_w1_b': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_cross_attn_q_proj.hada_w2_a': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_cross_attn_q_proj.hada_w2_b': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_cross_attn_q_proj.alpha': torch.tensor(float(RANK)),
    }


def sd_loha_kohya_self_attn_v():
    """Kohya LoHA on self-attention v_proj."""
    return {
        'lora_unet_blocks_1_self_attn_v_proj.hada_w1_a': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_1_self_attn_v_proj.hada_w1_b': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_1_self_attn_v_proj.hada_w2_a': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_1_self_attn_v_proj.hada_w2_b': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_1_self_attn_v_proj.alpha': torch.tensor(float(RANK)),
    }


def sd_loha_bfl_transformer():
    """BFL transformer LoHA - same Cosmos rename path as kohya, different file
    prefix. Less common in the wild but the resolver handles it."""
    return {
        'diffusion_model.blocks.0.self_attn.q_proj.hada_w1_a': torch.randn(HIDDEN, RANK),
        'diffusion_model.blocks.0.self_attn.q_proj.hada_w1_b': torch.randn(RANK, HIDDEN),
        'diffusion_model.blocks.0.self_attn.q_proj.hada_w2_a': torch.randn(HIDDEN, RANK),
        'diffusion_model.blocks.0.self_attn.q_proj.hada_w2_b': torch.randn(RANK, HIDDEN),
    }


# ============================================================
# Helpers: write state dict to disk, mock NetworkOnDisk
# ============================================================


class TempLora:
    """Context manager: writes a state dict to a temp safetensors file and
    yields a ``_MockNetworkOnDisk`` pointing at it. Cleans up on exit."""

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
    """Stand-in for ``network.NetworkOnDisk`` exposing only the attributes the
    native loaders read."""

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
# Tests - parsing primitives + Cosmos rename
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_all_prefixes():
    """parse_key recognizes each of the five Anima prefixes in priority order.

    Order matters: ``diffusion_model.llm_adapter.`` must precede
    ``diffusion_model.`` and ``text_encoders.qwen3_06b.transformer.model.``
    must precede ``text_encoders.*`` (none of which exist in Anima but the
    priority rule is the same generic mechanism).
    """
    cases = [
        ('lora_unet_blocks_0_self_attn_q_proj.lora_down.weight',
         A.LORA_SUFFIXES,
         ('lora_unet_', 'blocks_0_self_attn_q_proj', 'lora_down.weight')),
        ('lora_te_layers_0_self_attn_q_proj.lora_down.weight',
         A.LORA_SUFFIXES,
         ('lora_te_', 'layers_0_self_attn_q_proj', 'lora_down.weight')),
        ('diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight',
         A.LORA_SUFFIXES,
         ('diffusion_model.', 'blocks.0.self_attn.q_proj', 'lora_down.weight')),
        ('diffusion_model.llm_adapter.input_proj.lora_A.weight',
         A.LORA_SUFFIXES,
         ('diffusion_model.llm_adapter.', 'input_proj', 'lora_down.weight')),
        ('text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.lora_A.weight',
         A.LORA_SUFFIXES,
         ('text_encoders.qwen3_06b.transformer.model.', 'layers.0.self_attn.q_proj', 'lora_down.weight')),
        ('random.unrelated.key', A.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = A.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_marker_disambiguation():
    """Family markers must reject other families' files."""
    pure_lora = {
        'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': torch.zeros(1, 1),
        'lora_unet_blocks_0_self_attn_q_proj.lora_up.weight': torch.zeros(1, 1),
    }
    assert A.has_marker(pure_lora, A.LORA_MARKERS)
    assert not A.has_marker(pure_lora, A.LOHA_MARKERS)
    assert not A.has_marker(pure_lora, A.LOKR_MARKERS)
    assert not A.has_marker(pure_lora, A.OFT_MARKERS)

    pure_loha = {
        'lora_unet_blocks_0_self_attn_q_proj.hada_w1_a': torch.zeros(1, 1),
        'lora_unet_blocks_0_self_attn_q_proj.hada_w1_b': torch.zeros(1, 1),
        'lora_unet_blocks_0_self_attn_q_proj.hada_w2_a': torch.zeros(1, 1),
        'lora_unet_blocks_0_self_attn_q_proj.hada_w2_b': torch.zeros(1, 1),
    }
    assert A.has_marker(pure_loha, A.LOHA_MARKERS)
    assert not A.has_marker(pure_loha, A.LORA_MARKERS)
    return True


def test_cosmos_rename_full_coverage():
    """Every entry in COSMOS_2_FLAT_RENAME applies as a substring rename.

    Confirms order-sensitivity guards the longer-substring-first rule (e.g.
    ``t_embedder_1`` precedes any bare ``t_embedder`` lookalike).
    """
    cases = [
        # Module-level
        ('t_embedder_1', 'time_embed_t_embedder'),
        ('t_embedding_norm', 'time_embed_norm'),
        ('x_embedder_proj_1', 'patch_embed_proj'),
        ('final_layer_linear', 'proj_out'),
        ('final_layer_adaln_modulation_1', 'norm_out_linear_1'),
        ('final_layer_adaln_modulation_2', 'norm_out_linear_2'),
        # Block-level prefix
        ('blocks_0_self_attn_q_proj', 'transformer_blocks_0_attn1_to_q'),
        ('blocks_5_self_attn_k_proj', 'transformer_blocks_5_attn1_to_k'),
        ('blocks_3_self_attn_v_proj', 'transformer_blocks_3_attn1_to_v'),
        ('blocks_2_self_attn_output_proj', 'transformer_blocks_2_attn1_to_out_0'),
        ('blocks_1_cross_attn_q_proj', 'transformer_blocks_1_attn2_to_q'),
        ('blocks_4_cross_attn_k_proj', 'transformer_blocks_4_attn2_to_k'),
        ('blocks_0_cross_attn_v_proj', 'transformer_blocks_0_attn2_to_v'),
        ('blocks_0_cross_attn_output_proj', 'transformer_blocks_0_attn2_to_out_0'),
        # qk_norm RMSNorms
        ('blocks_0_self_attn_q_norm', 'transformer_blocks_0_attn1_norm_q'),
        ('blocks_0_self_attn_k_norm', 'transformer_blocks_0_attn1_norm_k'),
        # MLP
        ('blocks_0_mlp_layer1', 'transformer_blocks_0_ff_net_0_proj'),
        ('blocks_0_mlp_layer2', 'transformer_blocks_0_ff_net_2'),
        # adaLN modulation - six separate Linear modules
        ('blocks_0_adaln_modulation_self_attn_1', 'transformer_blocks_0_norm1_linear_1'),
        ('blocks_0_adaln_modulation_self_attn_2', 'transformer_blocks_0_norm1_linear_2'),
        ('blocks_0_adaln_modulation_cross_attn_1', 'transformer_blocks_0_norm2_linear_1'),
        ('blocks_0_adaln_modulation_cross_attn_2', 'transformer_blocks_0_norm2_linear_2'),
        ('blocks_0_adaln_modulation_mlp_1', 'transformer_blocks_0_norm3_linear_1'),
        ('blocks_0_adaln_modulation_mlp_2', 'transformer_blocks_0_norm3_linear_2'),
    ]
    for src, expected in cases:
        got = A.cosmos_rename_flat(src)
        assert got == expected, f'cosmos_rename_flat({src!r}) = {got!r}, expected {expected!r}'
    return True


def test_resolve_targets_per_prefix():
    """resolve_targets emits one (path, None) per call - Anima has no fused QKV."""
    # Each call returns exactly one target with no ChunkSpec.
    assert A.resolve_targets('lora_unet_', 'blocks_0_self_attn_q_proj') == [
        ('transformer_blocks_0_attn1_to_q', None),
    ]
    assert A.resolve_targets('diffusion_model.', 'blocks.0.self_attn.q_proj') == [
        ('transformer_blocks_0_attn1_to_q', None),
    ]
    assert A.resolve_targets('diffusion_model.llm_adapter.', 'input_proj') == [
        ('input_proj', None),
    ]
    assert A.resolve_targets('text_encoders.qwen3_06b.transformer.model.', 'layers.0.self_attn.q_proj') == [
        ('layers_0_self_attn_q_proj', None),
    ]
    assert A.resolve_targets('lora_te_', 'layers_0_self_attn_q_proj') == [
        ('layers_0_self_attn_q_proj', None),
    ]
    # Unknown prefix yields empty target list.
    assert A.resolve_targets('unknown_prefix', 'foo') == []
    return True


def test_network_prefix_for_routing():
    """network_prefix_for picks the namespace per matched prefix."""
    assert A.network_prefix_for('diffusion_model.llm_adapter.') == 'lora_llm_adapter_'
    assert A.network_prefix_for('text_encoders.qwen3_06b.transformer.model.') == 'lora_te_'
    assert A.network_prefix_for('lora_te_') == 'lora_te_'
    assert A.network_prefix_for('lora_unet_') == 'lora_transformer_'
    assert A.network_prefix_for('diffusion_model.') == 'lora_transformer_'
    # Anything else falls through to transformer (default).
    assert A.network_prefix_for(None) == 'lora_transformer_'
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def test_lora_kohya_self_attn_q():
    """Kohya transformer LoRA on self_attn.q_proj binds via Cosmos rename."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_self_attn_q())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_transformer_blocks_0_attn1_to_q' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lora.NetworkModuleLora)
    return True


def test_lora_kohya_cross_attn_k():
    """Kohya transformer LoRA on cross_attn.k_proj (asymmetric in/out shape)."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_cross_attn_k())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn2_to_k' in net.modules
    return True


def test_lora_kohya_self_attn_output():
    """Kohya transformer LoRA on self_attn.output_proj renames to attn1.to_out.0."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_self_attn_output())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn1_to_out_0' in net.modules
    return True


def test_lora_kohya_mlp_layer1():
    """Kohya transformer LoRA on mlp_layer1 renames to ff.net.0.proj."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_mlp_layer1())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_ff_net_0_proj' in net.modules
    return True


def test_lora_kohya_mlp_layer2():
    """Kohya transformer LoRA on mlp_layer2 renames to ff.net.2."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_mlp_layer2())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_ff_net_2' in net.modules
    return True


def test_lora_kohya_adaln_self_attn_1():
    """Kohya transformer LoRA on adaLN self-attn modulation linear 1.

    On-disk ``adaln_modulation_self_attn_1`` is renamed to ``norm1_linear_1``
    because the diffusers Cosmos2 transformer block exposes its self-attention
    adaLN as ``norm1.linear_1``.
    """
    net = _load_via(A.try_load_lora, sd_lora_kohya_adaln_self_attn_1())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_norm1_linear_1' in net.modules
    return True


def test_lora_kohya_te_q():
    """Kohya text-encoder LoRA (the branch the legacy resolver silently dropped).

    Real fixture: ``BlueArcStyle`` contained 588 such keys; the pre-refactor
    loader recognized only ``lora_unet_blocks_*`` and ignored everything else.
    """
    net = _load_via(A.try_load_lora, sd_lora_kohya_te_q())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_te_layers_0_self_attn_q_proj' in net.modules
    return True


def test_lora_kohya_te_mlp_gate():
    """Kohya TE LoRA on mlp.gate_proj."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_te_mlp_gate())
    assert net is not None and len(net.modules) == 1
    assert 'lora_te_layers_1_mlp_gate_proj' in net.modules
    return True


def test_lora_bfl_self_attn_q():
    """BFL transformer LoRA converges to the same network key as the kohya form."""
    net = _load_via(A.try_load_lora, sd_lora_bfl_self_attn_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn1_to_q' in net.modules
    return True


def test_lora_bfl_x_embedder():
    """BFL transformer LoRA on x_embedder_proj_1 renames to patch_embed.proj."""
    net = _load_via(A.try_load_lora, sd_lora_bfl_x_embedder())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_patch_embed_proj' in net.modules
    return True


def test_lora_bfl_final_layer_linear():
    """BFL transformer LoRA on final_layer_linear renames to proj_out."""
    net = _load_via(A.try_load_lora, sd_lora_bfl_final_layer_linear())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_proj_out' in net.modules
    return True


def test_lora_bfl_llm_adapter_in_proj():
    """BFL llm_adapter LoRA on the top-level Qwen3->model_dim projection.

    Confirms top-level (non-block) paths route into ``lora_llm_adapter_*``.
    """
    net = _load_via(A.try_load_lora, sd_lora_bfl_llm_adapter_in_proj())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_llm_adapter_in_proj' in net.modules
    return True


def test_lora_bfl_llm_adapter_self_attn_q():
    """BFL llm_adapter LoRA on a block's self_attn.q_proj.

    Confirms block-nested attention projections route correctly through the
    underscore-flattened path (no Cosmos rename on the adapter branch).
    """
    net = _load_via(A.try_load_lora, sd_lora_bfl_llm_adapter_self_attn_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_llm_adapter_blocks_0_self_attn_q_proj' in net.modules
    return True


def test_lora_bfl_llm_adapter_cross_attn_v():
    """BFL llm_adapter LoRA on cross_attn.v_proj.

    Cross-attention is what the adapter does: v_proj consumes the Qwen3
    hidden states from the source. Common LoRA target.
    """
    net = _load_via(A.try_load_lora, sd_lora_bfl_llm_adapter_cross_attn_v())
    assert net is not None and len(net.modules) == 1
    assert 'lora_llm_adapter_blocks_1_cross_attn_v_proj' in net.modules
    return True


def test_lora_bfl_llm_adapter_mlp_in():
    """BFL llm_adapter LoRA on mlp.0 (the Linear inside Sequential).

    Confirms numeric Sequential indices flatten correctly: ``mlp.0`` -> ``mlp_0``.
    """
    net = _load_via(A.try_load_lora, sd_lora_bfl_llm_adapter_mlp_in())
    assert net is not None and len(net.modules) == 1
    assert 'lora_llm_adapter_blocks_0_mlp_0' in net.modules
    return True


def test_lora_bfl_llm_adapter_out_proj():
    """BFL llm_adapter LoRA on the top-level output projection."""
    net = _load_via(A.try_load_lora, sd_lora_bfl_llm_adapter_out_proj())
    assert net is not None and len(net.modules) == 1
    assert 'lora_llm_adapter_out_proj' in net.modules
    return True


def test_lora_bfl_te_q():
    """BFL TE LoRA via the ``text_encoders.qwen3_06b.transformer.model.`` prefix.

    Converges to the same network key as the kohya ``lora_te_`` form.
    """
    net = _load_via(A.try_load_lora, sd_lora_bfl_te_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_te_layers_0_self_attn_q_proj' in net.modules
    return True


def test_lora_combined_kohya_transformer_plus_te():
    """A single file with both transformer and TE LoRAs binds modules into both
    namespaces in one load (mirrors the BlueArcStyle layout)."""
    net = _load_via(A.try_load_lora, sd_lora_combined_kohya())
    assert net is not None and len(net.modules) == 2, f'got {net.modules if net else None}'
    assert 'lora_transformer_transformer_blocks_0_attn1_to_q' in net.modules
    assert 'lora_te_layers_0_self_attn_q_proj' in net.modules
    return True


def test_lora_dora_threading():
    """dora_scale flows into NetworkModuleLora.dora_scale via NetworkWeights."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_with_dora())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None, 'dora_scale not threaded into NetworkModule'
    return True


def test_loha_kohya_cross_attn_q():
    """Kohya LoHA on cross_attn.q_proj (mirrors scenery-anima-base layout)."""
    net = _load_via(A.try_load_loha, sd_loha_kohya_cross_attn_q())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_transformer_blocks_0_attn2_to_q' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_hada.NetworkModuleHada)
    return True


def test_loha_kohya_self_attn_v():
    """Kohya LoHA on self_attn.v_proj."""
    net = _load_via(A.try_load_loha, sd_loha_kohya_self_attn_v())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_1_attn1_to_v' in net.modules
    return True


def test_loha_bfl_transformer():
    """BFL-format LoHA produces the same network key as the kohya form."""
    net = _load_via(A.try_load_loha, sd_loha_bfl_transformer())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_transformer_blocks_0_attn1_to_q' in net.modules
    return True


def test_loha_marker_rejection():
    """A pure LoRA file produces no LoHA modules (and vice versa).

    Loaders gate on family-specific markers; cross-family routing yields None
    rather than a partial network.
    """
    # LoRA file routed through LoHA loader -> no LoHA markers -> None
    net = _load_via(A.try_load_loha, sd_lora_kohya_self_attn_q())
    assert net is None, f'LoHA loader matched a LoRA file: {net.modules if net else None}'
    # LoHA file routed through LoRA loader -> no LoRA markers -> None
    net = _load_via(A.try_load_lora, sd_loha_kohya_cross_attn_q())
    assert net is None, f'LoRA loader matched a LoHA file: {net.modules if net else None}'
    return True


def test_try_load_chain_routes_lora_and_loha():
    """try_load (the umbrella) dispatches to whichever family matches."""
    # Pure LoRA file
    net = _load_via(A.try_load, sd_lora_kohya_self_attn_q())
    assert net is not None and len(net.modules) == 1
    assert isinstance(next(iter(net.modules.values())), network_lora.NetworkModuleLora)
    # Pure LoHA file
    net = _load_via(A.try_load, sd_loha_kohya_cross_attn_q())
    assert net is not None and len(net.modules) == 1
    assert isinstance(next(iter(net.modules.values())), network_hada.NetworkModuleHada)
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_transformer():
    """NetworkModuleLora.calc_updown shape sanity against a transformer target."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_self_attn_q())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA transformer calc_updown')
    return True


def test_lora_calc_updown_cross_attn_asymmetric():
    """Cross-attention k_proj has Linear(CROSS_DIM, HIDDEN); the LoRA update
    must match that asymmetric in/out shape."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_cross_attn_k())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, CROSS_DIM)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA cross-attn asymmetric calc_updown')
    return True


def test_lora_calc_updown_llm_adapter():
    """LoRA calc_updown shape sanity against an llm_adapter target."""
    net = _load_via(A.try_load_lora, sd_lora_bfl_llm_adapter_self_attn_q())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(ADAPTER_DIM, ADAPTER_DIM)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA llm_adapter calc_updown')
    return True


def test_lora_calc_updown_text_encoder():
    """LoRA calc_updown shape sanity against a TE target."""
    net = _load_via(A.try_load_lora, sd_lora_kohya_te_q())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(TE_HIDDEN, TE_HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA text-encoder calc_updown')
    return True


def test_loha_calc_updown_transformer():
    """NetworkModuleHada.calc_updown shape sanity against a transformer target."""
    net = _load_via(A.try_load_loha, sd_loha_kohya_cross_attn_q())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoHA transformer calc_updown')
    return True


# === DoRA convention tests (apply_weight_decompose dual-path) ===
#
# DoRA stores a per-axis magnitude vector that the apply path renormalizes
# against. Two conventions exist in the wild:
#
# - per-input (DoRA paper / kohya): dora_scale shape ``(1, in)`` or ``(in,)``,
#   ``W' = W * (m / ||W||_col)`` rescales each column to magnitude m[i].
# - per-output (LyCORIS / PEFT): dora_scale shape ``(out, 1)`` or ``(out,)``,
#   ``W' = W * (m / ||W||_row)`` rescales each row to magnitude m[o].
#
# The pre-fix apply_weight_decompose computed per-input norm regardless and
# silently broadcast against per-output dora_scale via ``(out, 1) / (1, in)
# -> (out, in)``, scrambling the update. These tests guard the dual-path
# detection from regressing.


def _per_input_dora_sd():
    """Synthetic LoRA + per-input dora_scale (kohya convention) on a transformer target."""
    return {
        'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_self_attn_q_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_self_attn_q_proj.dora_scale': torch.randn(1, HIDDEN).abs() + 0.1,
    }


def _per_output_dora_sd():
    """Synthetic LoRA + per-output dora_scale (LyCORIS / PEFT convention).

    Targets cross_attn.k_proj which has asymmetric shape ``(HIDDEN, CROSS_DIM)``,
    so the ``(HIDDEN, 1)`` dora_scale is structurally distinguishable from
    a ``(1, CROSS_DIM)`` per-input scale.
    """
    return {
        'lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight': torch.randn(RANK, CROSS_DIM),
        'lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_cross_attn_k_proj.dora_scale': torch.randn(HIDDEN, 1).abs() + 0.1,
    }


def test_dora_per_input_convention():
    """Per-input dora_scale (1, in): merged weight columns match dora_scale.

    Standard DoRA paper convention. ``W' = W * (m / ||W||_col)`` produces
    new column norms equal to dora_scale.
    """
    net = _load_via(A.try_load_lora, _per_input_dora_sd())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN) * 0.05
    updown, _ = mod.calc_updown(target)
    new_weight = target + updown
    dora_scale = mod.dora_scale.squeeze()  # (in,)
    col_norms = new_weight.norm(dim=0)  # (in,)
    # Allow some numerical slack; the calc_scale factor (alpha/dim) may multiply
    # updown after the DoRA path. Without alpha, calc_scale returns 1.0 and the
    # match is exact to numerical precision.
    rel_err = (col_norms - dora_scale).abs() / dora_scale.abs().clamp_min(1e-6)
    assert rel_err.max() < 1e-3, f'per-input col norms diverge from dora_scale: max rel err {rel_err.max():.2e}'
    return True


def test_dora_per_output_convention():
    """Per-output dora_scale (out, 1): merged weight rows match dora_scale.

    LyCORIS / PEFT convention. ``W' = W * (m / ||W||_row)`` produces new row
    norms equal to dora_scale. Regression target for the LoKR+DoRA file
    (カードキャプターさくらAnimaPreview3Base) that surfaced this bug.
    """
    net = _load_via(A.try_load_lora, _per_output_dora_sd())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, CROSS_DIM) * 0.05
    updown, _ = mod.calc_updown(target)
    new_weight = target + updown
    dora_scale = mod.dora_scale.squeeze()  # (out,)
    row_norms = new_weight.norm(dim=1)  # (out,)
    rel_err = (row_norms - dora_scale).abs() / dora_scale.abs().clamp_min(1e-6)
    assert rel_err.max() < 1e-3, f'per-output row norms diverge from dora_scale: max rel err {rel_err.max():.2e}'
    return True


def test_dora_square_weight_1d_defaults_to_per_input():
    """When the target is square (out == in) AND dora_scale is 1D, the length
    matches both axes and shape can't disambiguate. Default to per-input to
    preserve the dominant kohya save behavior.

    2D dora_scale on a square target is structurally unambiguous and is
    handled correctly by ``test_dora_per_output_convention`` against an
    asymmetric target. This test guards the 1D-on-square ambiguity branch.
    """
    sd = {
        'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': torch.randn(RANK, HIDDEN),
        'lora_unet_blocks_0_self_attn_q_proj.lora_up.weight': torch.randn(HIDDEN, RANK),
        'lora_unet_blocks_0_self_attn_q_proj.dora_scale': torch.randn(HIDDEN).abs() + 0.1,  # 1D, length HIDDEN
    }
    net = _load_via(A.try_load_lora, sd)
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN) * 0.05
    updown, _ = mod.calc_updown(target)
    new_weight = target + updown
    dora_scale = mod.dora_scale.squeeze()
    # Per-input branch: column norms match dora_scale.
    col_norms = new_weight.norm(dim=0)
    rel_err_col = (col_norms - dora_scale).abs() / dora_scale.abs().clamp_min(1e-6)
    assert rel_err_col.max() < 1e-3, f'1D square dora_scale should default to per-input: max rel err {rel_err_col.max():.2e}'
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing primitives + Cosmos rename ===')
    for fn in [
        test_parse_key_all_prefixes,
        test_marker_disambiguation,
        test_cosmos_rename_full_coverage,
        test_resolve_targets_per_prefix,
        test_network_prefix_for_routing,
    ]:
        run_test(CAT_PARSE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        # Kohya transformer (Cosmos rename)
        test_lora_kohya_self_attn_q,
        test_lora_kohya_cross_attn_k,
        test_lora_kohya_self_attn_output,
        test_lora_kohya_mlp_layer1,
        test_lora_kohya_mlp_layer2,
        test_lora_kohya_adaln_self_attn_1,
        # Kohya text-encoder (newly supported)
        test_lora_kohya_te_q,
        test_lora_kohya_te_mlp_gate,
        # BFL transformer
        test_lora_bfl_self_attn_q,
        test_lora_bfl_x_embedder,
        test_lora_bfl_final_layer_linear,
        # BFL llm_adapter (mirrors AnimaLLMAdapter's real module tree)
        test_lora_bfl_llm_adapter_in_proj,
        test_lora_bfl_llm_adapter_self_attn_q,
        test_lora_bfl_llm_adapter_cross_attn_v,
        test_lora_bfl_llm_adapter_mlp_in,
        test_lora_bfl_llm_adapter_out_proj,
        # BFL text-encoder
        test_lora_bfl_te_q,
        # Combined transformer + TE
        test_lora_combined_kohya_transformer_plus_te,
        # DoRA threading
        test_lora_dora_threading,
        # LoHA (new family)
        test_loha_kohya_cross_attn_q,
        test_loha_kohya_self_attn_v,
        test_loha_bfl_transformer,
        # Cross-family rejection
        test_loha_marker_rejection,
        # Umbrella dispatcher
        test_try_load_chain_routes_lora_and_loha,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [
        test_lora_calc_updown_transformer,
        test_lora_calc_updown_cross_attn_asymmetric,
        test_lora_calc_updown_llm_adapter,
        test_lora_calc_updown_text_encoder,
        test_loha_calc_updown_transformer,
        # DoRA convention (apply_weight_decompose dual-path)
        test_dora_per_input_convention,
        test_dora_per_output_convention,
        test_dora_square_weight_1d_defaults_to_per_input,
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
