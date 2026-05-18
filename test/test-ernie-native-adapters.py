#!/usr/bin/env python
"""
Offline unit tests for ERNIE-Image native adapter loaders.

Covers the four native families currently supported by ``pipelines.ernie.ernie_lora``
(LoRA, LoKR, LoHA, OFT) plus DoRA threading via the universal
``NetworkModule.finalize_updown`` hook.

ERNIE-Image is the simplest native arch among z-image / chroma / ernie / flux2:
``ErnieImageAttention`` has fully split ``to_q`` / ``to_k`` / ``to_v`` Linear
modules (no fused QKV layout), and ``ErnieImageFeedForward`` exposes three
separate Linear modules (``gate_proj``, ``up_proj``, ``linear_fc2``). The
loader has no chunking, no renames, no fused-target dispatch - just direct
path-to-network-key conversion.

Save formats are cross-referenced against real ERNIE-Image LoRAs:

- BFL / AI-toolkit (``diffusion_model.layers.16.mlp.gate_proj.lora_A.weight``):
  e.g. ``Ernie-Breast-Slider-v1``
- kohya (``lora_unet_layers_0_mlp_gate_proj.lora_down.weight``):
  e.g. ``ernie_image_radiancechromevoluptuous``
- BFL LoKR (``diffusion_model.layers.0.mlp.gate_proj.lokr_w1``):
  e.g. ``ERNIE_Anatomy_Male``

The diffusers ``ErnieImageTransformer2DModel`` layout
(``layers[i].self_attention.{to_q,to_k,to_v,to_out.0}``,
``layers[i].mlp.{gate_proj,up_proj,linear_fc2}``, plus the module-level
``adaLN_modulation.1`` Linear inside a Sequential, and top-level
``final_norm`` / ``final_linear``) is taken straight from
``diffusers.ErnieImageTransformer2DModel`` /
``ErnieImageSharedAdaLNBlock``.

No running server required.

Usage:
    python test/test-ernie-native-adapters.py
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
from pipelines.ernie import ernie_lora as E    # pylint: disable=wrong-import-position


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
# Mock ERNIE-Image transformer
# ============================================================
# ErnieImage upstream: hidden_size=3072, num_attention_heads=24,
# head_dim=128, ffn_hidden_size=8192. Test scale keeps proportions:
# HIDDEN=96, N_HEADS=3, HEAD_DIM=32, FFN_HIDDEN=256, ADALN_OUT=6*HIDDEN=576.

HIDDEN = 96
HEAD_DIM = 32
FFN_HIDDEN = 256
ADALN_OUT = 6 * HIDDEN
N_LAYERS = 2


# pylint: disable=attribute-defined-outside-init
class _Holder(torch.nn.Module):
    """Empty container module - we attach children dynamically."""


def build_ernie_block():
    """Mirror ``ErnieImageSharedAdaLNBlock`` (single block type, no variants)."""
    block = _Holder()

    block.self_attention = _Holder()
    block.self_attention.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.self_attention.to_k = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.self_attention.to_v = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.self_attention.to_out = torch.nn.ModuleList([
        torch.nn.Linear(HIDDEN, HIDDEN, bias=False),
        torch.nn.Dropout(0.0),
    ])
    block.self_attention.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    block.self_attention.norm_k = torch.nn.RMSNorm(HEAD_DIM)

    block.mlp = _Holder()
    block.mlp.gate_proj = torch.nn.Linear(HIDDEN, FFN_HIDDEN, bias=False)
    block.mlp.up_proj = torch.nn.Linear(HIDDEN, FFN_HIDDEN, bias=False)
    block.mlp.linear_fc2 = torch.nn.Linear(FFN_HIDDEN, HIDDEN, bias=False)

    # Block-level RMSNorms (not typically LoRA-targeted but present)
    block.adaLN_sa_ln = torch.nn.RMSNorm(HIDDEN)
    block.adaLN_mlp_ln = torch.nn.RMSNorm(HIDDEN)

    return block


def build_mock_transformer():
    """Build a torch.nn.Module mimicking ``ErnieImageTransformer2DModel``."""
    transformer = _Holder()
    transformer.layers = torch.nn.ModuleList([build_ernie_block() for _ in range(N_LAYERS)])
    # Module-level adaLN_modulation: Sequential(SiLU, Linear).
    # Real ernie LoRAs target ``adaLN_modulation.1`` (the Linear).
    transformer.adaLN_modulation = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True),
    )
    # final_linear at the model top level - also LoRA-targetable per real fixtures
    transformer.final_linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=True)
    return transformer


class _MockErniePipeline:
    """Class name carries 'ErnieImage' so name-based model-type dispatch routes correctly."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockErnieSdModel:
    """Outer wrapper holding pipe + network_layer_mapping."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'ErnieImagePipeline'


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing an ERNIE-Image-shaped transformer.

    Each test re-installs so any prior network_layer_name stamps don't leak.
    """
    transformer = build_mock_transformer()
    pipe = _MockErniePipeline(transformer)
    sd_model = _MockErnieSdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# State-dict synthesizers (one per family/format)
# ============================================================

RANK_LORA = 8
LOKR_W1_DIM = 8


def sd_lora_bfl_mlp_gate_proj():
    """BFL LoRA on layers.X.mlp.gate_proj. Mirrors Ernie-Breast-Slider-v1."""
    return {
        'diffusion_model.layers.0.mlp.gate_proj.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.mlp.gate_proj.lora_B.weight': torch.randn(FFN_HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_self_attention():
    """BFL LoRA on the split self_attention.to_q (no fusion in ernie)."""
    return {
        'diffusion_model.layers.1.self_attention.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.1.self_attention.to_q.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.1.self_attention.to_q.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_bfl_self_attention_to_out():
    """BFL LoRA on self_attention.to_out.0 (the Linear inside the ModuleList)."""
    return {
        'diffusion_model.layers.0.self_attention.to_out.0.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.self_attention.to_out.0.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_mlp_linear_fc2():
    """BFL LoRA on mlp.linear_fc2 (the down-projection)."""
    return {
        'diffusion_model.layers.0.mlp.linear_fc2.lora_A.weight': torch.randn(RANK_LORA, FFN_HIDDEN),
        'diffusion_model.layers.0.mlp.linear_fc2.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_kohya_mlp_gate_proj():
    """Kohya flat-underscore LoRA on layers.X.mlp.gate_proj."""
    return {
        'lora_unet_layers_0_mlp_gate_proj.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_layers_0_mlp_gate_proj.lora_up.weight': torch.randn(FFN_HIDDEN, RANK_LORA),
        'lora_unet_layers_0_mlp_gate_proj.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_kohya_adaLN_modulation():
    """Kohya LoRA on module-level adaLN_modulation.1 (the Linear inside Sequential).

    Real fixtures use ``lora_unet_adaLN_modulation_1`` -> the Linear at index 1.
    """
    return {
        'lora_unet_adaLN_modulation_1.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_adaLN_modulation_1.lora_up.weight': torch.randn(ADALN_OUT, RANK_LORA),
        'lora_unet_adaLN_modulation_1.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_peft_to_v():
    """PEFT-format LoRA on self_attention.to_v."""
    return {
        'transformer.layers.0.self_attention.to_v.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.0.self_attention.to_v.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_with_dora_scale():
    """LoRA with dora_scale companion to exercise DoRA threading."""
    return {
        'transformer.layers.0.mlp.up_proj.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.0.mlp.up_proj.lora_B.weight': torch.randn(FFN_HIDDEN, RANK_LORA),
        'transformer.layers.0.mlp.up_proj.dora_scale': torch.randn(FFN_HIDDEN),
    }


def sd_lokr_bfl_mlp_gate_proj():
    """BFL LoKR on mlp.gate_proj. Mirrors ERNIE_Anatomy_Male."""
    return {
        'diffusion_model.layers.0.mlp.gate_proj.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.layers.0.mlp.gate_proj.lokr_w2': torch.randn(FFN_HIDDEN // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.layers.0.mlp.gate_proj.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_lokr_bfl_self_attention():
    """BFL LoKR on self_attention.to_q (no fusion, straight binding)."""
    return {
        'diffusion_model.layers.1.self_attention.to_q.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.layers.1.self_attention.to_q.lokr_w2': torch.randn(HIDDEN // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.layers.1.self_attention.to_q.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_loha_bfl_mlp():
    """LoHA on mlp.linear_fc2."""
    return {
        'diffusion_model.layers.0.mlp.linear_fc2.hada_w1_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.mlp.linear_fc2.hada_w1_b': torch.randn(RANK_LORA, FFN_HIDDEN),
        'diffusion_model.layers.0.mlp.linear_fc2.hada_w2_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.mlp.linear_fc2.hada_w2_b': torch.randn(RANK_LORA, FFN_HIDDEN),
        'diffusion_model.layers.0.mlp.linear_fc2.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_oft_lycoris_self_attention():
    """OFT (LyCORIS oft_diag form) on self_attention.to_k."""
    num_blocks = 4
    block_size = HIDDEN // num_blocks
    return {
        'diffusion_model.layers.0.self_attention.to_k.oft_blocks': torch.randn(num_blocks, block_size, block_size) * 0.01,
        'diffusion_model.layers.0.self_attention.to_k.oft_diag': torch.ones(num_blocks, block_size),
        'diffusion_model.layers.0.self_attention.to_k.alpha': torch.tensor(0.001),
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
    """parse_key returns (prefix_used, base, suffix). ERNIE has no path renames
    so resolve_targets passes the base through verbatim."""
    bd = E.BARE_DIFFUSERS_PREFIX_USED
    cases = [
        ('diffusion_model.layers.0.mlp.gate_proj.lora_A.weight',
         E.LORA_SUFFIXES,
         ('diffusion_model.', 'layers.0.mlp.gate_proj', 'lora_down.weight')),
        ('transformer.layers.1.self_attention.to_q.lora_B.weight',
         E.LORA_SUFFIXES,
         ('transformer.', 'layers.1.self_attention.to_q', 'lora_up.weight')),
        ('lora_unet_layers_0_mlp_linear_fc2.lora_down.weight',
         E.LORA_SUFFIXES,
         ('lora_unet_', 'layers_0_mlp_linear_fc2', 'lora_down.weight')),
        # Bare path starting with a known block prefix
        ('layers.0.mlp.up_proj.lora_A.weight',
         E.LORA_SUFFIXES,
         (bd, 'layers.0.mlp.up_proj', 'lora_down.weight')),
        ('random.unrelated.key', E.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = E.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_marker_disambiguation():
    """Each family's markers reject other families' files."""
    pure_lora = {
        'lora_unet_layers_0_mlp_gate_proj.lora_down.weight': torch.zeros(1, 1),
        'lora_unet_layers_0_mlp_gate_proj.lora_up.weight': torch.zeros(1, 1),
    }
    assert E.has_marker(pure_lora, E.LORA_MARKERS)
    assert not E.has_marker(pure_lora, E.LOKR_MARKERS)
    assert not E.has_marker(pure_lora, E.LOHA_MARKERS)
    assert not E.has_marker(pure_lora, E.OFT_MARKERS)

    pure_lokr = {
        'diffusion_model.layers.0.mlp.gate_proj.lokr_w1': torch.zeros(1, 1),
        'diffusion_model.layers.0.mlp.gate_proj.lokr_w2': torch.zeros(1, 1),
    }
    assert E.has_marker(pure_lokr, E.LOKR_MARKERS)
    assert not E.has_marker(pure_lokr, E.LORA_MARKERS)
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def test_lora_bfl_mlp_gate_proj():
    """BFL LoRA on layers.X.mlp.gate_proj binds straight through."""
    net = _load_via(E.try_load_lora, sd_lora_bfl_mlp_gate_proj())
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_layers_0_mlp_gate_proj' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lora.NetworkModuleLora)
    return True


def test_lora_bfl_self_attention():
    """BFL LoRA on self_attention.to_q (no fusion in ernie - straight binding)."""
    net = _load_via(E.try_load_lora, sd_lora_bfl_self_attention())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_1_self_attention_to_q' in net.modules
    return True


def test_lora_bfl_self_attention_to_out():
    """BFL LoRA on self_attention.to_out.0 (Linear inside ModuleList)."""
    net = _load_via(E.try_load_lora, sd_lora_bfl_self_attention_to_out())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_self_attention_to_out_0' in net.modules
    return True


def test_lora_bfl_mlp_linear_fc2():
    """BFL LoRA on mlp.linear_fc2 (the down projection)."""
    net = _load_via(E.try_load_lora, sd_lora_bfl_mlp_linear_fc2())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_mlp_linear_fc2' in net.modules
    return True


def test_lora_kohya_mlp_gate_proj():
    """Kohya format converges to the same diffusers network_key as BFL."""
    net = _load_via(E.try_load_lora, sd_lora_kohya_mlp_gate_proj())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_mlp_gate_proj' in net.modules
    return True


def test_lora_kohya_adaLN_modulation():
    """Kohya LoRA on the module-level adaLN_modulation.1 (Linear in Sequential)."""
    net = _load_via(E.try_load_lora, sd_lora_kohya_adaLN_modulation())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_adaLN_modulation_1' in net.modules
    return True


def test_lora_peft_to_v():
    """PEFT format binds without rename or chunking."""
    net = _load_via(E.try_load_lora, sd_lora_peft_to_v())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_self_attention_to_v' in net.modules
    return True


def test_lora_dora_threading():
    """dora_scale flows into NetworkModuleLora.dora_scale."""
    net = _load_via(E.try_load_lora, sd_lora_with_dora_scale())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None
    return True


def test_lokr_bfl_mlp_gate_proj():
    """BFL LoKR on mlp.gate_proj binds via NetworkModuleLokr (no chunk class in ernie)."""
    net = _load_via(E.try_load_lokr, sd_lokr_bfl_mlp_gate_proj())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_mlp_gate_proj' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lokr.NetworkModuleLokr)
    # ernie LoKR never instantiates the chunk variants - no fused targets exist
    assert not isinstance(mod, network_lokr.NetworkModuleLokrChunk)
    return True


def test_lokr_bfl_self_attention():
    """BFL LoKR on self_attention.to_q. No fusion means straight NetworkModuleLokr."""
    net = _load_via(E.try_load_lokr, sd_lokr_bfl_self_attention())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_1_self_attention_to_q' in net.modules
    return True


def test_loha_bfl_mlp():
    """LoHA on mlp.linear_fc2 binds via NetworkModuleHada (no chunk path)."""
    net = _load_via(E.try_load_loha, sd_loha_bfl_mlp())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_hada.NetworkModuleHada)
    return True


def test_oft_lycoris_no_npe():
    """OFT LyCORIS oft_diag form loads on self_attention.to_k without NoneType errors."""
    net = _load_via(E.try_load_oft, sd_oft_lycoris_self_attention())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_oft.NetworkModuleOFT)
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_shape():
    net = _load_via(E.try_load_lora, sd_lora_bfl_mlp_gate_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(FFN_HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA calc_updown')
    return True


def test_lokr_calc_updown_shape():
    net = _load_via(E.try_load_lokr, sd_lokr_bfl_mlp_gate_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(FFN_HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoKR calc_updown')
    return True


def test_loha_calc_updown_shape():
    net = _load_via(E.try_load_loha, sd_loha_bfl_mlp())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, FFN_HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoHA calc_updown')
    return True


def test_oft_calc_updown_shape():
    net = _load_via(E.try_load_oft, sd_oft_lycoris_self_attention())
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
    for fn in [test_parse_key_all_prefixes, test_marker_disambiguation]:
        run_test(CAT_PARSE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_bfl_mlp_gate_proj,
        test_lora_bfl_self_attention,
        test_lora_bfl_self_attention_to_out,
        test_lora_bfl_mlp_linear_fc2,
        test_lora_kohya_mlp_gate_proj,
        test_lora_kohya_adaLN_modulation,
        test_lora_peft_to_v,
        test_lora_dora_threading,
        test_lokr_bfl_mlp_gate_proj,
        test_lokr_bfl_self_attention,
        test_loha_bfl_mlp,
        test_oft_lycoris_no_npe,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [
        test_lora_calc_updown_shape,
        test_lokr_calc_updown_shape,
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
