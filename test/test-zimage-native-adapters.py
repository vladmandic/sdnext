#!/usr/bin/env python
"""
Offline unit tests for Z-Image native adapter loaders.

Covers the four native families currently supported by ``pipelines.z_image.zimage_lora``
(LoRA, LoKR, LoHA, OFT) plus DoRA threading via the universal
``NetworkModule.finalize_updown`` hook.

Tests build a mock Z-Image-shaped transformer, write synthetic safetensors
files for each adapter format observed in the wild, and exercise the full
loader path from state dict to ``NetworkModule*`` instantiation.

Save formats are cross-referenced against real Z-Image LoRAs in the wild:

- BFL / AI-toolkit (``diffusion_model.layers.0.attention.to_q.lora_A.weight``):
  e.g. ``80sFantasyZBase``, ``zimagebase_blending_v1``
- kohya (``lora_unet_layers_0_attention_to_q.lora_down.weight``): pattern
  produced by kohya-ss/sd-scripts targeting Z-Image
- PEFT (``transformer.layers.0.attention.to_q.lora_down.weight``):
  e.g. ``FameGrid_Revolution_ZIB_BOLD``
- Legacy fused ``attention.qkv`` (Z-Image pre-refactor): the loader splits
  the up-weight along dim 0 at load time and emits Q / K / V targets

Loader correctness is verified against the documented LyCORIS / kohya
on-disk shapes. The Z-Image diffusers transformer layout
(``layers[i].attention.{to_q,to_k,to_v,to_out[0]}`` plus
``feed_forward.net.{0.proj,2}`` and ``adaLN_modulation.0``) is taken
straight from ``diffusers.ZImageTransformer2DModel`` /
``ZImageTransformerBlock``.

No running server required.

Usage:
    python test/test-zimage-native-adapters.py
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
from pipelines.z_image import zimage_lora as Z    # pylint: disable=wrong-import-position


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
# Mock Z-Image transformer
# ============================================================
# Shape constants chosen to mirror real Z-Image proportions while keeping
# tensors small: ``hidden_dim = int(dim / 3 * 8)`` matches the
# ZImageTransformerBlock FeedForward sizing, and adaLN_modulation outputs
# ``4 * dim`` per upstream.

HIDDEN = 96            # dim
HEAD_DIM = 32          # head_dim; n_heads = HIDDEN / HEAD_DIM = 3
MLP_HIDDEN = int(HIDDEN / 3 * 8)   # 256, matches ZImageTransformerBlock FeedForward
ADALN_OUT = 4 * HIDDEN # 384, matches Z-Image adaLN_modulation
N_LAYERS = 2           # main transformer blocks
N_REFINER = 1          # noise_refiner / context_refiner blocks


# pylint: disable=attribute-defined-outside-init
class _Holder(torch.nn.Module):
    """Empty container module - we attach children dynamically."""


def build_zimage_block(modulation: bool):
    """Mirror ``ZImageTransformerBlock``'s diffusers-side module layout."""
    block = _Holder()
    # Attention module (diffusers Attention class with custom processor)
    block.attention = _Holder()
    block.attention.to_q = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attention.to_k = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.attention.to_v = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    # to_out is a ModuleList in diffusers; index 0 is the Linear, index 1 is Dropout
    block.attention.to_out = torch.nn.ModuleList([
        torch.nn.Linear(HIDDEN, HIDDEN, bias=False),
        torch.nn.Dropout(0.0),
    ])
    # qk_norm RMSNorms - present when qk_norm=True
    block.attention.norm_q = torch.nn.RMSNorm(HEAD_DIM)
    block.attention.norm_k = torch.nn.RMSNorm(HEAD_DIM)

    # FeedForward (diffusers FeedForward class)
    block.feed_forward = _Holder()
    block.feed_forward.net = torch.nn.ModuleList()
    proj_act = _Holder()  # GELU activation wrapping a proj Linear
    proj_act.proj = torch.nn.Linear(HIDDEN, MLP_HIDDEN, bias=True)
    block.feed_forward.net.append(proj_act)
    block.feed_forward.net.append(torch.nn.Dropout(0.0))
    block.feed_forward.net.append(torch.nn.Linear(MLP_HIDDEN, HIDDEN, bias=True))

    # Per-block RMSNorms
    block.attention_norm1 = torch.nn.RMSNorm(HIDDEN)
    block.ffn_norm1 = torch.nn.RMSNorm(HIDDEN)
    block.attention_norm2 = torch.nn.RMSNorm(HIDDEN)
    block.ffn_norm2 = torch.nn.RMSNorm(HIDDEN)

    if modulation:
        block.adaLN_modulation = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN, ADALN_OUT, bias=True),
        )
    return block


def build_mock_transformer():
    """Build a torch.nn.Module mimicking ``ZImageTransformer2DModel``.

    Mirrors the paths real Z-Image LoRAs target:
    ``layers.X.attention.{to_q,to_k,to_v,to_out.0}``,
    ``layers.X.feed_forward.net.{0.proj,2}``,
    ``layers.X.adaLN_modulation.0``, plus the four per-block RMSNorms and
    refiner stacks (``noise_refiner``, ``context_refiner``).
    """
    transformer = _Holder()
    transformer.layers = torch.nn.ModuleList([build_zimage_block(modulation=False) for _ in range(N_LAYERS)])
    transformer.noise_refiner = torch.nn.ModuleList([build_zimage_block(modulation=True) for _ in range(N_REFINER)])
    transformer.context_refiner = torch.nn.ModuleList([build_zimage_block(modulation=True) for _ in range(N_REFINER)])
    return transformer


class _MockZImagePipeline:
    """Class name carries 'ZImage' so name-based model-type dispatch routes correctly."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockZImageSdModel:
    """Outer wrapper exposing pipe + network_layer_mapping for
    ``lora_convert.assign_network_names_to_compvis_modules`` to write onto."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'ZImagePipeline'  # belt-and-suspenders


def install_mock_pipe():
    """Set shared.sd_model to a mock exposing a Z-Image-shaped transformer.

    Each test calls this fresh so any prior network_layer_name stamps don't
    leak across tests. Writes directly to ``model_data.sd_model`` to bypass
    the ModelData lock that no-ops the public setter outside webui startup.
    """
    transformer = build_mock_transformer()
    pipe = _MockZImagePipeline(transformer)
    sd_model = _MockZImageSdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# State-dict synthesizers (one per family/format)
# ============================================================

RANK_LORA = 8
RANK_LOKR = 4
LOKR_W1_DIM = 8


def sd_lora_bfl_to_q():
    """BFL / AI-toolkit LoRA on a split-QKV target. Mirrors 80sFantasyZBase."""
    return {
        'diffusion_model.layers.0.attention.to_q.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.to_q.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_bfl_adaln():
    """BFL LoRA on adaLN_modulation (the modulation Linear in Sequential)."""
    # Only refiner blocks have modulation in our mock; index path with refiner
    return {
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.lora_B.weight': torch.randn(ADALN_OUT, RANK_LORA),
    }


def sd_lora_peft_to_out():
    """PEFT-format LoRA on attention.to_out.0. Mirrors FameGrid_Revolution_ZIB_BOLD."""
    return {
        'transformer.layers.1.attention.to_out.0.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.1.attention.to_out.0.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'transformer.layers.1.attention.to_out.0.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_kohya_to_q():
    """Kohya-format LoRA on attention.to_q (flat underscore path)."""
    return {
        'lora_unet_layers_0_attention_to_q.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'lora_unet_layers_0_attention_to_q.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'lora_unet_layers_0_attention_to_q.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_legacy_fused_qkv():
    """Legacy Z-Image fused ``attention.qkv`` (BFL prefix). Loader splits the up-weight."""
    # up has 3*HIDDEN rows (Q stacked over K stacked over V), down is shared
    return {
        'diffusion_model.layers.0.attention.qkv.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.qkv.lora_B.weight': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.qkv.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_lora_legacy_attention_out_alias():
    """Legacy ``attention.out`` alias - loader renames to ``attention.to_out.0``."""
    return {
        'diffusion_model.layers.1.attention.out.lora_A.weight': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.1.attention.out.lora_B.weight': torch.randn(HIDDEN, RANK_LORA),
    }


def sd_lora_with_dora_scale():
    """LoRA carrying a dora_scale companion vector to exercise DoRA threading."""
    return {
        'transformer.layers.0.attention.to_v.lora_down.weight': torch.randn(RANK_LORA, HIDDEN),
        'transformer.layers.0.attention.to_v.lora_up.weight': torch.randn(HIDDEN, RANK_LORA),
        'transformer.layers.0.attention.to_v.dora_scale': torch.randn(HIDDEN),
    }


def sd_lokr_bfl_adaln():
    """BFL LoKR on the modulation linear. Mirrors after_dark_2_zib_lokr's adaLN target."""
    # w1=(HIDDEN/LOKR_W1_DIM, LOKR_W1_DIM), w2=(LOKR_W1_DIM, ADALN_OUT*HIDDEN/(LOKR_W1_DIM*HIDDEN))
    # Pick a factorization that yields kron shape == (ADALN_OUT, HIDDEN)
    return {
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.lokr_w2': torch.randn(ADALN_OUT // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.noise_refiner.0.adaLN_modulation.0.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_lokr_legacy_fused_qkv():
    """Legacy fused qkv LoKR - loader defers split to apply time via NetworkModuleLokrChunk."""
    # w1 small, w2 must yield 3*HIDDEN rows after kron; pick LOKR_W1_DIM and (3*HIDDEN/LOKR_W1_DIM, HIDDEN/LOKR_W1_DIM)
    return {
        'diffusion_model.layers.0.attention.qkv.lokr_w1': torch.randn(LOKR_W1_DIM, LOKR_W1_DIM),
        'diffusion_model.layers.0.attention.qkv.lokr_w2': torch.randn((3 * HIDDEN) // LOKR_W1_DIM, HIDDEN // LOKR_W1_DIM),
        'diffusion_model.layers.0.attention.qkv.alpha': torch.tensor(float(LOKR_W1_DIM)),
    }


def sd_loha_bfl_proj():
    """LoHA on attention.to_out.0 (non-fused; LoHA on fused qkv is skipped by the loader)."""
    return {
        'diffusion_model.layers.1.attention.to_out.0.hada_w1_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.1.attention.to_out.0.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.1.attention.to_out.0.hada_w2_a': torch.randn(HIDDEN, RANK_LORA),
        'diffusion_model.layers.1.attention.to_out.0.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.1.attention.to_out.0.alpha': torch.tensor(float(RANK_LORA)),
    }


def sd_loha_legacy_fused_qkv_skipped():
    """LoHA on legacy fused qkv - loader skips with warning (no NetworkModuleHadaChunk for z-image's path)."""
    return {
        'diffusion_model.layers.0.attention.qkv.hada_w1_a': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.qkv.hada_w1_b': torch.randn(RANK_LORA, HIDDEN),
        'diffusion_model.layers.0.attention.qkv.hada_w2_a': torch.randn(3 * HIDDEN, RANK_LORA),
        'diffusion_model.layers.0.attention.qkv.hada_w2_b': torch.randn(RANK_LORA, HIDDEN),
    }


def sd_oft_lycoris_proj():
    """OFT LyCORIS-style oft_diag on attention.to_v (non-fused)."""
    # OFT 3-D oft_blocks shape: (num_blocks, block_size, block_size). oft_diag is per-block.
    num_blocks = 4
    block_size = HIDDEN // num_blocks
    return {
        'diffusion_model.layers.0.attention.to_v.oft_blocks': torch.randn(num_blocks, block_size, block_size) * 0.01,
        'diffusion_model.layers.0.attention.to_v.oft_diag': torch.ones(num_blocks, block_size),
        'diffusion_model.layers.0.attention.to_v.alpha': torch.tensor(0.001),
    }


def sd_oft_legacy_fused_qkv_skipped():
    """OFT on legacy fused qkv - loader skips with warning (no row-sliceable OFT structure)."""
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
    """parse_key recognizes BFL, PEFT, kohya, and bare-diffusers keys.

    Returns (prefix_used, base, suffix) - prefix_used is the matched
    KNOWN_PREFIXES element, BARE_DIFFUSERS_PREFIX_USED for bare paths
    matching BARE_DIFFUSERS_PREFIXES, or None when no prefix is recognized.
    """
    bd = Z.BARE_DIFFUSERS_PREFIX_USED
    cases = [
        ('diffusion_model.layers.0.attention.to_q.lora_A.weight',
         Z.LORA_SUFFIXES,
         ('diffusion_model.', 'layers.0.attention.to_q', 'lora_down.weight')),
        ('transformer.layers.0.attention.to_v.lora_B.weight',
         Z.LORA_SUFFIXES,
         ('transformer.', 'layers.0.attention.to_v', 'lora_up.weight')),
        ('lora_unet_layers_0_attention_to_k.lora_down.weight',
         Z.LORA_SUFFIXES,
         ('lora_unet_', 'layers_0_attention_to_k', 'lora_down.weight')),
        # Bare path starting with a known block prefix
        ('layers.0.attention.to_out.0.lora_A.weight',
         Z.LORA_SUFFIXES,
         (bd, 'layers.0.attention.to_out.0', 'lora_down.weight')),
        ('random.unrelated.key', Z.LORA_SUFFIXES, None),
    ]
    for key, suffixes, expected in cases:
        got = Z.parse_key(key, suffixes)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_marker_disambiguation():
    """Each family's markers reject other families' files."""
    pure_lora = {
        'lora_unet_layers_0_attention_to_q.lora_down.weight': torch.zeros(1, 1),
        'lora_unet_layers_0_attention_to_q.lora_up.weight': torch.zeros(1, 1),
    }
    assert Z.has_marker(pure_lora, Z.LORA_MARKERS)
    assert not Z.has_marker(pure_lora, Z.LOKR_MARKERS)
    assert not Z.has_marker(pure_lora, Z.LOHA_MARKERS)
    assert not Z.has_marker(pure_lora, Z.OFT_MARKERS)

    pure_lokr = {
        'diffusion_model.layers.0.attention.to_q.lokr_w1': torch.zeros(1, 1),
        'diffusion_model.layers.0.attention.to_q.lokr_w2': torch.zeros(1, 1),
    }
    assert Z.has_marker(pure_lokr, Z.LOKR_MARKERS)
    assert not Z.has_marker(pure_lokr, Z.LORA_MARKERS)
    assert not Z.has_marker(pure_lokr, Z.LOHA_MARKERS)
    assert not Z.has_marker(pure_lokr, Z.OFT_MARKERS)
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def test_lora_bfl_split_qkv():
    """BFL-format LoRA on attention.to_q binds correctly."""
    net = _load_via(Z.try_load_lora, sd_lora_bfl_to_q())
    assert net is not None and len(net.modules) == 1, f'expected 1 module, got {net.modules if net else None}'
    assert 'lora_transformer_layers_0_attention_to_q' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lora.NetworkModuleLora)
    return True


def test_lora_peft_to_out():
    """PEFT-format LoRA on attention.to_out.0 binds correctly."""
    net = _load_via(Z.try_load_lora, sd_lora_peft_to_out())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_1_attention_to_out_0' in net.modules
    return True


def test_lora_kohya_to_q():
    """Kohya flat-underscore LoRA on attention.to_q binds correctly."""
    net = _load_via(Z.try_load_lora, sd_lora_kohya_to_q())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_0_attention_to_q' in net.modules
    return True


def test_lora_bfl_adaln():
    """BFL LoRA on adaLN_modulation.0 (the Linear inside Sequential)."""
    net = _load_via(Z.try_load_lora, sd_lora_bfl_adaln())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_noise_refiner_0_adaLN_modulation_0' in net.modules
    return True


def test_lora_legacy_fused_qkv_chunked():
    """Legacy fused attention.qkv is split into 3 chunks targeting to_q/to_k/to_v.

    The up-weight is chunked along dim 0 at load time; the down-weight is shared.
    """
    net = _load_via(Z.try_load_lora, sd_lora_legacy_fused_qkv())
    assert net is not None and len(net.modules) == 3, f'expected 3 chunked modules, got {net.modules if net else None}'
    expected = {
        'lora_transformer_layers_0_attention_to_q',
        'lora_transformer_layers_0_attention_to_k',
        'lora_transformer_layers_0_attention_to_v',
    }
    assert set(net.modules) == expected, f'got {set(net.modules)}'
    # Each chunked module's up-weight is (HIDDEN, RANK), not (3*HIDDEN, RANK)
    for nk, mod in net.modules.items():
        assert_shape(mod.up_model.weight, (HIDDEN, RANK_LORA), label=nk)
    return True


def test_lora_legacy_attention_out_alias():
    """Legacy attention.out alias is renamed to attention.to_out.0."""
    net = _load_via(Z.try_load_lora, sd_lora_legacy_attention_out_alias())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_layers_1_attention_to_out_0' in net.modules
    return True


def test_lora_dora_threading():
    """dora_scale flows into NetworkModuleLora.dora_scale."""
    net = _load_via(Z.try_load_lora, sd_lora_with_dora_scale())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert mod.dora_scale is not None, 'dora_scale not threaded into NetworkModule'
    return True


def test_lokr_bfl_adaln():
    """BFL LoKR on adaLN_modulation.0 binds via NetworkModuleLokr."""
    net = _load_via(Z.try_load_lokr, sd_lokr_bfl_adaln())
    assert net is not None and len(net.modules) == 1
    assert 'lora_transformer_noise_refiner_0_adaLN_modulation_0' in net.modules
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lokr.NetworkModuleLokr)
    return True


def test_lokr_legacy_fused_qkv_chunked():
    """Legacy fused LoKR on attention.qkv emits 3 chunked modules via NetworkModuleLokrChunk."""
    net = _load_via(Z.try_load_lokr, sd_lokr_legacy_fused_qkv())
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


def test_loha_bfl_proj():
    """BFL LoHA on a non-fused proj target binds via NetworkModuleHada."""
    net = _load_via(Z.try_load_loha, sd_loha_bfl_proj())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_hada.NetworkModuleHada)
    return True


def test_loha_legacy_fused_qkv_chunked():
    """LoHA on legacy fused attention.qkv emits 3 HadaChunk modules.

    Post-migration the generic LoHA loader uses NetworkModuleHadaChunk for
    equal-chunks dispatch (the chunk class was added in the flux2 PR and is
    now shared infrastructure).
    """
    net = _load_via(Z.try_load_loha, sd_loha_legacy_fused_qkv_skipped())
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
    """OFT loader handles LyCORIS oft_diag files without NoneType-attr errors.

    Regression check for the constraint guard in network_oft.py:58.
    """
    net = _load_via(Z.try_load_oft, sd_oft_lycoris_proj())
    assert net is not None and len(net.modules) == 1
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_oft.NetworkModuleOFT)
    return True


def test_oft_legacy_fused_qkv_skipped():
    """OFT on legacy fused attention.qkv is skipped (no row-sliceable structure)."""
    net = _load_via(Z.try_load_oft, sd_oft_legacy_fused_qkv_skipped())
    assert net is None or len(net.modules) == 0, f'expected no modules, got {net.modules if net else None}'
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_shape():
    """NetworkModuleLora.calc_updown emits the right shape against a target weight."""
    net = _load_via(Z.try_load_lora, sd_lora_bfl_to_q())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA calc_updown')
    return True


def test_lokr_calc_updown_shape():
    """NetworkModuleLokr.calc_updown produces a tensor matching the target's shape."""
    net = _load_via(Z.try_load_lokr, sd_lokr_bfl_adaln())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(ADALN_OUT, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoKR calc_updown')
    return True


def test_lokr_chunk_calc_updown_shape():
    """LokrChunk returns the designated row range; chunk shape matches the split target."""
    net = _load_via(Z.try_load_lokr, sd_lokr_legacy_fused_qkv())
    assert net is not None
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)  # one Q/K/V projection's shape
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LokrChunk calc_updown')
    return True


def test_loha_calc_updown_shape():
    """NetworkModuleHada produces shapes matching the target."""
    net = _load_via(Z.try_load_loha, sd_loha_bfl_proj())
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(HIDDEN, HIDDEN)
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoHA calc_updown')
    return True


def test_oft_calc_updown_shape():
    """NetworkModuleOFT calc_updown shape sanity against the target weight."""
    net = _load_via(Z.try_load_oft, sd_oft_lycoris_proj())
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
        test_lora_bfl_split_qkv,
        test_lora_peft_to_out,
        test_lora_kohya_to_q,
        test_lora_bfl_adaln,
        test_lora_legacy_fused_qkv_chunked,
        test_lora_legacy_attention_out_alias,
        test_lora_dora_threading,
        test_lokr_bfl_adaln,
        test_lokr_legacy_fused_qkv_chunked,
        test_loha_bfl_proj,
        test_loha_legacy_fused_qkv_chunked,
        test_oft_lycoris_no_npe,
        test_oft_legacy_fused_qkv_skipped,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== calc_updown shape sanity ===')
    for fn in [
        test_lora_calc_updown_shape,
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
