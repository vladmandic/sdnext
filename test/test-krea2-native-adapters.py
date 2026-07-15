#!/usr/bin/env python
"""
Offline unit tests for Krea 2 native adapter loaders.

Krea 2 is the one native arch whose transformer keeps checkpoint-style module
names (``blocks.N.attn.wq``, ``txtfusion.*``, ``first``, ``last`` ...) that
differ from upstream ``diffusers.Krea2Transformer2DModel``
(``transformer_blocks.N.attn.to_q``, ``text_fusion.*``, ``img_in`` ...). So
``pipelines.krea2.krea2_lora`` carries a diffusers -> checkpoint rename that the
other native arches (which load the diffusers class) do not need. These tests
pin that rename plus the resolver-first verbatim fallback in
``modules.lora.native_adapter.resolve_group_targets``.

Save formats exercised, cross-referenced against real Krea 2 LoRAs:

- diffusers-PEFT (``transformer.transformer_blocks.0.attn.to_q.lora_A.weight``):
  the official ``krea/Krea-2-LoRA-*`` releases (renamed to checkpoint names).
- bare diffusers (``transformer_blocks.0.attn.to_q.lora_A.weight``): what
  ``Krea2Transformer2DModel.save_lora_adapter()`` emits (renamed).
- comfy checkpoint (``diffusion_model.blocks.14.attn.wq.lora_A.weight``): the
  CivitAI ecosystem's checkpoint-named LoRAs (bound verbatim).
- kohya (``lora_unet_blocks_0_attn_wq.lora_down.weight``): checkpoint-named,
  reconstructed to dotted paths.
- OneTrainer (``lora_transformer_blocks_0_mlp_up.*``): sdnext's own
  network-key namespace, passthrough-bound.

The reference module tree is the real ``Krea2Transformer2DModel`` at tiny dims,
so module names are authoritative rather than hand-mirrored. Target shapes are
read from that instance, so the loader's shape-match gate is exercised too.

No running server required.

Usage:
    python test/test-krea2-native-adapters.py
"""

import os
import sys
import tempfile
import time

import torch
import torch.nn as nn
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
from modules.lora import native_adapter          # pylint: disable=wrong-import-position
from pipelines.krea2 import krea2_lora as K       # pylint: disable=wrong-import-position
from pipelines.krea2.transformer_krea2 import Krea2Transformer2DModel  # pylint: disable=wrong-import-position


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
# Reference Krea 2 transformer (real class, tiny dims)
# ============================================================
# Upstream: features=6144, heads=48, kvheads=12, multiplier=4, layers=28,
# txtdim=2560, txtlayers=12. Test scale keeps the proportions that matter for
# module naming and shape-arithmetic while staying instant to build.

FEATURES = 64
HEADS = 4
KVHEADS = 2
LAYERS = 2

REF = Krea2Transformer2DModel(
    features=FEATURES, tdim=16, txtdim=32, heads=HEADS, kvheads=KVHEADS,
    multiplier=4, layers=LAYERS, patch=2, channels=16,
    txtlayers=12, txtheads=2, txtkvheads=2,
)

# {checkpoint dotted name: (out, in)} for every Linear, so synthesizers can size
# adapter tensors to the real module and the loader's shape gate is real.
LINEAR_SHAPES = {name: tuple(m.weight.shape) for name, m in REF.named_modules() if isinstance(m, nn.Linear)}
# {network key: module} exactly as assign_network_names_to_compvis_modules stamps it.
LINEAR_NETKEYS = {'lora_transformer_' + name.replace('.', '_') for name in LINEAR_SHAPES}


def ckpt_shape(path: str):
    assert path in LINEAR_SHAPES, f'reference has no Linear {path!r}'
    return LINEAR_SHAPES[path]


# ============================================================
# Mock pipeline wrapping the real transformer
# ============================================================


class _MockKrea2Pipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockKrea2SdModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'Krea2Pipeline'


def install_mock_pipe():
    """Point shared.sd_model at a mock exposing the reference Krea 2 transformer.

    Re-installed per load so prior network_layer_name stamps do not leak.
    """
    pipe = _MockKrea2Pipeline(REF)
    sd_model = _MockKrea2SdModel(pipe)
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# Adapter-tensor synthesizers (sized to the reference modules)
# ============================================================

RANK = 8
LOKR_DIM = 8


def lora_pair(key_base, ckpt_path, suffix=('lora_A.weight', 'lora_B.weight'), alpha=False):
    out, inp = ckpt_shape(ckpt_path)
    sd = {
        f'{key_base}.{suffix[0]}': torch.randn(RANK, inp),
        f'{key_base}.{suffix[1]}': torch.randn(out, RANK),
    }
    if alpha:
        sd[f'{key_base}.alpha'] = torch.tensor(float(RANK))
    return sd


def lokr_pair(key_base, ckpt_path):
    out, inp = ckpt_shape(ckpt_path)
    assert out % LOKR_DIM == 0 and inp % LOKR_DIM == 0
    return {
        f'{key_base}.lokr_w1': torch.randn(LOKR_DIM, LOKR_DIM),
        f'{key_base}.lokr_w2': torch.randn(out // LOKR_DIM, inp // LOKR_DIM),
        f'{key_base}.alpha': torch.tensor(float(LOKR_DIM)),
    }


def loha_pair(key_base, ckpt_path):
    out, inp = ckpt_shape(ckpt_path)
    return {
        f'{key_base}.hada_w1_a': torch.randn(out, RANK),
        f'{key_base}.hada_w1_b': torch.randn(RANK, inp),
        f'{key_base}.hada_w2_a': torch.randn(out, RANK),
        f'{key_base}.hada_w2_b': torch.randn(RANK, inp),
        f'{key_base}.alpha': torch.tensor(float(RANK)),
    }


def oft_pair(key_base, ckpt_path, num_blocks=4):
    out, _inp = ckpt_shape(ckpt_path)
    assert out % num_blocks == 0
    bs = out // num_blocks
    return {
        f'{key_base}.oft_blocks': torch.randn(num_blocks, bs, bs) * 0.01,
        f'{key_base}.oft_diag': torch.ones(num_blocks, bs),
        f'{key_base}.alpha': torch.tensor(0.001),
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


def _load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def resolve_netkeys(state_dict):
    """Return the set of network keys a state dict resolves to (rename applied)."""
    groups = K.group_by_suffixes(state_dict, K.LORA_SUFFIXES)
    keys = set()
    for (prefix, base), _w in groups.items():
        for path, _chunk in native_adapter.resolve_group_targets(K.resolve_targets, prefix, base):
            keys.add('lora_transformer_' + path.replace('.', '_'))
    return keys


# ============================================================
# Tests - parsing primitives
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_all_prefixes():
    bd = K.BARE_DIFFUSERS_PREFIX_USED

    def parse(key):
        return K.parse_key(key, K.LORA_SUFFIXES)
    cases = [
        # diffusers-PEFT: 'transformer.' stripped, base is the diffusers path
        ('transformer.transformer_blocks.0.attn.to_q.lora_A.weight',
         ('transformer.', 'transformer_blocks.0.attn.to_q', 'lora_down.weight')),
        # comfy checkpoint
        ('diffusion_model.blocks.0.attn.wq.lora_B.weight',
         ('diffusion_model.', 'blocks.0.attn.wq', 'lora_up.weight')),
        # kohya checkpoint
        ('lora_unet_blocks_0_attn_wq.lora_down.weight',
         ('lora_unet_', 'blocks_0_attn_wq', 'lora_down.weight')),
        # bare diffusers (save_lora_adapter output)
        ('transformer_blocks.0.ff.gate.lora_A.weight',
         (bd, 'transformer_blocks.0.ff.gate', 'lora_down.weight')),
        ('text_fusion.projector.lora_A.weight',
         (bd, 'text_fusion.projector', 'lora_down.weight')),
        ('random.unrelated.key', None),
    ]
    for key, expected in cases:
        got = parse(key)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_marker_disambiguation():
    pure_lora = {
        'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.zeros(1, 1),
        'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.zeros(1, 1),
    }
    assert K.has_marker(pure_lora, K.LORA_MARKERS)
    assert not K.has_marker(pure_lora, K.LOKR_MARKERS)
    assert not K.has_marker(pure_lora, K.LOHA_MARKERS)

    pure_lokr = {
        'diffusion_model.blocks.0.attn.wq.lokr_w1': torch.zeros(1, 1),
        'diffusion_model.blocks.0.attn.wq.lokr_w2': torch.zeros(1, 1),
    }
    assert K.has_marker(pure_lokr, K.LOKR_MARKERS)
    assert not K.has_marker(pure_lokr, K.LORA_MARKERS)
    return True


# ============================================================
# Tests - rename resolution (the arch-specific crux)
# ============================================================

CAT_RESOLVE = category('resolve')


def test_resolve_block_attn_leaf_renames():
    """Diffusers attn/ff leaves rewrite to checkpoint leaves under 'transformer.'."""
    pairs = {
        'transformer_blocks.0.attn.to_q': 'blocks.0.attn.wq',
        'transformer_blocks.0.attn.to_k': 'blocks.0.attn.wk',
        'transformer_blocks.0.attn.to_v': 'blocks.0.attn.wv',
        'transformer_blocks.0.attn.to_gate': 'blocks.0.attn.gate',
        'transformer_blocks.1.attn.to_out.0': 'blocks.1.attn.wo',
        'transformer_blocks.0.ff.gate': 'blocks.0.mlp.gate',
        'transformer_blocks.0.ff.up': 'blocks.0.mlp.up',
        'transformer_blocks.1.ff.down': 'blocks.1.mlp.down',
    }
    for diff_base, ckpt in pairs.items():
        got = native_adapter.resolve_group_targets(K.resolve_targets, 'transformer.', diff_base)
        assert got == [(ckpt, None)], f'{diff_base} -> {got}, expected {ckpt}'
    return True


def test_resolve_text_fusion_renames():
    """text_fusion.* rewrites to txtfusion.*; projector keeps its name."""
    pairs = {
        'text_fusion.layerwise_blocks.0.attn.to_q': 'txtfusion.layerwise_blocks.0.attn.wq',
        'text_fusion.refiner_blocks.1.ff.down': 'txtfusion.refiner_blocks.1.mlp.down',
        'text_fusion.projector': 'txtfusion.projector',
    }
    for diff_base, ckpt in pairs.items():
        got = native_adapter.resolve_group_targets(K.resolve_targets, 'transformer.', diff_base)
        assert got == [(ckpt, None)], f'{diff_base} -> {got}, expected {ckpt}'
    return True


def test_resolve_non_block_extras():
    """Embedders / MLP-sequential / final layer map to their checkpoint paths."""
    for diff_base, ckpt in K.DIFFUSERS_EXTRA_MAP.items():
        got = native_adapter.resolve_group_targets(K.resolve_targets, 'transformer.', diff_base)
        assert got == [(ckpt, None)], f'{diff_base} -> {got}, expected {ckpt}'
    # every renamed extra is a real Linear on the reference transformer
    for ckpt in K.DIFFUSERS_EXTRA_MAP.values():
        assert ckpt in LINEAR_SHAPES, f'extra target {ckpt!r} is not a reference Linear'
    return True


def test_resolve_checkpoint_names_verbatim():
    """Checkpoint-named bases fall through to the shared verbatim binding."""
    # comfy dotted (diffusion_model.) and transformer.-prefixed checkpoint names
    for prefix in ('diffusion_model.', 'transformer.'):
        got = native_adapter.resolve_group_targets(K.resolve_targets, prefix, 'blocks.14.attn.wq')
        assert got == [('blocks.14.attn.wq', None)], f'{prefix} verbatim -> {got}'
    # OneTrainer / lycoris passthrough
    got = native_adapter.resolve_group_targets(K.resolve_targets, 'lora_transformer_', 'blocks_0_mlp_up')
    assert got == [('blocks_0_mlp_up', None)], f'lora_transformer_ passthrough -> {got}'
    return True


def test_resolve_every_official_module_is_real():
    """A full official-style block-0 + extras set resolves onto real Linears only."""
    diff_bases = [
        'transformer_blocks.0.attn.to_q', 'transformer_blocks.0.attn.to_k',
        'transformer_blocks.0.attn.to_v', 'transformer_blocks.0.attn.to_gate',
        'transformer_blocks.0.attn.to_out.0', 'transformer_blocks.0.ff.gate',
        'transformer_blocks.0.ff.up', 'transformer_blocks.0.ff.down',
        'text_fusion.layerwise_blocks.0.attn.to_q', 'text_fusion.projector',
        'img_in', 'txt_in.linear_1', 'txt_in.linear_2', 'time_embed.linear_1',
        'time_embed.linear_2', 'time_mod_proj', 'final_layer.linear',
    ]
    sd = {}
    for b in diff_bases:
        sd.update(lora_pair(f'transformer.{b}', _diff_to_ckpt(b)))
    netkeys = resolve_netkeys(sd)
    missing = netkeys - LINEAR_NETKEYS
    assert not missing, f'unmapped: {missing}'
    assert len(netkeys) == len(diff_bases), f'got {len(netkeys)} keys for {len(diff_bases)} modules'
    return True


def _diff_to_ckpt(diff_base):
    """Test-side mirror of resolve_targets, to size synthetic tensors."""
    got = native_adapter.resolve_group_targets(K.resolve_targets, 'transformer.', diff_base)
    return got[0][0]


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def test_lora_official_diffusers_renamed():
    """Official diffusers-PEFT key renames and binds onto the checkpoint module."""
    sd = lora_pair('transformer.transformer_blocks.0.attn.to_q', 'blocks.0.attn.wq', alpha=True)
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and len(net.modules) == 1, f'got {net.modules if net else None}'
    assert 'lora_transformer_blocks_0_attn_wq' in net.modules, f'got {set(net.modules)}'
    assert isinstance(next(iter(net.modules.values())), network_lora.NetworkModuleLora)
    return True


def test_lora_bare_diffusers_renamed():
    """Bare diffusers key (save_lora_adapter output) renames and binds."""
    sd = lora_pair('transformer_blocks.1.ff.up', 'blocks.1.mlp.up')
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and 'lora_transformer_blocks_1_mlp_up' in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_lora_comfy_checkpoint_verbatim():
    """CivitAI comfy checkpoint key binds verbatim (no rename)."""
    sd = lora_pair('diffusion_model.blocks.0.attn.gate', 'blocks.0.attn.gate')
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and 'lora_transformer_blocks_0_attn_gate' in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_lora_kohya_checkpoint():
    """Kohya checkpoint underscore key reconstructs to the dotted path and binds."""
    out, inp = ckpt_shape('blocks.0.attn.wq')
    sd = {
        'lora_unet_blocks_0_attn_wq.lora_down.weight': torch.randn(RANK, inp),
        'lora_unet_blocks_0_attn_wq.lora_up.weight': torch.randn(out, RANK),
        'lora_unet_blocks_0_attn_wq.alpha': torch.tensor(float(RANK)),
    }
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and 'lora_transformer_blocks_0_attn_wq' in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_lora_kohya_compound_block_names():
    """Kohya reconstruction protects the compound layerwise_blocks / refiner_blocks names."""
    out, inp = ckpt_shape('txtfusion.refiner_blocks.1.mlp.down')
    sd = {
        'lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_down.weight': torch.randn(RANK, inp),
        'lora_unet_txtfusion_refiner_blocks_1_mlp_down.lora_up.weight': torch.randn(out, RANK),
    }
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and 'lora_transformer_txtfusion_refiner_blocks_1_mlp_down' in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_lora_onetrainer_passthrough():
    """OneTrainer lora_transformer_ keys bind via the shared passthrough, no rename."""
    out, inp = ckpt_shape('blocks.0.mlp.up')
    sd = {
        'lora_transformer_blocks_0_mlp_up.lora_down.weight': torch.randn(RANK, inp),
        'lora_transformer_blocks_0_mlp_up.lora_up.weight': torch.randn(out, RANK),
    }
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and 'lora_transformer_blocks_0_mlp_up' in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_lora_extras_renamed():
    """Non-block extras rename and bind (img_in->first, time_embed->tmlp, final_layer->last)."""
    sd = {}
    sd.update(lora_pair('transformer.img_in', 'first'))
    sd.update(lora_pair('transformer.time_embed.linear_1', 'tmlp.0'))
    sd.update(lora_pair('transformer.final_layer.linear', 'last.linear'))
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and len(net.modules) == 3, f'got {set(net.modules) if net else None}'
    assert set(net.modules) == {
        'lora_transformer_first', 'lora_transformer_tmlp_0', 'lora_transformer_last_linear',
    }, f'got {set(net.modules)}'
    return True


def test_lora_dora_threading():
    """dora_scale flows onto NetworkModuleLora.dora_scale after the rename."""
    out, _inp = ckpt_shape('blocks.0.mlp.down')
    sd = lora_pair('transformer.transformer_blocks.0.ff.down', 'blocks.0.mlp.down')
    sd['transformer.transformer_blocks.0.ff.down.dora_scale'] = torch.randn(out)
    net = _load_via(K.try_load_lora, sd)
    assert net is not None and len(net.modules) == 1
    assert next(iter(net.modules.values())).dora_scale is not None
    return True


def test_lokr_diffusers_renamed():
    """Diffusers-named LoKR renames and binds via NetworkModuleLokr (no chunk variant)."""
    sd = lokr_pair('transformer.transformer_blocks.0.ff.gate', 'blocks.0.mlp.gate')
    net = _load_via(K.try_load_lokr, sd)
    assert net is not None and 'lora_transformer_blocks_0_mlp_gate' in net.modules, f'got {set(net.modules) if net else None}'
    mod = next(iter(net.modules.values()))
    assert isinstance(mod, network_lokr.NetworkModuleLokr)
    assert not isinstance(mod, network_lokr.NetworkModuleLokrChunk)
    return True


def test_loha_diffusers_renamed():
    """Diffusers-named LoHA renames and binds via NetworkModuleHada."""
    sd = loha_pair('transformer.transformer_blocks.0.attn.to_q', 'blocks.0.attn.wq')
    net = _load_via(K.try_load_loha, sd)
    assert net is not None and 'lora_transformer_blocks_0_attn_wq' in net.modules, f'got {set(net.modules) if net else None}'
    assert isinstance(next(iter(net.modules.values())), network_hada.NetworkModuleHada)
    return True


def test_oft_diffusers_renamed():
    """Diffusers-named OFT renames and binds via NetworkModuleOFT without NoneType errors."""
    sd = oft_pair('transformer.transformer_blocks.0.attn.to_out.0', 'blocks.0.attn.wo')
    net = _load_via(K.try_load_oft, sd)
    assert net is not None and 'lora_transformer_blocks_0_attn_wo' in net.modules, f'got {set(net.modules) if net else None}'
    assert isinstance(next(iter(net.modules.values())), network_oft.NetworkModuleOFT)
    return True


def test_full_diff_chain():
    """Full-diff extraction loads through the try_load chain and yields finite updown."""
    out, inp = ckpt_shape('blocks.0.attn.wq')
    sd = {
        'transformer.transformer_blocks.0.attn.to_q.diff': torch.randn(out, inp),
        'transformer.transformer_blocks.0.attn.to_q.diff_b': torch.randn(out),
    }
    net = _load_via(K.try_load, sd)
    assert net is not None and 'lora_transformer_blocks_0_attn_wq' in net.modules, f'got {set(net.modules) if net else None}'
    mod = next(iter(net.modules.values()))
    updown, ex_bias = mod.calc_updown(mod.sd_module.weight)
    assert tuple(updown.shape) == (out, inp) and torch.isfinite(updown).all()
    assert ex_bias is not None and tuple(ex_bias.shape) == (out,)
    return True


def test_chain_merges_multiple_families():
    """try_load merges LoRA + LoKR groups from one file onto distinct modules."""
    sd = {}
    sd.update(lora_pair('transformer.transformer_blocks.0.attn.to_q', 'blocks.0.attn.wq'))
    sd.update(lokr_pair('transformer.transformer_blocks.0.ff.gate', 'blocks.0.mlp.gate'))
    net = _load_via(K.try_load, sd)
    assert net is not None and len(net.modules) == 2, f'got {set(net.modules) if net else None}'
    assert set(net.modules) == {'lora_transformer_blocks_0_attn_wq', 'lora_transformer_blocks_0_mlp_gate'}
    return True


# ============================================================
# Tests - calc_updown shape sanity
# ============================================================

CAT_MATH = category('math')


def test_lora_calc_updown_shape():
    net = _load_via(K.try_load_lora, lora_pair('transformer.transformer_blocks.0.attn.to_q', 'blocks.0.attn.wq'))
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(*ckpt_shape('blocks.0.attn.wq'))
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoRA calc_updown')
    return True


def test_lokr_calc_updown_shape():
    net = _load_via(K.try_load_lokr, lokr_pair('transformer.transformer_blocks.0.ff.gate', 'blocks.0.mlp.gate'))
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(*ckpt_shape('blocks.0.mlp.gate'))
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoKR calc_updown')
    return True


def test_loha_calc_updown_shape():
    net = _load_via(K.try_load_loha, loha_pair('transformer.transformer_blocks.0.attn.to_q', 'blocks.0.attn.wq'))
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(*ckpt_shape('blocks.0.attn.wq'))
    updown, _ = mod.calc_updown(target)
    assert_shape(updown, target.shape, label='LoHA calc_updown')
    return True


def test_oft_calc_updown_shape():
    net = _load_via(K.try_load_oft, oft_pair('transformer.transformer_blocks.0.attn.to_out.0', 'blocks.0.attn.wo'))
    mod = make_network_for_module(next(iter(net.modules.values())))
    target = torch.randn(*ckpt_shape('blocks.0.attn.wo'))
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

    log.warning('=== Rename resolution ===')
    for fn in [
        test_resolve_block_attn_leaf_renames,
        test_resolve_text_fusion_renames,
        test_resolve_non_block_extras,
        test_resolve_checkpoint_names_verbatim,
        test_resolve_every_official_module_is_real,
    ]:
        run_test(CAT_RESOLVE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_official_diffusers_renamed,
        test_lora_bare_diffusers_renamed,
        test_lora_comfy_checkpoint_verbatim,
        test_lora_kohya_checkpoint,
        test_lora_kohya_compound_block_names,
        test_lora_onetrainer_passthrough,
        test_lora_extras_renamed,
        test_lora_dora_threading,
        test_lokr_diffusers_renamed,
        test_loha_diffusers_renamed,
        test_oft_diffusers_renamed,
        test_full_diff_chain,
        test_chain_merges_multiple_families,
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
