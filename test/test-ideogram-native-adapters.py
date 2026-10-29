#!/usr/bin/env python
"""
Offline unit tests for Ideogram 4 native adapter loaders.

``pipelines.ideogram.ideogram_lora`` splits the checkpoint's fused
``layers.N.attention.qkv`` into equal q/k/v chunks on the diffusers
``to_q``/``to_k``/``to_v`` modules, renames ``attention.o`` to
``attention.to_out.0`` and binds the rest verbatim.

Save formats exercised, cross-referenced against real Ideogram 4 LoRAs:

- comfy fused (``diffusion_model.layers.0.attention.qkv.lora_A.weight``): ai-toolkit,
  every LoRA sampled on CivitAI (LoRA and LoKR).
- ``conditional_transformer.`` under the same names: the prefix diffusers'
  Ideogram 4 converter strips.
- kohya (``lora_unet_layers_0_attention_qkv.lora_down.weight``): sd-scripts style.
- diffusers-PEFT (``transformer.layers.0.attention.to_q.lora_A.weight``): diffusers
  trainers through the transformer's PEFT mixin.

The reference module tree is the real ``Ideogram4Transformer2DModel`` at tiny dims.
Chunked q/k/v deltas are checked against the same factors applied whole to a
plain ``Linear(dim, 3 * dim)``. A second instance stands in for the
unconditional tower, which carries its own layer names.

No running server required.

Usage:
    python test/test-ideogram-native-adapters.py
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
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from modules.errors import log   # pylint: disable=wrong-import-position
from modules import shared        # pylint: disable=wrong-import-position,unused-import
from modules.lora import network, network_lora, network_lokr, network_hada  # pylint: disable=wrong-import-position
from modules.lora import native_adapter          # pylint: disable=wrong-import-position
from pipelines.ideogram import ideogram_lora as I  # pylint: disable=wrong-import-position
from diffusers.models.transformers.transformer_ideogram4 import Ideogram4Transformer2DModel  # pylint: disable=wrong-import-position


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
# Reference Ideogram 4 transformer (real class, tiny dims)
# ============================================================
# Upstream: 34 layers, 18 heads of 256 (hidden 4608), intermediate 12288, adaln 512.

DIM = 64
FFN = 96
LAYERS = 2

torch.manual_seed(0)
REF = Ideogram4Transformer2DModel(in_channels=8, num_layers=LAYERS, attention_head_dim=16, num_attention_heads=4, intermediate_size=FFN, adaln_dim=32, llm_features_dim=48, mrope_section=(4, 2, 2))

# {dotted module name: (out, in)} for every Linear, so synthesizers size tensors to the real module.
LINEAR_SHAPES = {name: tuple(m.weight.shape) for name, m in REF.named_modules() if isinstance(m, nn.Linear)}
LINEAR_NETKEYS = {'lora_transformer_' + name.replace('.', '_') for name in LINEAR_SHAPES}

# Per-block module set of the ai-toolkit LoRAs on CivitAI (14 files, LoRA and LoKR), and what each binds to.
AITOOLKIT_MODULES = {
    'attention.qkv': ('attention.to_q', 'attention.to_k', 'attention.to_v'),
    'attention.o': ('attention.to_out.0',),
    'feed_forward.w1': ('feed_forward.w1',),
    'feed_forward.w2': ('feed_forward.w2',),
    'feed_forward.w3': ('feed_forward.w3',),
    'adaln_modulation': ('adaln_modulation',),
}
EXTRA_PATHS = ['input_proj', 'llm_cond_proj', 't_embedding.mlp_in', 't_embedding.mlp_out', 'adaln_proj', 'final_layer.linear', 'final_layer.adaln_modulation']


class FusedHolder(nn.Module):
    """A lone ``Linear(dim, 3 * dim)``: the fused reference the chunked q/k/v deltas must add up to."""

    def __init__(self):
        super().__init__()
        self.fused = nn.Linear(DIM, 3 * DIM, bias=False)


FUSED = FusedHolder()


def shape(path: str):
    assert path in LINEAR_SHAPES, f'reference has no Linear {path!r}'
    return LINEAR_SHAPES[path]


def netkey(path: str):
    return 'lora_transformer_' + path.replace('.', '_')


class MockIdeogramPipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class MockIdeogramSdModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'Ideogram4Pipeline'


def install_mock_pipe(transformer=REF):
    """Point shared.sd_model at a mock exposing the reference transformer; re-installed per load."""
    sd_model = MockIdeogramSdModel(MockIdeogramPipeline(transformer))
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# Adapter-tensor synthesizers
# ============================================================

RANK = 8
LOKR_DIM = 4


def lora_factors(out, inp):
    return torch.randn(RANK, inp), torch.randn(out, RANK)


def lora_sd(key_base, down, up, suffix=('lora_A.weight', 'lora_B.weight'), alpha=None):
    sd = {f'{key_base}.{suffix[0]}': down, f'{key_base}.{suffix[1]}': up}
    if alpha is not None:
        sd[f'{key_base}.alpha'] = torch.tensor(float(alpha))
    return sd


def lokr_sd(key_base, out, inp):
    return {
        f'{key_base}.lokr_w1': torch.randn(LOKR_DIM, LOKR_DIM),
        f'{key_base}.lokr_w2': torch.randn(out // LOKR_DIM, inp // LOKR_DIM),
        f'{key_base}.alpha': torch.tensor(float(LOKR_DIM)),
    }


def loha_sd(key_base, out, inp):
    return {
        f'{key_base}.hada_w1_a': torch.randn(out, RANK),
        f'{key_base}.hada_w1_b': torch.randn(RANK, inp),
        f'{key_base}.hada_w2_a': torch.randn(out, RANK),
        f'{key_base}.hada_w2_b': torch.randn(RANK, inp),
        f'{key_base}.alpha': torch.tensor(float(RANK)),
    }


def oft_sd(key_base, out, num_blocks=4):
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
        sd = {k: v.contiguous() for k, v in self.state_dict.items()}
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        safetensors.torch.save_file(sd, self.path)
        return MockNetworkOnDisk(self.path, self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class MockNetworkOnDisk:
    def __init__(self, filename, name):
        self.filename = filename
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'


def load_via(try_fn, state_dict, name='test', transformer=REF):
    install_mock_pipe(transformer)
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def delta(net, key, weight):
    """Scaled delta of one bound module against ``weight``."""
    mod = net.modules[key]
    mod.network.te_multiplier = 1.0
    mod.network.unet_multiplier = 1.0
    updown, _ = mod.calc_updown(weight.detach().clone())
    return updown


def chunked_matches_fused(try_fn, fused_sd_for):
    """A fused block save split onto q/k/v gives the delta the same factors give applied whole to a fused Linear."""
    block = 'diffusion_model.layers.1.attention.qkv'
    sd = fused_sd_for(block)
    net = load_via(try_fn, sd)
    paths = [f'layers.1.attention.{n}' for n in ('to_q', 'to_k', 'to_v')]
    assert net is not None and set(net.modules) == {netkey(p) for p in paths}, f'got {set(net.modules) if net else None}'
    split = torch.cat([delta(net, netkey(p), REF.get_submodule(p).weight) for p in paths], dim=0)
    whole = load_via(try_fn, {k.replace(block, 'diffusion_model.fused'): v for k, v in sd.items()}, transformer=FUSED)
    assert whole is not None, 'reference load on the fused Linear failed'
    expected = delta(whole, 'lora_transformer_fused', FUSED.fused.weight)
    assert split.shape == expected.shape, f'{tuple(split.shape)} vs {tuple(expected.shape)}'
    assert torch.allclose(split, expected, atol=1e-5), f'max diff {(split - expected).abs().max().item()}'
    return net


def resolve_netkeys(prefix_used, base):
    return ['lora_transformer_' + p.replace('.', '_') for p, _c in native_adapter.resolve_group_targets(I.resolve_targets, prefix_used, base)]


# ============================================================
# Tests - parsing and resolution
# ============================================================

CAT_RESOLVE = category('resolve')


def test_parse_key_all_prefixes():
    cases = [
        ('diffusion_model.layers.0.attention.qkv.lora_A.weight', ('diffusion_model.', 'layers.0.attention.qkv', 'lora_down.weight')),
        ('conditional_transformer.layers.1.attention.o.lora_B.weight', ('conditional_transformer.', 'layers.1.attention.o', 'lora_up.weight')),
        ('transformer.layers.0.attention.to_q.lora_A.weight', ('transformer.', 'layers.0.attention.to_q', 'lora_down.weight')),
        ('lora_unet_layers_0_attention_qkv.lora_down.weight', ('lora_unet_', 'layers_0_attention_qkv', 'lora_down.weight')),
        ('layers.0.feed_forward.w2.lora_A.weight', (native_adapter.BARE_DIFFUSERS_PREFIX_USED, 'layers.0.feed_forward.w2', 'lora_down.weight')),
    ]
    for key, expected in cases:
        got = native_adapter.parse_key(key, native_adapter.LORA_SUFFIXES, prefixes=I.KNOWN_PREFIXES)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_resolve_fused_every_spelling():
    """Fused qkv thirds and the ``o`` rename apply under every prefix, dotted or kohya-underscored."""
    bare = native_adapter.BARE_DIFFUSERS_PREFIX_USED
    split = [netkey(f'layers.0.attention.{n}') for n in ('to_q', 'to_k', 'to_v')]
    for prefix, sep in (('diffusion_model.', '.'), ('conditional_transformer.', '.'), ('transformer.', '.'), (bare, '.'), ('lora_unet_', '_')):
        got = resolve_netkeys(prefix, sep.join(('layers', '0', 'attention', 'qkv')))
        assert got == split, f'{prefix} qkv -> {got}'
        chunks = [c for _p, c in native_adapter.resolve_group_targets(I.resolve_targets, prefix, sep.join(('layers', '0', 'attention', 'qkv')))]
        assert [(c.idx, c.total) for c in chunks] == [(0, 3), (1, 3), (2, 3)], f'{prefix} chunks {chunks}'
        got = resolve_netkeys(prefix, sep.join(('layers', '0', 'attention', 'o')))
        assert got == [netkey('layers.0.attention.to_out.0')], f'{prefix} o -> {got}'
    return True


def test_resolve_every_module_is_real():
    """Every Linear of the reference, dotted or kohya-underscored, resolves to a stamped network key."""
    for path in LINEAR_SHAPES:
        for prefix, base in (('diffusion_model.', path), ('conditional_transformer.', path), ('transformer.', path), ('lora_unet_', path.replace('.', '_'))):
            keys = resolve_netkeys(prefix, base)
            assert keys and set(keys) <= LINEAR_NETKEYS, f'{prefix}{base} -> {keys}'
    return True


def test_resolve_aitoolkit_modules_are_real():
    """Every module name the ai-toolkit saves resolves onto modules of the reference."""
    for n in range(LAYERS):
        for module, targets in AITOOLKIT_MODULES.items():
            keys = resolve_netkeys('diffusion_model.', f'layers.{n}.{module}')
            assert keys == [netkey(f'layers.{n}.{t}') for t in targets], f'layers.{n}.{module} -> {keys}'
            assert set(keys) <= LINEAR_NETKEYS, f'layers.{n}.{module} -> {keys}'
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def test_lora_aitoolkit_layout():
    """The full ai-toolkit module set binds eight modules per block with no mismatch."""
    sd = {}
    for n in range(LAYERS):
        for module, targets in AITOOLKIT_MODULES.items():
            out = sum(shape(f'layers.{n}.{t}')[0] for t in targets)
            inp = shape(f'layers.{n}.{targets[0]}')[1]
            sd.update(lora_sd(f'diffusion_model.layers.{n}.{module}', *lora_factors(out, inp)))
    net = load_via(I.try_load_lora, sd)
    expected = {netkey(f'layers.{n}.{t}') for n in range(LAYERS) for targets in AITOOLKIT_MODULES.values() for t in targets}
    assert net is not None and net.mismatch == 0, f'mismatch={getattr(net, "mismatch", None)}'
    assert set(net.modules) == expected and len(expected) == 8 * LAYERS, f'missing {sorted(expected - set(net.modules))[:4]}'
    return True


def test_lora_fused_qkv_chunked():
    """Fused qkv binds three chunked modules with the up rows split in q, k, v order."""
    down, up = lora_factors(3 * DIM, DIM)
    net = load_via(I.try_load_lora, lora_sd('diffusion_model.layers.0.attention.qkv', down, up))
    assert net is not None and len(net.modules) == 3, f'got {set(net.modules) if net else None}'
    for idx, name in enumerate(('to_q', 'to_k', 'to_v')):
        mod = net.modules[netkey(f'layers.0.attention.{name}')]
        assert torch.equal(mod.up_model.weight, up[idx * DIM:(idx + 1) * DIM]), f'{name}: up rows differ'
        assert torch.equal(mod.down_model.weight, down), f'{name}: down differs'
    return True


def test_lora_fused_qkv_matches_fused_module():
    chunked_matches_fused(I.try_load_lora, lambda base: lora_sd(base, *lora_factors(3 * DIM, DIM), alpha=RANK / 2))
    return True


def test_lora_out_delta_exact():
    """``attention.o`` lands on ``to_out.0`` with up @ down scaled by alpha / rank."""
    down, up = lora_factors(DIM, DIM)
    net = load_via(I.try_load_lora, lora_sd('diffusion_model.layers.1.attention.o', down, up, alpha=RANK / 2))
    path = 'layers.1.attention.to_out.0'
    assert net is not None and set(net.modules) == {netkey(path)}, f'got {set(net.modules) if net else None}'
    got = delta(net, netkey(path), REF.get_submodule(path).weight)
    expected = (up @ down) * 0.5
    assert torch.allclose(got, expected, atol=1e-5), f'max diff {(got - expected).abs().max().item()}'
    return True


def test_lora_conditional_transformer_prefix():
    sd = lora_sd('conditional_transformer.layers.0.attention.qkv', *lora_factors(3 * DIM, DIM))
    sd.update(lora_sd('conditional_transformer.layers.0.feed_forward.w2', *lora_factors(*shape('layers.0.feed_forward.w2'))))
    net = load_via(I.try_load_lora, sd)
    expected = {netkey(f'layers.0.attention.{n}') for n in ('to_q', 'to_k', 'to_v')} | {netkey('layers.0.feed_forward.w2')}
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    return True


def test_lora_kohya_fused_qkv():
    sd = lora_sd('lora_unet_layers_1_attention_qkv', *lora_factors(3 * DIM, DIM), suffix=('lora_down.weight', 'lora_up.weight'), alpha=RANK / 4)
    sd.update(lora_sd('lora_unet_layers_1_attention_o', *lora_factors(DIM, DIM), suffix=('lora_down.weight', 'lora_up.weight')))
    net = load_via(I.try_load_lora, sd)
    expected = {netkey(f'layers.1.attention.{n}') for n in ('to_q', 'to_k', 'to_v', 'to_out.0')}
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    return True


def test_lora_diffusers_peft():
    paths = [f'layers.0.attention.{n}' for n in ('to_q', 'to_k', 'to_v', 'to_out.0')] + ['layers.1.adaln_modulation']
    sd = {}
    for path in paths:
        sd.update(lora_sd(f'transformer.{path}', *lora_factors(*shape(path))))
    net = load_via(I.try_load_lora, sd)
    assert net is not None and set(net.modules) == {netkey(p) for p in paths}, f'got {set(net.modules) if net else None}'
    assert all(isinstance(m, network_lora.NetworkModuleLora) for m in net.modules.values())
    return True


def test_lora_embedders_and_final_layer():
    """The embedders, the shared adaln projection and the final layer bind verbatim."""
    sd = {}
    for path in EXTRA_PATHS:
        sd.update(lora_sd(f'diffusion_model.{path}', *lora_factors(*shape(path))))
    net = load_via(I.try_load_lora, sd)
    expected = {netkey(p) for p in EXTRA_PATHS}
    assert net is not None and set(net.modules) == expected, f'missing {expected - set(net.modules) if net else expected}'
    return True


def test_lokr_fused_qkv_matches_fused_module():
    net = chunked_matches_fused(I.try_load_lokr, lambda base: lokr_sd(base, 3 * DIM, DIM))
    for mod in net.modules.values():
        assert isinstance(mod, network_lokr.NetworkModuleLokrChunk), type(mod).__name__
    return True


def test_loha_fused_qkv_matches_fused_module():
    net = chunked_matches_fused(I.try_load_loha, lambda base: loha_sd(base, 3 * DIM, DIM))
    for mod in net.modules.values():
        assert isinstance(mod, network_hada.NetworkModuleHadaChunk), type(mod).__name__
    return True


def test_oft_fused_qkv_skipped():
    """OFT has no chunk variant; a fused qkv binds nothing while a split projection binds."""
    net = load_via(I.try_load_oft, oft_sd('diffusion_model.layers.0.attention.qkv', 3 * DIM))
    assert net is None or len(net.modules) == 0, f'got {set(net.modules)}'
    net = load_via(I.try_load_oft, oft_sd('diffusion_model.layers.0.attention.o', DIM))
    assert net is not None and netkey('layers.0.attention.to_out.0') in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_chain_merges_families():
    """try_load runs every family and merges LoRA and LoKR groups from one file."""
    sd = lora_sd('diffusion_model.layers.0.attention.qkv', *lora_factors(3 * DIM, DIM))
    sd.update(lokr_sd('diffusion_model.layers.1.feed_forward.w1', *shape('layers.1.feed_forward.w1')))
    net = load_via(I.try_load, sd)
    expected = {netkey(f'layers.0.attention.{n}') for n in ('to_q', 'to_k', 'to_v')} | {netkey('layers.1.feed_forward.w1')}
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    assert isinstance(net, network.Network)
    return True


def test_chain_refuses_mismatch():
    """A factor sized for the wrong module refuses the file."""
    down, up = lora_factors(*shape('layers.0.feed_forward.w1'))
    net = load_via(I.try_load, lora_sd('diffusion_model.layers.0.feed_forward.w2', down, up))
    assert net is None, f'expected refusal, got {set(net.modules)}'
    return True


# ============================================================
# Tests - the unconditional tower
# ============================================================

CAT_TOWER = category('tower')


def test_unconditional_tower_stamped_apart():
    """The second transformer carries its own layer names, so a load binds only the conditional one."""
    torch.manual_seed(1)
    uncond = Ideogram4Transformer2DModel(in_channels=8, num_layers=LAYERS, attention_head_dim=16, num_attention_heads=4, intermediate_size=FFN, adaln_dim=32, llm_features_dim=48, mrope_section=(4, 2, 2))
    sd_model = install_mock_pipe()
    sd_model.pipe.unconditional_transformer = uncond
    from modules.lora import lora_convert
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    assert REF.layers[0].attention.to_q.network_layer_name == 'lora_transformer_layers_0_attention_to_q'
    assert uncond.layers[0].attention.to_q.network_layer_name == 'lora_unconditional_transformer_layers_0_attention_to_q'
    assert sd_model.network_layer_mapping['lora_unconditional_transformer_layers_1_feed_forward_w2'] is uncond.layers[1].feed_forward.w2
    with TempLora(lora_sd('diffusion_model.layers.0.attention.qkv', *lora_factors(3 * DIM, DIM))) as nod:
        net = I.try_load('tower', nod, lora_scale=1.0)
    assert net is not None and all(module.sd_module is REF.get_submodule(f'layers.0.attention.{n}') for n, module in zip(('to_q', 'to_k', 'to_v'), net.modules.values())), 'the load must bind the conditional transformer'
    sd_model.pipe.unconditional_transformer = REF # guidance off: the pipeline reuses the conditional transformer for the negative pass
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    assert REF.layers[0].attention.to_q.network_layer_name == 'lora_transformer_layers_0_attention_to_q', 'a reused transformer keeps its own names'
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing and resolution ===')
    for fn in [
        test_parse_key_all_prefixes,
        test_resolve_fused_every_spelling,
        test_resolve_every_module_is_real,
        test_resolve_aitoolkit_modules_are_real,
    ]:
        run_test(CAT_RESOLVE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_aitoolkit_layout,
        test_lora_fused_qkv_chunked,
        test_lora_fused_qkv_matches_fused_module,
        test_lora_out_delta_exact,
        test_lora_conditional_transformer_prefix,
        test_lora_kohya_fused_qkv,
        test_lora_diffusers_peft,
        test_lora_embedders_and_final_layer,
        test_lokr_fused_qkv_matches_fused_module,
        test_loha_fused_qkv_matches_fused_module,
        test_oft_fused_qkv_skipped,
        test_chain_merges_families,
        test_chain_refuses_mismatch,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== Tower ===')
    for fn in [
        test_unconditional_tower_stamped_apart,
    ]:
        run_test(CAT_TOWER, fn)

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
    sys.exit(0 if run_tests() else 1)
