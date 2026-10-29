#!/usr/bin/env python
"""
Offline unit tests for Boogu-Image native adapter loaders.

``pipelines.boogu.boogu_lora`` binds every prefix verbatim: the transformer
keeps the official checkpoint names that ComfyUI and its trainers save. These
tests pin that the names in real Boogu LoRAs exist in the module tree,
including the double-stream projections that live on the attention processor.

Save formats exercised, cross-referenced against real Boogu-Image LoRAs:

- comfy (``diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.lora_A.weight``):
  ai-toolkit, every LoRA sampled on CivitAI and the Hub, base and edit.
- LoKR under the same names: ai-toolkit LoKR exports.
- kohya (``lora_unet_single_stream_layers_0_attn_to_q.lora_down.weight``): sd-scripts style.
- diffusers-PEFT (``transformer.single_stream_layers.0.attn.to_q.lora_A.weight``):
  diffusers trainers through the transformer's PEFT mixin.

The reference module tree is the real ``BooguImageTransformer2DModel`` at tiny dims.

No running server required.

Usage:
    python test/test-boogu-native-adapters.py
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
from modules.lora import network, network_lora, network_lokr, network_hada, network_oft  # pylint: disable=wrong-import-position
from modules.lora import native_adapter          # pylint: disable=wrong-import-position
from pipelines.boogu import boogu_lora as B       # pylint: disable=wrong-import-position
from pipelines.boogu.transformer_boogu import BooguImageTransformer2DModel  # pylint: disable=wrong-import-position


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
# Reference Boogu-Image transformer (real class, tiny dims)
# ============================================================
# Upstream: hidden_size=3360, heads=28, kv_heads=7, num_layers=40, num_double_stream_layers=8.

DIM = 64
KV_DIM = 32
DOUBLE = 1
LAYERS = 3

torch.manual_seed(0)
REF = BooguImageTransformer2DModel(
    hidden_size=DIM, num_attention_heads=4, num_kv_heads=2, num_layers=LAYERS, num_double_stream_layers=DOUBLE, num_refiner_layers=1,
    multiple_of=16, axes_dim_rope=(4, 6, 6), instruction_feature_configs={'instruction_feat_dim': 32, 'reduce_type': 'mean', 'num_instruction_feat_layers': 1},
)

# {dotted module name: (out, in)} for every Linear, so synthesizers size tensors to the real module.
LINEAR_SHAPES = {name: tuple(m.weight.shape) for name, m in REF.named_modules() if isinstance(m, nn.Linear)}
LINEAR_NETKEYS = {'lora_transformer_' + name.replace('.', '_') for name in LINEAR_SHAPES}

# Per-block module set of the ai-toolkit LoRAs sampled on CivitAI and the Hub (14 files, LoRA and LoKR, base and edit).
DOUBLE_MODULES = (
    [f'img_feed_forward.linear_{i}' for i in (1, 2, 3)]
    + [f'img_instruct_attn.processor.{p}' for p in ('img_to_q', 'img_to_k', 'img_to_v', 'img_out', 'instruct_to_q', 'instruct_to_k', 'instruct_to_v', 'instruct_out')]
    + ['img_instruct_attn.to_out.0']
    + [f'img_norm{i}.linear' for i in (1, 2, 3)]
    + [f'img_self_attn.{p}' for p in ('to_q', 'to_k', 'to_v', 'to_out.0')]
    + [f'instruct_feed_forward.linear_{i}' for i in (1, 2, 3)]
    + [f'instruct_norm{i}.linear' for i in (1, 2)]
)
SINGLE_MODULES = [f'attn.{p}' for p in ('to_q', 'to_k', 'to_v', 'to_out.0')] + [f'feed_forward.linear_{i}' for i in (1, 2, 3)] + ['norm1.linear']
AITOOLKIT_PATHS = [f'double_stream_layers.{n}.{m}' for n in range(DOUBLE) for m in DOUBLE_MODULES] + [f'single_stream_layers.{n}.{m}' for n in range(LAYERS - DOUBLE) for m in SINGLE_MODULES]


def shape(path: str):
    assert path in LINEAR_SHAPES, f'reference has no Linear {path!r}'
    return LINEAR_SHAPES[path]


def netkey(path: str):
    return 'lora_transformer_' + path.replace('.', '_')


class MockBooguPipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class MockBooguSdModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'BooguImagePipeline'


def install_mock_pipe():
    """Point shared.sd_model at a mock exposing the reference transformer; re-installed per load."""
    sd_model = MockBooguSdModel(MockBooguPipeline(REF))
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


def load_via(try_fn, state_dict, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def delta(net, key, path):
    """Scaled delta of one bound module against the reference weight."""
    mod = net.modules[key]
    mod.network.te_multiplier = 1.0
    mod.network.unet_multiplier = 1.0
    updown, _ = mod.calc_updown(REF.get_submodule(path).weight.detach().clone())
    return updown


def resolve_netkeys(prefix_used, base):
    return ['lora_transformer_' + p.replace('.', '_') for p, _c in native_adapter.resolve_group_targets(B.resolve_targets, prefix_used, base)]


# ============================================================
# Tests - parsing and resolution
# ============================================================

CAT_RESOLVE = category('resolve')


def test_parse_key_all_prefixes():
    cases = [
        ('diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.lora_A.weight', ('diffusion_model.', 'double_stream_layers.0.img_instruct_attn.processor.img_to_q', 'lora_down.weight')),
        ('transformer.single_stream_layers.1.attn.to_k.lora_B.weight', ('transformer.', 'single_stream_layers.1.attn.to_k', 'lora_up.weight')),
        ('lora_unet_single_stream_layers_0_feed_forward_linear_2.lora_down.weight', ('lora_unet_', 'single_stream_layers_0_feed_forward_linear_2', 'lora_down.weight')),
        ('double_stream_layers.0.img_norm1.linear.lora_A.weight', (native_adapter.BARE_DIFFUSERS_PREFIX_USED, 'double_stream_layers.0.img_norm1.linear', 'lora_down.weight')),
    ]
    for key, expected in cases:
        got = native_adapter.parse_key(key, native_adapter.LORA_SUFFIXES, prefixes=B.KNOWN_PREFIXES)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_resolve_verbatim_every_prefix():
    bare = native_adapter.BARE_DIFFUSERS_PREFIX_USED
    for prefix, base in (('diffusion_model.', 'double_stream_layers.0.img_self_attn.to_out.0'), ('transformer.', 'single_stream_layers.0.attn.to_q'),
                         (bare, 'single_stream_layers.0.norm1.linear'), ('lora_unet_', 'double_stream_layers_0_img_instruct_attn_processor_instruct_out')):
        got = native_adapter.resolve_group_targets(B.resolve_targets, prefix, base)
        assert got == [(base, None)], f'{prefix}{base}: {got}'
    return True


def test_processor_projections_are_modules():
    """The double-stream processor is a registered submodule, so its projections carry stamped names."""
    for n in range(DOUBLE):
        for proj in ('img_to_q', 'img_to_k', 'img_to_v', 'img_out', 'instruct_to_q', 'instruct_to_k', 'instruct_to_v', 'instruct_out'):
            shape(f'double_stream_layers.{n}.img_instruct_attn.processor.{proj}')
        assert f'double_stream_layers.{n}.img_instruct_attn.to_q' not in LINEAR_SHAPES, 'instruction attention gained its own qkv'
    return True


def test_resolve_every_module_is_real():
    """Every Linear of the reference, dotted or kohya-underscored, resolves to a stamped network key."""
    for path in LINEAR_SHAPES:
        for prefix, base in (('diffusion_model.', path), ('transformer.', path), ('lora_unet_', path.replace('.', '_'))):
            keys = resolve_netkeys(prefix, base)
            assert keys and set(keys) <= LINEAR_NETKEYS, f'{prefix}{base} -> {keys}'
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def test_lora_aitoolkit_layout():
    """The full ai-toolkit module set binds on every block with no mismatch."""
    sd = {}
    for path in AITOOLKIT_PATHS:
        sd.update(lora_sd(f'diffusion_model.{path}', *lora_factors(*shape(path))))
    net = load_via(B.try_load_lora, sd)
    expected = {netkey(p) for p in AITOOLKIT_PATHS}
    assert net is not None and net.mismatch == 0, f'mismatch={getattr(net, "mismatch", None)}'
    assert set(net.modules) == expected, f'missing {sorted(expected - set(net.modules))[:4]}'
    return True


def test_lora_delta_exact():
    """A grouped-query kv projection gets up @ down scaled by alpha / rank."""
    path = 'double_stream_layers.0.img_instruct_attn.processor.instruct_to_k'
    assert shape(path) == (KV_DIM, DIM)
    down, up = lora_factors(KV_DIM, DIM)
    net = load_via(B.try_load_lora, lora_sd(f'diffusion_model.{path}', down, up, alpha=RANK / 2))
    got = delta(net, netkey(path), path)
    expected = (up @ down) * 0.5
    assert torch.allclose(got, expected, atol=1e-5), f'max diff {(got - expected).abs().max().item()}'
    return True


def test_lora_kohya_alpha():
    path = 'single_stream_layers.1.feed_forward.linear_2'
    sd = lora_sd('lora_unet_' + path.replace('.', '_'), *lora_factors(*shape(path)), suffix=('lora_down.weight', 'lora_up.weight'), alpha=RANK / 4)
    net = load_via(B.try_load_lora, sd)
    assert net is not None and set(net.modules) == {netkey(path)}, f'got {set(net.modules) if net else None}'
    return True


def test_lora_diffusers_peft():
    path = 'double_stream_layers.0.instruct_norm2.linear'
    net = load_via(B.try_load_lora, lora_sd(f'transformer.{path}', *lora_factors(*shape(path))))
    assert net is not None and set(net.modules) == {netkey(path)}, f'got {set(net.modules) if net else None}'
    assert isinstance(next(iter(net.modules.values())), network_lora.NetworkModuleLora)
    return True


def test_lora_refiners_and_embedders():
    """Refiner blocks, embedders and the output norm bind verbatim."""
    paths = ['noise_refiner.0.attn.to_q', 'ref_image_refiner.0.feed_forward.linear_1', 'context_refiner.0.attn.to_out.0',
             'x_embedder', 'ref_image_patch_embedder', 'norm_out.linear_1']
    sd = {}
    for path in paths:
        sd.update(lora_sd(f'diffusion_model.{path}', *lora_factors(*shape(path))))
    net = load_via(B.try_load_lora, sd)
    expected = {netkey(p) for p in paths}
    assert net is not None and set(net.modules) == expected, f'missing {expected - set(net.modules) if net else expected}'
    return True


def test_lokr_aitoolkit():
    path = 'double_stream_layers.0.img_instruct_attn.processor.img_to_v'
    net = load_via(B.try_load_lokr, lokr_sd(f'diffusion_model.{path}', *shape(path)))
    assert net is not None and set(net.modules) == {netkey(path)}, f'got {set(net.modules) if net else None}'
    mod = net.modules[netkey(path)]
    assert isinstance(mod, network_lokr.NetworkModuleLokr), type(mod).__name__
    assert tuple(delta(net, netkey(path), path).shape) == shape(path)
    return True


def test_loha_binds():
    path = 'single_stream_layers.0.attn.to_v'
    net = load_via(B.try_load_loha, loha_sd(f'diffusion_model.{path}', *shape(path)))
    assert net is not None and isinstance(net.modules.get(netkey(path)), network_hada.NetworkModuleHada), f'got {set(net.modules) if net else None}'
    return True


def test_oft_binds():
    path = 'double_stream_layers.0.img_self_attn.to_out.0'
    net = load_via(B.try_load_oft, oft_sd(f'diffusion_model.{path}', shape(path)[0]))
    assert net is not None and isinstance(net.modules.get(netkey(path)), network_oft.NetworkModuleOFT), f'got {set(net.modules) if net else None}'
    return True


def test_chain_merges_families():
    """try_load runs every family and merges LoRA and LoKR groups from one file."""
    lora_path = 'single_stream_layers.0.attn.to_q'
    lokr_path = 'double_stream_layers.0.img_instruct_attn.processor.instruct_out'
    sd = lora_sd(f'diffusion_model.{lora_path}', *lora_factors(*shape(lora_path)))
    sd.update(lokr_sd(f'diffusion_model.{lokr_path}', *shape(lokr_path)))
    net = load_via(B.try_load, sd)
    assert net is not None and set(net.modules) == {netkey(lora_path), netkey(lokr_path)}, f'got {set(net.modules) if net else None}'
    assert isinstance(net, network.Network)
    return True


def test_chain_refuses_mismatch():
    """A factor sized for the wrong module refuses the file."""
    down, up = lora_factors(*shape('single_stream_layers.0.attn.to_q'))
    net = load_via(B.try_load, lora_sd('diffusion_model.single_stream_layers.0.attn.to_k', down, up))
    assert net is None, f'expected refusal, got {set(net.modules)}'
    return True


# ============================================================
# Test runner
# ============================================================


def run_tests():
    t0 = time.time()

    log.warning('=== Parsing and resolution ===')
    for fn in [
        test_parse_key_all_prefixes,
        test_resolve_verbatim_every_prefix,
        test_processor_projections_are_modules,
        test_resolve_every_module_is_real,
    ]:
        run_test(CAT_RESOLVE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_aitoolkit_layout,
        test_lora_delta_exact,
        test_lora_kohya_alpha,
        test_lora_diffusers_peft,
        test_lora_refiners_and_embedders,
        test_lokr_aitoolkit,
        test_loha_binds,
        test_oft_binds,
        test_chain_merges_families,
        test_chain_refuses_mismatch,
    ]:
        run_test(CAT_LOADER, fn)

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
