#!/usr/bin/env python
"""
Offline unit tests for MOSS-Image native adapter loaders.

``pipelines.moss.moss_lora`` binds the official module names verbatim and
splits the fused ``layers.N.proj.qkv`` that ComfyUI-format trainers save into
equal q/k/v chunks. These tests pin both, plus the per-family chunk modules.

Save formats exercised, cross-referenced against real MOSS-Image LoRAs:

- comfy fused (``diffusion_model.layers.0.proj.qkv.lora_A.weight``): ai-toolkit
  0.13.29-0.13.30 and the kohya ``networks.lora`` exports on CivitAI.
- comfy split (``diffusion_model.layers.0.proj.q.lora_A.weight``): later
  ai-toolkit releases, Turbo LoRAs.
- kohya (``lora_unet_layers_0_proj_q.lora_down.weight``): sd-scripts, Fizgig.
- diffusers-PEFT (``transformer.layers.0.proj.q.lora_A.weight``): SimpleTuner
  and the diffusers trainers.
- LoKR on the fused qkv: ai-toolkit and LyCORIS exports.

The reference module tree is the real ``MOSSImageTransformer`` at tiny dims.
Its text embedder carries a natively fused ``qkv`` Linear of the same
``[3 * dim, dim]`` shape as a block's fused save, so the chunked block deltas
are checked against the same factors applied whole to that module.

No running server required.

Usage:
    python test/test-moss-native-adapters.py
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
from modules import shared        # pylint: disable=wrong-import-position,unused-import
from modules.lora import network, network_lora, network_lokr, network_hada  # pylint: disable=wrong-import-position
from modules.lora import native_adapter          # pylint: disable=wrong-import-position
from modules.lora.native_adapter import ChunkSpec  # pylint: disable=wrong-import-position
from pipelines.moss import moss_lora as M         # pylint: disable=wrong-import-position
from pipelines.moss.moss_image import MOSSImageTransformer  # pylint: disable=wrong-import-position
from pipelines.moss.moss_transformer_2d import MOSSImageTransformer2DModel  # pylint: disable=wrong-import-position


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
        test_result = fn()
        if test_result is False:
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
# Reference MOSS-Image transformer (real class, tiny dims)
# ============================================================
# Upstream: dim=3072, ffn_dim=8192, heads=24, layers=42, text_dim=4096.

DIM = 64
FFN = 96
LAYERS = 2

torch.manual_seed(0)
REF = MOSSImageTransformer(dim=DIM, ffn_dim=FFN, num_heads=4, num_layers=LAYERS, text_dim=32, freq_dim=16, latent_channels=8)
REF_UPSTREAM = MOSSImageTransformer2DModel(
    in_channels=8,
    num_layers=LAYERS,
    num_refiner_layers=2,
    num_attention_heads=4,
    attention_head_dim=16,
    ffn_dim=FFN,
    text_embed_dim=32,
    timestep_embed_dim=16,
)

# {dotted module name: (out, in)} for every Linear, so synthesizers size tensors to the real module.
LINEAR_SHAPES = {name: tuple(m.weight.shape) for name, m in REF.named_modules() if isinstance(m, nn.Linear)}
LINEAR_NETKEYS = {'lora_transformer_' + name.replace('.', '_') for name in LINEAR_SHAPES}
FUSED_REF = 'text_embedder.blocks.0.qkv'


def shape(path: str):
    assert path in LINEAR_SHAPES, f'reference has no Linear {path!r}'
    return LINEAR_SHAPES[path]


class _MockMossPipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockMossSdModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'MOSSImagePipeline'


def install_mock_pipe(transformer=REF):
    """Point shared.sd_model at a mock exposing the reference transformer; re-installed per load."""
    sd_model = _MockMossSdModel(_MockMossPipeline(transformer))
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


def rekey(sd, old, new):
    """Copy of ``sd`` with the module base ``old`` renamed to ``new``."""
    return {k.replace(old, new, 1): v for k, v in sd.items()}


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


def load_via(try_fn, state_dict, name='test', transformer=REF):
    install_mock_pipe(transformer)
    with TempLora(state_dict, name=name) as nod:
        return try_fn(name, nod, lora_scale=1.0)


def delta(net, netkey, path):
    """Scaled delta of one bound module against the reference weight."""
    mod = net.modules[netkey]
    mod.network.te_multiplier = 1.0
    mod.network.unet_multiplier = 1.0
    updown, _ = mod.calc_updown(REF.get_submodule(path).weight.detach().clone())
    return updown


def chunked_matches_fused(try_fn, fused_sd_for):
    """Fused block save split onto q/k/v gives the same delta as the factors applied whole to the fused text-embedder qkv."""
    block = 'diffusion_model.layers.1.proj.qkv'
    sd = fused_sd_for(block)
    net = load_via(try_fn, sd)
    keys = [f'lora_transformer_layers_1_proj_{n}' for n in ('q', 'k', 'v')]
    assert net is not None and set(net.modules) == set(keys), f'got {set(net.modules) if net else None}'
    split = torch.cat([delta(net, k, f'layers.1.proj.{k[-1]}') for k in keys], dim=0)
    whole = load_via(try_fn, rekey(sd, block, f'diffusion_model.{FUSED_REF}'))
    assert whole is not None, 'reference load on the fused text-embedder qkv failed'
    expected = delta(whole, 'lora_transformer_' + FUSED_REF.replace('.', '_'), FUSED_REF)
    assert split.shape == expected.shape, f'{tuple(split.shape)} vs {tuple(expected.shape)}'
    assert torch.allclose(split, expected, atol=1e-5), f'max diff {(split - expected).abs().max().item()}'
    return net


def resolve_netkeys(prefix_used, base):
    return ['lora_transformer_' + p.replace('.', '_') for p, _c in native_adapter.resolve_group_targets(M.resolve_targets, prefix_used, base)]


# ============================================================
# Tests - parsing and resolution
# ============================================================

CAT_RESOLVE = category('resolve')


def test_parse_key_all_prefixes():
    cases = [
        ('diffusion_model.layers.0.proj.qkv.lora_A.weight', ('diffusion_model.', 'layers.0.proj.qkv', 'lora_down.weight')),
        ('transformer.layers.0.proj.q.lora_B.weight', ('transformer.', 'layers.0.proj.q', 'lora_up.weight')),
        ('lora_unet_layers_0_proj_ffn_up_gate.lora_down.weight', ('lora_unet_', 'layers_0_proj_ffn_up_gate', 'lora_down.weight')),
        ('layers.1.proj.o.lora_A.weight', (native_adapter.BARE_DIFFUSERS_PREFIX_USED, 'layers.1.proj.o', 'lora_down.weight')),
    ]
    for key, expected in cases:
        got = native_adapter.parse_key(key, native_adapter.LORA_SUFFIXES, prefixes=M.KNOWN_PREFIXES)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_resolve_fused_qkv_every_spelling():
    install_mock_pipe(REF)
    bare = native_adapter.BARE_DIFFUSERS_PREFIX_USED
    for prefix, base, sep in (('diffusion_model.', 'layers.3.proj.qkv', '.'), ('transformer.', 'layers.3.proj.qkv', '.'), (bare, 'layers.3.proj.qkv', '.'), ('lora_unet_', 'layers_3_proj_qkv', '_')):
        got = native_adapter.resolve_group_targets(M.resolve_targets, prefix, base)
        expected = [(f'layers{sep}3{sep}proj{sep}{name}', ChunkSpec(idx=i, total=3)) for i, name in enumerate(('q', 'k', 'v'))]
        assert got == expected, f'{prefix}{base}: {got}'
    return True


def test_resolve_text_embedder_qkv_verbatim():
    """The text embedder's qkv is fused in the model, so it is not split."""
    for prefix, base in (('diffusion_model.', 'text_embedder.blocks.0.qkv'), ('lora_unet_', 'text_embedder_blocks_0_qkv'), ('transformer.', 'text_embedder.blocks.0.qkv')):
        got = native_adapter.resolve_group_targets(M.resolve_targets, prefix, base)
        assert got == [(base, None)], f'{prefix}{base}: {got}'
    return True


def test_resolve_upstream_refiner_qkv_verbatim():
    install_mock_pipe(REF_UPSTREAM)
    bare = native_adapter.BARE_DIFFUSERS_PREFIX_USED
    cases = (
        ('diffusion_model.', 'context_refiner.0.attn.to_qkv'),
        ('transformer.', 'context_refiner.1.attn.to_qkv'),
        (bare, 'context_refiner.0.attn.to_qkv'),
        ('lora_unet_', 'context_refiner_1_attn_to_qkv'),
    )
    for prefix, base in cases:
        assert native_adapter.resolve_group_targets(M.resolve_targets, prefix, base) == [(base, None)]
    for prefix, base, sep in (('diffusion_model.', 'layers.3.proj.qkv', '.'), ('lora_unet_', 'layers_3_proj_qkv', '_')):
        got = native_adapter.resolve_group_targets(M.resolve_targets, prefix, base)
        expected = [
            (f'transformer_blocks{sep}3{sep}attn{sep}to_{name}', ChunkSpec(idx=idx, total=3))
            for idx, name in enumerate(('q', 'k', 'v'))
        ]
        assert got == expected, f'{prefix}{base}: {got}'
    return True


def test_resolve_every_module_is_real():
    """Every Linear of the reference, dotted or kohya-underscored, resolves to a stamped network key."""
    install_mock_pipe(REF)
    for path in LINEAR_SHAPES:
        for prefix, base in (('diffusion_model.', path), ('transformer.', path), ('lora_unet_', path.replace('.', '_'))):
            keys = resolve_netkeys(prefix, base)
            legacy_key = 'lora_transformer_' + path.replace('.', '_')
            assert keys == [legacy_key] and legacy_key in LINEAR_NETKEYS, f'{prefix}{base} -> {keys}'
    return True


def test_resolve_every_upstream_module_is_real():
    """Every Linear in the upstream tree resolves to its exact named target."""
    install_mock_pipe(REF_UPSTREAM)
    linear_paths = {
        name for name, module in REF_UPSTREAM.named_modules() if isinstance(module, nn.Linear)
    }
    for path in linear_paths:
        for prefix, base in (('diffusion_model.', path), ('transformer.', path), ('lora_unet_', path.replace('.', '_'))):
            targets = native_adapter.resolve_group_targets(M.resolve_targets, prefix, base)
            assert targets == [(base, None)], f'{prefix}{base} -> {targets}'
    return True


# ============================================================
# Tests - loaders end-to-end
# ============================================================

CAT_LOADER = category('loader')


def test_lora_fused_qkv_chunked():
    """Fused qkv binds three chunked modules with the up rows split in q, k, v order."""
    down, up = lora_factors(3 * DIM, DIM)
    net = load_via(M.try_load_lora, lora_sd('diffusion_model.layers.0.proj.qkv', down, up))
    assert net is not None and len(net.modules) == 3, f'got {set(net.modules) if net else None}'
    for idx, name in enumerate(('q', 'k', 'v')):
        mod = net.modules[f'lora_transformer_layers_0_proj_{name}']
        assert torch.equal(mod.up_model.weight, up[idx * DIM:(idx + 1) * DIM]), f'{name}: up rows differ'
        assert torch.equal(mod.down_model.weight, down), f'{name}: down differs'
    return True


def test_lora_fused_qkv_matches_fused_module():
    chunked_matches_fused(M.try_load_lora, lambda base: lora_sd(base, *lora_factors(3 * DIM, DIM), alpha=RANK / 2))
    return True


def test_lora_comfy_split():
    sd = {}
    for name in ('q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down'):
        sd.update(lora_sd(f'diffusion_model.layers.1.proj.{name}', *lora_factors(*shape(f'layers.1.proj.{name}'))))
    net = load_via(M.try_load_lora, sd)
    assert net is not None and len(net.modules) == 6 and net.mismatch == 0, f'got {set(net.modules) if net else None}'
    return True


def test_lora_kohya_alpha():
    down, up = lora_factors(*shape('layers.0.proj.ffn_up_gate'))
    sd = lora_sd('lora_unet_layers_0_proj_ffn_up_gate', down, up, suffix=('lora_down.weight', 'lora_up.weight'), alpha=RANK / 4)
    net = load_via(M.try_load_lora, sd)
    assert net is not None and 'lora_transformer_layers_0_proj_ffn_up_gate' in net.modules, f'got {set(net.modules) if net else None}'
    return True


def test_lora_kohya_fused_qkv():
    sd = lora_sd('lora_unet_layers_1_proj_qkv', *lora_factors(3 * DIM, DIM), suffix=('lora_down.weight', 'lora_up.weight'))
    net = load_via(M.try_load_lora, sd)
    expected = {f'lora_transformer_layers_1_proj_{n}' for n in ('q', 'k', 'v')}
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    return True


def test_lora_diffusers_peft():
    sd = lora_sd('transformer.layers.0.proj.o', *lora_factors(*shape('layers.0.proj.o')))
    net = load_via(M.try_load_lora, sd)
    assert net is not None and 'lora_transformer_layers_0_proj_o' in net.modules, f'got {set(net.modules) if net else None}'
    assert isinstance(next(iter(net.modules.values())), network_lora.NetworkModuleLora)
    return True


def test_lora_extras():
    """Text embedder, time embedding, modulation and head modules bind verbatim."""
    paths = ['text_embedder.proj', FUSED_REF, 'text_embedder.blocks.1.ffn_gate', 'time_embedding.0', 'time_embedding.2',
             'shared_block_modulation', 'video_embedder', 'final_linear_video', 'video_head_modulation']
    sd = {}
    for path in paths:
        sd.update(lora_sd(f'diffusion_model.{path}', *lora_factors(*shape(path))))
    net = load_via(M.try_load_lora, sd)
    expected = {'lora_transformer_' + p.replace('.', '_') for p in paths}
    assert net is not None and set(net.modules) == expected, f'missing {expected - set(net.modules) if net else expected}'
    return True


def test_upstream_lora_targets_attach():
    paths = (
        ('diffusion_model.', 'context_refiner.0.attn.to_qkv', 'context_refiner.0.attn.to_qkv'),
        ('transformer.', 'transformer_blocks.0.attn.to_q', 'transformer_blocks.0.attn.to_q'),
        ('lora_unet_', 'transformer_blocks_1_ff_in', 'transformer_blocks.1.ff_in'),
        ('diffusion_model.', 'time_embedder.linear_1', 'time_embedder.linear_1'),
    )
    state_dict = {}
    expected = set()
    for prefix, path, module_path in paths:
        module = REF_UPSTREAM.get_submodule(module_path)
        down, up = lora_factors(*module.weight.shape)
        state_dict.update(lora_sd(prefix + path, down, up))
        expected.add('lora_transformer_' + module_path.replace('.', '_'))
    net = load_via(M.try_load_lora, state_dict, transformer=REF_UPSTREAM)
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    assert net.mismatch == 0, f'mismatches: {net.mismatch}'
    return True


def test_legacy_comfy_lora_targets_attach_upstream():
    state_dict = {}
    expected = set()
    for projection in ('q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down'):
        legacy_path = f'layers.0.proj.{projection}'
        if projection in ('q', 'k', 'v'):
            target_path = f'transformer_blocks.0.attn.to_{projection}'
        elif projection == 'o':
            target_path = 'transformer_blocks.0.attn.to_out'
        elif projection == 'ffn_up_gate':
            target_path = 'transformer_blocks.0.ff_in'
        else:
            target_path = 'transformer_blocks.0.ff_out'
        module = REF_UPSTREAM.get_submodule(target_path)
        down, up = lora_factors(*module.weight.shape)
        state_dict.update(lora_sd(f'diffusion_model.{legacy_path}', down, up))
        expected.add('lora_transformer_' + target_path.replace('.', '_'))

    net = load_via(M.try_load_lora, state_dict, transformer=REF_UPSTREAM)
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    assert net.mismatch == 0, f'mismatches: {net.mismatch}'
    return True


def test_lokr_fused_qkv_matches_fused_module():
    net = chunked_matches_fused(M.try_load_lokr, lambda base: lokr_sd(base, 3 * DIM, DIM))
    for mod in net.modules.values():
        assert isinstance(mod, network_lokr.NetworkModuleLokrChunk), type(mod).__name__
    return True


def test_loha_fused_qkv_matches_fused_module():
    net = chunked_matches_fused(M.try_load_loha, lambda base: loha_sd(base, 3 * DIM, DIM))
    for mod in net.modules.values():
        assert isinstance(mod, network_hada.NetworkModuleHadaChunk), type(mod).__name__
    return True


def test_oft_fused_qkv_skipped():
    """OFT has no chunk variant; a fused qkv binds nothing while a split projection binds."""
    net = load_via(M.try_load_oft, oft_sd('diffusion_model.layers.0.proj.qkv', 3 * DIM))
    assert net is None or len(net.modules) == 0, f'got {set(net.modules)}'
    net = load_via(M.try_load_oft, oft_sd('diffusion_model.layers.0.proj.o', DIM))
    assert net is not None and 'lora_transformer_layers_0_proj_o' in net.modules
    return True


def test_chain_merges_families():
    """try_load runs every family and merges LoRA and LoKR groups from one file."""
    sd = lora_sd('diffusion_model.layers.0.proj.qkv', *lora_factors(3 * DIM, DIM))
    sd.update(lokr_sd('diffusion_model.layers.1.proj.o', DIM, DIM))
    net = load_via(M.try_load, sd)
    expected = {f'lora_transformer_layers_0_proj_{n}' for n in ('q', 'k', 'v')} | {'lora_transformer_layers_1_proj_o'}
    assert net is not None and set(net.modules) == expected, f'got {set(net.modules) if net else None}'
    assert isinstance(net, network.Network)
    return True


def test_chain_refuses_mismatch():
    """A factor sized for the wrong module refuses the file."""
    down, up = lora_factors(*shape('layers.0.proj.ffn_down'))
    sd = lora_sd('diffusion_model.layers.0.proj.o', down, up)
    net = load_via(M.try_load, sd)
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
        test_resolve_fused_qkv_every_spelling,
        test_resolve_text_embedder_qkv_verbatim,
        test_resolve_upstream_refiner_qkv_verbatim,
        test_resolve_every_module_is_real,
        test_resolve_every_upstream_module_is_real,
    ]:
        run_test(CAT_RESOLVE, fn)

    log.warning('=== Loaders ===')
    for fn in [
        test_lora_fused_qkv_chunked,
        test_lora_fused_qkv_matches_fused_module,
        test_lora_comfy_split,
        test_lora_kohya_alpha,
        test_lora_kohya_fused_qkv,
        test_lora_diffusers_peft,
        test_lora_extras,
        test_upstream_lora_targets_attach,
        test_legacy_comfy_lora_targets_attach_upstream,
        test_lokr_fused_qkv_matches_fused_module,
        test_loha_fused_qkv_matches_fused_module,
        test_oft_fused_qkv_skipped,
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
    tests_passed = run_tests()
    sys.exit(0 if tests_passed else 1)
