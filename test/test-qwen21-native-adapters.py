#!/usr/bin/env python
"""
Offline unit tests for the Qwen-Image 2.1 native adapter loader.

``diffusers.QwenImage21Transformer2DModel`` keeps attention and the SwiGLU
split (``img_mlp.gate_layer`` / ``img_mlp.proj`` / ``img_mlp.out``), while
ComfyUI checkpoints fuse the SwiGLU input as ``img_mlp.gate_up`` with rows
``[gate_layer; proj]``, the layout ai-toolkit and musubi-tuner train against.
``pipelines.qwen.qwen21_lora`` splits that projection; everything else binds
verbatim. These tests pin the split (including a forward-pass check against
the fused SwiGLU), every save layout seen on CivitAI and the Hub, and the
file-level alpha read from PEFT / DiffSynth metadata.

Save formats exercised:

- ai-toolkit (``diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight``)
- musubi-tuner (``lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_down.weight`` + alpha)
- diffusers-PEFT (``transformer.transformer_blocks.0.img_mlp.proj.lora_A.weight``),
  PEFT named adapter (``...lora_A.default.weight``), bare factors (``...lora_A``)
- naive diffusers-to-ComfyUI conversion (``diffusion_model.transformer.``)
- LoKr and DoRA on the fused projection, rank-padded fused re-exports
- Qwen-Image 1.x files, which must be refused

The reference module tree is the real class at tiny dims. When the local
``Lora/Qwen 2.1`` folder exists, every file in it is also mapped header-only
against a full-size model built on the meta device.

No running server required.

Usage:
    python test/test-qwen21-native-adapters.py
"""

import os
import sys
import math
import json
import tempfile

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

import diffusers  # pylint: disable=wrong-import-position
from modules.errors import log   # pylint: disable=wrong-import-position
from modules.lora import network, network_lora, network_lokr  # pylint: disable=wrong-import-position
from modules.lora import native_adapter  # pylint: disable=wrong-import-position
from modules.lora.native_adapter import ChunkSpec  # pylint: disable=wrong-import-position
from pipelines.qwen import qwen21_lora as Q  # pylint: disable=wrong-import-position


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
        record(cat, ok is not False, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


# ============================================================
# Reference Qwen-Image 2.1 transformer (real class, tiny dims)
# ============================================================
# Upstream: 32 layers, 32 heads x 128 = 4096, mlp_ratio 3, context 4096, 64 latent channels.

INNER = 32
MLP = INNER * 3
LAYERS = 2
RANK = 4

REF = diffusers.QwenImage21Transformer2DModel(
    patch_size=1, in_channels=16, out_channels=16, num_layers=LAYERS,
    attention_head_dim=16, num_attention_heads=2, context_in_dim=24,
    mlp_ratio=3, axes_dims_rope=(4, 6, 6),
)

LINEAR_SHAPES = {name: tuple(m.weight.shape) for name, m in REF.named_modules() if isinstance(m, nn.Linear)}
LINEAR_NETKEYS = {'lora_transformer_' + name.replace('.', '_') for name in LINEAR_SHAPES}
BLOCK_LEAVES = ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0', 'img_mlp.gate_layer', 'img_mlp.proj', 'img_mlp.out')
EXTRAS = ('modulation.1', 'time_text_embed.timestep_embedder.linear_1', 'time_text_embed.timestep_embedder.linear_2',
          'txt_in.in_layer', 'txt_in.out_layer', 'img_in', 'norm_out.linear', 'proj_out')


def shape(path):
    assert path in LINEAR_SHAPES, f'reference has no Linear {path!r}'
    return LINEAR_SHAPES[path]


class MockPipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class MockSdModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'QwenImage21Pipeline'


def install_mock_pipe(transformer=REF):
    """Point shared.sd_model at a mock exposing ``transformer``; re-installed per load so stamps do not leak."""
    from modules.modeldata import model_data
    model_data.sd_model = MockSdModel(MockPipeline(transformer))
    return model_data.sd_model


# ============================================================
# Synthesizers and file helpers
# ============================================================


def lora(key_base, out, inp, rank=RANK, names=('lora_A.weight', 'lora_B.weight'), alpha=None):
    sd = {f'{key_base}.{names[0]}': torch.randn(rank, inp), f'{key_base}.{names[1]}': torch.randn(out, rank)}
    if alpha is not None:
        sd[f'{key_base}.alpha'] = torch.tensor(float(alpha))
    return sd


def ai_toolkit_block(i, prefix='diffusion_model.', names=('lora_A.weight', 'lora_B.weight'), alpha=None):
    """One block as ai-toolkit saves it: split attention, fused gate_up, out."""
    sd = {}
    for leaf in ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0'):
        sd.update(lora(f'{prefix}transformer_blocks.{i}.{leaf}', INNER, INNER, names=names, alpha=alpha))
    sd.update(lora(f'{prefix}transformer_blocks.{i}.img_mlp.gate_up', 2 * MLP, INNER, names=names, alpha=alpha))
    sd.update(lora(f'{prefix}transformer_blocks.{i}.img_mlp.out', INNER, MLP, names=names, alpha=alpha))
    return sd


def musubi_block(i, alpha=RANK):
    """One block as musubi-tuner saves it: kohya-flattened names, lora_down/up, alpha."""
    sd = {}
    for leaf, (out, inp) in (('attn_to_q', (INNER, INNER)), ('attn_to_k', (INNER, INNER)), ('attn_to_v', (INNER, INNER)),
                             ('attn_to_out_0', (INNER, INNER)), ('img_mlp_gate_up', (2 * MLP, INNER)), ('img_mlp_out', (INNER, MLP))):
        sd.update(lora(f'lora_unet_transformer_blocks_{i}_{leaf}', out, inp, names=('lora_down.weight', 'lora_up.weight'), alpha=alpha))
    return sd


def split_block(i, prefix='transformer.', names=('lora_A.weight', 'lora_B.weight')):
    """One block with diffusers names: split SwiGLU."""
    sd = {}
    for leaf in BLOCK_LEAVES:
        out, inp = shape(f'transformer_blocks.{i}.{leaf}')
        sd.update(lora(f'{prefix}transformer_blocks.{i}.{leaf}', out, inp, names=names))
    return sd


class TempLora:
    """Context manager: writes a state dict (and optional metadata) to a temp safetensors file."""

    def __init__(self, state_dict, name='test', metadata=None, cached_metadata=None):
        self.state_dict = state_dict
        self.name = name
        self.metadata = metadata
        self.cached_metadata = cached_metadata
        self.path = None

    def __enter__(self):
        sd = {k: v.contiguous() for k, v in self.state_dict.items()}
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        safetensors.torch.save_file(sd, self.path, metadata=self.metadata)
        return MockNetworkOnDisk(self.path, self.name, self.cached_metadata)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class MockNetworkOnDisk:
    def __init__(self, filename, name, metadata=None):
        self.filename = filename
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'
        if metadata is not None:
            self.metadata = metadata


def load_via(try_fn, state_dict, metadata=None, cached_metadata=None, name='test'):
    install_mock_pipe()
    with TempLora(state_dict, name=name, metadata=metadata, cached_metadata=cached_metadata) as nod:
        net = try_fn(name, nod, lora_scale=1.0)
    if net is not None:
        for module in net.modules.values():
            module.network.te_multiplier = 1.0
            module.network.unet_multiplier = 1.0
    return net


def netkey(path):
    return 'lora_transformer_' + path.replace('.', '_')


def updown(module, path):
    delta, _ = module.calc_updown(REF.get_submodule(path).weight.detach().clone())
    return delta


# ============================================================
# Tests - parsing
# ============================================================

CAT_PARSE = category('parse')


def test_parse_key_layouts():
    bd = Q.BARE_DIFFUSERS_PREFIX_USED
    cases = [
        ('diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight', ('diffusion_model.', 'transformer_blocks.0.img_mlp.gate_up', 'lora_down.weight')),
        ('transformer.transformer_blocks.0.attn.to_q.lora_B.weight', ('transformer.', 'transformer_blocks.0.attn.to_q', 'lora_up.weight')),
        ('diffusion_model.transformer.modulation.1.lora_A.weight', ('transformer.', 'modulation.1', 'lora_down.weight')),
        ('transformer_blocks.0.attn.to_k.lora_A.default.weight', (bd, 'transformer_blocks.0.attn.to_k', 'lora_down.weight')),
        ('transformer_blocks.0.attn.to_k.lora_B', (bd, 'transformer_blocks.0.attn.to_k', 'lora_up.weight')),
        ('lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_down.weight', ('lora_unet_', 'transformer_blocks_0_img_mlp_gate_up', 'lora_down.weight')),
        ('lora_unet_transformer_blocks_0_attn_to_q.alpha', ('lora_unet_', 'transformer_blocks_0_attn_to_q', 'alpha')),
        ('img_in.lora_down', (bd, 'img_in', 'lora_down.weight')),
        ('diffusion_model.transformer_blocks.3.img_mlp.gate_up.dora_scale', ('diffusion_model.', 'transformer_blocks.3.img_mlp.gate_up', 'dora_scale')),
        ('random.unrelated.key', None),
    ]
    for key, expected in cases:
        got = Q.parse_key(key, Q.LORA_SUFFIXES)
        assert got == expected, f'parse_key({key!r}) = {got}, expected {expected}'
    return True


def test_markers():
    assert Q.has_marker({'transformer_blocks.0.attn.to_q.lora_A': None}, Q.LORA_MARKERS), 'bare lora_A not detected'
    assert Q.has_marker({'transformer_blocks.0.attn.to_q.lora_A.default.weight': None}, Q.LORA_MARKERS)
    lokr = {'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lokr_w1': None}
    assert Q.has_marker(lokr, Q.LOKR_MARKERS) and not Q.has_marker(lokr, Q.LORA_MARKERS)
    return True


# ============================================================
# Tests - target resolution
# ============================================================

CAT_RESOLVE = category('resolve')

GATE_UP_SPLIT = [('transformer_blocks.0.img_mlp.gate_layer', ChunkSpec(idx=0, total=2)), ('transformer_blocks.0.img_mlp.proj', ChunkSpec(idx=1, total=2))]


def test_resolve_gate_up_every_prefix():
    for prefix in ('diffusion_model.', 'transformer.', Q.BARE_DIFFUSERS_PREFIX_USED, 'lora_transformer_'):
        got = native_adapter.resolve_group_targets(Q.resolve_targets, prefix, 'transformer_blocks.0.img_mlp.gate_up')
        assert got == GATE_UP_SPLIT, f'{prefix}: {got}'
    return True


def test_resolve_gate_up_flattened():
    expected = [('transformer_blocks_0_img_mlp_gate_layer', ChunkSpec(idx=0, total=2)), ('transformer_blocks_0_img_mlp_proj', ChunkSpec(idx=1, total=2))]
    for prefix in ('lora_unet_', 'lycoris_', 'lora_transformer_'):
        got = native_adapter.resolve_group_targets(Q.resolve_targets, prefix, 'transformer_blocks_0_img_mlp_gate_up')
        assert got == expected, f'{prefix}: {got}'
        assert {netkey(path) for path, _ in got} <= LINEAR_NETKEYS
    return True


def test_resolve_verbatim_is_real():
    for prefix in ('diffusion_model.', 'transformer.', Q.BARE_DIFFUSERS_PREFIX_USED):
        for path in [f'transformer_blocks.1.{leaf}' for leaf in BLOCK_LEAVES] + list(EXTRAS):
            got = native_adapter.resolve_group_targets(Q.resolve_targets, prefix, path)
            assert got == [(path, None)], f'{prefix}{path}: {got}'
            assert netkey(path) in LINEAR_NETKEYS, f'{path} is not a reference Linear'
    got = native_adapter.resolve_group_targets(Q.resolve_targets, 'lora_unet_', 'transformer_blocks_1_attn_to_out_0')
    assert got == [('transformer_blocks_1_attn_to_out_0', None)] and netkey('transformer_blocks.1.attn.to_out.0') in LINEAR_NETKEYS
    return True


# ============================================================
# Tests - loading real-world layouts
# ============================================================

CAT_LOAD = category('load')
FULL = 7 * LAYERS


def test_load_ai_toolkit_fused():
    sd = {}
    for i in range(LAYERS):
        sd.update(ai_toolkit_block(i))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL, f'modules={len(net.modules) if net else None}'
    assert set(net.modules) <= LINEAR_NETKEYS
    fused_up = sd['diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight']
    down = sd['diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight']
    gate = net.modules[netkey('transformer_blocks.0.img_mlp.gate_layer')]
    proj = net.modules[netkey('transformer_blocks.0.img_mlp.proj')]
    assert torch.equal(gate.up_model.weight, fused_up[:MLP]) and torch.equal(proj.up_model.weight, fused_up[MLP:]), 'gate_up rows landed on the wrong module'
    assert torch.equal(gate.down_model.weight, down) and torch.equal(proj.down_model.weight, down)
    return True


def test_load_musubi_flattened_alpha():
    sd = {}
    for i in range(LAYERS):
        sd.update(musubi_block(i, alpha=RANK / 2))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL, f'modules={len(net.modules) if net else None}'
    for key in (netkey('transformer_blocks.1.img_mlp.gate_layer'), netkey('transformer_blocks.1.img_mlp.proj')):
        assert net.modules[key].alpha == RANK / 2 and net.modules[key].calc_scale() == 0.5
    return True


def test_load_peft_named_adapter():
    sd = {}
    for i in range(LAYERS):
        sd.update(split_block(i, prefix='', names=('lora_A.default.weight', 'lora_B.default.weight')))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL
    return True


def test_load_bare_factor_names():
    sd = {}
    for i in range(LAYERS):
        sd.update(split_block(i, prefix='', names=('lora_A', 'lora_B')))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL, f'modules={len(net.modules) if net else None}'
    return True


def test_load_turbo_extras():
    sd = {}
    for i in range(LAYERS):
        sd.update(split_block(i))
    for path in ('modulation.1', 'time_text_embed.timestep_embedder.linear_1', 'time_text_embed.timestep_embedder.linear_2'):
        sd.update(lora(f'transformer.{path}', *shape(path)))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL + 3
    assert netkey('modulation.1') in net.modules
    return True


def test_load_comfy_wrapped_diffusers():
    sd = {}
    for i in range(LAYERS):
        sd.update(split_block(i, prefix='diffusion_model.transformer.'))
    sd.update(lora('diffusion_model.transformer.modulation.1', *shape('modulation.1')))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL + 1, f'modules={len(net.modules) if net else None}'
    return True


def test_load_pdd_backbone():
    """alibaba-pai exports: bare names, bare lora_down/lora_up, non-block targets."""
    sd = {}
    for i in range(LAYERS):
        sd.update(split_block(i, prefix='', names=('lora_down', 'lora_up')))
    for path in ('img_in', 'txt_in.in_layer', 'txt_in.out_layer', 'norm_out.linear', 'modulation.1'):
        sd.update(lora(path, *shape(path), names=('lora_down', 'lora_up')))
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == FULL + 5
    return True


def test_load_partial_blocks():
    net = load_via(Q.try_load, ai_toolkit_block(1))
    assert net is not None and len(net.modules) == 7
    assert all('_blocks_1_' in key for key in net.modules)
    return True


def test_load_lokr_fused():
    sd = {
        'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lokr_w1': torch.randn(8, 4),
        'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lokr_w2': torch.randn(2 * MLP // 8, INNER // 4),
        'diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha': torch.tensor(4.0),
    }
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == 2
    gate = net.modules[netkey('transformer_blocks.0.img_mlp.gate_layer')]
    proj = net.modules[netkey('transformer_blocks.0.img_mlp.proj')]
    assert isinstance(gate, network_lokr.NetworkModuleLokrChunk) and (gate.chunk_index, gate.num_chunks) == (0, 2)
    assert isinstance(proj, network_lokr.NetworkModuleLokrChunk) and (proj.chunk_index, proj.num_chunks) == (1, 2)
    return True


def test_load_dora_fused():
    sd = lora('diffusion_model.transformer_blocks.0.img_mlp.gate_up', 2 * MLP, INNER)
    sd['diffusion_model.transformer_blocks.0.img_mlp.gate_up.dora_scale'] = torch.rand(2 * MLP, 1) + 0.5
    net = load_via(Q.try_load, sd)
    assert net is not None and len(net.modules) == 2
    ds = sd['diffusion_model.transformer_blocks.0.img_mlp.gate_up.dora_scale']
    assert torch.equal(net.modules[netkey('transformer_blocks.0.img_mlp.gate_layer')].dora_scale, ds[:MLP])
    assert torch.equal(net.modules[netkey('transformer_blocks.0.img_mlp.proj')].dora_scale, ds[MLP:])
    return True


def test_refuse_qwen_image_1():
    """Qwen-Image 1.x names bind nothing; right names at the wrong width are a shape mismatch that refuses the file."""
    names_only = lora('transformer.transformer_blocks.0.img_mod.1', 6 * INNER, INNER)
    names_only.update(lora('transformer.transformer_blocks.0.txt_mlp.net.2', INNER, 4 * INNER))
    assert load_via(Q.try_load, names_only) is None
    wrong_width = lora('diffusion_model.transformer_blocks.0.attn.to_q', 3 * INNER, 3 * INNER)
    wrong_width.update(lora('diffusion_model.transformer_blocks.0.attn.to_k', INNER, INNER))
    assert load_via(Q.try_load, wrong_width) is None
    return True


# ============================================================
# Tests - delta math
# ============================================================

CAT_MATH = category('math')


def swiglu_fused(x, w_gate_up, w_out):
    """ai-toolkit / ComfyUI forward: ``gate, up = gate_up(x).chunk(2)``."""
    gate, up = (x @ w_gate_up.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ w_out.T


def swiglu_split(x, w_gate, w_proj, w_out):
    """diffusers forward: ``out(silu(gate_layer(x)) * proj(x))``."""
    return (torch.nn.functional.silu(x @ w_gate.T) * (x @ w_proj.T)) @ w_out.T


def test_gate_up_split_matches_fused_forward():
    """The split LoRA reproduces the fused SwiGLU output; the swapped split does not."""
    mlp = REF.transformer_blocks[0].img_mlp
    w_gate, w_proj, w_out = mlp.gate_layer.weight.detach(), mlp.proj.weight.detach(), mlp.out.weight.detach()
    sd = lora('diffusion_model.transformer_blocks.0.img_mlp.gate_up', 2 * MLP, INNER)
    delta = sd['diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight'] @ sd['diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight']
    x = torch.randn(5, INNER)
    reference = swiglu_fused(x, torch.cat([w_gate, w_proj]) + delta, w_out)
    net = load_via(Q.try_load, sd)
    d_gate = updown(net.modules[netkey('transformer_blocks.0.img_mlp.gate_layer')], 'transformer_blocks.0.img_mlp.gate_layer')
    d_proj = updown(net.modules[netkey('transformer_blocks.0.img_mlp.proj')], 'transformer_blocks.0.img_mlp.proj')
    native = swiglu_split(x, w_gate + d_gate, w_proj + d_proj, w_out)
    assert torch.allclose(native, reference, atol=1e-5), f'max diff {(native - reference).abs().max():.3e}'
    swapped = swiglu_split(x, w_gate + d_proj, w_proj + d_gate, w_out)
    assert not torch.allclose(swapped, reference, atol=1e-3), 'swapped halves also match; the check does not discriminate'
    return True


def test_lokr_chunks_tile_the_kron():
    w1, w2 = torch.randn(8, 4), torch.randn(2 * MLP // 8, INNER // 4)
    sd = {
        'diffusion_model.transformer_blocks.1.img_mlp.gate_up.lokr_w1': w1,
        'diffusion_model.transformer_blocks.1.img_mlp.gate_up.lokr_w2': w2,
    }
    net = load_via(Q.try_load, sd)
    full = torch.kron(w1, w2)
    d_gate = updown(net.modules[netkey('transformer_blocks.1.img_mlp.gate_layer')], 'transformer_blocks.1.img_mlp.gate_layer')
    d_proj = updown(net.modules[netkey('transformer_blocks.1.img_mlp.proj')], 'transformer_blocks.1.img_mlp.proj')
    assert torch.allclose(torch.cat([d_gate, d_proj]), full, atol=1e-5)
    return True


def test_rank_padded_fused_is_exact():
    """A split LoRA re-fused as ``down = [A_g; A_p]``, ``up = diag(B_g, B_p)`` returns each half exactly."""
    a_g, b_g, a_p, b_p = torch.randn(RANK, INNER), torch.randn(MLP, RANK), torch.randn(RANK, INNER), torch.randn(MLP, RANK)
    up = torch.zeros(2 * MLP, 2 * RANK)
    up[:MLP, :RANK], up[MLP:, RANK:] = b_g, b_p
    sd = {
        'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_down.weight': torch.cat([a_g, a_p]),
        'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_up.weight': up,
        'diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha': torch.tensor(float(2 * RANK)),
    }
    net = load_via(Q.try_load, sd)
    d_gate = updown(net.modules[netkey('transformer_blocks.0.img_mlp.gate_layer')], 'transformer_blocks.0.img_mlp.gate_layer')
    d_proj = updown(net.modules[netkey('transformer_blocks.0.img_mlp.proj')], 'transformer_blocks.0.img_mlp.proj')
    assert torch.allclose(d_gate, b_g @ a_g, atol=1e-5) and torch.allclose(d_proj, b_p @ a_p, atol=1e-5)
    return True


# ============================================================
# Tests - file-level alpha from metadata
# ============================================================

CAT_ALPHA = category('alpha')


def peft_metadata(**fields):
    return {'lora_adapter_metadata': json.dumps({f'transformer.{k}': v for k, v in fields.items()}), 'format': 'pt'}


def test_peft_metadata_alpha():
    """Pruna ships r=64 lora_alpha=128 without alpha tensors: PEFT applies scale 2."""
    net = load_via(Q.try_load, split_block(0), metadata=peft_metadata(r=RANK, lora_alpha=2 * RANK, alpha_pattern={}, use_rslora=False))
    assert all(module.calc_scale() == 2.0 for module in net.modules.values())
    return True


def test_peft_alpha_pattern():
    net = load_via(Q.try_load, split_block(0), metadata=peft_metadata(r=RANK, lora_alpha=RANK, alpha_pattern={'transformer_blocks.0.attn.to_q': RANK / 2}))
    assert net.modules[netkey('transformer_blocks.0.attn.to_q')].calc_scale() == 0.5
    assert net.modules[netkey('transformer_blocks.0.attn.to_k')].calc_scale() == 1.0
    return True


def test_peft_rslora():
    net = load_via(Q.try_load, split_block(0), metadata=peft_metadata(r=RANK, lora_alpha=RANK, use_rslora=True))
    assert all(math.isclose(module.calc_scale(), RANK / math.sqrt(RANK)) for module in net.modules.values())
    return True


def test_alpha_tensors_win_over_metadata():
    sd = {}
    for i in range(LAYERS):
        sd.update(musubi_block(i, alpha=RANK))
    net = load_via(Q.try_load, sd, metadata=peft_metadata(r=RANK, lora_alpha=4 * RANK))
    assert all(module.calc_scale() == 1.0 for module in net.modules.values())
    return True


def test_flat_alpha_metadata():
    """DiffSynth-Studio stores alpha and rank as flat metadata keys."""
    net = load_via(Q.try_load, split_block(0, prefix='', names=('lora_A.default.weight', 'lora_B.default.weight')), metadata={'alpha': str(RANK / 2), 'rank': str(RANK)})
    assert all(module.calc_scale() == 0.5 for module in net.modules.values())
    return True


def test_metadata_read_from_header_when_cache_empty():
    """An empty cached metadata dict (failed read, --no-metadata) falls back to the file header."""
    net = load_via(Q.try_load, split_block(0), metadata=peft_metadata(r=RANK, lora_alpha=2 * RANK), cached_metadata={})
    assert all(module.calc_scale() == 2.0 for module in net.modules.values())
    return True


def test_text_encoder_component_is_separate():
    components = native_adapter.peft_configs({'lora_adapter_metadata': json.dumps({'transformer.lora_alpha': 8, 'text_encoder.lora_alpha': 2})})
    alpha = native_adapter.peft_alpha(components)
    assert alpha('transformer.', 'transformer_blocks.0.attn.to_q', 4) == 8.0
    assert alpha('lora_te_', 'model_layers_0_self_attn_q_proj', 4) == 2.0
    return True


# ============================================================
# Tests - real files (header-only, full-size meta model)
# ============================================================

CAT_REAL = category('real-files')
LORA_DIR = '/home/ohiom/database/models/Lora/Qwen 2.1'
QWEN1_DIR = '/home/ohiom/database/models/Lora/Qwen'


def full_size_shapes():
    with torch.device('meta'):
        model = diffusers.QwenImage21Transformer2DModel()
    return {'lora_transformer_' + name.replace('.', '_'): tuple(m.weight.shape) for name, m in model.named_modules() if isinstance(m, nn.Linear)}


def header_shapes(filename):
    from safetensors import safe_open
    with safe_open(filename, framework='pt', device='cpu') as f:
        return {key: tuple(f.get_slice(key).get_shape()) for key in f.keys()}


def map_file(shapes, targets):
    """(mapped, unmapped, mismatch) for the LoRA, DoRA and LoKr groups of one file, resolved like the loaders do."""
    mapped = unmapped = mismatch = 0
    lora_groups = Q.group_by_suffixes(shapes, Q.LORA_SUFFIXES)
    lokr_groups = Q.group_by_suffixes(shapes, Q.LOKR_SUFFIXES)
    for (prefix, base), w in list(lora_groups.items()) + list(lokr_groups.items()):
        is_lora = 'lora_down.weight' in w and 'lora_up.weight' in w
        is_lokr = 'lokr_w1' in w and 'lokr_w2' in w
        if not (is_lora or is_lokr):
            continue
        for path, chunk in native_adapter.resolve_group_targets(Q.resolve_targets, prefix, base):
            key = 'lora_transformer_' + path.replace('.', '_')
            if key not in targets:
                unmapped += 1
                continue
            out, inp = targets[key]
            total = chunk.total if chunk is not None else 1
            if is_lora:
                fits = w['lora_down.weight'][1] == inp and w['lora_up.weight'][0] == out * total
            else:
                fits = w['lokr_w1'][0] * w['lokr_w2'][0] == out * total and w['lokr_w1'][1] * w['lokr_w2'][1] == inp
            mapped += fits
            mismatch += not fits
    return mapped, unmapped, mismatch


def test_real_qwen21_files_map_cleanly():
    if not os.path.isdir(LORA_DIR):
        log.warning(f'  SKIP: {LORA_DIR} not present')
        return True
    targets = full_size_shapes()
    files = sorted(f for f in os.listdir(LORA_DIR) if f.endswith('.safetensors'))
    bad = []
    for f in files:
        mapped, unmapped, mismatch = map_file(header_shapes(os.path.join(LORA_DIR, f)), targets)
        log.info(f'    {f}: mapped={mapped} unmapped={unmapped} mismatch={mismatch}')
        if mapped == 0 or unmapped or mismatch:
            bad.append(f)
    assert not bad, f'{len(bad)}/{len(files)} files do not map cleanly: {bad}'
    return True


def test_real_qwen1_files_do_not_fit():
    if not os.path.isdir(QWEN1_DIR):
        log.warning(f'  SKIP: {QWEN1_DIR} not present')
        return True
    targets = full_size_shapes()
    files = sorted(f for f in os.listdir(QWEN1_DIR) if f.endswith('.safetensors'))[:6]
    for f in files:
        mapped, unmapped, mismatch = map_file(header_shapes(os.path.join(QWEN1_DIR, f)), targets)
        log.info(f'    {f}: mapped={mapped} unmapped={unmapped} mismatch={mismatch}')
        assert unmapped or mismatch, f'{f} maps onto Qwen-Image 2.1 without a single mismatch'
    return True


# ============================================================
# Main
# ============================================================


def main():
    tests = {
        CAT_PARSE: [test_parse_key_layouts, test_markers],
        CAT_RESOLVE: [test_resolve_gate_up_every_prefix, test_resolve_gate_up_flattened, test_resolve_verbatim_is_real],
        CAT_LOAD: [
            test_load_ai_toolkit_fused, test_load_musubi_flattened_alpha, test_load_peft_named_adapter,
            test_load_bare_factor_names, test_load_turbo_extras, test_load_comfy_wrapped_diffusers,
            test_load_pdd_backbone, test_load_partial_blocks, test_load_lokr_fused, test_load_dora_fused,
            test_refuse_qwen_image_1,
        ],
        CAT_MATH: [test_gate_up_split_matches_fused_forward, test_lokr_chunks_tile_the_kron, test_rank_padded_fused_is_exact],
        CAT_ALPHA: [
            test_peft_metadata_alpha, test_peft_alpha_pattern, test_peft_rslora, test_alpha_tensors_win_over_metadata,
            test_flat_alpha_metadata, test_metadata_read_from_header_when_cache_empty, test_text_encoder_component_is_separate,
        ],
        CAT_REAL: [test_real_qwen21_files_map_cleanly, test_real_qwen1_files_do_not_fit],
    }
    torch.manual_seed(0)
    for cat, fns in tests.items():
        log.info(f'Category: {cat}')
        for fn in fns:
            run_test(cat, fn)
    passed = sum(r['passed'] for r in results.values())
    failed = sum(r['failed'] for r in results.values())
    for cat, r in results.items():
        log.info(f'{cat}: passed={r["passed"]} failed={r["failed"]}')
    log.info(f'Total: passed={passed} failed={failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
