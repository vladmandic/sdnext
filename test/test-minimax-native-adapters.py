#!/usr/bin/env python
"""
Offline unit tests for the MiniMax H3 native adapter loader.

Published MiniMax H3 LoRAs target the reference module names (fused
``attn.qkv_proj``, fused SwiGLU ``mlp.fc1``, ``token_refiner.blocks``), while
sdnext loads upstream ``diffusers.MiniMaxH3Transformer3DModel``. The reference
fc1 is ``[gate; value]`` and diffusers' SwiGLU is ``[value; gate]``, so the
native mapping has to permute fc1 output rows the same way the diffusers LoRA
converter does. These tests pin ``pipelines.minimax.minimax_lora`` against that
converter: same targets, same per-module deltas.

Save formats exercised, each seen in a published LoRA:

- comfy / ai-toolkit (``diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight``):
  the CivitAI ecosystem and the larryvrh turbo files.
- bare reference names (``blocks.0.attn.qkv_proj.lora_A.weight``): the
  unpruned larryvrh saves.
- musubi-tuner (``lora_unet_blocks_0_mlp_fc1.lora_down.weight`` + ``alpha``).
- comfy with block-diagonal fused qkv and tripled alpha: lightx2v's ComfyUI
  exports.
- peft dump (``transformer_blocks.0.attn.to_q.lora_A.default.weight``) with the
  training alpha in the file metadata: lightx2v's diffusers exports.
- diffusers names with kohya suffixes: the alibaba-pai Acc LoRAs.
- peft wrapper around a ``dit`` attribute (``base_model.model.dit.blocks.0...``):
  the mvp-lab RAVEN LoRA.

The reference module tree is the real ``MiniMaxH3Transformer3DModel`` at tiny
dims, so module names and target shapes are authoritative. Every LoRA file
under the MiniMax H3 LoRA folder is also mapped through both paths and compared
per target, exactly when the file carries no alpha and through random probes
when the converter folds an alpha into the weights.

No running server required.

Usage:
    python test/test-minimax-native-adapters.py
"""

import glob
import os
import sys
import tempfile
import time

import torch
import torch.nn as nn
import safetensors
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
from modules.lora import native_adapter          # pylint: disable=wrong-import-position
from pipelines.minimax import minimax_lora as M  # pylint: disable=wrong-import-position
from diffusers import MiniMaxH3Transformer3DModel  # pylint: disable=wrong-import-position
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers as convert_diffusers  # pylint: disable=wrong-import-position


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
# Reference MiniMax H3 transformer (real class, tiny dims)
# ============================================================
# Upstream: heads=56, head_dim=128, hidden=5376, layers=50, refiner=2, ffn=14336.
# The attention inner dim exceeds the hidden size upstream; the test dims keep that.

NUM_LAYERS = 2
NUM_REFINER = 1

REF = MiniMaxH3Transformer3DModel(
    num_attention_heads=2, attention_head_dim=8, hidden_size=12, num_layers=NUM_LAYERS, num_refiner_layers=NUM_REFINER,
    ffn_dim=16, in_channels=4, audio_in_channels=4, patch_size=(1, 2, 2), text_dim=10, freq_dim=8,
    time_embed_hidden_dim=12, time_embed_dim=6, rope_freq_dim=1,
)

# {diffusers dotted name: (out, in)} for every Linear.
LINEAR_SHAPES = {name: tuple(m.weight.shape) for name, m in REF.named_modules() if isinstance(m, nn.Linear)}
# {network key: module} exactly as assign_network_names_to_compvis_modules stamps it.
LINEAR_NETKEYS = {'lora_transformer_' + name.replace('.', '_') for name in LINEAR_SHAPES}

# Reference module names and where they land, used only to size synthetic tensors;
# correctness is judged against the diffusers converter, not this table.
STANDALONE = {
    'video_patch_proj': 'proj_in',
    'audio_patch_proj': 'audio_proj_in',
    'condition_proj': 'context_embedder',
    'time_embedder.proj_in': 'time_embedder.linear_1',
    'time_embedder.proj_out': 'time_embedder.linear_2',
    'final_layer.adaln_proj.linear': 'norm_out.linear',
    'final_layer.video_out': 'proj_out',
    'final_layer.audio_out': 'audio_proj_out',
}
BLOCK_LEAVES = {
    'attn.qkv_proj': 'attn.to_q',
    'attn.out_proj': 'attn.to_out.0',
    'mlp.fc1': 'ff.net.0.proj',
    'mlp.fc2': 'ff.net.2',
    'adaln_proj.linear': 'adaln_proj.linear',
}


def ref_shape(ref_path: str):
    """(out, in) of the reference Linear a LoRA key names, read from the diffusers module it maps to."""
    if ref_path in STANDALONE:
        return LINEAR_SHAPES[STANDALONE[ref_path]]
    if ref_path.startswith('blocks.'):
        _, idx, leaf = ref_path.split('.', 2)
        stack = 'transformer_blocks'
    else:
        _, _, idx, leaf = ref_path.split('.', 3)
        stack = 'token_refiner.refiner_blocks'
    out, inp = LINEAR_SHAPES[f'{stack}.{idx}.{BLOCK_LEAVES[leaf]}']
    if leaf == 'attn.qkv_proj':
        out *= 3
    return out, inp


def all_ref_paths():
    paths = list(STANDALONE)
    for i in range(NUM_LAYERS):
        paths += [f'blocks.{i}.{leaf}' for leaf in BLOCK_LEAVES]
    for i in range(NUM_REFINER):
        paths += [f'token_refiner.blocks.{i}.{leaf}' for leaf in BLOCK_LEAVES if leaf != 'adaln_proj.linear']
    return paths


# ============================================================
# Mock pipeline wrapping the real transformer
# ============================================================


class _MockPipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None


class _MockSdModel:
    def __init__(self, pipe):
        self.pipe = pipe
        self.network_layer_mapping = {}
        self.embedding_db = None
        self.__class__.__name__ = 'MiniMaxH3ModularPipeline'


def install_mock_pipe():
    """Point shared.sd_model at a mock exposing the reference transformer; re-installed per load so stamps do not leak."""
    sd_model = _MockSdModel(_MockPipeline(REF))
    from modules.modeldata import model_data
    model_data.sd_model = sd_model
    return sd_model


# ============================================================
# Synthesizers and helpers
# ============================================================

RANK = 4
PEFT = ('lora_A.weight', 'lora_B.weight')
KOHYA = ('lora_down.weight', 'lora_up.weight')


def lora_pair(key_base, shape, suffix=PEFT, alpha=None, rank=RANK):
    out, inp = shape
    sd = {
        f'{key_base}.{suffix[0]}': torch.randn(rank, inp),
        f'{key_base}.{suffix[1]}': torch.randn(out, rank),
    }
    if alpha is not None:
        sd[f'{key_base}.alpha'] = torch.tensor(float(alpha))
    return sd


def synth(prefix='diffusion_model.', suffix=PEFT, alpha=None, flatten=False):
    """One LoRA pair for every reference Linear, in the requested reference-name layout."""
    sd = {}
    for path in all_ref_paths():
        key = f'lora_unet_{path.replace(".", "_")}' if flatten else f'{prefix}{path}'
        sd.update(lora_pair(key, ref_shape(path), suffix, alpha))
    return sd


def synth_diffusers(prefix='', suffix=PEFT, infix='', alpha=None):
    """One LoRA pair for every Linear, keyed by its diffusers name; ``infix`` inserts a peft adapter slot."""
    sd = {}
    for path, shape in LINEAR_SHAPES.items():
        pair = lora_pair(f'{prefix}{path}', shape, suffix, alpha)
        for key, value in pair.items():
            for marker in ('lora_A', 'lora_B', 'lora_down', 'lora_up'):
                key = key.replace(f'.{marker}.weight', f'.{marker}{infix}.weight')
            sd[key] = value
    return sd


class TempLora:
    """Context manager: writes a state dict to a temp safetensors file."""

    def __init__(self, state_dict, name='test', metadata=None):
        self.state_dict = state_dict
        self.name = name
        self.metadata = metadata
        self.path = None

    def __enter__(self):
        sd = {k: v.contiguous() for k, v in self.state_dict.items()}
        fd, self.path = tempfile.mkstemp(suffix='.safetensors', prefix=f'{self.name}_')
        os.close(fd)
        safetensors.torch.save_file(sd, self.path)
        return _MockNetworkOnDisk(self.path, self.name, self.metadata)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class _MockNetworkOnDisk:
    def __init__(self, filename, name, metadata=None):
        self.filename = filename
        self.name = name
        self.shorthash = ''
        self.sd_version = 'unknown'
        self.metadata = metadata or {}


def load_native(state_dict, name='test', metadata=None):
    install_mock_pipe()
    with TempLora(state_dict, name=name, metadata=metadata) as nod:
        return M.try_load(name, nod, lora_scale=1.0)


def native_deltas(net):
    """{network key: full delta} as the apply pass would compute it at multiplier 1."""
    out = {}
    for key, module in net.modules.items():
        module.network.te_multiplier = 1.0
        module.network.unet_multiplier = 1.0
        up = module.up_model.weight.float()
        down = module.down_model.weight.float()
        out[key] = (up @ down) * module.calc_scale()
    return out


def diffusers_deltas(state_dict, rewrite=None):
    """{network key: full delta} from the diffusers converter's output."""
    if rewrite is not None:
        state_dict = {k.replace(rewrite[0], rewrite[1], 1) if k.startswith(rewrite[0]) else k: v for k, v in state_dict.items()}
    converted = convert_diffusers(dict(state_dict))
    out = {}
    for key, value in converted.items():
        if not key.endswith('.lora_A.weight'):
            continue
        path = key[len('transformer.'):-len('.lora_A.weight')]
        down = value.float()
        up = converted[f'transformer.{path}.lora_B.weight'].float()
        out['lora_transformer_' + path.replace('.', '_')] = up @ down
    return out


def identity_deltas(state_dict, scale=1.0):
    """{network key: full delta} for a file already on diffusers names, at a uniform scale."""
    out = {}
    for key, value in state_dict.items():
        for marker in ('lora_A', 'lora_down'):
            idx = key.find(f'.{marker}')
            if idx == -1:
                continue
            path = key[:idx].removeprefix('transformer.')
            up_key = key.replace(marker, 'lora_B' if marker == 'lora_A' else 'lora_up', 1)
            out['lora_transformer_' + path.replace('.', '_')] = (state_dict[up_key].float() @ value.float()) * scale
    return out


def assert_same_deltas(native, reference):
    assert set(native) == set(reference), f'targets differ: native-only={sorted(set(native) - set(reference))} reference-only={sorted(set(reference) - set(native))}'
    for key, ref in reference.items():
        got = native[key]
        assert got.shape == ref.shape, f'{key}: shape {tuple(got.shape)} vs {tuple(ref.shape)}'
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), f'{key}: delta differs, max abs {float((got - ref).abs().max()):.3e}'


def native_mapping(state_dict, network_alpha=None):
    """{diffusers path: (down, up, scale)} the native loader would bind, without a model: parse, resolve, chunk, file alpha."""
    groups = M.group_by_suffixes(state_dict, M.LORA_SUFFIXES)
    if network_alpha is not None and any('alpha' in w for w in groups.values()):
        network_alpha = None
    out = {}
    for (prefix, base), w in groups.items():
        if 'lora_down.weight' not in w or 'lora_up.weight' not in w:
            continue
        for path, chunk in native_adapter.resolve_group_targets(M.resolve_targets, prefix, base):
            target = native_adapter._slice_lora_chunk(w, chunk) if chunk is not None else w # pylint: disable=protected-access
            alpha = network_alpha if 'alpha' not in target else float(target['alpha'])
            scale = 1.0 if alpha is None else alpha / target['lora_down.weight'].shape[0]
            out[path] = (target['lora_down.weight'], target['lora_up.weight'], scale)
    return out


REFERENCE_PREFIXES = ('diffusion_model.', 'blocks.', 'token_refiner.blocks.', 'final_layer.', 'lora_unet_', 'base_model.model.dit.',
                      'video_patch_proj.', 'audio_patch_proj.', 'condition_proj.', 'time_embedder.proj_')


def oracle_mapping(state_dict, network_alpha=None):
    """{diffusers path: (down, up, scale)} plus the keys no LoRA pair claims, from the layout's reference loader.

    Reference-name layouts go through the diffusers converter, which folds any
    alpha into the weights. Files already on diffusers names bind verbatim, with
    a per-key alpha or else the file-level one applied as ``alpha / rank``.
    """
    if any(k.startswith(REFERENCE_PREFIXES) for k in state_dict):
        sd = {k.replace('base_model.model.dit.', 'diffusion_model.', 1) if k.startswith('base_model.model.dit.') else k: v for k, v in state_dict.items()}
        converted = convert_diffusers(sd)
        out = {}
        for key, value in converted.items():
            if key.endswith('.lora_A.weight'):
                path = key[len('transformer.'):-len('.lora_A.weight')]
                out[path] = (value, converted[f'transformer.{path}.lora_B.weight'], 1.0)
        return out, []
    has_alpha = any(k.endswith('.alpha') for k in state_dict)
    out, ignored = {}, []
    for key, value in state_dict.items():
        marker = next((m for m in ('lora_A', 'lora_down') if f'.{m}' in key), None)
        if marker is None:
            if not any(f'.{m}' in key for m in ('lora_B', 'lora_up', 'alpha')):
                ignored.append(key)
            continue
        path = key[:key.find(f'.{marker}')].removeprefix('transformer.')
        up = state_dict[key.replace(marker, 'lora_B' if marker == 'lora_A' else 'lora_up', 1)]
        alpha = state_dict.get(f'{path}.alpha', state_dict.get(f'transformer.{path}.alpha'))
        if alpha is not None:
            scale = float(alpha) / value.shape[0]
        elif network_alpha is not None and not has_alpha:
            scale = network_alpha / value.shape[0]
        else:
            scale = 1.0
        out[path] = (value, up, scale)
    return out, ignored


def assert_same_factors(native, oracle, label):
    """Exact when neither side scales; probe-equal on the effective delta otherwise."""
    assert set(native) == set(oracle), f'{label}: targets differ: native-only={sorted(set(native) - set(oracle))[:4]} oracle-only={sorted(set(oracle) - set(native))[:4]}'
    probes = 0
    for path, (down_o, up_o, scale_o) in oracle.items():
        down_n, up_n, scale_n = native[path]
        if scale_n == 1.0 and scale_o == 1.0:
            assert torch.equal(down_n, down_o), f'{label}: {path} down differs'
            assert torch.equal(up_n, up_o), f'{label}: {path} up differs'
            continue
        probes += 1
        x = torch.randn(4, down_o.shape[1])
        got = (x @ down_n.float().t() @ up_n.float().t()) * scale_n
        ref = (x @ down_o.float().t() @ up_o.float().t()) * scale_o
        err = float((got - ref).abs().max() / ref.abs().max().clamp(min=1e-12))
        assert err < 1e-4, f'{label}: {path} effective delta differs, relative error {err:.2e}'
    return probes


# ============================================================
# Tests - resolution
# ============================================================

CAT_RESOLVE = category('resolve')
CAT_LOADER = category('loader')
CAT_REAL = category('real-files')


def test_every_reference_module_resolves_to_a_real_linear():
    """Every reference key layout resolves onto a Linear that exists in the diffusers model."""
    for layout in ({'prefix': 'diffusion_model.'}, {'prefix': ''}, {'flatten': True, 'suffix': KOHYA}, {'prefix': 'base_model.model.dit.'}):
        sd = synth(**layout)
        got = set(native_mapping(sd))
        expected = set(oracle_mapping(sd)[0])
        assert got == expected, f'{layout}: native={sorted(got - expected)} missing={sorted(expected - got)}'
        assert got <= set(LINEAR_SHAPES), f'{layout}: not real modules: {sorted(got - set(LINEAR_SHAPES))}'
    return True


def test_fc1_output_halves_swapped():
    """fc1 gate rows land on the second half of ff.net.0.proj and value rows on the first."""
    sd = lora_pair('diffusion_model.blocks.0.mlp.fc1', ref_shape('blocks.0.mlp.fc1'))
    up = sd['diffusion_model.blocks.0.mlp.fc1.lora_B.weight']
    down, bound_up, _scale = native_mapping(sd)['transformer_blocks.0.ff.net.0.proj']
    half = up.shape[0] // 2
    assert torch.equal(bound_up[:half], up[half:]), 'value rows must lead'
    assert torch.equal(bound_up[half:], up[:half]), 'gate rows must trail'
    assert torch.equal(down, sd['diffusion_model.blocks.0.mlp.fc1.lora_A.weight']), 'down is untouched'
    return True


def test_fc1_row_extras_follow_the_swap():
    """Per-output extras on fc1 (bias delta, DoRA magnitude) are permuted with the up rows."""
    sd = lora_pair('diffusion_model.blocks.0.mlp.fc1', ref_shape('blocks.0.mlp.fc1'))
    out = ref_shape('blocks.0.mlp.fc1')[0]
    sd['diffusion_model.blocks.0.mlp.fc1.diff_b'] = torch.arange(out, dtype=torch.float32)
    sd['diffusion_model.blocks.0.mlp.fc1.dora_scale'] = torch.arange(out, dtype=torch.float32).reshape(out, 1)
    net = load_native(sd, name='fc1extras')
    module = net.modules['lora_transformer_transformer_blocks_0_ff_net_0_proj']
    half = out // 2
    assert torch.equal(module.ex_bias[:half], torch.arange(half, out, dtype=torch.float32))
    assert torch.equal(module.dora_scale[:half, 0], torch.arange(half, out, dtype=torch.float32))
    return True


def test_chunk_reorder_composes_with_slice():
    """A ChunkSpec reorder applies to the rows its slice selects."""
    t = torch.arange(8).reshape(8, 1)
    got = native_adapter.slice_chunk_rows(t, native_adapter.ChunkSpec(idx=1, total=2, reorder=(1, 0)))
    assert got.flatten().tolist() == [6, 7, 4, 5], got.flatten().tolist()
    got = native_adapter.slice_chunk_rows(t, native_adapter.ChunkSpec(reorder=(1, 0)))
    assert got.flatten().tolist() == [4, 5, 6, 7, 0, 1, 2, 3], got.flatten().tolist()
    return True


def test_qkv_split_order():
    """Fused qkv rows split as [q; k; v] onto to_q / to_k / to_v."""
    sd = lora_pair('diffusion_model.blocks.1.attn.qkv_proj', ref_shape('blocks.1.attn.qkv_proj'))
    up = sd['diffusion_model.blocks.1.attn.qkv_proj.lora_B.weight']
    mapping = native_mapping(sd)
    for i, proj in enumerate(('to_q', 'to_k', 'to_v')):
        _down, bound_up, _scale = mapping[f'transformer_blocks.1.attn.{proj}']
        assert torch.equal(bound_up, torch.chunk(up, 3, dim=0)[i]), f'{proj} rows'
    return True


def test_diffusers_peft_keys_bind_verbatim():
    """A diffusers-PEFT save is already on diffusers names and gets no permutation."""
    sd = lora_pair('transformer.transformer_blocks.0.ff.net.0.proj', ref_shape('blocks.0.mlp.fc1'))
    down, up, _scale = native_mapping(sd)['transformer_blocks.0.ff.net.0.proj']
    assert torch.equal(up, sd['transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight'])
    assert torch.equal(down, sd['transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight'])
    return True


def test_unknown_bare_key_reaches_the_resolver():
    """A bare key that names no module is offered verbatim and left unbound, not dropped at parse time."""
    sd = lora_pair('transformer_blocks.0.attn.to_q', LINEAR_SHAPES['transformer_blocks.0.attn.to_q'])
    sd.update(lora_pair('nowhere.proj', (4, 4)))
    mapping = native_mapping(sd)
    assert set(mapping) == {'transformer_blocks.0.attn.to_q', 'nowhere.proj'}, sorted(mapping)
    net = load_native(sd, name='stray')
    assert set(net.modules) == {'lora_transformer_transformer_blocks_0_attn_to_q'}, sorted(net.modules)
    assert net.mismatch == 0
    return True


# ============================================================
# Tests - loader against the reference loaders
# ============================================================


def _loader_matches_converter(layout, name, rewrite=None):
    sd = synth(**layout)
    net = load_native(sd, name=name)
    assert net is not None, 'nothing bound'
    assert net.mismatch == 0, f'mismatch={net.mismatch}'
    reference = diffusers_deltas(sd, rewrite=rewrite)
    assert set(net.modules) <= LINEAR_NETKEYS, f'bound to non-linear keys: {sorted(set(net.modules) - LINEAR_NETKEYS)}'
    assert_same_deltas(native_deltas(net), reference)
    assert len(net.modules) == len(reference), f'bound {len(net.modules)} modules, converter has {len(reference)}'
    return True


def test_comfy_layout_matches_converter():
    """diffusion_model.* keys with PEFT suffixes: the published turbo LoRA layout."""
    return _loader_matches_converter({'prefix': 'diffusion_model.'}, 'comfy')


def test_bare_reference_layout_matches_converter():
    """Bare reference names, as the reference generate.py saves them."""
    return _loader_matches_converter({'prefix': ''}, 'bare')


def test_musubi_kohya_alpha_matches_converter():
    """Flattened lora_unet_ names with kohya suffixes and a non-trivial alpha."""
    return _loader_matches_converter({'flatten': True, 'suffix': KOHYA, 'alpha': RANK / 2}, 'musubi')


def test_peft_wrapped_dit_layout_matches_converter():
    """A peft dump wrapping the reference model under a dit attribute maps like diffusion_model."""
    return _loader_matches_converter({'prefix': 'base_model.model.dit.'}, 'dit', rewrite=('base_model.model.dit.', 'diffusion_model.'))


def test_block_diagonal_qkv_with_tripled_alpha_matches_converter():
    """A fused qkv stored as stacked A and block-diagonal B with alpha tripled applies each projection at alpha / rank."""
    out_q, inp = LINEAR_SHAPES['transformer_blocks.0.attn.to_q']
    alpha = RANK / 2
    downs = [torch.randn(RANK, inp) for _ in range(3)]
    ups = [torch.randn(out_q, RANK) for _ in range(3)]
    sd = {
        'diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight': torch.cat(downs, dim=0),
        'diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight': torch.block_diag(*ups),
        'diffusion_model.blocks.0.attn.qkv_proj.alpha': torch.tensor(3 * alpha),
    }
    net = load_native(sd, name='blockdiag')
    got = native_deltas(net)
    assert_same_deltas(got, diffusers_deltas(sd))
    for i, proj in enumerate(('to_q', 'to_k', 'to_v')):
        expected = (ups[i] @ downs[i]) * (alpha / RANK)
        assert torch.allclose(got[f'lora_transformer_transformer_blocks_0_attn_{proj}'], expected, rtol=1e-5, atol=1e-5), f'{proj} is not its own projection at alpha / rank'
    return True


def test_peft_dump_layout_binds_verbatim():
    """Diffusers names carrying peft's .default. slot and no component prefix bind one to one."""
    sd = synth_diffusers(infix='.default')
    net = load_native(sd, name='peftdump')
    assert net is not None and net.mismatch == 0
    assert_same_deltas(native_deltas(net), identity_deltas({k.replace('.default.', '.'): v for k, v in sd.items()}))
    assert len(net.modules) == len(LINEAR_SHAPES)
    return True


def test_diffusers_names_with_kohya_suffixes_bind():
    """Diffusers names with lora_down / lora_up suffixes and no alpha bind at scale 1."""
    sd = synth_diffusers(suffix=KOHYA)
    net = load_native(sd, name='kohyadiff')
    assert net is not None and net.mismatch == 0
    assert_same_deltas(native_deltas(net), identity_deltas(sd))
    return True


def test_metadata_alpha_scales_an_alphaless_file():
    """A file-level alpha in the safetensors metadata scales every module by alpha / rank."""
    sd = synth_diffusers(infix='.default')
    net = load_native(sd, name='metaalpha', metadata={'alpha': '2'})
    assert_same_deltas(native_deltas(net), identity_deltas({k.replace('.default.', '.'): v for k, v in sd.items()}, scale=2 / RANK))
    return True


def test_metadata_alpha_yields_to_alpha_tensors():
    """A file carrying any alpha tensor keeps its own scaling and ignores the metadata alpha."""
    sd = synth_diffusers()
    sd['proj_out.alpha'] = torch.tensor(RANK / 2)
    net = load_native(sd, name='mixedalpha', metadata={'alpha': '2'})
    got = native_deltas(net)
    expected = identity_deltas({k: v for k, v in sd.items() if not k.endswith('.alpha')})
    expected['lora_transformer_proj_out'] = expected['lora_transformer_proj_out'] * 0.5
    assert_same_deltas(got, expected)
    return True


def test_non_numeric_metadata_alpha_is_ignored():
    """A metadata alpha that is not a number leaves the file at alpha == rank."""
    sd = synth_diffusers()
    net = load_native(sd, name='badalpha', metadata={'alpha': 'n/a'})
    assert_same_deltas(native_deltas(net), identity_deltas(sd))
    return True


# ============================================================
# Tests - real files
# ============================================================

REAL_FILES = sorted(glob.glob(os.path.join(shared.opts.lora_dir, 'MiniMax H3', '*.safetensors')))


def test_real_files_match_reference_loaders():
    """Every local MiniMax H3 LoRA maps onto the same targets as its reference loader, with the same factors."""
    if not REAL_FILES:
        log.warning('  no local MiniMax H3 LoRA files, skipped')
        return True
    for path in REAL_FILES:
        name = os.path.basename(path)
        with safetensors.safe_open(path, framework='pt') as f:
            metadata = f.metadata() or {}
        network_alpha = M.file_alpha(_MockNetworkOnDisk(path, name, metadata))
        sd = safetensors.torch.load_file(path)
        native = native_mapping(sd, network_alpha)
        oracle, ignored = oracle_mapping(sd, network_alpha)
        probes = assert_same_factors(native, oracle, name)
        note = f' alpha=file:{network_alpha}' if network_alpha is not None else (' alpha=keys' if probes else '')
        note += f' ignored={len(ignored)}' if ignored else ''
        log.info(f'  {name}: targets={len(native)} {"probed" if probes else "identical"}{note}')
        del sd, native, oracle
    return True


# ============================================================
# Runner
# ============================================================


def run_tests():
    t0 = time.time()
    log.warning('=== MiniMax H3 native adapter tests ===')
    log.warning('=== Resolution ===')
    for fn in [
        test_every_reference_module_resolves_to_a_real_linear,
        test_fc1_output_halves_swapped,
        test_fc1_row_extras_follow_the_swap,
        test_chunk_reorder_composes_with_slice,
        test_qkv_split_order,
        test_diffusers_peft_keys_bind_verbatim,
        test_unknown_bare_key_reaches_the_resolver,
    ]:
        run_test(CAT_RESOLVE, fn)

    log.warning('=== Loader vs reference loaders ===')
    for fn in [
        test_comfy_layout_matches_converter,
        test_bare_reference_layout_matches_converter,
        test_musubi_kohya_alpha_matches_converter,
        test_peft_wrapped_dit_layout_matches_converter,
        test_block_diagonal_qkv_with_tripled_alpha_matches_converter,
        test_peft_dump_layout_binds_verbatim,
        test_diffusers_names_with_kohya_suffixes_bind,
        test_metadata_alpha_scales_an_alphaless_file,
        test_metadata_alpha_yields_to_alpha_tensors,
        test_non_numeric_metadata_alpha_is_ignored,
    ]:
        run_test(CAT_LOADER, fn)

    log.warning('=== Real files ===')
    run_test(CAT_REAL, test_real_files_match_reference_loaders)

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
