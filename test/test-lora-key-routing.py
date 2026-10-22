#!/usr/bin/env python
"""
Offline tests for where LoRA keys land, run through ``lora_load.load_safetensors``.

The model trees are meta-device diffusers and transformers modules, so the
tests check bindings against the real module layout of each architecture
whichever loader serves it. Key names are copied from real files; tensors are
synthetic except in the fixture case, which reads a real file from the shared
models directory and skips when it is absent.

Covered:

- SD3 kohya: ``context_block`` carries the text stream (``add_*_proj``),
  ``x_block`` the image stream (``to_q/k/v``).
- SD3 kohya: the pre-only last block's context modulation is stored
  (shift, scale) and read (scale, shift).
- Flux kohya without alpha tensors loads at scale 1.
- A load leaves the cached state dict whole.
- SD 1.x: ``lora_te1_`` keys bind to the single text encoder.
- The native chain warns once per file when some targets name no module.

No running server required.

Usage:
    python test/test-lora-key-routing.py
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

import diffusers  # pylint: disable=wrong-import-position
import transformers  # pylint: disable=wrong-import-position
from modules.errors import log  # pylint: disable=wrong-import-position
from modules import shared, sd_models_utils  # pylint: disable=wrong-import-position
from modules.modeldata import model_data  # pylint: disable=wrong-import-position
from modules.lora import lora_load, network, native_adapter  # pylint: disable=wrong-import-position

FIXTURE_SD35L = '/home/ohiom/database/models/Lora/SD3/Photorealistic-SD3.5-Large-LoRA.safetensors'
RANK = 4  # not a multiple of three, so fused qkv takes the dense split


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, status: str, name: str, detail: str = ''):
    key = {'PASS': 'passed', 'FAIL': 'failed', 'SKIP': 'skipped'}[status]
    results[cat][key] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}' + (f' ({detail})' if detail else '')
    if status == 'FAIL':
        log.error(msg)
    elif status == 'SKIP':
        log.warning(msg)
    else:
        log.info(msg)


class Skip(Exception):
    pass


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        record(cat, 'FAIL' if ok is False else 'PASS', name)
    except Skip as e:
        record(cat, 'SKIP', name, str(e))
    except AssertionError as e:
        record(cat, 'FAIL', name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, 'FAIL', name, f'exception: {e}')
        import traceback
        traceback.print_exc()


# ============================================================
# Model trees, named like the pipelines so the model type resolves
# ============================================================

class StableDiffusion3Pipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.embedding_db = None


class FluxPipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.embedding_db = None


class StableDiffusionPipeline:
    def __init__(self, unet, text_encoder):
        self.unet = unet
        self.text_encoder = text_encoder
        self.embedding_db = None


def install(pipe):
    model_data.sd_model = pipe
    lora_load.lora_cache.clear()
    sd_models_utils.state_dict_cache.disable() # drops entries from earlier tests
    sd_models_utils.state_dict_cache.enable()
    return pipe


def sd3_tree(num_layers=2, heads=4, head_dim=8):
    with torch.device('meta'):
        return diffusers.SD3Transformer2DModel(
            sample_size=8, patch_size=2, in_channels=4, out_channels=4, num_layers=num_layers,
            attention_head_dim=head_dim, num_attention_heads=heads, joint_attention_dim=32,
            caption_projection_dim=heads * head_dim, pooled_projection_dim=16, pos_embed_max_size=8,
        )


def sd35_large_tree():
    with torch.device('meta'):
        return diffusers.SD3Transformer2DModel(
            sample_size=128, patch_size=2, in_channels=16, out_channels=16, num_layers=38,
            attention_head_dim=64, num_attention_heads=38, joint_attention_dim=4096,
            caption_projection_dim=2432, pooled_projection_dim=2048, pos_embed_max_size=192, qk_norm='rms_norm',
        )


def flux_tree():
    with torch.device('meta'):
        return diffusers.FluxTransformer2DModel(
            patch_size=1, in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=8,
            num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(2, 2, 4),
        )


def sd15_trees():
    with torch.device('meta'):
        unet = diffusers.UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
            down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'), up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
            cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8,
        )
        te = transformers.CLIPTextModel(transformers.CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, vocab_size=64, max_position_embeddings=16))
    return unet, te


def write(state_dict):
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    os.close(fd)
    safetensors.torch.save_file({k: v.clone().contiguous() for k, v in state_dict.items()}, path)
    return path


def load(path, name):
    return lora_load.load_safetensors(name, network.NetworkOnDisk(name, path))


def factors(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(RANK, cols, generator=g), torch.randn(rows, RANK, generator=g)


def fitted_residual(got, want):
    """||got - c * want|| / ||got|| for the best constant c; 0 for a pure rescale, ~1.4 for an unrelated tensor."""
    c = (got * want).sum() / (want * want).sum()
    return ((got - c * want).norm() / got.norm()).item()


def up_of(net, key):
    module = net.modules.get(key)
    assert module is not None, f'{key} not bound; bound={sorted(net.modules)[:6]}'
    return module.up_model.weight.float()


# ============================================================
# Tests
# ============================================================

def test_sd3_kohya_streams():
    install(StableDiffusion3Pipeline(sd3_tree()))
    dim = 32
    sd = {}
    for stream, seed in (('context_block', 1), ('x_block', 2)):
        down, up = factors(3 * dim, dim, seed)
        base = f'lora_unet_joint_blocks_0_{stream}_attn_qkv'
        sd[f'{base}.lora_down.weight'], sd[f'{base}.lora_up.weight'], sd[f'{base}.alpha'] = down, up, torch.tensor(float(RANK))
    path = write(sd)
    try:
        net = load(path, 'sd3-streams')
        assert net is not None, 'not loaded'
        for stream, targets in (('context_block', ('add_q_proj', 'add_k_proj', 'add_v_proj')), ('x_block', ('to_q', 'to_k', 'to_v'))):
            chunks = torch.chunk(sd[f'lora_unet_joint_blocks_0_{stream}_attn_qkv.lora_up.weight'], 3, dim=0)
            for chunk, target in zip(chunks, targets, strict=True):
                got = up_of(net, f'lora_transformer_transformer_blocks_0_attn_{target}')
                assert torch.equal(got, chunk), f'{stream} chunk did not land on {target}'
    finally:
        os.unlink(path)


def test_sd3_kohya_last_block_modulation_halves():
    install(StableDiffusion3Pipeline(sd3_tree(num_layers=2)))
    dim = 32
    down, up = factors(2 * dim, dim, 3)
    base = 'lora_unet_joint_blocks_1_context_block_adaLN_modulation_1'
    path = write({f'{base}.lora_down.weight': down, f'{base}.lora_up.weight': up, f'{base}.alpha': torch.tensor(float(RANK))})
    try:
        net = load(path, 'sd3-last-block')
        assert net is not None, 'not loaded'
        got = up_of(net, 'lora_transformer_transformer_blocks_1_norm1_context_linear')
        shift, scale = torch.chunk(up, 2, dim=0)
        assert torch.equal(got, torch.cat([scale, shift], dim=0)), 'pre-only modulation halves not swapped'
    finally:
        os.unlink(path)


def test_sd3_kohya_full_block_modulation_unswapped():
    install(StableDiffusion3Pipeline(sd3_tree(num_layers=2)))
    dim = 32
    down, up = factors(6 * dim, dim, 4)
    base = 'lora_unet_joint_blocks_0_context_block_adaLN_modulation_1'
    path = write({f'{base}.lora_down.weight': down, f'{base}.lora_up.weight': up, f'{base}.alpha': torch.tensor(float(RANK))})
    try:
        net = load(path, 'sd3-full-block')
        assert net is not None, 'not loaded'
        assert torch.equal(up_of(net, 'lora_transformer_transformer_blocks_0_norm1_context_linear'), up), 'full-block modulation must keep its order'
    finally:
        os.unlink(path)


def test_sd35_large_kohya_fixture():
    if not os.path.isfile(FIXTURE_SD35L):
        raise Skip(f'fixture absent: {FIXTURE_SD35L}')
    install(StableDiffusion3Pipeline(sd35_large_tree()))
    sd = safetensors.torch.load_file(FIXTURE_SD35L)
    net = load(FIXTURE_SD35L, 'sd35l-fixture')
    assert net is not None, 'not loaded'
    groups = {k.rsplit('.', 2)[0] for k in sd if k.endswith('.lora_down.weight')}
    fused = sum(1 for g in groups if g.endswith('_attn_qkv'))
    expected = len(groups) + 2 * fused
    assert len(net.modules) == expected, f'bound {len(net.modules)} modules, file has {len(groups)} groups ({fused} fused qkv) so {expected} targets'
    # an alpha fold may rescale the up factor, so compare up to a fitted constant; the key rows are checked because
    # the last block's context query feeds no output, trains to zero and cannot tell the streams apart
    for i in (0, 37):
        context_k = torch.chunk(sd[f'lora_unet_joint_blocks_{i}_context_block_attn_qkv.lora_up.weight'].float(), 3, dim=0)[1]
        image_k = torch.chunk(sd[f'lora_unet_joint_blocks_{i}_x_block_attn_qkv.lora_up.weight'].float(), 3, dim=0)[1]
        assert fitted_residual(up_of(net, f'lora_transformer_transformer_blocks_{i}_attn_add_k_proj'), context_k) < 1e-2, f'block {i}: context k rows not on add_k_proj'
        assert fitted_residual(up_of(net, f'lora_transformer_transformer_blocks_{i}_attn_to_k'), image_k) < 1e-2, f'block {i}: image k rows not on to_k'
    up = sd['lora_unet_joint_blocks_37_context_block_adaLN_modulation_1.lora_up.weight'].float()
    got = up_of(net, 'lora_transformer_transformer_blocks_37_norm1_context_linear')
    shift, scale = torch.chunk(up, 2, dim=0)
    residual = fitted_residual(got, torch.cat([scale, shift], dim=0))
    assert residual < 1e-2, f'last block modulation residual {residual:.3g} against the swapped halves'


def test_flux_kohya_without_alpha():
    install(FluxPipeline(flux_tree()))
    inner = 16
    down, up = factors(inner, inner, 5)
    base = 'lora_unet_double_blocks_0_img_attn_proj'
    path = write({f'{base}.lora_down.weight': down, f'{base}.lora_up.weight': up})
    try:
        net = load(path, 'flux-no-alpha')
        assert net is not None, 'a file without alpha tensors must load'
        module = net.modules.get('lora_transformer_transformer_blocks_0_attn_to_out_0')
        assert module is not None, f'bound={sorted(net.modules)}'
        assert torch.equal(module.up_model.weight.float(), up) and torch.equal(module.down_model.weight.float(), down), 'factors changed without an alpha to fold'
        assert module.calc_scale() == 1.0
    finally:
        os.unlink(path)


def test_cached_state_dict_left_whole():
    install(StableDiffusion3Pipeline(sd3_tree()))
    dim = 32
    down, up = factors(3 * dim, dim, 6)
    base = 'lora_unet_joint_blocks_0_x_block_attn_qkv'
    sd = {f'{base}.lora_down.weight': down, f'{base}.lora_up.weight': up, f'{base}.alpha': torch.tensor(float(RANK)), 'lora_unet_joint_blocks_0_x_block_attn2_qkv.lora_down.weight': down}
    path = write(sd)
    try:
        load(path, 'sd3-cache')
        cached = sd_models_utils.state_dict_cache.get(path)
        if cached is not None:
            assert set(cached) == set(sd), f'cached dict lost keys: {sorted(set(sd) - set(cached))}'
    finally:
        os.unlink(path)


def test_sd15_te1_prefix_binds_single_encoder():
    unet, te = sd15_trees()
    install(StableDiffusionPipeline(unet, te))
    q_proj = next(m for n, m in te.named_modules() if n.endswith('layers.0.self_attn.q_proj'))
    down, up = factors(32, 32, 7)
    sd = {
        'lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight': down,
        'lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight': up,
        'lora_te1_text_model_encoder_layers_0_self_attn_q_proj.alpha': torch.tensor(float(RANK)),
        'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight': down,
        'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight': up[:32],
        'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.alpha': torch.tensor(float(RANK)),
    }
    path = write(sd)
    try:
        net = load(path, 'sd15-te1')
        assert net is not None, 'not loaded'
        stamped = q_proj.network_layer_name
        assert stamped in net.modules, f'te key not bound to {stamped}; bound={sorted(net.modules)}'
        assert any(k.startswith('lora_unet_') for k in net.modules), 'unet half missing'
    finally:
        os.unlink(path)


def chain_warnings(family_loaders):
    """Run try_load_chain over stub family loaders; return the WARNING messages it logged."""
    warnings = []
    real_warning = native_adapter.log.warning
    native_adapter.log.warning = lambda msg, *a, **k: warnings.append(msg)
    try:
        net = native_adapter.try_load_chain('chain', None, 1.0, family_loaders)
    finally:
        native_adapter.log.warning = real_warning
    return net, warnings


def stub_net(modules, unmapped=()):
    def family(name, network_on_disk, lora_scale): # pylint: disable=unused-argument
        for target in unmapped:
            native_adapter.note_unmapped(target)
        if not modules:
            return None
        net = network.Network(name, network.NetworkOnDisk(name, __file__))
        net.modules = dict.fromkeys(modules)
        return net
    return family


def test_native_chain_warns_on_unmapped():
    net, warnings = chain_warnings((stub_net(['a', 'b'], unmapped=['x']), stub_net([], unmapped=['y', 'x'])))
    assert net is not None and len(net.modules) == 2
    assert len(warnings) == 1, f'expected one warning per file, got {warnings}'
    assert 'native modules=2 unmapped=2' in warnings[0], warnings[0]


def test_native_chain_quiet_when_all_bound():
    net, warnings = chain_warnings((stub_net(['a']),))
    assert net is not None and not warnings, warnings


def test_native_chain_quiet_when_nothing_bound():
    net, warnings = chain_warnings((stub_net([], unmapped=['x']),))
    assert net is None and not warnings, 'a file that binds nothing is reported by the not-loaded error, not this warning'


# ============================================================
# Runner
# ============================================================

def run_tests():
    t0 = time.time()
    cat = category('routing')
    log.warning('=== LoRA key routing through load_safetensors ===')
    for fn in [
        test_sd3_kohya_streams,
        test_sd3_kohya_last_block_modulation_halves,
        test_sd3_kohya_full_block_modulation_unswapped,
        test_sd35_large_kohya_fixture,
        test_flux_kohya_without_alpha,
        test_cached_state_dict_left_whole,
        test_sd15_te1_prefix_binds_single_encoder,
        test_native_chain_warns_on_unmapped,
        test_native_chain_quiet_when_all_bound,
        test_native_chain_quiet_when_nothing_bound,
    ]:
        run_test(cat, fn)
    model_data.sd_model = None
    total_fail = sum(info['failed'] for info in results.values())
    for name, info in results.items():
        log.info(f'  {name}: {info["passed"]} passed, {info["failed"]} failed, {info["skipped"]} skipped')
    log.warning(f'Total: {sum(i["passed"] for i in results.values())} passed, {total_fail} failed in {time.time() - t0:.2f}s')
    return total_fail == 0


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
