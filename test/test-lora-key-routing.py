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
- Narrower-input deltas zero-pad onto widened modules (inpaint ``conv_in``,
  Flux Fill ``x_embedder``); wider or per-input DoRA deltas stay mismatches.
- Text-encoder prefixes route by encoder role: kohya ``lora_te3_`` is the T5,
  ``lora_te_`` names only a lone encoder, the ``text_model`` wrapper matches
  either tree shape.
- Underscore-namespace keys with a dotted tail group with their factors.
- A pure row reorder applies exactly to LoKr, LoHA and per-input DoRA.
- Every family checks its delta against the module shape, so a misfit file
  is refused whole.
- ``SD_LORA_DUMP`` writes the key lists for native arches too.
- ``module=`` and ``high``/``low`` tags are reported on the native method;
  ``cond``/``uncond`` name the asymmetric-CFG towers.

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
from modules.lora import lora_convert, lora_load, network, native_adapter  # pylint: disable=wrong-import-position

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
    path = write({'unrelated': torch.zeros(1)})
    real_warning = native_adapter.log.warning
    native_adapter.log.warning = lambda msg, *a, **k: warnings.append(msg)
    try:
        net = native_adapter.try_load_chain('chain', network.NetworkOnDisk('chain', path), 1.0, family_loaders)
    finally:
        native_adapter.log.warning = real_warning
        os.unlink(path)
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


def load_native(state_dict, mapping):
    """try_load_lora over a file holding state_dict, with the mapping replaced by the given modules."""
    path = write(state_dict)
    real = native_adapter.resolve_mapping
    native_adapter.resolve_mapping = lambda: mapping
    try:
        return native_adapter.try_load_lora('widen', network.NetworkOnDisk('widen', path), 1.0, resolve_targets=lambda prefix, base: [])
    finally:
        native_adapter.resolve_mapping = real
        os.unlink(path)


def widen_case(module, down, up, key='lora_transformer_target', **extra):
    state_dict = {f'{key}.lora_down.weight': down, f'{key}.lora_up.weight': up, **{f'{key}.{k}': v for k, v in extra.items()}}
    net = load_native(state_dict, {key: module})
    return None if net is None else net.modules.get(key)


def test_widen_inpaint_conv_in():
    conv = torch.nn.Conv2d(9, 32, 3, padding=1)
    down, up = torch.randn(RANK, 4, 3, 3), torch.randn(32, RANK, 1, 1)
    module = widen_case(conv, down, up)
    assert module is not None, 'a 4-channel conv_in delta must bind to a 9-channel conv_in'
    updown, _ = module.calc_updown(conv.weight.detach())
    assert updown.shape == conv.weight.shape, updown.shape
    assert updown[:, 4:].abs().max() == 0, 'the added channels must not move'
    want = (up.reshape(32, RANK) @ down.reshape(RANK, -1)).reshape(32, 4, 3, 3)
    assert torch.allclose(updown[:, :4], want, atol=1e-5), 'the trained channels must carry the delta'


def test_widen_fill_x_embedder():
    linear = torch.nn.Linear(384, 48)
    down, up = torch.randn(RANK, 64), torch.randn(48, RANK)
    module = widen_case(linear, down, up)
    assert module is not None, 'a 64-wide x_embedder delta must bind to a 384-wide x_embedder'
    updown, _ = module.calc_updown(linear.weight.detach())
    assert updown[:, 64:].abs().max() == 0 and torch.allclose(updown[:, :64], up @ down, atol=1e-5)


def test_widen_refuses_wider_delta():
    linear = torch.nn.Linear(64, 48)
    assert widen_case(linear, torch.randn(RANK, 384), torch.randn(48, RANK)) is None, 'a wider delta is a mismatch'


def test_widen_per_input_dora():
    conv = torch.nn.Conv2d(9, 32, 3, padding=1)
    down, up = torch.randn(RANK, 4, 3, 3), torch.randn(32, RANK, 1, 1)
    dora = torch.rand(1, 4, 1, 1) + 0.5
    module = widen_case(conv, down, up, dora_scale=dora)
    assert module is not None, 'a per-input DoRA conv_in must bind to a 9-channel conv_in'
    updown, _ = module.calc_updown(conv.weight.detach())
    assert updown.shape == conv.weight.shape, updown.shape
    assert updown[:, 4:].abs().max() < 1e-5, 'the added channels must not move'
    narrow = torch.nn.Conv2d(4, 32, 3, padding=1)
    narrow.weight.data.copy_(conv.weight.data[:, :4])
    want, _ = widen_case(narrow, down, up, dora_scale=dora).calc_updown(narrow.weight.detach())
    assert torch.allclose(updown[:, :4], want, atol=1e-5), 'the trained channels must match DoRA on the narrow weight'


def test_narrow_padded_input():
    linear = torch.nn.Linear(576, 48)
    down, up = torch.randn(RANK, 768), torch.randn(48, RANK)
    down[:, 576:] = 0
    module = widen_case(linear, down, up)
    assert module is not None, 'a delta whose columns past the input width are zero must bind to the unpadded module'
    updown, _ = module.calc_updown(linear.weight.detach())
    assert updown.shape == linear.weight.shape, updown.shape
    assert torch.allclose(updown, up @ down[:, :576], atol=1e-5), 'the live columns must carry the delta'


def test_narrow_refuses_trained_tail():
    linear = torch.nn.Linear(576, 48)
    down = torch.randn(RANK, 768)
    down[:, 576:] = 0
    down[0, 700] = 1.0
    assert widen_case(linear, down, torch.randn(48, RANK)) is None, 'a value past the input width is a mismatch, not padding'


def test_narrow_per_input_dora():
    linear = torch.nn.Linear(576, 48)
    down, up = torch.randn(RANK, 768), torch.randn(48, RANK)
    down[:, 576:] = 0
    dora = torch.rand(1, 768) + 0.5
    module = widen_case(linear, down, up, dora_scale=dora)
    assert module is not None, 'a per-input DoRA saved at the padded width must bind'
    updown, _ = module.calc_updown(linear.weight.detach())
    want, _ = widen_case(linear, down[:, :576].clone(), up, dora_scale=dora[:, :576].clone()).calc_updown(linear.weight.detach())
    assert torch.allclose(updown, want, atol=1e-5), 'trimming must match a delta saved at the module width'


def clip(hidden, projection=False):
    config = transformers.CLIPTextConfig(hidden_size=hidden, intermediate_size=4 * hidden, num_hidden_layers=1, num_attention_heads=hidden // 64, projection_dim=hidden)
    with torch.device('meta'):
        return transformers.CLIPTextModelWithProjection(config) if projection else transformers.CLIPTextModel(config)


def t5_encoder():
    with torch.device('meta'):
        return transformers.T5EncoderModel(transformers.T5Config(d_model=64, d_kv=32, d_ff=128, num_layers=1, num_heads=2, vocab_size=64))


class TextEncoders:
    def __init__(self, **components):
        self.embedding_db = None
        for name, component in components.items():
            setattr(self, name, component)


def router_for(**components):
    pipe = TextEncoders(**components)
    model_data.sd_model = pipe
    lora_convert.assign_network_names_to_compvis_modules(pipe)
    return native_adapter.TextEncoderRouter(pipe, shared.sd_model.network_layer_mapping)


def test_te_routing_flux():
    router = router_for(text_encoder=clip(768), text_encoder_2=t5_encoder())
    cases = {'lora_te1_': 'lora_te1_', 'lora_te3_': 'lora_te2_', 'lora_te2_': 'lora_te2_', 'text_encoder_2.': 'lora_te2_', 't5.': 'lora_te2_', 'lora_te_': None, 'clip_g.': None}
    for prefix, want in cases.items():
        assert router.prefix(prefix) == want, f'{prefix}: {router.prefix(prefix)} != {want}'


def test_te_routing_sdxl():
    router = router_for(text_encoder=clip(768), text_encoder_2=clip(1280, projection=True))
    cases = {'lora_te1_': 'lora_te1_', 'lora_te2_': 'lora_te2_', 'clip_l.': 'lora_te1_', 'clip_g.': 'lora_te2_', 'text_encoder.': 'lora_te1_', 'lora_te_': None, 'lora_te3_': None}
    for prefix, want in cases.items():
        assert router.prefix(prefix) == want, f'{prefix}: {router.prefix(prefix)} != {want}'


def test_te_routing_sd15():
    router = router_for(text_encoder=clip(768))
    cases = {'lora_te_': 'lora_te_', 'lora_te1_': 'lora_te_', 'text_encoder.': 'lora_te_', 'clip_l.': 'lora_te_', 'lora_te2_': None}
    for prefix, want in cases.items():
        assert router.prefix(prefix) == want, f'{prefix}: {router.prefix(prefix)} != {want}'


def test_te_routing_sd3():
    router = router_for(text_encoder=clip(768, projection=True), text_encoder_2=clip(1280, projection=True), text_encoder_3=t5_encoder())
    assert router.prefix('lora_te1_') == 'lora_te1_' and router.prefix('lora_te2_') == 'lora_te2_'
    assert router.prefix('lora_te3_') == 'lora_te3_' and router.prefix('t5.') == 'lora_te3_' and router.prefix('text_encoder_3.') == 'lora_te3_'


def test_te_path_text_model_either_shape():
    router = router_for(text_encoder=clip(768), text_encoder_2=clip(1280, projection=True))
    mapping = shared.sd_model.network_layer_mapping
    for prefix in ('lora_te1_', 'lora_te2_'):
        for base in ('text_model_encoder_layers_0_self_attn_q_proj', 'encoder.layers.0.self_attn.q_proj', 'text_model.encoder.layers.0.self_attn.q_proj'):
            path = router.path(prefix, base)
            assert path is not None and router.prefix(prefix) + path in mapping, f'{prefix}{base} -> {path}'
    assert router.path('lora_te1_', 'encoder_layers_9_self_attn_q_proj') is None, 'a layer the tree lacks stays unmapped'


FIXTURE_MERVE_KOHYA = '/home/ohiom/database/models/Lora/SDXL 1.0/merve_lego-sdxl-dora_kohya.safetensors'


def test_dotted_tail_groups_with_factors():
    sd = {
        'lora_unet_blk_attn1_to_k.lora_down.weight': torch.zeros(2, 4),
        'lora_unet_blk_attn1_to_k.lora_up.weight': torch.zeros(4, 2),
        'lora_unet_blk_attn1.to_k.lora_magnitude_vector': torch.ones(4),
        'lora_unet_blk_attn1_to_out_0.lora_down.weight': torch.zeros(2, 4),
        'lora_unet_blk_attn1_to_out_0.lora_up.weight': torch.zeros(4, 2),
        'lora_unet_blk_attn1_to_out.0.lora_magnitude_vector': torch.ones(4),
    }
    groups = native_adapter.group_by_suffixes(sd, native_adapter.LORA_SUFFIXES)
    assert set(groups) == {('lora_unet_', 'blk_attn1_to_k'), ('lora_unet_', 'blk_attn1_to_out_0')}, sorted(groups)
    assert all('lora_magnitude_vector' in w and 'lora_down.weight' in w for w in groups.values())


def test_dotted_tail_fixture():
    if not os.path.isfile(FIXTURE_MERVE_KOHYA):
        raise Skip(f'fixture absent: {FIXTURE_MERVE_KOHYA}')
    groups = native_adapter.group_by_suffixes(safetensors.torch.load_file(FIXTURE_MERVE_KOHYA), native_adapter.LORA_SUFFIXES)
    factored = [w for w in groups.values() if 'lora_down.weight' in w]
    assert factored and all('lora_magnitude_vector' in w for w in factored), 'every factored group carries its magnitude'
    assert len(factored) == len(groups), f'{len(groups) - len(factored)} groups hold a magnitude without factors'


def load_reordered(loader, state_dict, module):
    """Run a family loader with one target that swaps the halves of its rows."""
    key = 'lora_transformer_target'
    path = write({f'{key}.{k}': v for k, v in state_dict.items()})
    real = native_adapter.resolve_mapping
    native_adapter.resolve_mapping = lambda: {key: module}
    try:
        net = loader('reorder', network.NetworkOnDisk('reorder', path), 1.0, resolve_targets=lambda prefix, base: [(base, native_adapter.ChunkSpec(reorder=(1, 0)))])
    finally:
        native_adapter.resolve_mapping = real
        os.unlink(path)
    return None if net is None else net.modules.get(key)


def swap_halves(d):
    a, b = torch.chunk(d, 2, dim=0)
    return torch.cat([b, a], dim=0)


def test_reorder_lokr():
    linear = torch.nn.Linear(12, 8)
    w1, w2 = torch.randn(2, 3), torch.randn(4, 4)
    module = load_reordered(native_adapter.try_load_lokr, {'lokr_w1': w1, 'lokr_w2': w2}, linear)
    assert module is not None, 'a reordered LoKr target must bind'
    updown, _ = module.calc_updown(linear.weight.detach())
    assert torch.allclose(updown, swap_halves(torch.kron(w1, w2)), atol=1e-5)


def test_reorder_lokr_uneven_skips():
    linear = torch.nn.Linear(12, 8)
    module = load_reordered(native_adapter.try_load_lokr, {'lokr_w1': torch.randn(1, 3), 'lokr_w2': torch.randn(8, 4)}, linear)
    assert module is None, 'w1 with one row cannot carry a half swap'


def test_reorder_loha():
    linear = torch.nn.Linear(6, 8)
    w = {'hada_w1_a': torch.randn(8, 2), 'hada_w1_b': torch.randn(2, 6), 'hada_w2_a': torch.randn(8, 2), 'hada_w2_b': torch.randn(2, 6)}
    module = load_reordered(native_adapter.try_load_loha, w, linear)
    assert module is not None, 'a reordered LoHA target must bind'
    updown, _ = module.calc_updown(linear.weight.detach())
    want = (w['hada_w1_a'] @ w['hada_w1_b']) * (w['hada_w2_a'] @ w['hada_w2_b'])
    assert torch.allclose(updown, swap_halves(want), atol=1e-5)


def test_reorder_per_input_dora():
    linear = torch.nn.Linear(6, 8)
    down, up, magnitude = torch.randn(RANK, 6), torch.randn(8, RANK), torch.rand(1, 6) + 0.5
    module = load_reordered(native_adapter.try_load_lora, {'lora_down.weight': down, 'lora_up.weight': up, 'dora_scale': magnitude}, linear)
    assert module is not None, 'per-input DoRA survives a row reorder'
    weight = linear.weight.detach()
    updown, _ = module.calc_updown(weight)
    merged = weight + swap_halves(up @ down)
    want = merged * (magnitude / merged.norm(dim=0, keepdim=True)) - weight
    assert torch.allclose(updown, want, atol=1e-4)


def load_split(loader, state_dict, targets):
    """Run a family loader with one fused key split in equal chunks onto ``targets``; the modules in that order."""
    key = 'lora_transformer_target'
    path = write({f'{key}.{k}': v for k, v in state_dict.items()})
    real = native_adapter.resolve_mapping
    native_adapter.resolve_mapping = lambda: {f'{key}_{i}': module for i, module in enumerate(targets)}
    try:
        net = loader('split', network.NetworkOnDisk('split', path), 1.0, resolve_targets=lambda prefix, base: [(f'{base}_{i}', native_adapter.ChunkSpec(idx=i, total=len(targets))) for i in range(len(targets))])
    finally:
        native_adapter.resolve_mapping = real
        os.unlink(path)
    return None if net is None else [net.modules.get(f'{key}_{i}') for i in range(len(targets))]


def check_split_per_input_dora(modules, targets, delta, magnitude):
    """Each chunk must equal its rows of the DoRA on the fused weight."""
    assert modules is not None and all(m is not None for m in modules), 'every chunk of a per-input DoRA must bind'
    fused = torch.cat([t.weight.detach() for t in targets])
    merged = fused + delta
    want = merged * (magnitude / merged.norm(dim=0, keepdim=True)) - fused
    for module, target, rows in zip(modules, targets, torch.chunk(want, len(targets))):
        updown, _ = module.calc_updown(target.weight.detach())
        assert torch.allclose(updown, rows, atol=1e-4), f'max |got-want|={float((updown - rows).abs().max()):.2e}'


def test_split_per_input_dora_lora():
    targets = [torch.nn.Linear(6, 4) for _ in range(3)]
    down, up, magnitude = torch.randn(RANK, 6), torch.randn(12, RANK), torch.rand(1, 6) + 0.5
    modules = load_split(native_adapter.try_load_lora, {'lora_down.weight': down, 'lora_up.weight': up, 'dora_scale': magnitude}, targets)
    check_split_per_input_dora(modules, targets, up @ down, magnitude)


def test_split_per_input_dora_lokr():
    targets = [torch.nn.Linear(6, 4) for _ in range(3)]
    w1, w2, magnitude = torch.randn(3, 2), torch.randn(4, 3), torch.rand(1, 6) + 0.5
    modules = load_split(native_adapter.try_load_lokr, {'lokr_w1': w1, 'lokr_w2': w2, 'dora_scale': magnitude}, targets)
    check_split_per_input_dora(modules, targets, torch.kron(w1, w2), magnitude)


FAMILY_CASES = { # family: (loader, factors for a Linear(6, 8))
    'loha': ('try_load_loha', lambda: {'hada_w1_a': torch.randn(8, 2), 'hada_w1_b': torch.randn(2, 6), 'hada_w2_a': torch.randn(8, 2), 'hada_w2_b': torch.randn(2, 6)}),
    'oft': ('try_load_oft', lambda: {'oft_diag': torch.zeros(2, 4, 4)}),
    'ia3': ('try_load_ia3', lambda: {'weight': torch.randn(8), 'on_input': torch.tensor(False)}),
    'glora': ('try_load_glora', lambda: {'a1.weight': torch.randn(2, 6), 'a2.weight': torch.randn(6, 2), 'b1.weight': torch.randn(2, 6), 'b2.weight': torch.randn(8, 2), 'alpha': torch.tensor(2.0)}),
    'full': ('try_load_full', lambda: {'diff': torch.randn(8, 6)}),
}


def chain_one(loader_name, factors, module):
    key = 'lora_transformer_target'
    path = write({f'{key}.{k}': v for k, v in factors.items()})
    real = native_adapter.resolve_mapping
    native_adapter.resolve_mapping = lambda: {key: module}
    loader = getattr(native_adapter, loader_name)
    family = lambda name, nod, scale: loader(name, nod, scale, resolve_targets=lambda prefix, base: [])
    try:
        return native_adapter.try_load_chain('family', network.NetworkOnDisk('family', path), 1.0, (family,))
    finally:
        native_adapter.resolve_mapping = real
        os.unlink(path)


def test_family_shape_checks():
    for family, (loader, factors) in FAMILY_CASES.items():
        fits = chain_one(loader, factors(), torch.nn.Linear(6, 8))
        assert fits is not None and len(fits.modules) == 1, f'{family}: a fitting delta must bind'
        misfit = chain_one(loader, factors(), torch.nn.Linear(10, 16))
        assert misfit is None, f'{family}: a misfit delta must refuse the file'


def test_full_widens_inpaint_conv_in():
    conv = torch.nn.Conv2d(9, 8, 3)
    net = chain_one('try_load_full', {'diff': torch.randn(8, 4, 3, 3)}, conv)
    assert net is not None, 'a 4-channel conv_in diff must bind to a 9-channel conv_in'
    module = next(iter(net.modules.values()))
    updown, _ = module.calc_updown(conv.weight.detach())
    assert updown[:, 4:].abs().max() == 0 and updown.shape == conv.weight.shape


def test_full_narrows_padded_input():
    linear = torch.nn.Linear(576, 8)
    diff = torch.randn(8, 768)
    diff[:, 576:] = 0
    net = chain_one('try_load_full', {'diff': diff}, linear)
    assert net is not None, 'a diff whose columns past the input width are zero must bind to the unpadded module'
    module = next(iter(net.modules.values()))
    updown, _ = module.calc_updown(linear.weight.detach())
    assert updown.shape == linear.weight.shape and torch.allclose(updown, diff[:, :576], atol=1e-5)


class ZImagePipeline:
    def __init__(self, transformer):
        self.transformer = transformer
        self.embedding_db = None


def test_dump_covers_native_arch():
    install(ZImagePipeline(torch.nn.Sequential(torch.nn.Linear(4, 4))))
    path = write({'lora_transformer_0.lora_down.weight': torch.zeros(2, 4), 'lora_transformer_0.lora_up.weight': torch.zeros(4, 2)})
    real = lora_load.dump_lora_keys
    lora_load.dump_lora_keys = True
    try:
        lora_load.load_safetensors('dump', network.NetworkOnDisk('dump', path))
    finally:
        lora_load.dump_lora_keys = real
    try:
        mapping_file = os.path.join(tempfile.gettempdir(), 'Model-zimage-ZImagePipeline.txt')
        keys_file = os.path.join(tempfile.gettempdir(), f'LoRA-zimage-ZImagePipeline-{os.path.splitext(os.path.basename(path))[0]}.txt')
        assert os.path.isfile(keys_file), 'file keys not dumped for a native arch'
        with open(mapping_file, encoding='utf8') as f:
            assert 'lora_transformer_0' in f.read().split(), 'the dumped mapping lacks the stamped module'
        os.unlink(keys_file)
        os.unlink(mapping_file)
    finally:
        os.unlink(path)


def test_component_tags_parsed_and_missing_reported():
    from modules import extra_networks
    from modules.lora import extra_networks_lora
    install(ZImagePipeline(torch.nn.Sequential(torch.nn.Linear(4, 4))))
    params = [extra_networks.ExtraNetworkParams(items) for items in (['a', '1.0'], ['b_lowres', '1.0', 'high'], ['c', 'module=te'], ['d', '0.8', 'low', 'module=transformer'])]
    requested = extra_networks_lora.requested_components(params)
    assert requested == {'b_lowres': ['transformer'], 'c': ['text_encoder'], 'd': ['transformer_2', 'transformer']}, requested
    warnings = []
    real = extra_networks_lora.log.warning
    extra_networks_lora.log.warning = warnings.append
    extra_networks_lora.warned_components.clear()
    try:
        extra_networks_lora.warn_missing_components(requested)
        extra_networks_lora.warn_missing_components(requested)
    finally:
        extra_networks_lora.log.warning = real
    assert len(warnings) == 2 and all('missing' in w for w in warnings), f'text_encoder and transformer_2 are absent, each reported once: {warnings}'


def test_tower_tags_parsed():
    """cond and uncond name the asymmetric-CFG towers; a bare reference names no component."""
    from modules import extra_networks
    from modules.lora import extra_networks_lora
    install(ZImagePipeline(torch.nn.Sequential(torch.nn.Linear(4, 4))))
    params = [extra_networks.ExtraNetworkParams(items) for items in (['a', '1.0'], ['b', '0.7', 'uncond'], ['c', '1.0', 'cond', 'uncond'])]
    names, _te, _unet, _dyn, lora_modules, _blocks, components = extra_networks_lora.parse(None, params)
    assert dict(zip(names, lora_modules)) == {'a': [], 'b': ['unconditional_transformer'], 'c': ['transformer', 'unconditional_transformer']}, lora_modules
    assert components == [None, ['unconditional_transformer'], ['transformer', 'unconditional_transformer']], components
    assert extra_networks_lora.requested_components(params) == {'b': ['unconditional_transformer'], 'c': ['transformer', 'unconditional_transformer']}


def test_native_activate_stages_each_reference_its_tags():
    """The prompt path hands network_load one component list per reference, so one file named twice keeps both and the change signature tells the orders apart."""
    from modules import extra_networks
    from modules.lora import extra_networks_lora, lora_overrides, networks
    install(ZImagePipeline(torch.nn.Sequential(torch.nn.Linear(4, 4))))
    staged = []
    saved = lora_load.network_load, lora_overrides.get_method, networks.network_activate, networks.network_deactivate
    lora_load.network_load = lambda names, te, unet, dyn, lora_modules=None, block_specs=None, activate=True: staged.append((list(names), list(te), lora_modules))
    lora_overrides.get_method = lambda shorthash='': ('native', 'test')
    networks.network_activate = networks.network_deactivate = lambda *args, **kwargs: None
    extra_networks_lora.warned_components.clear()
    real_warning = extra_networks_lora.log.warning
    extra_networks_lora.log.warning = lambda *args, **kwargs: None # the mock pipeline has no unconditional tower
    try:
        lora = extra_networks_lora.ExtraNetworkLora()
        lora.activate(None, [extra_networks.ExtraNetworkParams(items) for items in (['x', '0.9', 'cond'], ['x', '0.4', 'uncond'])])
        lora.activate(None, [extra_networks.ExtraNetworkParams(items) for items in (['x', '0.9', 'uncond'], ['x', '0.4', 'uncond'])])
    finally:
        lora_load.network_load, lora_overrides.get_method, networks.network_activate, networks.network_deactivate = saved
        extra_networks_lora.log.warning = real_warning
    assert staged[0] == (['x', 'x'], [0.9, 0.4], [['transformer'], ['unconditional_transformer']]), staged[0]
    assert staged[1][2] == [['unconditional_transformer'], ['unconditional_transformer']], staged[1]
    sig = lora.signature
    assert sig(['x', 'x'], [0.9, 0.4], [[0.9] * 3, [0.4] * 3], None, [['transformer'], ['unconditional_transformer']]) != sig(['x', 'x'], [0.9, 0.4], [[0.9] * 3, [0.4] * 3], None, [['unconditional_transformer'], ['unconditional_transformer']])


def test_request_options_fold_into_overrides():
    """The per-request LoRA keywords become overrides for that request; an explicit override of the same key wins."""
    import inspect
    import types
    from modules import processing, processing_class
    p = types.SimpleNamespace(override_settings={'lora_force_reload': True}, lora_fuse_native=False, lora_fuse_diffusers=None, lora_force_reload=False, extra_networks_default_multiplier=0.6, lora_apply_tags=None)
    processing.apply_request_options(p)
    assert p.override_settings == {'lora_force_reload': True, 'lora_fuse_native': False, 'extra_networks_default_multiplier': 0.6}, p.override_settings
    keywords = inspect.signature(processing_class.StableDiffusionProcessing.__init__).parameters
    assert all(key in keywords for key in processing.REQUEST_OPTIONS), 'every request option must be a processing keyword'
    assert all(key in shared.opts.data_labels for key in processing.REQUEST_OPTIONS), 'every request option must be a setting, or the override is skipped'


def test_chain_modules_share_the_returned_network():
    install(ZImagePipeline(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))))
    sd = {'lora_transformer_0.lora_down.weight': torch.randn(2, 4), 'lora_transformer_0.lora_up.weight': torch.randn(4, 2), 'lora_transformer_1.diff': torch.randn(4, 4)}
    path = write(sd)
    try:
        net = lora_load.load_safetensors('mixed', network.NetworkOnDisk('mixed', path))
    finally:
        os.unlink(path)
    assert {type(m).__name__ for m in net.modules.values()} == {'NetworkModuleLora', 'NetworkModuleFull'}, sorted(net.modules)
    assert all(m.network is net for m in net.modules.values()), 'a module of a later family ignores the strength set on the returned network'
    net.unet_multiplier = 0.25
    full = net.modules['lora_transformer_1']
    updown, _ = full.calc_updown(torch.zeros(4, 4))
    assert torch.allclose(updown, sd['lora_transformer_1.diff'] * 0.25), 'the Full delta must scale with the network strength'


def test_unreadable_file_declines():
    install(ZImagePipeline(torch.nn.Sequential(torch.nn.Linear(4, 4))))
    path = write({'lora_transformer_0.lora_down.weight': torch.randn(2, 4), 'lora_transformer_0.lora_up.weight': torch.randn(4, 2)})
    with open(path, 'r+b') as f:
        f.truncate(os.path.getsize(path) - 8) # header promises more bytes than the file holds
    try:
        assert lora_load.load_safetensors('truncated', network.NetworkOnDisk('truncated', path)) is None
    finally:
        os.unlink(path)


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
        test_widen_inpaint_conv_in,
        test_widen_fill_x_embedder,
        test_widen_refuses_wider_delta,
        test_widen_per_input_dora,
        test_narrow_padded_input,
        test_narrow_refuses_trained_tail,
        test_narrow_per_input_dora,
        test_te_routing_flux,
        test_te_routing_sdxl,
        test_te_routing_sd15,
        test_te_routing_sd3,
        test_te_path_text_model_either_shape,
        test_dotted_tail_groups_with_factors,
        test_dotted_tail_fixture,
        test_reorder_lokr,
        test_reorder_lokr_uneven_skips,
        test_reorder_loha,
        test_reorder_per_input_dora,
        test_split_per_input_dora_lora,
        test_split_per_input_dora_lokr,
        test_family_shape_checks,
        test_full_widens_inpaint_conv_in,
        test_full_narrows_padded_input,
        test_dump_covers_native_arch,
        test_chain_modules_share_the_returned_network,
        test_unreadable_file_declines,
        test_component_tags_parsed_and_missing_reported,
        test_tower_tags_parsed,
        test_native_activate_stages_each_reference_its_tags,
        test_request_options_fold_into_overrides,
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
