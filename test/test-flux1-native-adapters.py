#!/usr/bin/env python
"""
Offline tests for the Flux.1 native adapter loader (``pipelines/flux/flux1_lora.py``).

The model is a small meta-device ``FluxTransformer2DModel`` with real module
names, plus CLIP-L and T5 text encoders. Key names follow real files: kohya
(``lora_unet_double_blocks_0_img_attn_qkv``), BFL dotted, ComfyUI
``diffusion_model.``, diffusers PEFT ``transformer.``, OneTrainer
``lora_transformer_``, XLabs processor keys, kohya text encoders
(``lora_te1_`` CLIP-L, ``lora_te3_`` T5). Correctness against diffusers on the
real files is checked by the harness in the LoRA retirement campaign folder;
these tests pin the routing.

No running server required.

Usage:
    python test/test-flux1-native-adapters.py
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
from modules import shared, sd_models_utils  # pylint: disable=wrong-import-position,unused-import
from modules.modeldata import model_data  # pylint: disable=wrong-import-position
from modules.lora import lora_load, network  # pylint: disable=wrong-import-position
from pipelines.flux import flux1_lora  # pylint: disable=wrong-import-position

HEADS, HEAD_DIM = 2, 8
INNER = HEADS * HEAD_DIM # 16
MLP = 4 * INNER # 64
RANK = 4


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'tests': []}
    return name


def record(cat: str, passed: bool, name: str, detail: str = ''):
    results[cat]['passed' if passed else 'failed'] += 1
    results[cat]['tests'].append(('PASS' if passed else 'FAIL', name))
    msg = f'  {"PASS" if passed else "FAIL"}: {name}' + (f' ({detail})' if detail else '')
    (log.info if passed else log.error)(msg)


def run_test(cat: str, fn):
    try:
        ok = fn()
        record(cat, ok is not False, fn.__name__)
    except AssertionError as e:
        record(cat, False, fn.__name__, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, fn.__name__, f'exception: {e}')
        import traceback
        traceback.print_exc()


# ============================================================
# Model
# ============================================================

class FluxPipeline:
    pass


def install(guidance=True, in_channels=8):
    pipe = FluxPipeline()
    pipe.embedding_db = None
    with torch.device('meta'):
        pipe.transformer = diffusers.FluxTransformer2DModel(
            patch_size=1, in_channels=in_channels, num_layers=1, num_single_layers=1, attention_head_dim=HEAD_DIM, num_attention_heads=HEADS,
            joint_attention_dim=32, pooled_projection_dim=16, guidance_embeds=guidance, axes_dims_rope=(2, 2, 4),
        )
        clip = transformers.CLIPTextConfig(hidden_size=768, intermediate_size=3072, num_hidden_layers=1, num_attention_heads=12)
        pipe.text_encoder = transformers.CLIPTextModel(clip)
        pipe.text_encoder_2 = transformers.T5EncoderModel(transformers.T5Config(d_model=64, d_kv=32, d_ff=128, num_layers=1, num_heads=2, vocab_size=64))
    model_data.sd_model = pipe
    lora_load.lora_cache.clear()
    sd_models_utils.state_dict_cache.disable()
    return pipe


def write(state_dict):
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    os.close(fd)
    safetensors.torch.save_file({k: v.clone().contiguous() for k, v in state_dict.items()}, path)
    return path


def load(state_dict):
    path = write(state_dict)
    try:
        return lora_load.load_safetensors('flux1', network.NetworkOnDisk('flux1', path))
    finally:
        os.unlink(path)


def lora(stem, rows, cols, seed, down='lora_down.weight', up='lora_up.weight', alpha=True):
    g = torch.Generator().manual_seed(seed)
    out = {f'{stem}.{down}': torch.randn(RANK, cols, generator=g), f'{stem}.{up}': torch.randn(rows, RANK, generator=g)}
    if alpha:
        out[f'{stem}.alpha'] = torch.tensor(float(RANK))
    return out


def up_of(net, key):
    assert net is not None, 'not loaded'
    module = net.modules.get(key)
    assert module is not None, f'{key} not bound; bound={sorted(net.modules)[:8]}'
    return module.up_model.weight.float()


T = 'lora_transformer_'


# ============================================================
# Tests
# ============================================================

def test_kohya_double_qkv_split():
    install()
    sd = lora('lora_unet_double_blocks_0_img_attn_qkv', 3 * INNER, INNER, 1) | lora('lora_unet_double_blocks_0_txt_attn_qkv', 3 * INNER, INNER, 2)
    net = load(sd)
    for stream, targets in (('img', ('to_q', 'to_k', 'to_v')), ('txt', ('add_q_proj', 'add_k_proj', 'add_v_proj'))):
        chunks = torch.chunk(sd[f'lora_unet_double_blocks_0_{stream}_attn_qkv.lora_up.weight'], 3, dim=0)
        for chunk, target in zip(chunks, targets, strict=True):
            assert torch.equal(up_of(net, f'{T}transformer_blocks_0_attn_{target}'), chunk), f'{stream} -> {target}'


def test_kohya_single_linear1_widths_from_model():
    install()
    sd = lora('lora_unet_single_blocks_0_linear1', 3 * INNER + MLP, INNER, 3)
    net = load(sd)
    up = sd['lora_unet_single_blocks_0_linear1.lora_up.weight']
    for target, (start, end) in zip(('attn_to_q', 'attn_to_k', 'attn_to_v', 'proj_mlp'), ((0, 16), (16, 32), (32, 48), (48, 112)), strict=True):
        assert torch.equal(up_of(net, f'{T}single_transformer_blocks_0_{target}'), up[start:end]), target


def test_bfl_dotted_and_comfy_names():
    install()
    for prefix in ('', 'diffusion_model.'):
        net = load(lora(f'{prefix}double_blocks.0.img_mlp.0', MLP, INNER, 4) | lora(f'{prefix}single_blocks.0.linear2', INNER, INNER + MLP, 5))
        up_of(net, f'{T}transformer_blocks_0_ff_net_0_proj')
        up_of(net, f'{T}single_transformer_blocks_0_proj_out')


def test_non_block_targets():
    install()
    names = {
        'lora_unet_img_in': ('x_embedder', INNER, 8),
        'lora_unet_txt_in': ('context_embedder', INNER, 32),
        'lora_unet_time_in_in_layer': ('time_text_embed_timestep_embedder_linear_1', INNER, 256),
        'lora_unet_vector_in_in_layer': ('time_text_embed_text_embedder_linear_1', INNER, 16),
        'lora_unet_guidance_in_out_layer': ('time_text_embed_guidance_embedder_linear_2', INNER, INNER),
        'lora_unet_final_layer_linear': ('proj_out', 8, INNER),
        'lora_unet_double_blocks_0_img_mod_lin': ('transformer_blocks_0_norm1_linear', 6 * INNER, INNER),
        'lora_unet_single_blocks_0_modulation_lin': ('single_transformer_blocks_0_norm_linear', 3 * INNER, INNER),
    }
    sd = {}
    for i, (stem, (_, rows, cols)) in enumerate(names.items()):
        sd |= lora(stem, rows, cols, 10 + i)
    net = load(sd)
    for stem, (target, _, _) in names.items():
        assert torch.equal(up_of(net, T + target), sd[f'{stem}.lora_up.weight']), stem


def test_final_layer_modulation_swaps_halves():
    install()
    sd = lora('lora_unet_final_layer_adaLN_modulation_1', 2 * INNER, INNER, 20)
    up = sd['lora_unet_final_layer_adaLN_modulation_1.lora_up.weight']
    shift, scale = torch.chunk(up, 2, dim=0)
    assert torch.equal(up_of(load(sd), f'{T}norm_out_linear'), torch.cat([scale, shift], dim=0))


def test_lycoris_whole_model_wrapper():
    install()
    sd = {'lycoris__transformer_blocks_0_attn_to_q.lokr_w1': torch.randn(4, 4), 'lycoris__transformer_blocks_0_attn_to_q.lokr_w2': torch.randn(4, 4), 'lycoris__transformer_blocks_0_attn_to_q.alpha': torch.tensor(1.0)}
    net = load(sd)
    assert net is not None and f'{T}transformer_blocks_0_attn_to_q' in net.modules, 'lycoris__ keys name the module after the doubled underscore'


def test_peft_onetrainer_and_wrapper_verbatim():
    install()
    for stem in ('transformer.transformer_blocks.0.attn.to_q', 'base_model.model.transformer_blocks.0.attn.to_q', 'lora_transformer_transformer_blocks_0_attn_to_q'):
        down, up = ('lora_A.weight', 'lora_B.weight') if not stem.startswith('lora_') else ('lora_down.weight', 'lora_up.weight')
        net = load(lora(stem, INNER, INNER, 21, down=down, up=up, alpha=False))
        up_of(net, f'{T}transformer_blocks_0_attn_to_q')


def test_xlabs_processor_keys():
    install()
    sd = {}
    for name, rows, cols in (('double_blocks.0.processor.qkv_lora1', 3 * INNER, INNER), ('double_blocks.0.processor.proj_lora2', INNER, INNER),
                             ('single_blocks.0.processor.qkv_lora', 3 * INNER, INNER), ('single_blocks.0.processor.proj_lora', INNER, INNER + MLP)):
        sd |= lora(name, rows, cols, 30 + len(sd), down='down.weight', up='up.weight', alpha=False)
    net = load(sd)
    qkv = torch.chunk(sd['double_blocks.0.processor.qkv_lora1.up.weight'], 3, dim=0)
    assert torch.equal(up_of(net, f'{T}transformer_blocks_0_attn_to_k'), qkv[1]), 'qkv_lora1 is the image stream'
    up_of(net, f'{T}transformer_blocks_0_attn_to_add_out')
    single = sd['single_blocks.0.processor.qkv_lora.up.weight']
    assert torch.equal(up_of(net, f'{T}single_transformer_blocks_0_attn_to_v'), single[2 * INNER:]), 'single qkv_lora covers the Q/K/V rows'
    assert f'{T}single_transformer_blocks_0_proj_mlp' not in net.modules
    up_of(net, f'{T}single_transformer_blocks_0_proj_out')


def test_text_encoders_by_role():
    install()
    sd = lora('lora_te1_text_model_encoder_layers_0_self_attn_q_proj', 768, 768, 40) | lora('lora_te3_encoder_block_0_layer_0_SelfAttention_q', 64, 64, 41)
    net = load(sd)
    clip_key = next(k for k in net.modules if k.startswith('lora_te1_'))
    t5_key = next(k for k in net.modules if k.startswith('lora_te2_'))
    assert clip_key.endswith('layers_0_self_attn_q_proj') and t5_key == 'lora_te2_encoder_block_0_layer_0_SelfAttention_q', sorted(net.modules)


def test_component_strengths_by_namespace():
    install()
    net = load(lora('lora_te1_text_model_encoder_layers_0_self_attn_q_proj', 768, 768, 45) | lora('lora_unet_double_blocks_0_img_attn_proj', INNER, INNER, 46))
    net.te_multiplier, net.unet_multiplier = 0.3, [0.7, 0.7, 0.7]
    clip_key = next(k for k in net.modules if k.startswith('lora_te1_'))
    assert net.modules[clip_key].multiplier() == 0.3, 'te= scales the text encoder'
    assert net.modules[f'{T}transformer_blocks_0_attn_to_out_0'].multiplier() == 0.7, 'unet= scales the transformer'


def test_component_filter_leaves_other_components_out():
    from modules.lora import networks, lora_common
    install()
    net = load(lora('lora_te1_text_model_encoder_layers_0_self_attn_q_proj', 768, 768, 47) | lora('lora_unet_double_blocks_0_img_attn_proj', INNER, INNER, 48))
    clip_key = next(k for k in net.modules if k.startswith('lora_te1_'))
    transformer_key = f'{T}transformer_blocks_0_attn_to_out_0'
    real = list(lora_common.loaded_networks)
    try:
        lora_common.loaded_networks[:] = [net]
        for components, keys in ((('transformer',), {transformer_key}), (('text_encoder',), {clip_key}), (None, {clip_key, transformer_key})):
            net.pending_config = {'te': 1.0, 'unet': [1.0] * 3, 'dyn': None, 'components': components}
            networks.promote_pending()
            assert set(net.modules) == keys, f'{components}: {sorted(net.modules)}'
            assert set(net.bound()) == {clip_key, transformer_key}, 'the bound set must survive a narrowed request'
            assert all(module.multiplier() == 1.0 for module in net.modules.values()), components
            assert (networks.component_signature() != '') == (components is not None), 'a component filter must change the apply stamp'
    finally:
        lora_common.loaded_networks[:] = real


def test_guidance_keys_unmapped_on_schnell():
    install(guidance=False)
    net = load(lora('lora_unet_guidance_in_in_layer', INNER, 256, 50) | lora('lora_unet_img_in', INNER, 8, 51))
    assert f'{T}x_embedder' in net.modules
    assert not any('guidance' in k or 'timestep' in k for k in net.modules), f'guidance must not land on the timestep embedder: {sorted(net.modules)}'


def test_fill_widens_x_embedder():
    install(in_channels=24)
    net = load(lora('lora_unet_img_in', INNER, 8, 60))
    module = net.modules.get(f'{T}x_embedder')
    assert module is not None and module.down_model.weight.shape[1] == 24, 'a dev img_in delta binds to the wider Fill x_embedder'


def test_kohya_without_alpha_scale_one():
    install()
    net = load(lora('lora_unet_double_blocks_0_img_attn_proj', INNER, INNER, 70, alpha=False))
    assert net.modules[f'{T}transformer_blocks_0_attn_to_out_0'].calc_scale() == 1.0


def test_prefixed_factor_names():
    install()
    q, k = 'transformer.transformer_blocks.0.attn.to_q', 'transformer.transformer_blocks.0.attn.to_k'
    sd = lora(q, INNER, INNER, 80, down='lora_A.weight', up='lora_B.weight', alpha=False) | lora(k, INNER, INNER, 81)
    sd[f'{k}.alpha'] = torch.tensor(2.0)
    net = load(sd)
    assert torch.equal(up_of(net, f'{T}transformer_blocks_0_attn_to_q'), sd[f'{q}.lora_B.weight']), 'lora_B is the up factor'
    assert torch.equal(net.modules[f'{T}transformer_blocks_0_attn_to_q'].down_model.weight.float(), sd[f'{q}.lora_A.weight']), 'lora_A is the down factor'
    assert torch.equal(up_of(net, f'{T}transformer_blocks_0_attn_to_k'), sd[f'{k}.lora_up.weight']), 'prefixed lora_up binds'
    assert net.modules[f'{T}transformer_blocks_0_attn_to_k'].calc_scale() == 2.0 / RANK, 'prefixed alpha is read'


def test_absent_block_binds_nothing():
    install()
    net = load(lora('transformer.transformer_blocks.9.attn.to_q', INNER, INNER, 82, down='lora_A.weight', up='lora_B.weight', alpha=False))
    assert net is None, f'a block the model lacks must not bind: {sorted(net.modules) if net else None}'


# ============================================================
# Runner
# ============================================================

def run_tests():
    t0 = time.time()
    cat = category('flux1')
    log.warning('=== Flux.1 native adapter routing ===')
    for fn in [
        test_kohya_double_qkv_split,
        test_kohya_single_linear1_widths_from_model,
        test_bfl_dotted_and_comfy_names,
        test_non_block_targets,
        test_final_layer_modulation_swaps_halves,
        test_lycoris_whole_model_wrapper,
        test_peft_onetrainer_and_wrapper_verbatim,
        test_xlabs_processor_keys,
        test_text_encoders_by_role,
        test_component_strengths_by_namespace,
        test_component_filter_leaves_other_components_out,
        test_guidance_keys_unmapped_on_schnell,
        test_fill_widens_x_embedder,
        test_kohya_without_alpha_scale_one,
        test_prefixed_factor_names,
        test_absent_block_binds_nothing,
    ]:
        run_test(cat, fn)
    model_data.sd_model = None
    total_fail = sum(info['failed'] for info in results.values())
    log.warning(f'Total: {sum(i["passed"] for i in results.values())} passed, {total_fail} failed in {time.time() - t0:.2f}s')
    return total_fail == 0


if __name__ == '__main__':
    sys.exit(0 if run_tests() else 1)
