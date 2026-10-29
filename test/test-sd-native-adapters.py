#!/usr/bin/env python
"""
Offline tests for the SD 1.x and SDXL native adapter loader (``pipelines/sd/sd_lora.py``).

The models are small meta-device ``UNet2DConditionModel`` trees in the SD 1.x
and SDXL block layouts with CLIP text encoders. Key names follow real files:
kohya diffusers-flat and CompVis names, OFT ``oft_unet_``, the diffusers
training scripts, the 2023 attention-processor save, OMI component prefixes
and kohya text encoders. The campaign harness checks every local file against
the generic loop and an independent reference; these tests pin the routing.

No running server required.

Usage:
    python test/test-sd-native-adapters.py
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
# Models
# ============================================================

class StableDiffusionPipeline:
    pass


class StableDiffusionXLPipeline:
    pass


def clip(hidden, projection=False):
    config = transformers.CLIPTextConfig(hidden_size=hidden, intermediate_size=4 * hidden, num_hidden_layers=1, num_attention_heads=hidden // 64, projection_dim=hidden)
    return transformers.CLIPTextModelWithProjection(config) if projection else transformers.CLIPTextModel(config)


def install(xl=False, in_channels=4):
    pipe = StableDiffusionXLPipeline() if xl else StableDiffusionPipeline()
    pipe.embedding_db = None
    with torch.device('meta'):
        if xl: # SDXL layout: no attention in the first down block, three in the last up block
            pipe.unet = diffusers.UNet2DConditionModel(
                sample_size=8, in_channels=in_channels, out_channels=4, layers_per_block=2, block_out_channels=(32, 64),
                down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
                cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8, transformer_layers_per_block=1,
            )
            pipe.text_encoder, pipe.text_encoder_2 = clip(768), clip(1280, projection=True)
        else: # SD 1.x layout: the first up block has no attention and holds the upsampler at output_blocks.2.1
            pipe.unet = diffusers.UNet2DConditionModel(
                sample_size=8, in_channels=in_channels, out_channels=4, layers_per_block=2, block_out_channels=(32, 64),
                down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'), up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
                cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8,
            )
            pipe.text_encoder = clip(768)
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
        return lora_load.load_safetensors('sd', network.NetworkOnDisk('sd', path))
    finally:
        os.unlink(path)


def lora(stem, rows, cols, seed, down='lora_down.weight', up='lora_up.weight', alpha=True, kernel=None):
    g = torch.Generator().manual_seed(seed)
    if kernel is None:
        out = {f'{stem}.{down}': torch.randn(RANK, cols, generator=g), f'{stem}.{up}': torch.randn(rows, RANK, generator=g)}
    else:
        out = {f'{stem}.{down}': torch.randn(RANK, cols, kernel, kernel, generator=g), f'{stem}.{up}': torch.randn(rows, RANK, 1, 1, generator=g)}
    if alpha:
        out[f'{stem}.alpha'] = torch.tensor(float(RANK))
    return out


def bound(net, key):
    assert net is not None, 'not loaded'
    assert key in net.modules, f'{key} not bound; bound={sorted(net.modules)[:8]}'
    return net.modules[key]


U = 'lora_unet_'


# ============================================================
# Tests
# ============================================================

def test_kohya_flat_names_verbatim():
    install()
    net = load(lora('lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q', 32, 32, 1) | lora('lora_unet_down_blocks_0_resnets_0_conv1', 32, 32, 2, kernel=3))
    bound(net, f'{U}down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q')
    bound(net, f'{U}down_blocks_0_resnets_0_conv1')


def test_compvis_names_both_layouts():
    install(xl=True)
    net = load(lora('lora_unet_input_blocks_4_1_proj_in', 64, 64, 3) | lora('lora_unet_output_blocks_2_2_conv', 64, 64, 4, kernel=3))
    bound(net, f'{U}down_blocks_1_attentions_0_proj_in')
    bound(net, f'{U}up_blocks_0_upsamplers_0_conv')
    install()
    net = load(lora('lora_unet_input_blocks_1_1_proj_in', 32, 32, 5) | lora('lora_unet_output_blocks_2_1_conv', 64, 64, 6, kernel=3) | lora('lora_unet_middle_block_1_proj_out', 64, 64, 7))
    bound(net, f'{U}down_blocks_0_attentions_0_proj_in')
    bound(net, f'{U}up_blocks_0_upsamplers_0_conv') # SD 1.x upsampler sits where the SDXL table has an attention block
    bound(net, f'{U}mid_block_attentions_0_proj_out')


def test_oft_unet_and_comfy_prefixes():
    install()
    net = load({'oft_unet_down_blocks_0_attentions_0_proj_in.oft_blocks': torch.zeros(4, 8, 8), 'oft_unet_down_blocks_0_attentions_0_proj_in.alpha': torch.tensor(1e-3)}
               | lora('diffusion_model_input_blocks_1_1_proj_out', 32, 32, 8) | lora('diffusion_model.middle_block.1.proj_in', 64, 64, 9))
    bound(net, f'{U}down_blocks_0_attentions_0_proj_in')
    bound(net, f'{U}down_blocks_0_attentions_0_proj_out')
    bound(net, f'{U}mid_block_attentions_0_proj_in')


def test_diffusers_training_script_formats():
    install(xl=True)
    sd = lora('unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k', 64, 64, 10, down='lora.down.weight', up='lora.up.weight', alpha=False)
    sd['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_magnitude_vector'] = torch.rand(64) + 0.5
    sd |= lora('text_encoder.text_model.encoder.layers.0.self_attn.k_proj', 768, 768, 11, down='lora_linear_layer.down.weight', up='lora_linear_layer.up.weight', alpha=False)
    sd |= lora('unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_out_lora', 64, 64, 12, down='down.weight', up='up.weight', alpha=False)
    net = load(sd)
    dora = bound(net, f'{U}down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k')
    assert dora.dora_scale is not None, 'the magnitude vector is the DoRA scale'
    assert any(k.startswith('lora_te1_') and k.endswith('self_attn_k_proj') for k in net.modules), sorted(net.modules)
    bound(net, f'{U}down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0')


def test_text_encoders_by_role():
    install()
    net = load(lora('lora_te1_text_model_encoder_layers_0_self_attn_q_proj', 768, 768, 20) | lora('lora_te_text_model_encoder_layers_0_mlp_fc1', 3072, 768, 21))
    assert sum(k.startswith('lora_te_') for k in net.modules) == 2, sorted(net.modules)
    install(xl=True)
    net = load(lora('lora_te1_text_model_encoder_layers_0_mlp_fc1', 3072, 768, 22) | lora('lora_te2_text_model_encoder_layers_0_mlp_fc1', 5120, 1280, 23) | lora('clip_g.text_model.encoder.layers.0.self_attn.v_proj', 1280, 1280, 24) | lora('lora_te2_text_projection', 1280, 1280, 25))
    assert sum(k.startswith('lora_te1_') for k in net.modules) == 1 and sum(k.startswith('lora_te2_') for k in net.modules) == 3, sorted(net.modules)


def test_sd15_text_encoder_keys_unmapped_on_sdxl():
    install(xl=True)
    net = load(lora('lora_te_text_model_encoder_layers_0_mlp_fc1', 3072, 768, 30) | lora('lora_unet_down_blocks_1_attentions_0_proj_in', 64, 64, 31))
    assert not any(k.startswith('lora_te') for k in net.modules), 'lora_te_ names a lone encoder, which SDXL lacks'


def test_inpaint_conv_in_widens():
    install(in_channels=9)
    net = load(lora('lora_unet_conv_in', 32, 4, 40, kernel=3))
    assert bound(net, f'{U}conv_in').down_model.weight.shape[1] == 9


def test_wrong_base_refused():
    install(xl=True)
    assert load(lora('lora_unet_down_blocks_1_attentions_0_proj_in', 32, 32, 50)) is None, 'an SD 1.x width on SDXL must refuse the file'


def test_glora_on_pointwise_conv():
    install()
    g = torch.Generator().manual_seed(70)
    stem = 'lora_unet_down_blocks_0_attentions_0_proj_in' # 1x1 conv on SD 1.x; factors in the kohya lycoris layout
    a1, a2, b1, b2 = torch.randn(RANK, 32, 1, 1, generator=g), torch.randn(32, RANK, 1, 1, generator=g), torch.randn(RANK, 32, 1, 1, generator=g), torch.randn(32, RANK, 1, 1, generator=g)
    net = load({f'{stem}.a1.weight': a1, f'{stem}.a2.weight': a2, f'{stem}.b1.weight': b1, f'{stem}.b2.weight': b2, f'{stem}.alpha': torch.tensor(float(RANK))})
    module = bound(net, f'{U}down_blocks_0_attentions_0_proj_in')
    weight = torch.randn(32, 32, 1, 1, generator=g)
    updown, _ = module.calc_updown(weight)
    flat = b2.flatten(1) @ b1.flatten(1) + (weight.flatten(1) @ a2.flatten(1)) @ a1.flatten(1)
    assert updown.shape == weight.shape and torch.allclose(updown.flatten(1), flat * module.calc_scale() * module.multiplier(), atol=1e-4), 'a pointwise conv must match the linear form'


def test_bundled_embeddings():
    install()
    sd = lora('lora_unet_down_blocks_0_attentions_0_proj_in', 32, 32, 60) | {'bundle_emb.mytoken.string_to_param.*': torch.randn(2, 768)}
    net = load(sd)
    assert set(net.bundle_embeddings) == {'mytoken'} and 'string_to_param.*' in net.bundle_embeddings['mytoken']


# ============================================================
# Runner
# ============================================================

def run_tests():
    t0 = time.time()
    cat = category('sd')
    log.warning('=== SD and SDXL native adapter routing ===')
    for fn in [
        test_kohya_flat_names_verbatim,
        test_compvis_names_both_layouts,
        test_oft_unet_and_comfy_prefixes,
        test_diffusers_training_script_formats,
        test_text_encoders_by_role,
        test_sd15_text_encoder_keys_unmapped_on_sdxl,
        test_inpaint_conv_in_widens,
        test_wrong_base_refused,
        test_glora_on_pointwise_conv,
        test_bundled_embeddings,
    ]:
        run_test(cat, fn)
    model_data.sd_model = None
    total_fail = sum(info['failed'] for info in results.values())
    log.warning(f'Total: {sum(i["passed"] for i in results.values())} passed, {total_fail} failed in {time.time() - t0:.2f}s')
    return total_fail == 0


if __name__ == '__main__':
    sys.exit(0 if run_tests() else 1)
