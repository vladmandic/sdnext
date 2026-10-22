#!/usr/bin/env python
"""
Offline tests for the SD3 and SD3.5 native adapter loader (``pipelines/sd3/sd3_lora.py``).

The model is a small meta-device ``SD3Transformer2DModel`` with a dual-attention
block and a pre-only last block, plus CLIP-L and CLIP-G text encoders. Key names
follow real kohya SD3.5 files (``lora_unet_joint_blocks_0_context_block_attn_qkv``),
the original dotted layout, diffusers PEFT ``transformer.`` and kohya text
encoders. These tests pin the routing; the campaign harness checks real files
against an independent reference.

No running server required.

Usage:
    python test/test-sd3-native-adapters.py
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

HEADS, HEAD_DIM = 2, 8
INNER = HEADS * HEAD_DIM # 16
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

class StableDiffusion3Pipeline:
    pass


def install():
    pipe = StableDiffusion3Pipeline()
    pipe.embedding_db = None
    with torch.device('meta'):
        pipe.transformer = diffusers.SD3Transformer2DModel(
            sample_size=8, patch_size=2, in_channels=4, out_channels=4, num_layers=2, attention_head_dim=HEAD_DIM, num_attention_heads=HEADS,
            joint_attention_dim=32, caption_projection_dim=INNER, pooled_projection_dim=16, pos_embed_max_size=8, dual_attention_layers=(0,), qk_norm='rms_norm',
        )
        pipe.text_encoder = transformers.CLIPTextModelWithProjection(transformers.CLIPTextConfig(hidden_size=768, intermediate_size=3072, num_hidden_layers=1, num_attention_heads=12, projection_dim=768))
        pipe.text_encoder_2 = transformers.CLIPTextModelWithProjection(transformers.CLIPTextConfig(hidden_size=1280, intermediate_size=5120, num_hidden_layers=1, num_attention_heads=20, projection_dim=1280))
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
        return lora_load.load_safetensors('sd3', network.NetworkOnDisk('sd3', path))
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

def test_kohya_streams_and_dual_attention():
    install()
    sd = {}
    for stream, seed in (('context_block_attn_qkv', 1), ('x_block_attn_qkv', 2), ('x_block_attn2_qkv', 3)):
        sd |= lora(f'lora_unet_joint_blocks_0_{stream}', 3 * INNER, INNER, seed)
    net = load(sd)
    for stream, targets in (('context_block_attn_qkv', ('attn_add_q_proj', 'attn_add_k_proj', 'attn_add_v_proj')), ('x_block_attn_qkv', ('attn_to_q', 'attn_to_k', 'attn_to_v')), ('x_block_attn2_qkv', ('attn2_to_q', 'attn2_to_k', 'attn2_to_v'))):
        chunks = torch.chunk(sd[f'lora_unet_joint_blocks_0_{stream}.lora_up.weight'], 3, dim=0)
        for chunk, target in zip(chunks, targets, strict=True):
            assert torch.equal(up_of(net, f'{T}transformer_blocks_0_{target}'), chunk), f'{stream} -> {target}'


def test_kohya_block_targets():
    install()
    names = {
        'joint_blocks_0_context_block_attn_proj': ('transformer_blocks_0_attn_to_add_out', INNER, INNER),
        'joint_blocks_0_x_block_attn_proj': ('transformer_blocks_0_attn_to_out_0', INNER, INNER),
        'joint_blocks_0_x_block_attn2_proj': ('transformer_blocks_0_attn2_to_out_0', INNER, INNER),
        'joint_blocks_0_context_block_mlp_fc1': ('transformer_blocks_0_ff_context_net_0_proj', 4 * INNER, INNER),
        'joint_blocks_0_x_block_mlp_fc2': ('transformer_blocks_0_ff_net_2', INNER, 4 * INNER),
        'joint_blocks_0_x_block_adaLN_modulation_1': ('transformer_blocks_0_norm1_linear', 9 * INNER, INNER),
        'joint_blocks_0_context_block_adaLN_modulation_1': ('transformer_blocks_0_norm1_context_linear', 6 * INNER, INNER),
    }
    sd = {}
    for i, (stem, (_, rows, cols)) in enumerate(names.items()):
        sd |= lora(f'lora_unet_{stem}', rows, cols, 10 + i)
    net = load(sd)
    for stem, (target, _, _) in names.items():
        assert torch.equal(up_of(net, T + target), sd[f'lora_unet_{stem}.lora_up.weight']), stem


def test_pre_only_and_final_modulation_swap_halves():
    install()
    sd = lora('lora_unet_joint_blocks_1_context_block_adaLN_modulation_1', 2 * INNER, INNER, 20) | lora('lora_unet_final_layer_adaLN_modulation_1', 2 * INNER, INNER, 21)
    net = load(sd)
    for stem, target in (('lora_unet_joint_blocks_1_context_block_adaLN_modulation_1', 'transformer_blocks_1_norm1_context_linear'), ('lora_unet_final_layer_adaLN_modulation_1', 'norm_out_linear')):
        shift, scale = torch.chunk(sd[f'{stem}.lora_up.weight'], 2, dim=0)
        assert torch.equal(up_of(net, T + target), torch.cat([scale, shift], dim=0)), stem


def test_non_block_targets_and_conv_patch_embed():
    install()
    sd = lora('lora_unet_context_embedder', INNER, 32, 30) | lora('lora_unet_t_embedder_mlp_0', INNER, 256, 31) | lora('lora_unet_y_embedder_mlp_2', INNER, INNER, 32)
    g = torch.Generator().manual_seed(33)
    sd |= {'transformer.pos_embed.proj.lora_A.weight': torch.randn(RANK, 4, 2, 2, generator=g), 'transformer.pos_embed.proj.lora_B.weight': torch.randn(INNER, RANK, 1, 1, generator=g)}
    net = load(sd)
    for target in ('context_embedder', 'time_text_embed_timestep_embedder_linear_1', 'time_text_embed_text_embedder_linear_2', 'pos_embed_proj'):
        assert T + target in net.modules, f'{target} not bound: {sorted(net.modules)}'


def test_original_dotted_and_peft():
    install()
    net = load(lora('joint_blocks.0.x_block.attn.qkv', 3 * INNER, INNER, 40) | lora('transformer.transformer_blocks.1.attn.to_q', INNER, INNER, 41, down='lora_A.weight', up='lora_B.weight', alpha=False))
    up_of(net, f'{T}transformer_blocks_0_attn_to_k')
    up_of(net, f'{T}transformer_blocks_1_attn_to_q')


def test_text_encoders_by_role():
    install()
    sd = lora('lora_te1_text_model_encoder_layers_0_self_attn_q_proj', 768, 768, 50) | lora('lora_te2_text_model_encoder_layers_0_self_attn_q_proj', 1280, 1280, 51) | lora('lora_te3_encoder_block_0_layer_0_SelfAttention_q', 64, 64, 52)
    net = load(sd)
    assert any(k.startswith('lora_te1_') for k in net.modules) and any(k.startswith('lora_te2_') for k in net.modules), sorted(net.modules)
    assert not any(k.startswith('lora_te3_') for k in net.modules), 'the SD3 T5 is not stamped'


# ============================================================
# Runner
# ============================================================

def run_tests():
    t0 = time.time()
    cat = category('sd3')
    log.warning('=== SD3 native adapter routing ===')
    for fn in [
        test_kohya_streams_and_dual_attention,
        test_kohya_block_targets,
        test_pre_only_and_final_modulation_swap_halves,
        test_non_block_targets_and_conv_patch_embed,
        test_original_dotted_and_peft,
        test_text_encoders_by_role,
    ]:
        run_test(cat, fn)
    model_data.sd_model = None
    total_fail = sum(info['failed'] for info in results.values())
    log.warning(f'Total: {sum(i["passed"] for i in results.values())} passed, {total_fail} failed in {time.time() - t0:.2f}s')
    return total_fail == 0


if __name__ == '__main__':
    sys.exit(0 if run_tests() else 1)
