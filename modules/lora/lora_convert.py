import os
import re
import bisect
import torch
from modules import shared


debug = os.environ.get('SD_LORA_DEBUG', None) is not None
suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}
re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}


def make_unet_conversion_map() -> dict[str, str]:
    unet_conversion_map_layer = []

    for i in range(4):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))
            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))
            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))
            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j + 1}."
        sd_time_embed_prefix = f"time_embed.{j * 2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j + 1}."
        sd_label_embed_prefix = f"label_emb.0.{j * 2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    sd_hf_conversion_map = {sd.replace(".", "_")[:-1]: hf.replace(".", "_")[:-1] for sd, hf in unet_conversion_map}
    return sd_hf_conversion_map


class KeyConvert:
    def __init__(self):
        self.is_sdxl = True if shared.sd_model_type == "sdxl" else False
        self.UNET_CONVERSION_MAP = make_unet_conversion_map()
        self.LORA_PREFIX_UNET = "lora_unet_"
        self.LORA_PREFIX_TEXT_ENCODER = "lora_te_"
        self.OFT_PREFIX_UNET = "oft_unet_"
        # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
        self.LORA_PREFIX_TEXT_ENCODER1 = "lora_te1_"
        self.LORA_PREFIX_TEXT_ENCODER2 = "lora_te2_"

    def __call__(self, key):
        if "diffusion_model" in key:  # Fix NTC Slider naming error
            key = key.replace("diffusion_model", "lora_unet")
        map_keys = list(self.UNET_CONVERSION_MAP.keys())  # prefix of U-Net modules
        map_keys.sort()
        search_key = key.replace(self.LORA_PREFIX_UNET, "").replace(self.OFT_PREFIX_UNET, "").replace(self.LORA_PREFIX_TEXT_ENCODER1, "").replace(self.LORA_PREFIX_TEXT_ENCODER2, "")
        position = bisect.bisect_right(map_keys, search_key)
        map_key = map_keys[position - 1]
        if search_key.startswith(map_key):
            key = key.replace(map_key, self.UNET_CONVERSION_MAP[map_key]).replace("oft", "lora") # pylint: disable=unsubscriptable-object
        if "lycoris" in key and "transformer" in key:
            key = key.replace("lycoris", "lora_transformer")
        sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        if sd_module is None:
            sd_module = shared.sd_model.network_layer_mapping.get(key.replace("guidance", "timestep"), None)  # FLUX1 fix
        if debug and sd_module is None:
            raise RuntimeError(f"LoRA key not found in network_layer_mapping: key={key} mapping={shared.sd_model.network_layer_mapping.keys()}")
        return key, sd_module


# Taken from https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py
# Modified from 'lora_A' and 'lora_B' to 'lora_down' and 'lora_up'
# Added early exit
# The utilities under `_convert_kohya_flux_lora_to_diffusers()`
# are taken from https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
# All credits go to `kohya-ss`.
def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
    if sds_key + ".lora_down.weight" not in sds_sd:
        return
    down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

    # scale weight by alpha and dim
    rank = down_weight.shape[0]
    alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
    scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

    # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2 < scale_up:
        scale_down *= 2
        scale_up /= 2

    ait_sd[ait_key + ".lora_down.weight"] = down_weight * scale_down
    ait_sd[ait_key + ".lora_up.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up

def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
    if sds_key + ".lora_down.weight" not in sds_sd:
        return
    down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
    up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
    sd_lora_rank = down_weight.shape[0]

    # scale weight by alpha and dim
    alpha = sds_sd.pop(sds_key + ".alpha")
    scale = alpha / sd_lora_rank

    # calculate scale_down and scale_up
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2 < scale_up:
        scale_down *= 2
        scale_up /= 2

    down_weight = down_weight * scale_down
    up_weight = up_weight * scale_up

    # calculate dims if not provided
    num_splits = len(ait_keys)
    if dims is None:
        dims = [up_weight.shape[0] // num_splits] * num_splits
    else:
        assert sum(dims) == up_weight.shape[0]

    # check upweight is sparse or not
    is_sparse = False
    if sd_lora_rank % num_splits == 0:
        ait_rank = sd_lora_rank // num_splits
        is_sparse = True
        i = 0
        for j in range(len(dims)):
            for k in range(len(dims)):
                if j == k:
                    continue
                is_sparse = is_sparse and torch.all(
                    up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0
                )
            i += dims[j]

    # make ai-toolkit weight
    ait_down_keys = [k + ".lora_down.weight" for k in ait_keys]
    ait_up_keys = [k + ".lora_up.weight" for k in ait_keys]
    if not is_sparse:
        # down_weight is copied to each split
        ait_sd.update({k: down_weight for k in ait_down_keys})

        # up_weight is split to each split
        ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0), strict=False)})  # noqa: C416 # pylint: disable=unnecessary-comprehension
    else:
        # down_weight is chunked to each split
        ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0), strict=False)})  # noqa: C416 # pylint: disable=unnecessary-comprehension

        # up_weight is sparse: only non-zero values are copied to each split
        i = 0
        for j in range(len(dims)):
            ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
            i += dims[j]

def _convert_text_encoder_lora_key(key, lora_name):
    """
    Converts a text encoder LoRA key to a Diffusers compatible key.
    """
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        key_to_replace = "lora_te2_"

    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        pass
    elif "mlp" in diffusers_name:
        # Be aware that this is the new diffusers convention and the rest of the code might
        # not utilize it yet.
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
    return diffusers_name

def _convert_kohya_flux_lora_to_diffusers(state_dict):
    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        if len(sds_sd) > 0:
            return None

        return ait_sd

    return _convert_sd_scripts_to_ai_toolkit(state_dict)

def _convert_kohya_sd3_lora_to_diffusers(state_dict):
    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_context_block_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_context_block_mlp_fc1",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_context_block_mlp_fc2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_x_block_mlp_fc1",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_x_block_mlp_fc2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_context_block_adaLN_modulation_1",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_x_block_adaLN_modulation_1",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_context_block_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_x_block_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out_0",
            )

            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_joint_blocks_{i}_x_block_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
        remaining_keys = list(sds_sd.keys())
        te_state_dict = {}
        if remaining_keys:
            if not all(k.startswith("lora_te1") for k in remaining_keys):
                raise ValueError(f"Incompatible keys detected: \n\n {', '.join(remaining_keys)}")
            for key in remaining_keys:
                if not key.endswith("lora_down.weight"):
                    continue

                lora_name = key.split(".")[0]
                lora_name_up = f"{lora_name}.lora_up.weight"
                lora_name_alpha = f"{lora_name}.alpha"
                diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

                sd_lora_rank = 1
                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    down_weight = sds_sd.pop(key)
                    sd_lora_rank = down_weight.shape[0]
                    te_state_dict[diffusers_name] = down_weight
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = sds_sd.pop(lora_name_up)

                if lora_name_alpha in sds_sd:
                    alpha = sds_sd.pop(lora_name_alpha).item()
                    scale = alpha / sd_lora_rank

                    scale_down = scale
                    scale_up = 1.0
                    while scale_down * 2 < scale_up:
                        scale_down *= 2
                        scale_up /= 2

                    te_state_dict[diffusers_name] *= scale_down
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] *= scale_up

        if len(sds_sd) > 0:
            print(f"Unsupported keys for ai-toolkit: {sds_sd.keys()}")

        if te_state_dict:
            te_state_dict = {f"text_encoder.{module_name}": params for module_name, params in te_state_dict.items()}

        new_state_dict = {**ait_sd, **te_state_dict}
        return new_state_dict

    return _convert_sd_scripts_to_ai_toolkit(state_dict)


def assign_network_names_to_compvis_modules(sd_model):
    if sd_model is None:
        return
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatiblility
    network_layer_mapping = {}
    if hasattr(sd_model, 'text_encoder') and sd_model.text_encoder is not None:
        for name, module in sd_model.text_encoder.named_modules():
            prefix = "lora_te1_" if hasattr(sd_model, 'text_encoder_2') else "lora_te_"
            network_name = prefix + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'text_encoder_2'):
        for name, module in sd_model.text_encoder_2.named_modules():
            network_name = "lora_te2_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'unet'):
        for name, module in sd_model.unet.named_modules():
            network_name = "lora_unet_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'transformer'):
        for name, module in sd_model.transformer.named_modules():
            network_name = "lora_transformer_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            if "norm" in network_name and "linear" not in network_name and shared.sd_model_type != "sd3":
                continue
            module.network_layer_name = network_name
    shared.sd_model.network_layer_mapping = network_layer_mapping
