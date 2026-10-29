from modules import shared


TOWERS = ('unconditional_transformer',) # second instances of the transformer (Ideogram 4's CFG prior); stamped lora_<name>_, so a network reaches one only by naming it


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


def assign_network_names_to_compvis_modules(sd_model):
    if sd_model is None:
        return
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatibility
    network_layer_mapping = {}
    if hasattr(sd_model, 'text_encoder') and sd_model.text_encoder is not None:
        for name, module in sd_model.text_encoder.named_modules() :
            prefix = "lora_te1_" if hasattr(sd_model, 'text_encoder_2') else "lora_te_"
            network_name = prefix + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'text_encoder_2') and sd_model.text_encoder_2 is not None:
        for name, module in sd_model.text_encoder_2.named_modules():
            network_name = "lora_te2_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'text_encoder_3') and sd_model.text_encoder_3 is not None:
        for name, module in sd_model.text_encoder_3.named_modules():
            network_name = "lora_te3_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'unet') and sd_model.unet is not None:
        for name, module in sd_model.unet.named_modules():
            network_name = "lora_unet_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    if hasattr(sd_model, 'transformer') and sd_model.transformer is not None:
        for name, module in sd_model.transformer.named_modules():
            network_name = "lora_transformer_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            if "norm" in network_name and "linear" not in network_name and shared.sd_model_type != "sd3":
                continue
            module.network_layer_name = network_name
    if hasattr(sd_model, 'transformer_ref') and sd_model.transformer_ref is not None:
        for name, module in sd_model.transformer_ref.named_modules():
            network_name = "lora_transformer_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            if "norm" in network_name and "linear" not in network_name and shared.sd_model_type != "sd3":
                continue
            module.network_layer_name = network_name
    for tower in TOWERS:
        component = getattr(sd_model, tower, None)
        if component is None or component is getattr(sd_model, 'transformer', None): # a pipeline that reuses the transformer as its tower keeps the transformer's names
            continue
        for name, module in component.named_modules():
            network_name = f"lora_{tower}_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            if "norm" in network_name and "linear" not in network_name and shared.sd_model_type != "sd3":
                continue
            module.network_layer_name = network_name
    if hasattr(sd_model, 'llm_adapter') and sd_model.llm_adapter is not None:
        for name, module in sd_model.llm_adapter.named_modules():
            network_name = "lora_llm_adapter_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    shared.sd_model.network_layer_mapping = network_layer_mapping
