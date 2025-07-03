from safetensors.torch import load_file
from scripts.layerdiffuse.layerdiffuse_model import LoraLoader, AttentionSharingProcessor


def merge_delta_weights_into_unet(pipe, delta_weights):
    unet_weights = pipe.unet.state_dict()

    for k in delta_weights.keys():
        assert k in unet_weights.keys(), k

    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        unet_weights[key] = unet_weights[key].to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def load_lora_to_unet(unet, model_path, frames, device, dtype):
    module_mapping_sd15 = {0: 'input_blocks.1.1.transformer_blocks.0.attn1', 1: 'input_blocks.1.1.transformer_blocks.0.attn2', 2: 'input_blocks.2.1.transformer_blocks.0.attn1', 3: 'input_blocks.2.1.transformer_blocks.0.attn2', 4: 'input_blocks.4.1.transformer_blocks.0.attn1', 5: 'input_blocks.4.1.transformer_blocks.0.attn2', 6: 'input_blocks.5.1.transformer_blocks.0.attn1', 7: 'input_blocks.5.1.transformer_blocks.0.attn2', 8: 'input_blocks.7.1.transformer_blocks.0.attn1', 9: 'input_blocks.7.1.transformer_blocks.0.attn2', 10: 'input_blocks.8.1.transformer_blocks.0.attn1', 11: 'input_blocks.8.1.transformer_blocks.0.attn2', 12: 'output_blocks.3.1.transformer_blocks.0.attn1', 13: 'output_blocks.3.1.transformer_blocks.0.attn2', 14: 'output_blocks.4.1.transformer_blocks.0.attn1', 15: 'output_blocks.4.1.transformer_blocks.0.attn2', 16: 'output_blocks.5.1.transformer_blocks.0.attn1', 17: 'output_blocks.5.1.transformer_blocks.0.attn2', 18: 'output_blocks.6.1.transformer_blocks.0.attn1', 19: 'output_blocks.6.1.transformer_blocks.0.attn2', 20: 'output_blocks.7.1.transformer_blocks.0.attn1', 21: 'output_blocks.7.1.transformer_blocks.0.attn2', 22: 'output_blocks.8.1.transformer_blocks.0.attn1', 23: 'output_blocks.8.1.transformer_blocks.0.attn2', 24: 'output_blocks.9.1.transformer_blocks.0.attn1', 25: 'output_blocks.9.1.transformer_blocks.0.attn2', 26: 'output_blocks.10.1.transformer_blocks.0.attn1', 27: 'output_blocks.10.1.transformer_blocks.0.attn2', 28: 'output_blocks.11.1.transformer_blocks.0.attn1', 29: 'output_blocks.11.1.transformer_blocks.0.attn2', 30: 'middle_block.1.transformer_blocks.0.attn1', 31: 'middle_block.1.transformer_blocks.0.attn2'}

    sd15_to_diffusers = {
        'input_blocks.1.1.transformer_blocks.0.attn1': 'down_blocks.0.attentions.0.transformer_blocks.0.attn1',
        'input_blocks.1.1.transformer_blocks.0.attn2': 'down_blocks.0.attentions.0.transformer_blocks.0.attn2',
        'input_blocks.2.1.transformer_blocks.0.attn1': 'down_blocks.0.attentions.1.transformer_blocks.0.attn1',
        'input_blocks.2.1.transformer_blocks.0.attn2': 'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
        'input_blocks.4.1.transformer_blocks.0.attn1': 'down_blocks.1.attentions.0.transformer_blocks.0.attn1',
        'input_blocks.4.1.transformer_blocks.0.attn2': 'down_blocks.1.attentions.0.transformer_blocks.0.attn2',
        'input_blocks.5.1.transformer_blocks.0.attn1': 'down_blocks.1.attentions.1.transformer_blocks.0.attn1',
        'input_blocks.5.1.transformer_blocks.0.attn2': 'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
        'input_blocks.7.1.transformer_blocks.0.attn1': 'down_blocks.2.attentions.0.transformer_blocks.0.attn1',
        'input_blocks.7.1.transformer_blocks.0.attn2': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2',
        'input_blocks.8.1.transformer_blocks.0.attn1': 'down_blocks.2.attentions.1.transformer_blocks.0.attn1',
        'input_blocks.8.1.transformer_blocks.0.attn2': 'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
        'output_blocks.3.1.transformer_blocks.0.attn1': "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        'output_blocks.3.1.transformer_blocks.0.attn2': "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
        'output_blocks.4.1.transformer_blocks.0.attn1': "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        'output_blocks.4.1.transformer_blocks.0.attn2': "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
        'output_blocks.5.1.transformer_blocks.0.attn1': "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        'output_blocks.5.1.transformer_blocks.0.attn2': "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
        'output_blocks.6.1.transformer_blocks.0.attn1': "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        'output_blocks.6.1.transformer_blocks.0.attn2': "up_blocks.2.attentions.0.transformer_blocks.0.attn2",
        'output_blocks.7.1.transformer_blocks.0.attn1': "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        'output_blocks.7.1.transformer_blocks.0.attn2': "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
        'output_blocks.8.1.transformer_blocks.0.attn1': "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        'output_blocks.8.1.transformer_blocks.0.attn2': "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
        'output_blocks.9.1.transformer_blocks.0.attn1': "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        'output_blocks.9.1.transformer_blocks.0.attn2': "up_blocks.3.attentions.0.transformer_blocks.0.attn2",
        'output_blocks.10.1.transformer_blocks.0.attn1': "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        'output_blocks.10.1.transformer_blocks.0.attn2': "up_blocks.3.attentions.1.transformer_blocks.0.attn2",
        'output_blocks.11.1.transformer_blocks.0.attn1': "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        'output_blocks.11.1.transformer_blocks.0.attn2': "up_blocks.3.attentions.2.transformer_blocks.0.attn2",
        'middle_block.1.transformer_blocks.0.attn1': "mid_block.attentions.0.transformer_blocks.0.attn1",
        'middle_block.1.transformer_blocks.0.attn2': "mid_block.attentions.0.transformer_blocks.0.attn2",
    }

    layer_list = []
    for i in range(32):
        real_key = module_mapping_sd15[i]
        diffuser_key = sd15_to_diffusers[real_key]
        attn_module = get_attr(unet, diffuser_key)
        u = AttentionSharingProcessor(attn_module, frames=frames, use_control=False).to(device=device, dtype=dtype)
        layer_list.append(u)
        attn_module.set_processor(u)

    loader = LoraLoader(layer_list)
    lora_state_dict = load_file(model_path)
    loader.load_state_dict(lora_state_dict)
