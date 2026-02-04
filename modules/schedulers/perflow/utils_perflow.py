import os
from collections import OrderedDict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint


def merge_delta_weights_into_unet(pipe, delta_weights):
    unet_weights = pipe.unet.state_dict()
    assert unet_weights.keys() == delta_weights.keys()
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        unet_weights[key] = unet_weights[key].to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe


def load_delta_weights_into_unet(
    pipe,
    model_path = "hsyan/piecewise-rectified-flow-v0-1",
    base_path = "runwayml/stable-diffusion-v1-5",
):
    ## load delta_weights
    if os.path.exists(os.path.join(model_path, "delta_weights.safetensors")):
        print("### delta_weights exists, loading...")
        delta_weights = OrderedDict()
        with safe_open(os.path.join(model_path, "delta_weights.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                delta_weights[key] = f.get_tensor(key)

    elif os.path.exists(os.path.join(model_path, "diffusion_pytorch_model.safetensors")):
        print("### merged_weights exists, loading...")
        merged_weights = OrderedDict()
        with safe_open(os.path.join(model_path, "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_weights[key] = f.get_tensor(key)

        base_weights = StableDiffusionPipeline.from_pretrained(
            base_path, torch_dtype=torch.float16, safety_checker=None).unet.state_dict()
        assert base_weights.keys() == merged_weights.keys()

        delta_weights = OrderedDict()
        for key in merged_weights.keys():
            delta_weights[key] = merged_weights[key] - base_weights[key].to(device=merged_weights[key].device, dtype=merged_weights[key].dtype)

        print("### saving delta_weights...")
        save_file(delta_weights, os.path.join(model_path, "delta_weights.safetensors"))

    else:
        raise ValueError(f"{model_path} does not contain delta weights or merged weights")

    ## merge delta_weights to the target pipeline
    pipe = merge_delta_weights_into_unet(pipe, delta_weights)
    return pipe


def load_dreambooth_into_pipeline(pipe, sd_dreambooth):
    assert sd_dreambooth.endswith(".safetensors")
    state_dict = {}
    with safe_open(sd_dreambooth, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    unet_config = {} # unet, line 449 in convert_ldm_unet_checkpoint
    for key in pipe.unet.config.keys():
        if key != 'num_class_embeds':
            unet_config[key] = pipe.unet.config[key]

    pipe.unet.load_state_dict(convert_ldm_unet_checkpoint(state_dict, unet_config), strict=False)
    pipe.vae.load_state_dict(convert_ldm_vae_checkpoint(state_dict, pipe.vae.config))
    pipe.text_encoder = convert_ldm_clip_checkpoint(state_dict, text_encoder=pipe.text_encoder)
    return pipe
