from collections import namedtuple, OrderedDict
import torch
from safetensors import safe_open
from transformers import (
    AutoModel, AutoImageProcessor,
    CLIPVisionModelWithProjection, CLIPImageProcessor)
from .attention_processor import init_attn_proc
from .ip_adapter import MultiIPAdapterImageProjection
from .resampler import Resampler


def init_adapter_in_unet(
        unet,
        image_proj_model=None,
        pretrained_model_path_or_dict=None,
        adapter_tokens=64,
        embedding_dim=None,
        use_lcm=False,
        use_adaln=True,
    ):
        device = unet.device
        dtype = unet.dtype
        if image_proj_model is None:
            assert embedding_dim is not None, "embedding_dim must be provided if image_proj_model is None."
            image_proj_model = Resampler(
                embedding_dim=embedding_dim,
                output_dim=unet.config.cross_attention_dim,
                num_queries=adapter_tokens,
            )
        if pretrained_model_path_or_dict is not None:
            if not isinstance(pretrained_model_path_or_dict, dict):
                if pretrained_model_path_or_dict.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(pretrained_model_path_or_dict, framework="pt", device=unet.device) as f:
                        for key in f.keys():
                            if key.startswith("image_proj."):
                                state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                            elif key.startswith("ip_adapter."):
                                state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
                else:
                    state_dict = torch.load(pretrained_model_path_or_dict, map_location=unet.device)
            else:
                state_dict = pretrained_model_path_or_dict
            keys = list(state_dict.keys())
            if "image_proj" not in keys and "ip_adapter" not in keys:
                state_dict = revise_state_dict(state_dict)

        # Creat IP cross-attention in unet.
        attn_procs = init_attn_proc(unet, adapter_tokens, use_lcm, use_adaln)
        unet.set_attn_processor(attn_procs)

        # Load pretrinaed model if needed.
        if pretrained_model_path_or_dict is not None:
            if "ip_adapter" in state_dict.keys():
                adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
                missing, unexpected = adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
                for mk in missing:
                    if "ln" not in mk:
                        raise ValueError(f"Missing keys in adapter_modules: {missing}")
            if "image_proj" in state_dict.keys():
                image_proj_model.load_state_dict(state_dict["image_proj"])

        # Load image projectors into iterable ModuleList.
        image_projection_layers = []
        image_projection_layers.append(image_proj_model)
        unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)

        # Adjust unet config to handle addtional ip hidden states.
        unet.config.encoder_hid_dim_type = "ip_image_proj"
        unet.to(dtype=dtype, device=device)


def load_adapter_to_pipe(
        pipe,
        pretrained_model_path_or_dict,
        image_encoder_or_path=None,
        feature_extractor_or_path=None,
        use_clip_encoder=False,
        adapter_tokens=64,
        use_lcm=False,
        use_adaln=True,
    ):
    if not isinstance(pretrained_model_path_or_dict, dict):
        if pretrained_model_path_or_dict.endswith(".safetensors"):
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(pretrained_model_path_or_dict, framework="pt", device=pipe.device) as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(pretrained_model_path_or_dict, map_location=pipe.device)
    else:
        state_dict = pretrained_model_path_or_dict
    keys = list(state_dict.keys())
    if "image_proj" not in keys and "ip_adapter" not in keys:
        state_dict = revise_state_dict(state_dict)

    # load CLIP image encoder here if it has not been registered to the pipeline yet
    if image_encoder_or_path is not None:
        if isinstance(image_encoder_or_path, str):
            feature_extractor_or_path = image_encoder_or_path if feature_extractor_or_path is None else feature_extractor_or_path

            image_encoder_or_path = (
                CLIPVisionModelWithProjection.from_pretrained(
                    image_encoder_or_path
                ) if use_clip_encoder else
                AutoModel.from_pretrained(image_encoder_or_path)
            )

    if feature_extractor_or_path is not None:
        if isinstance(feature_extractor_or_path, str):
            feature_extractor_or_path = (
                CLIPImageProcessor() if use_clip_encoder else
                AutoImageProcessor.from_pretrained(feature_extractor_or_path)
            )

    # create image encoder if it has not been registered to the pipeline yet
    if hasattr(pipe, "image_encoder") and getattr(pipe, "image_encoder", None) is None:
        image_encoder = image_encoder_or_path.to(pipe.device, dtype=pipe.dtype)
        pipe.register_modules(image_encoder=image_encoder)
    else:
        image_encoder = pipe.image_encoder

    # create feature extractor if it has not been registered to the pipeline yet
    if hasattr(pipe, "feature_extractor") and getattr(pipe, "feature_extractor", None) is None:
        feature_extractor = feature_extractor_or_path
        pipe.register_modules(feature_extractor=feature_extractor)
    else:
        feature_extractor = pipe.feature_extractor

    # load adapter into unet
    unet = getattr(pipe, pipe.unet_name) if not hasattr(pipe, "unet") else pipe.unet
    attn_procs = init_attn_proc(unet, adapter_tokens, use_lcm, use_adaln)
    unet.set_attn_processor(attn_procs)
    image_proj_model = Resampler(
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        num_queries=adapter_tokens,
    )

    # Load pretrinaed model if needed.
    if "ip_adapter" in state_dict.keys():
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        missing, unexpected = adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
        for mk in missing:
            if "ln" not in mk:
                raise ValueError(f"Missing keys in adapter_modules: {missing}")
    if "image_proj" in state_dict.keys():
        image_proj_model.load_state_dict(state_dict["image_proj"])

    # convert IP-Adapter Image Projection layers to diffusers
    image_projection_layers = []
    image_projection_layers.append(image_proj_model)
    unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)

    # Adjust unet config to handle addtional ip hidden states.
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.to(dtype=pipe.dtype, device=pipe.device)


def revise_state_dict(old_state_dict_or_path, map_location="cpu"):
    new_state_dict = OrderedDict()
    new_state_dict["image_proj"] = OrderedDict()
    new_state_dict["ip_adapter"] = OrderedDict()
    if isinstance(old_state_dict_or_path, str):
        old_state_dict = torch.load(old_state_dict_or_path, map_location=map_location)
    else:
        old_state_dict = old_state_dict_or_path
    for name, weight in old_state_dict.items():
        if name.startswith("image_proj_model."):
            new_state_dict["image_proj"][name[len("image_proj_model."):]] = weight
        elif name.startswith("adapter_modules."):
            new_state_dict["ip_adapter"][name[len("adapter_modules."):]] = weight
    return new_state_dict


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
def encode_image(image_encoder, feature_extractor, image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        return image_enc_hidden_states
    else:
        if isinstance(image_encoder, CLIPVisionModelWithProjection):
            # CLIP image encoder.
            image_embeds = image_encoder(image).image_embeds
        else:
            # DINO image encoder.
            image_embeds = image_encoder(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds
