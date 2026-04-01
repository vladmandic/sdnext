import os
import torch
from typing import List
from collections import namedtuple, OrderedDict
from utils import revise_state_dict


def is_torch2_available():
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
    from .attention_processor import (
        TA_IPAttnProcessor2_0 as TA_IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor, TA_IPAttnProcessor


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=2048, clip_embeddings_dim=1280, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=2048, clip_embeddings_dim=1280):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class MultiIPAdapterImageProjection(torch.nn.Module):
    def __init__(self, IPAdapterImageProjectionLayers):
        super().__init__()
        self.image_projection_layers = torch.nn.ModuleList(IPAdapterImageProjectionLayers)

    def forward(self, image_embeds: List[torch.FloatTensor]):
        projected_image_embeds = []

        # currently, we accept `image_embeds` as
        #  1. a tensor (deprecated) with shape [batch_size, embed_dim] or [batch_size, sequence_length, embed_dim]
        #  2. list of `n` tensors where `n` is number of ip-adapters, each tensor can hae shape [batch_size, num_images, embed_dim] or [batch_size, num_images, sequence_length, embed_dim]
        if not isinstance(image_embeds, list):
            image_embeds = [image_embeds.unsqueeze(1)]

        if len(image_embeds) != len(self.image_projection_layers):
            raise ValueError(
                f"image_embeds must have the same length as image_projection_layers, got {len(image_embeds)} and {len(self.image_projection_layers)}"
            )

        for image_embed, image_projection_layer in zip(image_embeds, self.image_projection_layers):
            batch_size, num_images = image_embed.shape[0], image_embed.shape[1]
            image_embed = image_embed.reshape((batch_size * num_images,) + image_embed.shape[2:])
            image_embed = image_projection_layer(image_embed)
            # image_embed = image_embed.reshape((batch_size, num_images) + image_embed.shape[1:])

            projected_image_embeds.append(image_embed)

        return projected_image_embeds


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj = image_proj_model
        self.ip_adapter = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ip_adapter.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")
        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            state_dict = revise_state_dict(state_dict)

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj.load_state_dict(state_dict["image_proj"], strict=True)
        self.ip_adapter.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ip_adapter.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"


class IPAdapterPlus(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj = image_proj_model
        self.ip_adapter = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ip_adapter.parameters()]))
        org_unet_sum = []
        for attn_name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, (TA_IPAttnProcessor, IPAttnProcessor)):
                org_unet_sum.append(torch.sum(torch.stack([torch.sum(p) for p in attn_proc.parameters()])))
        org_unet_sum = torch.sum(torch.stack(org_unet_sum))

        state_dict = torch.load(ckpt_path, map_location="cpu")
        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            state_dict = revise_state_dict(state_dict)

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj.state_dict()["latents"].shape:
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        missing_key, unexpected_key = self.ip_adapter.load_state_dict(state_dict["ip_adapter"], strict=False)
        if len(missing_key) > 0:
            for ms in missing_key:
                if "ln" not in ms:
                    raise ValueError(f"Missing key in adapter_modules: {len(missing_key)}")
        if len(unexpected_key) > 0:
            raise ValueError(f"Unexpected key in adapter_modules: {len(unexpected_key)}")

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ip_adapter.parameters()]))

        # Verify if the weights loaded to unet
        unet_sum = []
        for attn_name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, (TA_IPAttnProcessor, IPAttnProcessor)):
                unet_sum.append(torch.sum(torch.stack([torch.sum(p) for p in attn_proc.parameters()])))
        unet_sum = torch.sum(torch.stack(unet_sum))

        assert org_unet_sum != unet_sum, "Weights of adapter_modules in unet did not change!"
        assert (unet_sum - new_adapter_sum < 1e-4), "Weights of adapter_modules did not load to unet!"

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_mod`ules did not change!"


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred


class IPAdapterPlusXL(IPAdapterPlus):
    """IP-Adapter with fine-grained features"""

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
