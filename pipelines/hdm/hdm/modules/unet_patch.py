import json
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import (
    ResnetBlock2D,
)
from diffusers.models.transformers.transformer_2d import (
    Transformer2DModel,
    Transformer2DModelOutput,
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    Attention,
    XFormersAttnProcessor,
    AttnProcessor2_0,
)

try:
    import xformers
    import xformers.ops
except ImportError:
    xformers = None

from .rope import AxialRoPE, make_axial_pos
from ..utils import instantiate


class RoPEAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        head_dim = self.inner_dim // self.heads
        self.axial_rope = AxialRoPE(head_dim, self.heads)
        self.set_processor(RoPEAttnProcessor2_0())

    @classmethod
    def apply_to(cls, original: Attention):
        original.axial_rope = AxialRoPE(
            original.inner_dim // original.heads, original.heads
        )
        original.set_processor(RoPEAttnProcessor2_0())
        original.forward = lambda *args, **kwargs: cls.forward(
            original, *args, **kwargs
        )
        return original

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_map: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            position_map,
            encoder_hidden_states,
            attention_mask,
            **cross_attention_kwargs,
        )


class RoPEAttnProcessor2_0(AttnProcessor2_0):
    def __call__(
        self,
        attn: RoPEAttention,
        hidden_states: torch.Tensor,
        position_map: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        rotary_k = False
        if encoder_hidden_states is None:
            rotary_k = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)

        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query = attn.axial_rope(query, position_map).transpose(1, 2)
        if rotary_k:
            key = attn.axial_rope(key, position_map).transpose(1, 2)
        else:
            key = key.transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RoPEXFormersAttnProcessor(XFormersAttnProcessor):
    def __call__(
        self,
        attn: RoPEAttention,
        hidden_states: torch.Tensor,
        position_map: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask.reshape(
                batch_size, -1, *attention_mask.shape[-2:]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        rotary_k = False
        if encoder_hidden_states is None:
            rotary_k = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.reshape(batch_size, -1, attn.heads, head_dim)
        key = key.reshape(batch_size, -1, attn.heads, head_dim)
        value = value.reshape(batch_size, -1, attn.heads, head_dim)

        query = attn.axial_rope(query, position_map)
        if rotary_k:
            key = attn.axial_rope(key, position_map)

        if attention_mask is not None:
            attention_mask = attention_mask.to(query)
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, inner_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RoPEBasicTransformerBlock(BasicTransformerBlock):
    @classmethod
    def apply_to(cls, original: BasicTransformerBlock):
        original.forward = lambda *args, **kwargs: cls.forward(
            original, *args, **kwargs
        )
        for module in original.modules():
            if isinstance(module, Attention):
                RoPEAttention.apply_to(module)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_map: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            position_map,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                position_map,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class RoPETransformer2DModel(Transformer2DModel):
    _org_init = Transformer2DModel.__init__

    def __init__(self, *args, **kwargs):
        RoPETransformer2DModel._org_init(self, *args, **kwargs)
        for block in self.transformer_blocks:
            if isinstance(block, BasicTransformerBlock):
                RoPEBasicTransformerBlock.apply_to(block)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_map: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            height = self.latent_image_embedding.height
            width = self.latent_image_embedding.width
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = (
                hidden_states.shape[-2] // self.patch_size,
                hidden_states.shape[-1] // self.patch_size,
            )
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = (
                self._operate_on_patched_inputs(
                    hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
                )
            )
        if position_map is None:
            position_map = make_axial_pos(
                h=height,
                w=width,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            position_map = position_map.to(hidden_states)
            assert position_map.shape[-3:] == (height, width, 2)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    position_map,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    position_map,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


org_init = Transformer2DModel.__init__
org_forward = Transformer2DModel.forward


def apply_patch():
    import diffusers.models.transformers.transformer_2d as transformer_2d

    transformer_2d.Transformer2DModel.__init__ = RoPETransformer2DModel.__init__
    transformer_2d.Transformer2DModel.forward = RoPETransformer2DModel.forward


def restore():
    import diffusers.models.transformers.transformer_2d as transformer_2d

    transformer_2d.Transformer2DModel.__init__ = org_init
    transformer_2d.Transformer2DModel.forward = org_forward


class HDUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                nn.init.constant_(module.attn1.to_out[0].weight, 0.0)
                if module.attn2 is not None:
                    nn.init.constant_(module.attn2.to_out[0].weight, 0.0)
                if isinstance(module.ff.net[-2], nn.Linear):
                    nn.init.constant_(module.ff.net[-2].weight, 0.0)
                    nn.init.constant_(module.ff.net[-2].bias, 0.0)
                else:
                    nn.init.constant_(module.ff.net[-1].weight, 0.0)
                    nn.init.constant_(module.ff.net[-1].bias, 0.0)
            if isinstance(module, ResnetBlock2D):
                nn.init.constant_(module.conv2.weight, 0.0)
                nn.init.constant_(module.conv2.bias, 0.0)
        nn.init.constant_(self.conv_out.weight, 0.0)

    @classmethod
    def from_config(cls, arch: dict):
        if isinstance(arch, str):
            with open(arch, "r") as f:
                arch = json.load(f)
        return cls(**instantiate(arch))


class RoPEUNet2DConditionModel(HDUNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        apply_patch()
        super().__init__(*args, **kwargs)
        restore()
        if xformers is not None:
            self.set_attn_processor(RoPEXFormersAttnProcessor())

    @classmethod
    def from_config(cls, arch: dict):
        if isinstance(arch, str):
            with open(arch, "r") as f:
                arch = json.load(f)
        return cls(**instantiate(arch))

    def forward(self, *args, **kwargs):
        apply_patch()
        result = super().forward(*args, **kwargs)
        restore()
        return result
