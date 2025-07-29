from types import MethodType
from typing import Optional
from diffusers.models.attention_processor import Attention
import torch
import torch.nn.functional as F
from .features import feature_injection, normalize, appearance_transfer, get_elem, get_schedule


def get_control_config(structure_schedule, appearance_schedule):
    s = structure_schedule
    a = appearance_schedule

    control_config =\
f"""control_schedule:
    #       structure_conv   structure_attn   appearance_attn  conv/attn
    encoder:                                                # (num layers)
        0: [[             ], [             ], [             ]]  # 2/0
        1: [[             ], [             ], [{a}, {a}     ]]  # 2/2
        2: [[             ], [             ], [{a}, {a}     ]]  # 2/2
    middle: [[            ], [             ], [             ]]  # 2/1
    decoder:
        0: [[{s}          ], [{s}, {s}, {s}], [0.0, {a}, {a}]]  # 3/3
        1: [[             ], [             ], [{a}, {a}     ]]  # 3/3
        2: [[             ], [             ], [             ]]  # 3/0

control_target:
    - [output_tensor]  # structure_conv   choices: {{hidden_states, output_tensor}}
    - [query, key]     # structure_attn   choices: {{query, key, value}}
    - [before]         # appearance_attn  choices: {{before, value, after}}

self_recurrence_schedule:
    - [0.1, 0.5, 2]  # format: [start, end, num_recurrence]"""

    return control_config


def convolution_forward(  # From <class 'diffusers.models.resnet.ResnetBlock2D'>, forward (diffusers==0.28.0)
    self,
    input_tensor: torch.Tensor,
    temb: torch.Tensor,
    *args, # pylint: disable=unused-argument
    **kwargs, # pylint: disable=unused-argument
) -> torch.Tensor:
    do_structure_control = self.do_control and self.t in self.structure_schedule

    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = self.upsample(input_tensor)
        hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
        input_tensor = self.downsample(input_tensor)
        hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if self.time_emb_proj is not None:
        if not self.skip_time_act:
            temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

    if self.time_embedding_norm == "default":
        if temb is not None:
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
    elif self.time_embedding_norm == "scale_shift":
        if temb is None:
            raise ValueError(
                f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
            )
        time_scale, time_shift = torch.chunk(temb, 2, dim=1)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * (1 + time_scale) + time_shift
    else:
        hidden_states = self.norm2(hidden_states)

    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    # Feature injection and AdaIN (hidden_states)
    if do_structure_control and "hidden_states" in self.structure_target:
        hidden_states = feature_injection(hidden_states, batch_order=self.batch_order)

    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

    # Feature injection and AdaIN (output_tensor)
    if do_structure_control and "output_tensor" in self.structure_target:
        output_tensor = feature_injection(output_tensor, batch_order=self.batch_order)

    return output_tensor


class AttnProcessor2_0:  # From <class 'diffusers.models.attention_processor.AttnProcessor2_0'> (diffusers==0.28.0)

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__( # pylint: disable=keyword-arg-before-vararg
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        do_structure_control = attn.do_control and attn.t in attn.structure_schedule
        do_appearance_control = attn.do_control and attn.t in attn.appearance_schedule

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        no_encoder_hidden_states = encoder_hidden_states is None
        if no_encoder_hidden_states:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if do_appearance_control:  # Assume we only have this for self attention
            hidden_states_normed = normalize(hidden_states, dim=-2)  # B H D C
            encoder_hidden_states_normed = normalize(encoder_hidden_states, dim=-2)

            query_normed = attn.to_q(hidden_states_normed)
            key_normed = attn.to_k(encoder_hidden_states_normed)

            inner_dim = key_normed.shape[-1]
            head_dim = inner_dim // attn.heads
            query_normed = query_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_normed = key_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Match query and key injection with structure injection (if injection is happening this layer)
            if do_structure_control:
                if "query" in attn.structure_target:
                    query_normed = feature_injection(query_normed, batch_order=attn.batch_order)
                if "key" in attn.structure_target:
                    key_normed = feature_injection(key_normed, batch_order=attn.batch_order)

        # Appearance transfer (before)
        if do_appearance_control and "before" in attn.appearance_target:
            hidden_states = hidden_states.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = appearance_transfer(hidden_states, query_normed, key_normed, batch_order=attn.batch_order)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            if no_encoder_hidden_states:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Feature injection (query, key, and/or value)
        if do_structure_control:
            if "query" in attn.structure_target:
                query = feature_injection(query, batch_order=attn.batch_order)
            if "key" in attn.structure_target:
                key = feature_injection(key, batch_order=attn.batch_order)
            if "value" in attn.structure_target:
                value = feature_injection(value, batch_order=attn.batch_order)

        # Appearance transfer (value)
        if do_appearance_control and "value" in attn.appearance_target:
            value = appearance_transfer(value, query_normed, key_normed, batch_order=attn.batch_order)

        # The output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Appearance transfer (after)
        if do_appearance_control and "after" in attn.appearance_target:
            hidden_states = appearance_transfer(hidden_states, query_normed, key_normed, batch_order=attn.batch_order)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_control(
    model,
    timesteps,
    control_schedule,  # structure_conv, structure_attn, appearance_attn
    control_target = [["output_tensor"], ["query", "key"], ["before"]],
):
    # Assume timesteps in reverse order (T -> 0)
    for block_type in ["encoder", "decoder", "middle"]:
        blocks = {
            "encoder": model.unet.down_blocks,
            "decoder": model.unet.up_blocks,
            "middle": [model.unet.mid_block],
        }[block_type]

        control_schedule_block = control_schedule[block_type]
        if block_type == "middle":
            control_schedule_block = [control_schedule_block]

        for layer in range(len(control_schedule_block)):
            # Convolution
            num_blocks = len(blocks[layer].resnets) if hasattr(blocks[layer], "resnets") else 0
            for block in range(num_blocks):
                convolution = blocks[layer].resnets[block]
                convolution.structure_target = control_target[0]
                convolution.structure_schedule = get_schedule(
                    timesteps, get_elem(control_schedule_block[layer][0], block)
                )
                convolution.forward = MethodType(convolution_forward, convolution)

            # Self-attention
            num_blocks = len(blocks[layer].attentions) if hasattr(blocks[layer], "attentions") else 0
            for block in range(num_blocks):
                for transformer_block in blocks[layer].attentions[block].transformer_blocks:
                    attention = transformer_block.attn1
                    attention.structure_target = control_target[1]
                    attention.structure_schedule = get_schedule(
                        timesteps, get_elem(control_schedule_block[layer][1], block)
                    )
                    attention.appearance_target = control_target[2]
                    attention.appearance_schedule = get_schedule(
                        timesteps, get_elem(control_schedule_block[layer][2], block)
                    )
                    attention.processor = AttnProcessor2_0()


def register_attr(model, t, do_control, batch_order):
    for layer_type in ["encoder", "decoder", "middle"]:
        blocks = {"encoder": model.unet.down_blocks, "decoder": model.unet.up_blocks,
                  "middle": [model.unet.mid_block]}[layer_type]
        for layer in blocks:
            # Convolution
            for module in layer.resnets:
                module.t = t
                module.do_control = do_control
                module.batch_order = batch_order
            # Self-attention
            if hasattr(layer, "attentions"):
                for block in layer.attentions:
                    for module in block.transformer_blocks:
                        module.attn1.t = t
                        module.attn1.do_control = do_control
                        module.attn1.batch_order = batch_order
