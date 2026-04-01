# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from . import ras_manager


class RASLuminaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LuminaNextDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        if ras_manager.MANAGER.sample_ratio < 1.0:
            self.k_cache = None
            self.v_cache = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: torch.Tensor | None = None,
        key_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        is_self_attention = True if hidden_states.shape == encoder_hidden_states.shape else False

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.current_step == 0 and is_self_attention:
            self.k_cache = None
            self.v_cache = None

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.current_step > ras_manager.MANAGER.scheduler_end_step and is_self_attention:
            self.k_cache = None
            self.v_cache = None

        # Apply RoPE if needed
        if query_rotary_emb is not None:
            if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
                query = apply_rotary_emb(query, ras_manager.MANAGER.image_rotary_emb_skip, use_real=False)
            else:
                query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
                key = apply_rotary_emb(key, ras_manager.MANAGER.image_rotary_emb_skip, use_real=False)
            else:
                key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        if ras_manager.MANAGER.sample_ratio < 1.0 and (ras_manager.MANAGER.current_step == ras_manager.MANAGER.scheduler_start_step - 1 or ras_manager.MANAGER.current_step in ras_manager.MANAGER.error_reset_steps) and is_self_attention:
            self.k_cache = key
            self.v_cache = value

        if ras_manager.MANAGER.sample_ratio < 1.0 and is_self_attention and ras_manager.MANAGER.is_RAS_step:
            self.k_cache[:, ras_manager.MANAGER.other_patchified_index] = key
            self.v_cache[:, ras_manager.MANAGER.other_patchified_index] = value
            key = self.k_cache
            value = self.v_cache

        query, key = query.to(dtype), key.to(dtype)

        if ras_manager.MANAGER.sample_ratio < 1.0 and is_self_attention and ras_manager.MANAGER.is_RAS_step:
            if is_self_attention:
                sequence_length = key.shape[1]
            else:
                sequence_length = base_sequence_length


        # Apply proportional attention if true
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # perform Grouped-qurey Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        if ras_manager.MANAGER.sample_ratio < 1.0 and is_self_attention and ras_manager.MANAGER.is_RAS_step:
            attention_mask = attention_mask.expand(-1, attn.heads, query.shape[1], -1)
        else:
            attention_mask = attention_mask.expand(-1, attn.heads, sequence_length, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)
        return hidden_states


class RASJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        if ras_manager.MANAGER.sample_ratio < 1.0:
            self.k_cache = None
            self.v_cache = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.current_step == 0:
            self.k_cache = None
            self.v_cache = None

        if ras_manager.MANAGER.sample_ratio < 1.0 and (ras_manager.MANAGER.current_step == ras_manager.MANAGER.scheduler_start_step - 1 or ras_manager.MANAGER.current_step in ras_manager.MANAGER.error_reset_steps):
            self.k_cache = key
            self.v_cache = value

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
            self.k_cache[:, :, ras_manager.MANAGER.other_patchified_index] = key
            self.v_cache[:, :, ras_manager.MANAGER.other_patchified_index] = value
            key = self.k_cache
            value = self.v_cache

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.current_step > ras_manager.MANAGER.scheduler_end_step:
            self.k_cache = None
            self.v_cache = None

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
