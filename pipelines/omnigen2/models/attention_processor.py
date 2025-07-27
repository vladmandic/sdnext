"""
OmniGen2 Attention Processor Module

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from einops import repeat

from diffusers.models.attention_processor import Attention
from .embeddings import apply_rotary_emb


class OmniGen2AttnProcessor:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor is optimized for PyTorch 2.0 and implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling

    Args:
        None

    Raises:
        ImportError: If PyTorch version is less than 2.0
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "OmniGen2AttnProcessor requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
