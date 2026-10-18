# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm


class MingImageConnector(ModelMixin, ConfigMixin):
    r"""
    Turns Ming-Image text-encoder hidden states into the two conditioning streams of the transformer.

    The query-token hidden states go through `proj_in`, a bidirectional Qwen2 stack and `proj_out` to give
    `prompt_embeds`. The text-token hidden states of the selected encoder layers, concatenated along the feature axis,
    go through `proj_directvlm` to give `prompt_embeds_2`.

    Args:
        text_hidden_size (`int`, defaults to `2048`): Hidden size of the text encoder.
        hidden_state_layers (`list[int]`, defaults to `[5, 12, 20]`):
            Indices into the text encoder's hidden states (0 = input embeddings) whose text-token states are
            concatenated into the text stream.
        cond_dim (`int`, defaults to `2560`): Width of `prompt_embeds`.
        inner_dim (`int`, defaults to `3840`): Width of `prompt_embeds_2` (the transformer's model width).
        qwen2_config (`dict`): Config of the Qwen2 stack, as accepted by `transformers.Qwen2Config`.
    """

    _no_split_modules = ["Qwen2DecoderLayer"]

    @register_to_config
    def __init__(
        self,
        text_hidden_size: int = 2048,
        hidden_state_layers: list[int] = [5, 12, 20],
        cond_dim: int = 2560,
        inner_dim: int = 3840,
        qwen2_config: dict | None = None,
    ):
        super().__init__()
        qwen2_config = Qwen2Config(**qwen2_config)
        qwen2_config._attn_implementation = "sdpa"

        self.proj_in = nn.Linear(text_hidden_size, qwen2_config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(qwen2_config, layer_idx) for layer_idx in range(qwen2_config.num_hidden_layers)]
        )
        for layer in self.layers:
            # the connector attends bidirectionally over the query tokens
            layer.self_attn.is_causal = False
        self.norm = Qwen2RMSNorm(qwen2_config.hidden_size, eps=qwen2_config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=qwen2_config)
        self.proj_out = nn.Linear(qwen2_config.hidden_size, cond_dim)

        directvlm_dim = text_hidden_size * len(hidden_state_layers)
        self.proj_directvlm = nn.Sequential(RMSNorm(directvlm_dim, eps=1e-5), nn.Linear(directvlm_dim, inner_dim))

    def forward(
        self, query_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            query_hidden_states (`torch.Tensor`):
                Last-layer text-encoder hidden states at the query-token positions, of shape
                `(batch_size, num_query_tokens, text_hidden_size)`.
            text_hidden_states (`torch.Tensor`):
                Text-encoder hidden states at the text positions for each selected layer, concatenated along the last
                axis, of shape `(batch_size, text_seq_len, text_hidden_size * len(hidden_state_layers))`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`:
                `prompt_embeds` of shape `(batch_size, num_query_tokens, cond_dim)` and `prompt_embeds_2` of shape
                `(batch_size, text_seq_len, inner_dim)`.
        """
        hidden_states = self.proj_in(query_hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # The original model adds an all-ones mask to the attention scores. It leaves the softmax unchanged
        # mathematically but changes its rounding, so it is kept to reproduce the original outputs exactly.
        attention_mask = hidden_states.new_ones(batch_size, 1, seq_len, seq_len)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states)
        prompt_embeds = self.proj_out(hidden_states)

        prompt_embeds_2 = self.proj_directvlm(text_hidden_states)
        return prompt_embeds, prompt_embeds_2
