# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
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
"""LLaDA2 MoE model configuration."""

from transformers.configuration_utils import PretrainedConfig


class LLaDA2MoeConfig(PretrainedConfig):
    r"""
    Configuration class for the LLaDA2 MoE model.

    ```python
    >>> from configuration_llada2uni_moe import LLaDA2MoeConfig
    >>> config = LLaDA2MoeConfig()
    ```
    """

    # Keep the original value because it selects the fused-expert implementation.
    model_type = "llada2_moe_veomni"

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        head_dim=None,
        hidden_act="silu",
        use_qkv_bias=False,
        use_qk_norm=True,
        use_bias=True,
        rms_norm_eps=1e-05,
        tie_word_embeddings=False,
        attention_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        rope_parameters=None,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        use_cache=True,
        sliding_window=None,
        pad_token_id=126081,
        # Image
        image_token_offset=157184,
        # MoE
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        norm_topk_prob=True,
        router_dtype="fp32",
        score_function="sigmoid",
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        output_router_logits=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.use_cache = use_cache
        self.sliding_window = sliding_window

        # Image token offset: VQ codebook indices are shifted by this amount in the vocabulary
        self.image_token_offset = image_token_offset

        # RoPE parameters dict — used by LLaDA2MoeRotaryEmbedding
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            }
        self.rope_parameters = rope_parameters

        # MoE
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.norm_topk_prob = norm_topk_prob
        self.router_dtype = router_dtype
        self.score_function = score_function
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        # FP8 quantization flag — set to True to use FP8Linear for experts
        self.use_fp8_experts = kwargs.pop("use_fp8_experts", False)

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["LLaDA2MoeConfig"]
