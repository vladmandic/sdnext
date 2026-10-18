# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class MingImageRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def video_rope_cos_sin(cos: torch.Tensor, sin: torch.Tensor, mrope_section: list[int]):
    """Interleave the (frame, height, width) rotary tables into the "video_rope" channel layout.

    `cos` / `sin` are `(3, batch, seq_len, rotary_dim)` for the frame, height and width axes. Height and width
    alternate channel by channel, and the frame axis takes the last channels of each half.
    """
    sections = [mrope_section[0], mrope_section[1] + mrope_section[2]] * 2
    index = 0
    result_cos, result_sin = [], []
    for i, section in enumerate(sections[::-1]):
        if i % 2 == 0:
            for j in range(section):
                row = 1 if j % 2 == 0 else 2
                result_cos.append(cos[row, ..., index : index + 1])
                result_sin.append(sin[row, ..., index : index + 1])
                index += 1
        else:
            result_cos.append(cos[0, ..., index : index + section])
            result_sin.append(sin[0, ..., index : index + section])
            index += section
    return torch.cat(result_cos, dim=-1), torch.cat(result_sin, dim=-1)


class MingImageRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (3, batch, seq_len) for the frame, height and width axes
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        position_ids = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq @ position_ids).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class MingImageMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MingImageGate(nn.Module):
    """Sigmoid router with expert bias and group-limited top-k selection."""

    def __init__(self, hidden_size, num_experts, top_k, n_group, topk_group, routed_scaling_factor):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.zeros((num_experts, hidden_size)))
        self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(logits)

        # pick the top `topk_group` groups by the sum of their two best (bias-corrected) scores, then the top-k
        # experts inside those groups
        scores_for_routing = scores + self.expert_bias
        num_tokens = scores_for_routing.shape[0]
        group_scores = scores_for_routing.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = scores_for_routing.masked_fill(~score_mask.bool(), float("-inf"))
        topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)[1]

        topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


class MingImageSparseMoeBlock(nn.Module):
    """Routed experts plus a shared expert. Tokens at image positions are routed by `image_gate`."""

    def __init__(self, config):
        super().__init__()
        gate_kwargs = {
            "hidden_size": config.hidden_size,
            "num_experts": config.num_experts,
            "top_k": config.num_experts_per_tok,
            "n_group": config.n_group,
            "topk_group": config.topk_group,
            "routed_scaling_factor": config.routed_scaling_factor,
        }
        self.gate = MingImageGate(**gate_kwargs)
        self.image_gate = MingImageGate(**gate_kwargs)
        self.experts = nn.ModuleList(
            [MingImageMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)]
        )
        self.shared_experts = MingImageMLP(
            config.hidden_size, config.moe_intermediate_size * config.num_shared_experts
        )

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        image_topk_idx, image_topk_weight = self.image_gate(hidden_states)
        image_mask = image_mask.view(-1, 1)
        topk_idx = torch.where(image_mask, image_topk_idx, topk_idx)
        topk_weight = torch.where(image_mask, image_topk_weight, topk_weight)

        routed = self.moe_infer(hidden_states.view(-1, hidden_size), topk_idx, topk_weight)
        return routed.view(batch_size, seq_len, hidden_size) + self.shared_experts(hidden_states)

    def moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        tokens_per_expert = torch.bincount(topk_ids.view(-1), minlength=len(self.experts)).tolist()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []
        start_idx = 0
        for expert, num_tokens in zip(self.experts, tokens_per_expert):
            if num_tokens == 0:
                continue
            outputs.append(expert(sorted_tokens[start_idx : start_idx + num_tokens]))
            start_idx += num_tokens

        outs = torch.cat(outputs, dim=0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


class MingImageTextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.mrope_section = config.mrope_section

        self.query_key_value = nn.Linear(
            config.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False
        )
        self.q_norm = MingImageRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MingImageRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states).view(
            batch_size, seq_len, self.num_heads + 2 * self.num_kv_heads, self.head_dim
        )
        query, key, value = qkv.split([self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2)
        query = self.q_norm(query)
        key = self.k_norm(key)

        # partial rotary: only the first `rotary_dim` channels of each head are rotated
        cos, sin = video_rope_cos_sin(*position_embeddings, self.mrope_section)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        query = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
        key = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)

        # (batch, heads, seq_len, head_dim), with the key/value heads repeated for grouped-query attention
        query = query.transpose(1, 2)
        key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2).transpose(1, 2)
        value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2).transpose(1, 2)

        # Explicit attention matching the original model's default ("eager") implementation operation for operation.
        # A fused kernel changes the rounding, which is enough to flip some near-tied expert choices in the MoE
        # layers; at the text encoder's sequence lengths the explicit form costs next to nothing.
        attn_weights = torch.matmul(query / math.sqrt(self.head_dim), key.transpose(2, 3)) + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        hidden_states = torch.matmul(attn_weights, value).transpose(1, 2)
        return self.dense(hidden_states.flatten(2, 3))


class MingImageTextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attention = MingImageTextAttention(config)
        self.is_moe = layer_idx >= config.first_k_dense_replace
        self.mlp = (
            MingImageSparseMoeBlock(config)
            if self.is_moe
            else MingImageMLP(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = MingImageRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MingImageRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embeddings, image_mask) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.input_layernorm(hidden_states), attention_mask, position_embeddings
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, image_mask) if self.is_moe else self.mlp(hidden_states)
        return residual + hidden_states


class MingImageTextEncoder(ModelMixin, ConfigMixin):
    r"""
    The Ming-Image text encoder: the language model of the Ming multimodal LLM (a BailingMoeV2 mixture-of-experts
    decoder with a separate router for image positions and 3D "video_rope" positions), together with the learnable
    image-generation query tokens it is prompted with.

    The query tokens are placed in `input_ids` as a block of `image_patch_token` ids between `image_start_token` and
    `image_end_token`. They are treated as a `1 x num_query_tokens` image: they are routed by the image router and get
    image positions.

    Args:
        vocab_size (`int`): Vocabulary size.
        hidden_size (`int`): Hidden size.
        intermediate_size (`int`): FFN width of the dense layers.
        moe_intermediate_size (`int`): FFN width of each expert.
        num_hidden_layers (`int`): Number of decoder layers.
        num_attention_heads (`int`): Number of query heads.
        num_key_value_heads (`int`): Number of key/value heads.
        head_dim (`int`): Head dimension.
        rms_norm_eps (`float`): RMSNorm epsilon.
        rope_theta (`float`): RoPE base.
        partial_rotary_factor (`float`): Fraction of each head's channels that are rotated.
        mrope_section (`list[int]`): Rotary channels of the frame, height and width axes.
        rope_time_scale (`float`): Frame-axis position step of image tokens.
        spatial_merge_size (`int`): Merge factor between an image grid and its tokens.
        num_experts (`int`): Number of routed experts.
        num_experts_per_tok (`int`): Experts per token.
        n_group (`int`): Number of expert groups for group-limited routing.
        topk_group (`int`): Number of groups a token may route to.
        routed_scaling_factor (`float`): Scale applied to the routing weights.
        num_shared_experts (`int`): Width multiplier of the shared expert.
        first_k_dense_replace (`int`): Number of leading layers with a dense FFN instead of experts.
        pad_token_id (`int`): Padding token id.
        image_patch_token (`int`): Placeholder id of image (and query) tokens.
        image_start_token (`int`): Id that opens an image block.
        image_end_token (`int`): Id that closes an image block.
        num_query_tokens (`int`): Number of image-generation query tokens.
    """

    _no_split_modules = ["MingImageTextDecoderLayer"]

    @register_to_config
    def __init__(
        self,
        vocab_size: int = 157184,
        hidden_size: int = 2048,
        intermediate_size: int = 5120,
        moe_intermediate_size: int = 512,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 600000.0,
        partial_rotary_factor: float = 0.5,
        mrope_section: list[int] = [8, 12, 12],
        rope_time_scale: float = 2.0,
        spatial_merge_size: int = 2,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
        num_shared_experts: int = 1,
        first_k_dense_replace: int = 1,
        pad_token_id: int = 156892,
        image_patch_token: int = 157157,
        image_start_token: int = 157158,
        image_end_token: int = 157159,
        num_query_tokens: int = 256,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.query_tokens = nn.Parameter(torch.zeros(num_query_tokens, hidden_size))
        self.layers = nn.ModuleList(
            [MingImageTextDecoderLayer(self.config, layer_idx) for layer_idx in range(num_hidden_layers)]
        )
        self.norm = MingImageRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = MingImageRotaryEmbedding(int(head_dim * partial_rotary_factor), rope_theta)

    def get_position_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """3D (frame, height, width) positions: text tokens advance all three axes together; each query-token block
        is laid out as a `1 x num_query_tokens` image grid, centred and offset by the text before it."""
        position_ids = torch.ones(3, *input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)
        query_grid_w = self.config.num_query_tokens
        for i in range(input_ids.shape[0]):
            tokens = input_ids[i][attention_mask[i] == 1].tolist()
            num_images = sum(
                1
                for idx, token in enumerate(tokens[:-1])
                if token == self.config.image_start_token and tokens[idx + 1] == self.config.image_patch_token
            )
            positions = []
            start = 0
            for _ in range(num_images):
                end = tokens.index(self.config.image_patch_token, start)
                text_len = end - start
                st_idx = positions[-1][0].max() + 1 if positions else 0
                positions.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.zeros(query_grid_w, dtype=torch.long) * self.config.rope_time_scale + text_len + st_idx
                h_index = torch.zeros(query_grid_w, dtype=torch.long) + t_index
                w_index = torch.arange(query_grid_w) - (query_grid_w - 1) // 2 + t_index
                positions.append(torch.stack([t_index, h_index, w_index]))
                start = end + query_grid_w

            if start < len(tokens):
                st_idx = positions[-1][0].max() + 1 if positions else 0
                positions.append(torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + st_idx)

            position_ids[..., i, attention_mask[i] == 1] = torch.cat(positions, dim=1).to(position_ids)
        return position_ids

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        r"""
        Args:
            input_ids (`torch.Tensor`):
                Token ids of shape `(batch_size, seq_len)`, with each item's query-token block already inserted.
            attention_mask (`torch.Tensor`):
                Padding mask of shape `(batch_size, seq_len)`, 1 for real tokens.

        Returns:
            `tuple[torch.Tensor, ...]`:
                `num_hidden_layers + 1` hidden states of shape `(batch_size, seq_len, hidden_size)`: the input
                embeddings, the outputs of layers 1 to `num_hidden_layers - 1`, and the final normalized output.
        """
        image_mask = input_ids == self.config.image_patch_token
        inputs_embeds = self.word_embeddings(input_ids.clip(0, self.config.vocab_size - 1))
        query_tokens = self.query_tokens.to(inputs_embeds.dtype).repeat(input_ids.shape[0], 1)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), query_tokens)

        position_ids = self.get_position_ids(input_ids, attention_mask)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Additive causal mask that also excludes padded keys, (batch, 1, seq_len, seq_len)
        seq_len = input_ids.shape[1]
        attend = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device).tril()[None, None]
        attend = attend & attention_mask.bool()[:, None, None, :]
        mask = torch.zeros(attend.shape, dtype=inputs_embeds.dtype, device=input_ids.device)
        mask = mask.masked_fill(~attend, torch.finfo(inputs_embeds.dtype).min)

        hidden_states = inputs_embeds
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, mask, position_embeddings, image_mask)
        all_hidden_states += (self.norm(hidden_states),)
        return all_hidden_states
