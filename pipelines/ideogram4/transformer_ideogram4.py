"""Ideogram 4 flow-matching DiT transformer.

A single-stream Diffusion Transformer: Qwen3-VL text features and noisy image
latent tokens are concatenated into one sequence and processed by 34 shared
blocks (QK-RMSNorm attention, SwiGLU MLP, tanh-gated AdaLN), with 3D multimodal
RoPE giving text and image tokens a unified positional space. The model predicts
a flow-matching velocity on the image tokens.

Ported from the reference implementation (github.com/ideogram-oss/ideogram4) as a
diffusers ModelMixin so it loads from the shipped diffusers-layout checkpoint and
can be SDNQ-quantized at load. Parameter names match the reference state dict
1:1, so no key remapping is needed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from pipelines.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, num_heads, L, head_dim); cos/sin: (B, L, head_dim).
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4MRoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: int, mrope_section: tuple[int, ...]) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L, 3) of int (t, h, w).
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, _, _ = position_ids.shape

        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)  # (3, B, L)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
        freqs = inv_freq @ pos.unsqueeze(2)  # (3, B, F, L)
        freqs = freqs.transpose(2, 3)  # (3, B, L, F)

        # interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        # SDPA expects (B, num_heads, L, head_dim).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Block-diagonal mask from segment ids: (B, 1, L, L), True = attend.
        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, norm_eps: float, adaln_dim: int) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adaln_dim, 4 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, adaln_input: torch.Tensor) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(self.attention_norm1(x) * scale_msa, segment_ids=segment_ids, cos=cos, sin=sin)
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x holds a scalar per token; keep its (float) dtype as the compute dtype so
        # SDNQ-quantized Linears (whose .weight is int) don't drive an int cast.
        compute_dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(getattr(self.mlp_in, "compute_dtype", None) or compute_dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adaln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adaln_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer2DModel(ModelMixin, ConfigMixin):
    """Ideogram 4 flow-matching transformer (single-stream DiT)."""

    _no_split_modules = ["Ideogram4TransformerBlock"]
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 18,
        attention_head_dim: int = 256,
        num_layers: int = 34,
        intermediate_size: int = 12288,
        adaln_dim: int = 512,
        in_channels: int = 128,
        llm_features_dim: int = 53248,
        mrope_section: tuple[int, ...] = (24, 20, 20),
        rope_theta: int = 5_000_000,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        emb_dim = num_attention_heads * attention_head_dim
        head_dim = attention_head_dim
        self.num_heads = num_attention_heads
        self.emb_dim = emb_dim

        self.input_proj = nn.Linear(in_channels, emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(llm_features_dim, emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(emb_dim, adaln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, emb_dim)

        self.rotary_emb = Ideogram4MRoPE(head_dim=head_dim, base=rope_theta, mrope_section=tuple(mrope_section))

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=emb_dim,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    norm_eps=norm_eps,
                    adaln_dim=adaln_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(hidden_size=emb_dim, out_channels=in_channels, adaln_dim=adaln_dim)

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity prediction.

        Args:
            llm_features: (B, L, llm_features_dim) Qwen3-VL conditioning features.
            x: (B, L, in_channels) noise tokens.
            t: (B,) or (B, L) flow-matching time in [0, 1].
            position_ids: (B, L, 3) (t, h, w) positions for MRoPE.
            segment_ids: (B, L) sample id within a packed batch.
            indicator: (B, L) per-token role (LLM_TOKEN_INDICATOR / OUTPUT_IMAGE_INDICATOR).

        Returns:
            (B, L, in_channels) velocity in float32; only OUTPUT_IMAGE_INDICATOR
            positions are meaningful.
        """
        _, _, in_channels = x.shape
        assert in_channels == self.config["in_channels"]

        # Compute dtype from a norm weight (never SDNQ-quantized), honoring an
        # explicit Fp8Linear compute_dtype if one is present.
        param_dtype = getattr(self.input_proj, "compute_dtype", None)
        if param_dtype is None:
            w = self.input_proj.weight
            param_dtype = w.dtype if torch.is_floating_point(w) else self.llm_cond_norm.weight.dtype

        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        # Keep shape (B, 1, ...) when t is per-sample so the adaln projections don't
        # pay for L identical copies.
        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        for layer in self.layers:
            h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)

        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
