from typing import Any

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import apply_lora_scale
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device


def situ(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden_states / 25.0))


def apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return hidden_states * freqs_cos + torch.cat((-second, first), dim=-1) * freqs_sin


class MOSSImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, freqs_cos, freqs_sin):
        if attn.to_qkv is not None:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        query = apply_rotary_emb(query, freqs_cos, freqs_sin)
        key = apply_rotary_emb(key, freqs_cos, freqs_sin)
        hidden_states = dispatch_attention_fn(
            query.to(value.dtype),
            key.to(value.dtype),
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.to_out(hidden_states.flatten(2, 3))


class MOSSImageAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MOSSImageAttnProcessor
    _available_processors = [MOSSImageAttnProcessor]

    def __init__(self, dim: int, heads: int, bias: bool, fused_qkv: bool, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        if fused_qkv:
            self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
            self.to_q = self.to_k = self.to_v = None
        else:
            self.to_qkv = None
            self.to_q = nn.Linear(dim, dim, bias=bias)
            self.to_k = nn.Linear(dim, dim, bias=bias)
            self.to_v = nn.Linear(dim, dim, bias=bias)
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(dim, dim, bias=bias)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states, freqs_cos, freqs_sin):
        return self.processor(self, hidden_states, freqs_cos, freqs_sin)


class MOSSImageTextRefinerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_attention_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = MOSSImageAttention(dim, num_attention_heads, bias=True, fused_qkv=True, eps=eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff_gate = nn.Linear(dim, ffn_dim)
        self.ff_up = nn.Linear(dim, ffn_dim)
        self.ff_down = nn.Linear(ffn_dim, dim)

    def forward(self, hidden_states, freqs_cos, freqs_sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), freqs_cos, freqs_sin)
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff_down(situ(self.ff_up(norm_hidden_states), self.ff_gate(norm_hidden_states)))
        return hidden_states + ff_output


class MOSSImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_attention_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = MOSSImageAttention(dim, num_attention_heads, bias=False, fused_qkv=False, eps=eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff_in = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ff_out = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, hidden_states, modulation, freqs_cos, freqs_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(1)
        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn_output = self.attn(norm_hidden_states, freqs_cos, freqs_sin)
        hidden_states = hidden_states + attn_output * gate_msa[:, None]
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_hidden_states, ff_gate = self.ff_in(norm_hidden_states).chunk(2, dim=-1)
        ff_output = self.ff_out(situ(ff_hidden_states, ff_gate))
        return hidden_states + ff_output * gate_mlp[:, None]


class MOSSImageRotaryPosEmbed(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.theta = theta

    def forward(self, height: int, width: int, text_seq_len: int, device):
        exponent = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
        bands = (1 / (self.theta**exponent)).to(device)
        grid_h, grid_w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        image_ids = torch.stack(
            (torch.zeros_like(grid_h), torch.full_like(grid_h, text_seq_len), grid_h, grid_w), dim=-1
        ).flatten(0, 1).float()
        image_ids[:, 2] -= (height - 1) / 2
        image_ids[:, 3] -= (width - 1) / 2
        text_ids = torch.zeros(text_seq_len, 4, device=device, dtype=torch.float32)
        text_ids[:, 1] = torch.arange(text_seq_len, device=device)
        phase = (torch.cat((image_ids, text_ids), dim=0).unsqueeze(-1) * bands).flatten(1)
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos()[None, :, None], phase.sin()[None, :, None]


class MOSSImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    """Upstream MOSS-Image transformer layout used by SDNQ checkpoints."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MOSSImageTransformerBlock", "MOSSImageTextRefinerBlock"]
    _repeated_blocks = ["MOSSImageTransformerBlock", "MOSSImageTextRefinerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 42,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        ffn_dim: int = 8192,
        text_embed_dim: int = 4096,
        timestep_embed_dim: int = 256,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.timestep_embed_dim = timestep_embed_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = MOSSImageRotaryPosEmbed(attention_head_dim, rope_theta)
        self.x_embedder = nn.Linear(in_channels, inner_dim)
        self.context_embedder = nn.Linear(text_embed_dim, inner_dim)
        self.context_refiner = nn.ModuleList(
            [MOSSImageTextRefinerBlock(inner_dim, ffn_dim, num_attention_heads) for _ in range(num_refiner_layers)]
        )
        self.time_embedder = TimestepEmbedding(timestep_embed_dim, inner_dim)
        self.block_modulation = nn.Linear(inner_dim, 6 * inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [MOSSImageTransformerBlock(inner_dim, ffn_dim, num_attention_heads) for _ in range(num_layers)]
        )
        self.norm_out = RMSNorm(inner_dim, eps=1e-6)
        self.norm_out_modulation = nn.Linear(inner_dim, 2 * inner_dim)
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.gradient_checkpointing = False

    @property
    def layers(self):
        return self.transformer_blocks

    def timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.timestep_embed_dim // 2
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, timestep.device)
        freqs = 10000 ** (-torch.arange(half, device=timestep.device, dtype=freqs_dtype) / half)
        phase = timestep.to(freqs_dtype)[:, None] * freqs
        return torch.cat((phase.cos(), phase.sin()), dim=-1).float()

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        skip_layers: list[int] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        attention_kwargs = attention_kwargs or {}
        batch_size, channels, height, width = hidden_states.shape
        if channels != self.in_channels:
            raise ValueError(f"hidden_states must have {self.in_channels} channels, got {channels}.")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("encoder_hidden_states must have shape [batch, sequence, channels].")
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2], dtype=torch.bool, device=encoder_hidden_states.device
            )
        if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_attention_mask must match the first two encoder_hidden_states dimensions.")
        if skip_layers is not None and any(type(index) is not int or index < 0 or index >= len(self.layers) for index in skip_layers):
            raise ValueError(f"skip_layers must contain block indices in [0, {len(self.layers)}).")

        device = hidden_states.device
        timestep = torch.as_tensor(timestep, device=device).reshape(-1).expand(batch_size)
        temb = self.time_embedder(self.timestep_embedding(timestep).to(dtype=self.dtype))
        modulation = self.block_modulation(temb).view(batch_size, 6, -1)
        shift_out, scale_out = self.norm_out_modulation(temb).view(batch_size, 2, -1).unbind(1)
        image_seq_len = height * width
        outputs = []

        from modules.attention.sparse import layout as sparse_layout

        for index in range(batch_size):
            text_mask = encoder_attention_mask[index].bool()
            text_states = encoder_hidden_states[index, text_mask]
            text_seq_len = text_states.shape[0]
            if text_seq_len == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            freqs_cos, freqs_sin = self.rope(height, width, text_seq_len, device)
            freqs_cos, freqs_sin = freqs_cos.to(self.dtype), freqs_sin.to(self.dtype)
            text_freqs = (freqs_cos[:, image_seq_len:], freqs_sin[:, image_seq_len:])
            image_tokens = self.x_embedder(
                hidden_states[index : index + 1].flatten(2).transpose(1, 2).to(dtype=self.dtype).contiguous()
            )
            text_tokens = self.context_embedder(text_states[None].to(dtype=self.dtype))
            for block in self.context_refiner:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    text_tokens = self._gradient_checkpointing_func(block, text_tokens, *text_freqs)
                else:
                    text_tokens = block(text_tokens, *text_freqs)

            tokens = torch.cat((image_tokens, text_tokens), dim=1)
            sparse_layout.publish_segments([("image", image_seq_len), ("text", text_seq_len)], source="moss")
            sample_modulation = modulation[index : index + 1]
            for block_index, block in enumerate(self.transformer_blocks):
                if skip_layers is not None and block_index in skip_layers:
                    continue
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    tokens = self._gradient_checkpointing_func(
                        block, tokens, sample_modulation, freqs_cos, freqs_sin
                    )
                else:
                    tokens = block(tokens, sample_modulation, freqs_cos, freqs_sin)

            image_tokens = self.norm_out(tokens[:, :image_seq_len])
            image_tokens = image_tokens * (1 + scale_out[index : index + 1, None]) + shift_out[index : index + 1, None]
            image_tokens = self.proj_out(image_tokens)
            outputs.append(image_tokens.transpose(1, 2).reshape(1, -1, height, width))

        output = torch.cat(outputs, dim=0)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
