# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
PADDING_MODES = ("zero_masked", "learned")


# Copied from diffusers.models.transformers.transformer_z_image.TimestepEmbedder
class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class MingImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MingImageAttention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # (batch, seq_len) -> (batch, 1, 1, seq_len), broadcast over heads and queries
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        dtype = query.dtype
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MingImageAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MingImageAttnProcessor
    _available_processors = [MingImageAttnProcessor]

    def __init__(self, dim: int, heads: int, eps: float = 1e-5, processor=None):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, freqs_cis)


# Copied from diffusers.models.transformers.transformer_z_image.FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x) -> torch.Tensor:
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


@maybe_allow_in_graph
class MingImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, norm_eps: float, modulation: bool = True):
        super().__init__()
        self.attention = MingImageAttention(dim, n_heads, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class MingImageFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        return self.linear(x)


class MingImageRopeEmbedder:
    def __init__(self, theta: float, axes_dims: list[int], axes_lens: list[int]):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: list[int], end: list[int], theta: float):
        # Precomputed once on CPU in float64: the frame axis reaches position 20480, where float32 angles drift
        # by ~1e-3 rad. The complex64 tables are then moved to the model's device.
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, dtype=torch.float64, device="cpu")
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
        if self.freqs_cis[0].device != ids.device:
            self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def create_coordinate_grid(size, start, device):
    axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
    return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1).flatten(0, -2)


class MingImageTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    The single-stream diffusion transformer used by Ming-Image.

    It is the Z-Image transformer with three additions: a second caption stream (`encoder_hidden_states_2`) that is
    already in model width and skips `cap_embedder`, reference latents concatenated along the frame axis
    (`ref_hidden_states`), and a choice of how sequence-alignment padding is handled (`alignment_padding_mode`).

    Args:
        patch_size (`int`, defaults to `2`): Spatial patch size.
        f_patch_size (`int`, defaults to `1`): Frame patch size.
        in_channels (`int`, defaults to `16`): Latent channels.
        dim (`int`, defaults to `3840`): Model width.
        n_layers (`int`, defaults to `30`): Number of main transformer blocks.
        n_refiner_layers (`int`, defaults to `2`): Number of noise and context refiner blocks each.
        n_heads (`int`, defaults to `30`): Number of attention heads.
        norm_eps (`float`, defaults to `1e-5`): Epsilon of the RMS norms.
        cap_feat_dim (`int`, defaults to `2560`): Width of `encoder_hidden_states`.
        rope_theta (`float`, defaults to `256.0`): RoPE base.
        t_scale (`float`, defaults to `1000.0`): Multiplier applied to the timestep before embedding.
        axes_dims (`list[int]`, defaults to `[32, 48, 48]`): RoPE dimensions of the frame, height and width axes.
        axes_lens (`list[int]`, defaults to `[20480, 512, 512]`): RoPE table length of each axis.
        alignment_padding_mode (`str`, defaults to `"zero_masked"`):
            How the tokens that pad each sequence to a multiple of 32 are handled. `"zero_masked"` zeroes them and
            masks them out of attention; `"learned"` replaces them with learned pad tokens that stay attended.
        multi_frame_output (`bool`, defaults to `False`):
            Return every frame of the latent sequence instead of only the first one.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MingImageTransformerBlock"]
    _repeated_blocks = ["MingImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        f_patch_size: int = 1,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: list[int] = [32, 48, 48],
        axes_lens: list[int] = [20480, 512, 512],
        alignment_padding_mode: str = "zero_masked",
        multi_frame_output: bool = False,
    ) -> None:
        super().__init__()
        if alignment_padding_mode not in PADDING_MODES:
            raise ValueError(f"alignment_padding_mode must be one of {PADDING_MODES}, got {alignment_padding_mode!r}")
        if dim // n_heads != sum(axes_dims):
            raise ValueError(f"head dim {dim // n_heads} must equal sum(axes_dims)={sum(axes_dims)}")

        self.out_channels = in_channels
        self.gradient_checkpointing = False

        patch_key = f"{patch_size}-{f_patch_size}"
        patch_dim = f_patch_size * patch_size * patch_size * in_channels
        self.all_x_embedder = nn.ModuleDict({patch_key: nn.Linear(patch_dim, dim, bias=True)})
        self.all_final_layer = nn.ModuleDict({patch_key: MingImageFinalLayer(dim, patch_dim)})

        self.noise_refiner = nn.ModuleList(
            [MingImageTransformerBlock(dim, n_heads, norm_eps, modulation=True) for _ in range(n_refiner_layers)]
        )
        self.context_refiner = nn.ModuleList(
            [MingImageTransformerBlock(dim, n_heads, norm_eps, modulation=False) for _ in range(n_refiner_layers)]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        if alignment_padding_mode == "learned":
            self.x_pad_token = nn.Parameter(torch.zeros((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.zeros((1, dim)))

        self.layers = nn.ModuleList([MingImageTransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)])
        self.rope_embedder = MingImageRopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def _attention_mask(self, keep: list[torch.Tensor]) -> torch.Tensor | None:
        """Boolean (batch, max_len) mask from per-item boolean `keep` vectors; `None` when nothing is masked."""
        mask = pad_sequence(keep, batch_first=True, padding_value=False)
        return None if mask.all() else mask

    def _run_blocks(self, blocks, hidden_states, attn_mask, freqs_cis, adaln_input=None):
        for block in blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attn_mask, freqs_cis, adaln_input
                )
            else:
                hidden_states = block(hidden_states, attn_mask, freqs_cis, adaln_input)
        return hidden_states

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        encoder_hidden_states_2: list[torch.Tensor],
        ref_hidden_states: list[torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[list[torch.Tensor]]:
        r"""
        The [`MingImageTransformer2DModel`] forward method.

        Args:
            hidden_states (`list[torch.Tensor]`):
                Noisy latents, one `(in_channels, num_frames, height, width)` tensor per batch item.
            timestep (`torch.Tensor`):
                Normalized denoising time in `[0, 1]` of shape `(batch_size,)`, where 0 is pure noise.
            encoder_hidden_states (`list[torch.Tensor]`):
                Query-token conditioning, one `(seq_len, cap_feat_dim)` tensor per batch item. Passed through
                `cap_embedder`.
            encoder_hidden_states_2 (`list[torch.Tensor]`):
                Text-token conditioning already in model width, one `(seq_len, dim)` tensor per batch item.
            ref_hidden_states (`list[torch.Tensor]`, *optional*):
                Reference latents, one `(in_channels, num_ref_frames, height, width)` tensor per batch item, appended
                after the noisy frames along the frame axis.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The predicted velocity as a list of `(in_channels, num_frames, height, width)` tensors, where
                `num_frames` is 1 unless `multi_frame_output` is set. Returned as a one-element tuple when
                `return_dict` is `False`.
        """
        p = self.config.patch_size
        pf = self.config.f_patch_size
        patch_key = f"{p}-{pf}"
        learned_padding = self.config.alignment_padding_mode == "learned"
        device = hidden_states[0].device

        if ref_hidden_states is None:
            ref_hidden_states = [None] * len(hidden_states)

        # Patchify the latents and lay out caption and image positions. Each sequence is padded to a multiple of
        # SEQ_MULTI_OF; caption positions start at 1 on the frame axis, image positions right after the captions.
        x, x_sizes, x_pos_ids, x_pad_masks = [], [], [], []
        cap_feats_2, cap_pos_ids, cap_pad_masks = [], [], []
        for image, ref_image, cap_feat, cap_feat_2 in zip(
            hidden_states, ref_hidden_states, encoder_hidden_states, encoder_hidden_states_2
        ):
            cap_len = len(cap_feat) + len(cap_feat_2)
            cap_padding = (-cap_len) % SEQ_MULTI_OF
            cap_pos_ids.append(create_coordinate_grid((cap_len + cap_padding, 1, 1), (1, 0, 0), device))
            cap_pad_masks.append(
                torch.cat(
                    [
                        torch.zeros(cap_len, dtype=torch.bool, device=device),
                        torch.ones(cap_padding, dtype=torch.bool, device=device),
                    ]
                )
            )
            cap_feats_2.append(torch.cat([cap_feat_2, cap_feat_2.new_zeros(cap_padding, cap_feat_2.shape[-1])]))

            if ref_image is not None:
                image = torch.cat([image, ref_image], dim=1)
            channels, frames, height, width = image.shape
            x_sizes.append((frames, height, width))
            f_tokens, h_tokens, w_tokens = frames // pf, height // p, width // p
            image = image.view(channels, f_tokens, pf, h_tokens, p, w_tokens, p)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(f_tokens * h_tokens * w_tokens, -1)

            image_len = len(image)
            image_padding = (-image_len) % SEQ_MULTI_OF
            x_pos_ids.append(
                torch.cat(
                    [
                        create_coordinate_grid(
                            (f_tokens, h_tokens, w_tokens), (cap_len + cap_padding + 1, 0, 0), device
                        ),
                        torch.zeros((image_padding, 3), dtype=torch.int32, device=device),
                    ]
                )
            )
            x_pad_masks.append(
                torch.cat(
                    [
                        torch.zeros(image_len, dtype=torch.bool, device=device),
                        torch.ones(image_padding, dtype=torch.bool, device=device),
                    ]
                )
            )
            x.append(torch.cat([image, image[-1:].repeat(image_padding, 1)]))

        # Masks: padded batch positions are always excluded; alignment padding only in zero_masked mode.
        x_keep = [~m if not learned_padding else torch.ones_like(m) for m in x_pad_masks]
        cap_keep = [~m if not learned_padding else torch.ones_like(m) for m in cap_pad_masks]

        adaln_input = self.t_embedder(timestep * self.config.t_scale)

        # Image embed + noise refiner
        x_lens = [len(item) for item in x]
        x = self.all_x_embedder[patch_key](torch.cat(x))
        adaln_input = adaln_input.type_as(x)
        x[torch.cat(x_pad_masks)] = self.x_pad_token if learned_padding else 0.0
        x_pos_ids_flat = torch.cat(x_pos_ids)
        x_freqs = list(self.rope_embedder(x_pos_ids_flat).split(x_lens))
        x = pad_sequence(list(x.split(x_lens)), batch_first=True, padding_value=0.0)
        x_freqs_padded = pad_sequence(x_freqs, batch_first=True, padding_value=0.0)[:, : x.shape[1]]
        x = self._run_blocks(self.noise_refiner, x, self._attention_mask(x_keep), x_freqs_padded, adaln_input)

        # Caption embed + context refiner. Only the query-token stream goes through cap_embedder.
        cap_lens_1 = [len(item) for item in encoder_hidden_states]
        cap = list(self.cap_embedder(torch.cat(encoder_hidden_states)).split(cap_lens_1))
        cap = [torch.cat([c1, c2]) for c1, c2 in zip(cap, cap_feats_2)]
        cap_lens = [len(item) for item in cap]
        cap = torch.cat(cap)
        cap[torch.cat(cap_pad_masks)] = self.cap_pad_token if learned_padding else 0.0
        cap_freqs = list(self.rope_embedder(torch.cat(cap_pos_ids)).split(cap_lens))
        cap = pad_sequence(list(cap.split(cap_lens)), batch_first=True, padding_value=0.0)
        cap_freqs_padded = pad_sequence(cap_freqs, batch_first=True, padding_value=0.0)[:, : cap.shape[1]]
        cap = self._run_blocks(self.context_refiner, cap, self._attention_mask(cap_keep), cap_freqs_padded)

        # Unified [image, caption] sequence through the main blocks
        unified = pad_sequence(
            [torch.cat([x[i][: x_lens[i]], cap[i][: cap_lens[i]]]) for i in range(len(x_lens))],
            batch_first=True,
            padding_value=0.0,
        )
        unified_freqs = pad_sequence(
            [torch.cat([x_freqs[i], cap_freqs[i]]) for i in range(len(x_lens))],
            batch_first=True,
            padding_value=0.0,
        )
        unified_keep = [torch.cat([xk, ck]) for xk, ck in zip(x_keep, cap_keep)]
        unified = self._run_blocks(
            self.layers, unified, self._attention_mask(unified_keep), unified_freqs, adaln_input
        )
        unified = self.all_final_layer[patch_key](unified, adaln_input)

        # Unpatchify, dropping alignment padding and caption tokens
        output = []
        for item, (frames, height, width) in zip(unified.unbind(dim=0), x_sizes):
            f_tokens, h_tokens, w_tokens = frames // pf, height // p, width // p
            item = (
                item[: f_tokens * h_tokens * w_tokens]
                .view(f_tokens, h_tokens, w_tokens, pf, p, p, self.out_channels)
                # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, frames, height, width)
            )
            if not self.config.multi_frame_output:
                item = item[:, :1]
            output.append(item)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
