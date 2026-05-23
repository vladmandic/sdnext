"""Lens denoising transformer (DiT).

The model uses a double-stream architecture with joint image+text attention,
RoPE on both streams, and SwiGLU MLPs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm


# ---------------------------------------------------------------------------
# Embeddings & RoPE
# ---------------------------------------------------------------------------


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (DDPM-style)."""
    assert timesteps.ndim == 1, "Timesteps should be 1-D"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_lens(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply complex-valued RoPE (Lens variant).

    Args:
        x:         [B, S, H, D] query or key tensor.
        freqs_cis: [S, D/2] complex tensor of rotation factors.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)  # broadcast over heads
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class GateMLP(nn.Module):
    """SwiGLU MLP used by the transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LensTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj(timestep)
        return self.timestep_embedder(proj.to(dtype=hidden_states.dtype))


class LensEmbedRope(nn.Module):
    """Frame/H/W axial RoPE shared between image and text streams."""

    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [self._rope_params(pos_index, d, theta) for d in axes_dim], dim=1
        )
        self.neg_freqs = torch.cat(
            [self._rope_params(neg_index, d, theta) for d in axes_dim], dim=1
        )
        # Note: we deliberately do NOT register these as buffers - registering
        # complex tensors as buffers strips the imaginary component on save/load.
        self.rope_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(
            index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: Union[List[Tuple[int, int, int]], Tuple[int, int, int]],
        txt_seq_lens: Union[List[int], int],
        device: torch.device = torch.device("cuda"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]
        if not isinstance(txt_seq_lens, list):
            txt_seq_lens = [txt_seq_lens]
        assert len(video_fhw) == 1, "video_fhw must have length 1"

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if rope_key not in self.rope_cache:
                self.rope_cache[rope_key] = (
                    self._compute_video_freqs(frame, height, width, idx=0).to("cpu")
                )
            video_freq = self.rope_cache[rope_key].to(device)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)
            vid_freqs.append(video_freq)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
            ).view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0
            ).view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


# ---------------------------------------------------------------------------
# Attention (joint image + text, plain SDPA)
# ---------------------------------------------------------------------------


class LensJointAttention(nn.Module):
    """Joint image+text attention with fused QKV and SDPA backend."""

    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = self.inner_dim // dim_head
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

        self.img_qkv = nn.Linear(query_dim, 3 * self.inner_dim, bias=True)
        self.txt_qkv = nn.Linear(added_kv_proj_dim, 3 * self.inner_dim, bias=True)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.out_dim, bias=True), nn.Identity()])
        self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_img, _ = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]

        # Fused QKV per stream -> split.
        img_qkv = self.img_qkv(hidden_states).view(bsz, seq_img, 3, self.heads, self.dim_head)
        txt_qkv = self.txt_qkv(encoder_hidden_states).view(bsz, seq_txt, 3, self.heads, self.dim_head)
        img_q, img_k, img_v = img_qkv.unbind(dim=2)
        txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)

        # QK RMSNorm.
        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        # RoPE.
        img_freqs, txt_freqs = image_rotary_emb
        if img_freqs.shape[0] < seq_img:
            raise ValueError(
                f"Image RoPE length {img_freqs.shape[0]} is shorter than "
                f"image sequence length {seq_img}."
            )
        img_freqs = img_freqs[:seq_img]
        img_q = apply_rotary_emb_lens(img_q, img_freqs)
        img_k = apply_rotary_emb_lens(img_k, img_freqs)
        if seq_txt > 0:
            if txt_freqs.shape[0] < seq_txt:
                raise ValueError(
                    f"Text RoPE length {txt_freqs.shape[0]} is shorter than "
                    f"text sequence length {seq_txt}."
                )
            txt_freqs = txt_freqs[:seq_txt]
            txt_q = apply_rotary_emb_lens(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_lens(txt_k, txt_freqs)

        # Joint sequence per sample, then SDPA in [B, H, S, D] layout.
        q = torch.cat([img_q, txt_q], dim=1).transpose(1, 2)
        k = torch.cat([img_k, txt_k], dim=1).transpose(1, 2)
        v = torch.cat([img_v, txt_v], dim=1).transpose(1, 2)

        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, 1, seq_img + seq_txt)
            if attention_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"attention_mask must have shape {expected_mask_shape}, "
                    f"got {tuple(attention_mask.shape)}."
                )
            attention_mask = attention_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(bsz, seq_img + seq_txt, -1)

        img_out = self.to_out[1](self.to_out[0](out[:, :seq_img, :]))
        txt_out = self.to_add_out(out[:, seq_img:, :])
        return img_out, txt_out


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class LensTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        rms_norm: bool = False,
        gate_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.attn = LensJointAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            eps=eps,
        )

        norm_cls = (lambda d: RMSNorm(d, eps=eps)) if rms_norm else (
            lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=eps)
        )
        if gate_mlp:
            mlp_cls = lambda: GateMLP(dim, int(dim / 3 * 8)) # pylint: disable=unnecessary-lambda-assignment
        else:
            mlp_cls = lambda: FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate") # pylint: disable=unnecessary-lambda-assignment

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = norm_cls(dim)
        self.img_norm2 = norm_cls(dim)
        self.img_mlp = mlp_cls()

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = norm_cls(dim)
        self.txt_norm2 = norm_cls(dim)
        self.txt_mlp = mlp_cls()

    @staticmethod
    def _modulate(x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class LensTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    """The Lens text-to-image DiT.

    Supports a single conditioning stream of multi-layer text features. The
    text features are normalized per layer, concatenated along the channel
    axis, and projected to `inner_dim` before joining the image stream.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["LensTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["LensTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 128,
        out_channels: Optional[int] = 32,
        num_layers: int = 48,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        inner_dim: int = 1536, # pylint: disable=unused-argument
        enc_hidden_dim: int = 2880,
        axes_dims_rope: Tuple[int, int, int] = (8, 28, 28),
        gate_mlp: bool = True,
        rms_norm: bool = True,
        multi_layer_encoder_feature: bool = True,
        selected_layer_index: Tuple[int, ...] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.multi_layer_encoder_feature = multi_layer_encoder_feature
        self.selected_layer_index = list(selected_layer_index)

        self.pos_embed = LensEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = LensTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        if self.multi_layer_encoder_feature:
            self.txt_norm = nn.ModuleList(
                [RMSNorm(enc_hidden_dim, eps=1e-5) for _ in self.selected_layer_index]
            )
            self.txt_in = nn.Linear(enc_hidden_dim * len(self.selected_layer_index), self.inner_dim)
        else:
            self.txt_norm = RMSNorm(enc_hidden_dim, eps=1e-5)
            self.txt_in = nn.Linear(enc_hidden_dim, self.inner_dim)

        self.img_in = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                LensTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rms_norm=rms_norm,
                    gate_mlp=gate_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: List[Tuple[int, int, int]],
        attention_kwargs: Optional[Dict[str, Any]] = None, # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states:                 [B, S_img, in_channels] image latents.
            encoder_hidden_states:         either a Tensor [B, S_txt, enc_dim]
                                           (single-layer) or a list of such
                                           tensors (multi-layer).
            encoder_hidden_states_mask:    bool [B, S_txt] (True = valid).
            timestep:                      [B] in [0, 1].
            img_shapes:                    list with a single (frame, h_lat, w_lat).
        """
        bsz, img_len, _ = hidden_states.shape
        if self.multi_layer_encoder_feature:
            if not isinstance(encoder_hidden_states, (list, tuple)):
                raise ValueError(
                    "multi_layer_encoder_feature=True expects a list of "
                    "per-layer text tensors."
                )
            if len(encoder_hidden_states) != len(self.selected_layer_index):
                raise ValueError(
                    f"Expected {len(self.selected_layer_index)} text feature "
                    f"layers, got {len(encoder_hidden_states)}."
                )
            text_seq_len = encoder_hidden_states[0].shape[1]
            for i, feat in enumerate(encoder_hidden_states):
                if feat.shape[0] != bsz:
                    raise ValueError(
                        f"Text feature layer {i} batch size {feat.shape[0]} "
                        f"does not match hidden_states batch size {bsz}."
                    )
                if feat.shape[1] != text_seq_len:
                    raise ValueError(
                        f"Text feature layer {i} sequence length {feat.shape[1]} "
                        f"does not match layer 0 length {text_seq_len}."
                    )
        else:
            if not isinstance(encoder_hidden_states, torch.Tensor):
                raise ValueError(
                    "multi_layer_encoder_feature=False expects a single text "
                    "feature tensor."
                )
            if encoder_hidden_states.shape[0] != bsz:
                raise ValueError(
                    f"Text feature batch size {encoder_hidden_states.shape[0]} "
                    f"does not match hidden_states batch size {bsz}."
                )
            text_seq_len = encoder_hidden_states.shape[1]
        if encoder_hidden_states_mask.shape != (bsz, text_seq_len):
            raise ValueError(
                "encoder_hidden_states_mask must have shape "
                f"{(bsz, text_seq_len)}, got {tuple(encoder_hidden_states_mask.shape)}."
            )
        attention_mask = self._build_joint_attention_mask(
            encoder_hidden_states_mask, img_len
        )

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.multi_layer_encoder_feature:
            normed = [
                self.txt_norm[i](encoder_hidden_states[i])
                for i in range(len(self.selected_layer_index))
            ]
            encoder_hidden_states = torch.cat(normed, dim=-1)
        else:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(
            img_shapes, [text_seq_len], device=hidden_states.device
        )

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)

    @staticmethod
    def _build_joint_attention_mask(
        text_mask: torch.Tensor, img_len: int
    ) -> torch.Tensor:
        """Additive joint mask of shape ``[B, 1, 1, img_len + S_txt]``.

        Image tokens are always valid; text positions follow ``text_mask``.
        Padded positions hold ``-inf`` so SDPA's softmax masks them out.
        """
        if text_mask.dtype != torch.bool:
            text_mask = text_mask.bool()
        bsz = text_mask.shape[0]
        img_ones = torch.ones(
            (bsz, img_len), dtype=torch.bool, device=text_mask.device
        )
        joint = torch.cat([img_ones, text_mask], dim=1)
        additive = torch.zeros_like(joint, dtype=torch.float32)
        additive.masked_fill_(~joint, float("-inf"))
        return additive[:, None, None, :]
