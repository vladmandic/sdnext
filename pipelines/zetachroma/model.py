import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file


DEFAULT_TEXT_ENCODER_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_CHECKPOINT_FILENAME = "zeta-chroma-base-x0-pixel-dino-distance.safetensors"


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t.dtype) if torch.is_floating_point(t) else embedding


def rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    rot = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    return rot.reshape(*out.shape, 2, 2).float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: float, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)


def _apply_rope_single(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return _apply_rope_single(xq, freqs_cis), _apply_rope_single(xk, freqs_cis)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, output_size: int | None = None):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        return self.mlp(t_freq)


class JointAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int | None = None, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.qkv = nn.Linear(dim, (n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x)
        xq, xk, xv = torch.split(
            qkv,
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        xq = self.q_norm(xq.view(batch_size, sequence_length, self.n_heads, self.head_dim))
        xk = self.k_norm(xk.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim))
        xv = xv.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rope(xq, xk, freqs_cis)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.n_rep > 1:
            xk = xk.unsqueeze(2).repeat(1, 1, self.n_rep, 1, 1).flatten(1, 2)
            xv = xv.unsqueeze(2).repeat(1, 1, self.n_rep, 1, 1).flatten(1, 2)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        ffn_hidden_dim: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        z_image_modulation: bool = False,
    ):
        super().__init__()
        self.modulation = modulation
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(dim, ffn_hidden_dim)
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        if modulation:
            if z_image_modulation:
                self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, 256), 4 * dim, bias=True))
            else:
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(dim, 1024), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(self.attention_norm1(x) * (1 + scale_msa.unsqueeze(1)), mask, freqs_cis)
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * (1 + scale_mlp.unsqueeze(1)))
            )
            return x

        x = x + self.attention_norm2(self.attention(self.attention_norm1(x), mask, freqs_cis))
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input))
        self._pos_cache: dict[tuple[int, str, str], torch.Tensor] = {}

    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (patch_size, str(device), str(dtype))
        if cache_key in self._pos_cache:
            return self._pos_cache[cache_key]
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs**2)
        self._pos_cache[cache_key] = dct
        return dct

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, patch_area, channels = inputs.shape
        _ = channels
        patch_size = int(patch_area**0.5)
        dct = self.fetch_pos(patch_size, inputs.device, inputs.dtype).repeat(batch_size, 1, 1)
        return self.embedder(torch.cat((inputs, dct), dim=-1))


class PixelResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        hidden = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(hidden)


class DCTFinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        max_freqs: int = 8,
    ):
        super().__init__()
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_embedder = NerfEmbedder(in_channels, model_channels, max_freqs)
        self.res_blocks = nn.ModuleList([PixelResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = DCTFinalLayer(model_channels, out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_embedder(x)
        y = self.cond_embed(c).unsqueeze(1)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x)


def pad_to_patch_size(x: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    _, _, height, width = x.shape
    patch_h, patch_w = patch_size
    pad_h = (patch_h - height % patch_h) % patch_h
    pad_w = (patch_w - width % patch_w) % patch_w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x


def pad_zimage(feats: torch.Tensor, pad_token: torch.Tensor, pad_tokens_multiple: int) -> tuple[torch.Tensor, int]:
    pad_extra = (-feats.shape[1]) % pad_tokens_multiple
    if pad_extra > 0:
        feats = torch.cat(
            [
                feats,
                pad_token.to(device=feats.device, dtype=feats.dtype).unsqueeze(0).repeat(feats.shape[0], pad_extra, 1),
            ],
            dim=1,
        )
    return feats, pad_extra


class NextDiTPixelSpace(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):  # type: ignore[misc]
    @register_to_config
    def __init__(
        self,
        patch_size: int = 32,
        in_channels: int = 3,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int | None = None,
        ffn_hidden_dim: int = 10240,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        axes_dims: list[int] | None = None,
        axes_lens: list[int] | None = None,
        rope_theta: float = 256.0,
        time_scale: float = 1000.0,
        pad_tokens_multiple: int | None = 128,
        decoder_hidden_size: int = 3840,
        decoder_num_res_blocks: int = 4,
        decoder_max_freqs: int = 8,
    ):
        super().__init__()
        axes_dims = axes_dims or [32, 48, 48]
        axes_lens = axes_lens or [1536, 512, 512]
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim,
                    n_heads,
                    n_kv_heads,
                    ffn_hidden_dim,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=True,
                )
                for _ in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim,
                    n_heads,
                    n_kv_heads,
                    ffn_hidden_dim,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for _ in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(
            hidden_size=min(dim, 1024),
            frequency_embedding_size=256,
            output_size=256,
        )
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim,
                    n_heads,
                    n_kv_heads,
                    ffn_hidden_dim,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=True,
                )
                for _ in range(n_layers)
            ]
        )

        if pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        assert dim // n_heads == sum(axes_dims)
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=list(axes_dims))
        dec_in_ch = patch_size**2 * in_channels
        self.dec_net = SimpleMLPAdaLN(
            in_channels=dec_in_ch,
            model_channels=decoder_hidden_size,
            out_channels=dec_in_ch,
            z_channels=dim,
            num_res_blocks=decoder_num_res_blocks,
            max_freqs=decoder_max_freqs,
        )
        self.register_buffer("__x0__", torch.tensor([]))

    def embed_cap(self, cap_feats: torch.Tensor, offset: float = 0, bsz: int = 1, device=None, dtype=None):
        _ = dtype
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats_len = cap_feats.shape[1]
        if self.pad_tokens_multiple is not None:
            cap_feats, _ = pad_zimage(cap_feats, self.cap_pad_token, self.pad_tokens_multiple)

        cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0 + offset
        freqs_cis = self.rope_embedder(cap_pos_ids).movedim(1, 2)
        return cap_feats, freqs_cis, cap_feats_len

    def pos_ids_x(self, start_t: float, h_tokens: int, w_tokens: int, batch_size: int, device):
        x_pos_ids = torch.zeros((batch_size, h_tokens * w_tokens, 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = start_t
        x_pos_ids[:, :, 1] = (
            torch.arange(h_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, w_tokens).flatten()
        )
        x_pos_ids[:, :, 2] = (
            torch.arange(w_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(h_tokens, 1).flatten()
        )
        return x_pos_ids

    def unpatchify(self, x: torch.Tensor, img_size: list[tuple[int, int]], cap_size: list[int]) -> torch.Tensor:
        patch_h = patch_w = self.patch_size
        imgs = []
        for index in range(x.size(0)):
            height, width = img_size[index]
            begin = cap_size[index]
            end = begin + (height // patch_h) * (width // patch_w)
            imgs.append(
                x[index][begin:end]
                .view(height // patch_h, width // patch_w, patch_h, patch_w, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        return torch.stack(imgs, dim=0)

    def _forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, num_tokens: int | None) -> torch.Tensor:
        _ = num_tokens
        timesteps = 1.0 - timesteps
        _, _, original_h, original_w = x.shape
        x = pad_to_patch_size(x, (self.patch_size, self.patch_size))
        batch_size, channels, height, width = x.shape
        t_emb = self.t_embedder(timesteps * self.time_scale, dtype=x.dtype)
        patch_h = patch_w = self.patch_size

        pixel_patches = (
            x.view(batch_size, channels, height // patch_h, patch_h, width // patch_w, patch_w)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )
        num_image_tokens = pixel_patches.shape[1]
        pixel_values = pixel_patches.reshape(batch_size * num_image_tokens, 1, patch_h * patch_w * channels)

        cap_feats_emb, cap_freqs_cis, _ = self.embed_cap(context, offset=0, bsz=batch_size, device=x.device, dtype=x.dtype)

        x_tokens = self.x_embedder(
            x.view(batch_size, channels, height // patch_h, patch_h, width // patch_w, patch_w)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )
        cap_feats_len_total = cap_feats_emb.shape[1]
        x_pos_ids = self.pos_ids_x(cap_feats_len_total + 1, height // patch_h, width // patch_w, batch_size, x.device)
        if self.pad_tokens_multiple is not None:
            x_tokens, x_pad_extra = pad_zimage(x_tokens, self.x_pad_token, self.pad_tokens_multiple)
            x_pos_ids = F.pad(x_pos_ids, (0, 0, 0, x_pad_extra))
        x_freqs_cis = self.rope_embedder(x_pos_ids).movedim(1, 2)

        for layer in self.context_refiner:
            cap_feats_emb = layer(cap_feats_emb, None, cap_freqs_cis)

        for layer in self.noise_refiner:
            x_tokens = layer(x_tokens, None, x_freqs_cis, t_emb)

        padded_full_embed = torch.cat([cap_feats_emb, x_tokens], dim=1)
        full_freqs_cis = torch.cat([cap_freqs_cis, x_freqs_cis], dim=1)
        img_len = x_tokens.shape[1]
        cap_size = [padded_full_embed.shape[1] - img_len] * batch_size

        hidden_states = padded_full_embed
        for layer in self.layers:
            hidden_states = layer(hidden_states, None, full_freqs_cis.to(hidden_states.device), t_emb)

        img_hidden = hidden_states[:, cap_size[0] : cap_size[0] + num_image_tokens, :]
        decoder_cond = img_hidden.reshape(batch_size * num_image_tokens, self.dim)
        output = self.dec_net(pixel_values, decoder_cond).reshape(batch_size, num_image_tokens, -1)
        cap_placeholder = torch.zeros(
            batch_size,
            cap_size[0],
            output.shape[-1],
            device=output.device,
            dtype=output.dtype,
        )
        img_size = [(height, width)] * batch_size
        img_out = self.unpatchify(torch.cat([cap_placeholder, output], dim=1), img_size, cap_size)
        return -img_out[:, :, :original_h, :original_w]

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, num_tokens: int | None = None):
        x0_pred = self._forward(x, timesteps, context, num_tokens)
        return (x - x0_pred) / timesteps.view(-1, 1, 1, 1)


def remap_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_sd: dict[str, torch.Tensor] = {}
    qkv_parts: dict[str, dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        if key == "__x0__":
            new_sd[key] = value
            continue
        if ".attention.to_q." in key:
            prefix = key.split(".attention.to_q.")[0]
            suffix = key.split(".attention.to_q.")[1]
            qkv_parts.setdefault(f"{prefix}.attention.{suffix}", {})["q"] = value
            continue
        if ".attention.to_k." in key:
            prefix = key.split(".attention.to_k.")[0]
            suffix = key.split(".attention.to_k.")[1]
            qkv_parts.setdefault(f"{prefix}.attention.{suffix}", {})["k"] = value
            continue
        if ".attention.to_v." in key:
            prefix = key.split(".attention.to_v.")[0]
            suffix = key.split(".attention.to_v.")[1]
            qkv_parts.setdefault(f"{prefix}.attention.{suffix}", {})["v"] = value
            continue
        if ".attention.to_out.0." in key:
            new_sd[key.replace(".attention.to_out.0.", ".attention.out.")] = value
            continue
        if ".attention.norm_q." in key:
            new_sd[key.replace(".attention.norm_q.", ".attention.q_norm.")] = value
            continue
        if ".attention.norm_k." in key:
            new_sd[key.replace(".attention.norm_k.", ".attention.k_norm.")] = value
            continue
        new_sd[key] = value

    for combined_key, parts in qkv_parts.items():
        if not {"q", "k", "v"}.issubset(parts):
            raise ValueError(f"Incomplete QKV for {combined_key}: got {list(parts.keys())}")
        prefix = combined_key.rsplit(".", 1)[0]
        suffix = combined_key.rsplit(".", 1)[1]
        new_sd[f"{prefix}.qkv.{suffix}"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)

    return new_sd


def default_axes_dims_for_head_dim(head_dim: int) -> list[int]:
    if head_dim % 8 == 0:
        first = head_dim // 4
        second = (head_dim * 3) // 8
        third = head_dim - first - second
        if first % 2 == 0 and second % 2 == 0 and third % 2 == 0:
            return [first, second, third]
    if head_dim == 128:
        return [32, 48, 48]
    if head_dim % 6 == 0:
        split = head_dim // 3
        return [split, split, split]
    raise ValueError(f"Unable to infer axes_dims for head_dim={head_dim}; provide axes_dims explicitly")


def _count_module_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    indices = set()
    needle = f"{prefix}."
    prefix_parts = prefix.count(".") + 1
    for key in state_dict:
        if not key.startswith(needle):
            continue
        parts = key.split(".")
        if len(parts) <= prefix_parts:
            continue
        try:
            indices.add(int(parts[prefix_parts]))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def infer_model_config(
    state_dict: dict[str, torch.Tensor],
    axes_dims: list[int] | None = None,
    axes_lens: list[int] | None = None,
    rope_theta: float = 256.0,
    time_scale: float = 1000.0,
    pad_tokens_multiple: int | None = 128,
    norm_eps: float = 1e-5,
) -> dict:
    dim = state_dict["x_embedder.weight"].shape[0]
    patch_channels = state_dict["x_embedder.weight"].shape[1]
    in_channels = 3 if patch_channels % 3 == 0 else 1
    patch_size = int(round((patch_channels / in_channels) ** 0.5))
    head_dim = state_dict["layers.0.attention.q_norm.weight"].shape[0]
    n_heads = dim // head_dim
    total_proj_heads = state_dict["layers.0.attention.qkv.weight"].shape[0] // head_dim
    n_kv_heads = (total_proj_heads - n_heads) // 2
    if n_kv_heads == n_heads:
        n_kv_heads = None
    qk_norm = "layers.0.attention.q_norm.weight" in state_dict
    decoder_input = state_dict["dec_net.input_embedder.embedder.0.weight"].shape[1]
    decoder_max_freqs = int(round((decoder_input - patch_channels) ** 0.5))
    config = {
        "patch_size": patch_size,
        "in_channels": in_channels,
        "dim": dim,
        "n_layers": _count_module_blocks(state_dict, "layers"),
        "n_refiner_layers": _count_module_blocks(state_dict, "noise_refiner"),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "ffn_hidden_dim": state_dict["layers.0.feed_forward.w1.weight"].shape[0],
        "norm_eps": norm_eps,
        "qk_norm": qk_norm,
        "cap_feat_dim": state_dict["cap_embedder.1.weight"].shape[1],
        "axes_dims": axes_dims or default_axes_dims_for_head_dim(head_dim),
        "axes_lens": axes_lens or [1536, 512, 512],
        "rope_theta": rope_theta,
        "time_scale": time_scale,
        "pad_tokens_multiple": pad_tokens_multiple,
        "decoder_hidden_size": state_dict["dec_net.cond_embed.weight"].shape[0],
        "decoder_num_res_blocks": _count_module_blocks(state_dict, "dec_net.res_blocks"),
        "decoder_max_freqs": decoder_max_freqs,
    }
    return config


def resolve_checkpoint_path(
    pretrained_model_name_or_path: str,
    checkpoint_filename: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
) -> str:
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    if os.path.isdir(pretrained_model_name_or_path):
        if checkpoint_filename is not None:
            candidate = os.path.join(pretrained_model_name_or_path, checkpoint_filename)
            if os.path.isfile(candidate):
                return candidate
        candidates = [
            os.path.join(pretrained_model_name_or_path, name)
            for name in os.listdir(pretrained_model_name_or_path)
            if name.lower().endswith(".safetensors")
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No .safetensors checkpoint found in {pretrained_model_name_or_path}")
        raise FileNotFoundError(
            f"Multiple .safetensors checkpoints found in {pretrained_model_name_or_path}; provide checkpoint_filename"
        )

    filename = checkpoint_filename
    if filename is None:
        repo_files = [name for name in list_repo_files(pretrained_model_name_or_path) if name.lower().endswith(".safetensors")]
        if not repo_files and DEFAULT_CHECKPOINT_FILENAME:
            repo_files = [DEFAULT_CHECKPOINT_FILENAME]
        if len(repo_files) == 1:
            filename = repo_files[0]
        elif DEFAULT_CHECKPOINT_FILENAME in repo_files:
            filename = DEFAULT_CHECKPOINT_FILENAME
        else:
            raise FileNotFoundError(
                f"Unable to determine checkpoint filename for {pretrained_model_name_or_path}; provide checkpoint_filename"
            )

    return hf_hub_download(
        repo_id=pretrained_model_name_or_path,
        filename=filename,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_zetachroma_transformer(
    pretrained_model_name_or_path: str,
    checkpoint_filename: str | None = None,
    torch_dtype: torch.dtype | None = None,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
    model_config: dict | None = None,
    axes_dims: list[int] | None = None,
    axes_lens: list[int] | None = None,
    rope_theta: float = 256.0,
    time_scale: float = 1000.0,
    pad_tokens_multiple: int | None = 128,
) -> NextDiTPixelSpace:
    checkpoint_path = resolve_checkpoint_path(
        pretrained_model_name_or_path,
        checkpoint_filename=checkpoint_filename,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    raw_state_dict = load_file(checkpoint_path, device="cpu")
    state_dict = remap_checkpoint_keys(raw_state_dict)
    del raw_state_dict
    config = infer_model_config(
        state_dict,
        axes_dims=axes_dims,
        axes_lens=axes_lens,
        rope_theta=rope_theta,
        time_scale=time_scale,
        pad_tokens_multiple=pad_tokens_multiple,
    )
    if model_config is not None:
        config.update(model_config)

    model = NextDiTPixelSpace(**config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = {"__x0__"}
    relevant_missing = [key for key in missing if key not in ignored_missing]
    if relevant_missing or unexpected:
        raise ValueError(
            "Zeta-Chroma checkpoint load mismatch: "
            f"missing={relevant_missing[:20]} unexpected={unexpected[:20]}"
        )
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    return model
