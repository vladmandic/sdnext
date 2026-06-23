"""Krea 2 (K2) single-stream DiT, ported to the diffusers ModelMixin contract.

The module tree mirrors the original `SingleStreamDiT` checkpoint exactly (``first``,
``blocks.N.attn.{wq,wk,wv,gate,wo}``, ``txtfusion.*``, ``last`` ...), so the safetensors
load is an identity map (no key conversion). Architecture specifics: gated attention,
grouped-query attention, per-head QK RMSNorm, 3-axis rotary embedding, a shared+per-block
modulation, and a text-fusion stage that collapses several text-encoder hidden-state layers
into one conditioning stream.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin


def rope(pos: torch.Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> torch.Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def rope_apply(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def time_embed(t: torch.Tensor, dim: int, period: float = 1e4, tfactor: float = 1e3, device=None, dtype=None) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    # t: (B,) -> (B, 1, half) so the embedding broadcasts as a per-sample vector.
    args = (t.float() * tfactor)[:, None, None] * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1).to(dtype=dtype)


def expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat each KV head ``n_rep`` times so grouped-query attention runs on any SDPA backend."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)  # x: (B, kvheads, L, D) -> (B, heads, L, D)


def segment_mask(mask: torch.Tensor) -> torch.Tensor:
    """Expand a (B, L) key-padding mask into a (B, 1, L, L) attention mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


class RMSNorm(nn.Module):
    """RMSNorm whose stored weight is a zero-init delta applied as ``scale + 1`` and computed in fp32."""

    def __init__(self, features: int, eps: float = 1e-05):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        t = F.rms_norm(x.float(), (self.features,), eps=self.eps, weight=self.scale.float() + 1.0)
        return t.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.qnorm(q), self.knorm(k), v


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: torch.Tensor):
        out = vec + self.lin
        return out.chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # split of the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return torch.cat([rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)], dim=-3)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Gated grouped-query attention with per-head QK RMSNorm and optional 3-axis RoPE."""

    def __init__(self, dim: int, heads: int, kvheads: int | None = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.n_rep = self.heads // self.kvheads

        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = rope_apply(q, k, freqs)
        k, v = expand_kv(k, self.n_rep), expand_kv(v, self.n_rep)

        # dispatch_attention_fn expects (B, L, H, D); the q/k/v math above stays in (B, H, L, D).
        out = dispatch_attention_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        # A fully-masked query row (padding token) yields NaN on the CUDA SDPA backends; zero it so
        # it cannot propagate through the next layer's `0 * NaN`. cuDNN returns 0 here already.
        out = torch.nan_to_num(out)
        out = rearrange(out, "B L H D -> B L (H D)")
        return self.wo(out * F.sigmoid(gate))


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)
        self.down = nn.Linear(features, features, bias=False)
        self.up = nn.Linear(features, features, bias=False)

    def forward(self, x: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift + self.up(self.down(x))
        return self.linear(x)


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    """Fuse ``num_txt_layers`` stacked text-encoder hidden states into a single conditioning stream.

    ``num_txt_layers`` is the count of selected encoder layers fed in (projected down to 1),
    not the transformer depth, which is fixed at 2 layerwise + 2 refiner blocks.
    """

    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class Krea2Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""Single-stream flow-matching DiT backbone for Krea 2.

    The transformer consumes patchified noisy image tokens plus stacked text-encoder hidden
    states, fuses the text layers internally, concatenates text and image tokens into one
    stream, and predicts the flow-matching velocity for the image-token positions.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SingleStreamBlock", "TextFusionBlock"]
    _repeated_blocks = ["SingleStreamBlock"]

    @register_to_config
    def __init__(
        self,
        features: int = 6144,
        tdim: int = 256,
        txtdim: int = 2560,
        heads: int = 48,
        kvheads: int = 12,
        multiplier: int = 4,
        layers: int = 28,
        patch: int = 2,
        channels: int = 16,
        bias: bool = False,
        theta: float = 1e3,
        txtlayers: int = 12,
        txtheads: int = 20,
        txtkvheads: int = 20,
        is_distilled: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.tdim = tdim
        self.is_distilled = is_distilled

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"sum(axes)={sum(axes)} != headdim={headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes={axes}"

        self.posemb = PositionalEncoding(axes, theta=theta, ntk=1.0)
        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(features, features * 6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        r"""
        Args:
            hidden_states: `(B, L_img, channels * patch ** 2)` patchified noisy image tokens.
            encoder_hidden_states: `(B, L_txt, txtlayers, txtdim)` stacked text-encoder hidden states.
            timestep: `(B,)` flow-matching time in `[0, 1]`.
            position_ids: `(B, L_txt + L_img, 3)` `(t, h, w)` coordinates for the 3-axis RoPE.
            attention_mask: `(B, L_txt + L_img)` boolean key-padding mask (True = valid token).
        """
        img = self.first(hidden_states)
        t = self.tmlp(time_embed(timestep, self.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtmask = segment_mask(attention_mask[:, : encoder_hidden_states.shape[1]])
        context = self.txtfusion(encoder_hidden_states, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Pad the joint sequence to a multiple of 256 to keep compiled attention kernel shapes stable.
        padlen = (-combined.shape[1]) % 256
        if padlen > 0:
            combined = F.pad(combined, (0, 0, 0, padlen))
            attention_mask = F.pad(attention_mask, (0, padlen), value=False)
            position_ids = F.pad(position_ids, (0, 0, 0, padlen))

        mask = segment_mask(attention_mask)
        freqs = self.posemb(position_ids)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                combined = self._gradient_checkpointing_func(block, combined, tvec, freqs, mask)
            else:
                combined = block(combined, tvec, freqs, mask)

        output = self.last(combined, t)[:, txtlen : txtlen + imglen, :]
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
