# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from ....common.half_precision_fixes import safe_pad_operation
from ....common.distributed.ops import (
    gather_heads,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    scatter_heads,
)

from ..attention import TorchAttention
from ..mlp import get_mlp
from ..mm import MMArg, MMModule
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import RotaryEmbedding3d


class MMWindowAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3

        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.head_dim = head_dim
        self.proj_qkv = MMModule(nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_qkv)
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_qkv)
        self.norm_q = MMModule(qk_norm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.norm_k = MMModule(qk_norm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.rope = RotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = TorchAttention()

    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
        txt: torch.FloatTensor,  # b L c
        txt_mask: torch.BoolTensor,  # b L
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # Project q, k, v.
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        _, L, _ = txt.shape

        if self.window_method == "win":
            nt, nh, nw = self.window
            tt, hh, ww = T // nt, H // nh, W // nw
        elif self.window_method == "win_by_size":
            tt, hh, ww = self.window
            tt, hh, ww = (
                tt if tt > 0 else T,
                hh if hh > 0 else H,
                ww if ww > 0 else W,
            )
            nt, nh, nw = T // tt, H // hh, W // ww
        else:
            raise NotImplementedError

        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "b L (o h d) -> o b h L d", o=3, d=self.head_dim)
        txt_qkv = scatter_heads(txt_qkv, dim=2)

        vid_q, vid_k, vid_v = vid_qkv.unbind()
        txt_q, txt_k, txt_v = txt_qkv.unbind()

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        def vid_window(v):
            return rearrange(
                v,
                "b h (nt tt nh hh nw ww) d -> b h (nt nh nw) (tt hh ww) d",
                hh=hh,
                ww=ww,
                tt=tt,
                nh=nh,
                nw=nw,
                nt=nt,
            )

        def txt_window(t):
            return rearrange(t, "b h L d -> b h 1 L d").expand(-1, -1, nt * nh * nw, -1, -1)

        # Process video attention.
        vid_msk = safe_pad_operation(txt_mask, (tt * hh * ww, 0), value=True)
        vid_msk = rearrange(vid_msk, "b l -> b 1 1 1 l").expand(-1, 1, 1, tt * hh * ww, -1)
        vid_out = self.attn(
            vid_window(vid_q),
            torch.cat([vid_window(vid_k), txt_window(txt_k)], dim=-2),
            torch.cat([vid_window(vid_v), txt_window(txt_v)], dim=-2),
            vid_msk,
        )
        vid_out = rearrange(
            vid_out,
            "b h (nt nh nw) (tt hh ww) d -> b (nt tt) (nh hh) (nw ww) (h d)",
            hh=hh,
            ww=ww,
            tt=tt,
            nh=nh,
            nw=nw,
        )
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)

        # Process text attention.
        txt_msk = safe_pad_operation(txt_mask, (T * H * W, 0), value=True)
        txt_msk = rearrange(txt_msk, "b l -> b 1 1 l").expand(-1, 1, L, -1)
        txt_out = self.attn(
            txt_q,
            torch.cat([vid_k, txt_k], dim=-2),
            torch.cat([vid_v, txt_v], dim=-2),
            txt_msk,
        )
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)

        # Project output.
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class MMWindowTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False)
        self.attn = MMWindowAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
        )
        self.mlp_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_mlp,
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"])

    def forward(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
        txt_mask: torch.BoolTensor,
        emb: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, emb=emb, layer="attn", mode="in")
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, txt_mask=txt_mask)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, emb=emb, layer="attn", mode="out")
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, emb=emb, layer="mlp", mode="in")
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, emb=emb, layer="mlp", mode="out")
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp
