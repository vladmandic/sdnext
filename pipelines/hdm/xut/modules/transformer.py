import torch.nn as nn

from .layers import SwiGLU
from .attention import SelfAttention, CrossAttention
from .norm import RMSNorm
from .adaln import AdaLN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ctx_dim,
        heads,
        dim_head,
        mlp_dim,
        pos_dim,
        use_adaln=False,
        use_shared_adaln=False,
        ctx_from_self=False,
        norm_layer=RMSNorm,
    ):
        super().__init__()
        self.use_adaln = use_adaln
        self.attn = SelfAttention(dim, heads, dim_head, pos_dim)
        if ctx_dim is None:
            self.xattn_pre_norm = None
            self.xattn = None
        else:
            self.ctx_from_self = ctx_from_self
            self.xattn = CrossAttention(dim, ctx_dim, heads, dim_head, pos_dim)
        self.mlp = SwiGLU(dim, mlp_dim, dim)

        if self.use_adaln:
            self.attn_pre_norm = AdaLN(
                dim, dim, norm_layer=norm_layer, shared=use_shared_adaln
            )
            self.mlp_pre_norm = AdaLN(
                dim, dim, norm_layer=norm_layer, shared=use_shared_adaln
            )
            if self.xattn is not None:
                self.xattn_pre_norm = AdaLN(
                    dim, dim, norm_layer=norm_layer, shared=use_shared_adaln
                )
        else:
            self.attn_pre_norm = norm_layer(dim)
            self.mlp_pre_norm = norm_layer(dim)
            if self.xattn is not None:
                self.xattn_pre_norm = norm_layer(dim)

    def forward(
        self,
        x,
        ctx,
        pos_map=None,
        ctx_pos_map=None,
        y=None,
        x_mask=None,
        ctx_mask=None,
        shared_adaln=None,
    ):
        y = [y] if y is not None else []
        y = y if shared_adaln is None else [y[0], shared_adaln[0]]
        x, gate = self.attn_pre_norm(x, *y)
        x = x + self.attn(x, pos_map, mask=x_mask) * gate

        if self.xattn is not None:
            if shared_adaln is not None:
                y[1] = shared_adaln[1]
            x, gate = self.xattn_pre_norm(x, *y)
            if self.ctx_from_self:
                ctx_mask = x_mask
            x = x + self.xattn(x, ctx, pos_map, ctx_pos_map, mask=ctx_mask) * gate

        if shared_adaln is not None:
            y[1] = shared_adaln[-1]
        x, gate = self.mlp_pre_norm(x, *y)
        x = x + self.mlp(x) * gate
        return x
