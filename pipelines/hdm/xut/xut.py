import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modules.norm import RMSNorm
from .modules.transformer import TransformerBlock
from .modules.patch import PatchEmbed, UnPatch
from .modules.axial_rope import make_axial_pos
from .modules.time_emb import TimestepEmbedding
from .modules.norm import RMSNorm, DyT
from .utils import isiterable


class TBackBone(nn.Module):
    """
    Basic backbone of transformer
    """

    def __init__(
        self,
        dim=1024,
        ctx_dim=1024,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        pos_dim=2,
        depth=8,
        use_adaln=False,
        use_shared_adaln=False,
        use_dyt=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    ctx_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    pos_dim,
                    use_adaln,
                    use_shared_adaln,
                    norm_layer=DyT if use_dyt else RMSNorm,
                )
                for _ in range(depth)
            ]
        )
        self.grad_ckpt = False

    def init_weight(self):
        for param in self.parameters():
            if param.ndim == 1:
                nn.init.normal_(param, mean=0.0, std=(1 / param.size(0)) ** 0.5)
            elif param.ndim == 2:
                fan_in = param.size(1)
                nn.init.normal_(param, mean=0.0, std=(1 / fan_in) ** 0.5)
            elif param.ndim >= 3:
                fan_out, *fan_ins = param.shape
                # cumprod
                fan_in = 1
                for f in fan_ins:
                    fan_in *= f
                nn.init.normal_(param, mean=0.0, std=(1 / fan_in) ** 0.5)

    def forward(
        self,
        x,
        ctx=None,
        x_mask=None,
        ctx_mask=None,
        pos_map=None,
        y=None,
        shared_adaln=None,
    ):
        if pos_map is not None:
            assert pos_map.size(1) == x.size(1)

        for block in self.blocks:
            if self.grad_ckpt:
                x = checkpoint(
                    block,
                    x,
                    ctx,
                    pos_map,
                    None,
                    y,
                    x_mask,
                    ctx_mask,
                    shared_adaln,
                    use_reentrant=False,
                )
            else:
                x = block(x, ctx, pos_map, None, y, x_mask, ctx_mask, shared_adaln)

        return x


class XUTBackBone(nn.Module):
    """
    Basic backbone of cross-U-transformer.
    """

    def __init__(
        self,
        dim=1024,
        ctx_dim=None,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        pos_dim=2,
        depth=8,
        enc_blocks=1,
        dec_blocks=2,
        dec_ctx=False,
        use_adaln=False,
        use_shared_adaln=False,
        use_dyt=False,
    ):
        super().__init__()
        if isiterable(enc_blocks):
            enc_blocks = list(enc_blocks)
            assert len(enc_blocks) == depth
        else:
            enc_blocks = [int(enc_blocks)] * depth
        if isiterable(dec_blocks):
            dec_blocks = list(dec_blocks)
            assert len(dec_blocks) == depth
        else:
            dec_blocks = [int(dec_blocks)] * depth

        self.enc_blocks = nn.ModuleList()
        for i in range(depth):
            blocks = [
                TransformerBlock(
                    dim,
                    ctx_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    pos_dim,
                    use_adaln,
                    use_shared_adaln,
                    norm_layer=DyT if use_dyt else RMSNorm,
                )
                for _ in range(enc_blocks[i])
            ]
            self.enc_blocks.append(nn.ModuleList(blocks))

        self.dec_ctx = dec_ctx
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            blocks = [
                TransformerBlock(
                    dim,
                    dim if bid == 0 else ctx_dim if dec_ctx else None,
                    heads,
                    dim_head,
                    mlp_dim,
                    pos_dim,
                    use_adaln,
                    use_shared_adaln,
                    ctx_from_self=bid == 0,
                    norm_layer=DyT if use_dyt else RMSNorm,
                )
                for bid in range(dec_blocks[i])
            ]
            self.dec_blocks.append(nn.ModuleList(blocks))

        self.grad_ckpt = False

    def init_weight(self):
        for param in self.parameters():
            if param.ndim == 1:
                nn.init.normal_(param, mean=0.0, std=(1 / param.size(0)) ** 0.5)
            elif param.ndim == 2:
                fan_in = param.size(1)
                nn.init.normal_(param, mean=0.0, std=(1 / fan_in) ** 0.5)
            elif param.ndim >= 3:
                fan_out, *fan_ins = param.shape
                # cumprod
                fan_in = 1
                for f in fan_ins:
                    fan_in *= f
                nn.init.normal_(param, mean=0.0, std=(1 / fan_in) ** 0.5)

    def forward(
        self,
        x,
        ctx=None,
        x_mask=None,
        ctx_mask=None,
        pos_map=None,
        y=None,
        shared_adaln=None,
        return_enc_out=False,
    ):
        if pos_map is not None:
            assert pos_map.size(1) == x.size(1)

        self_ctx = []
        for blocks in self.enc_blocks:
            for block in blocks:
                if self.grad_ckpt:
                    x = checkpoint(
                        block,
                        x,
                        ctx,
                        pos_map,
                        None,
                        y,
                        x_mask,
                        ctx_mask,
                        shared_adaln,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, ctx, pos_map, None, y, x_mask, ctx_mask, shared_adaln)
            self_ctx.append(x)
        enc_out = x

        for blocks in self.dec_blocks:
            first_block = blocks[0]
            if self.grad_ckpt:
                x = checkpoint(
                    first_block,
                    x,
                    self_ctx[-1],
                    pos_map,
                    pos_map,
                    y,
                    x_mask,
                    ctx_mask,
                    shared_adaln,
                    use_reentrant=False,
                )
            else:
                x = first_block(
                    x, self_ctx[-1], pos_map, pos_map, y, x_mask, ctx_mask, shared_adaln
                )

            for block in blocks[1:]:
                if self.grad_ckpt:
                    x = checkpoint(
                        block,
                        x,
                        ctx if self.dec_ctx else None,
                        pos_map,
                        None,
                        y,
                        x_mask,
                        ctx_mask,
                        shared_adaln,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        ctx if self.dec_ctx else None,
                        pos_map,
                        None,
                        y,
                        x_mask,
                        ctx_mask,
                        shared_adaln,
                    )

        if return_enc_out:
            return x, enc_out
        return x


class XUDiT(nn.Module):
    """
    Xross-U-Transformer for Image Gen (XUDiT).
    """

    def __init__(
        self,
        patch_size=2,
        input_dim=4,
        dim=1024,
        ctx_dim=1024,
        ctx_size=256,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        depth=8,
        enc_blocks=1,
        dec_blocks=2,
        dec_ctx=False,
        class_cond=0,
        shared_adaln=True,
        concat_ctx=True,
        use_dyt=False,
        double_t=False,
        addon_info_embs_dim=None,
        tread_config=None,
    ):
        super().__init__()
        self.backbone = XUTBackBone(
            dim,
            None if concat_ctx else ctx_dim,
            heads,
            dim_head,
            mlp_dim,
            2,
            depth,
            enc_blocks,
            dec_blocks,
            use_adaln=True,
            use_shared_adaln=shared_adaln,
            dec_ctx=dec_ctx,
            use_dyt=use_dyt,
        )

        self.use_tread = False
        if tread_config is not None:
            self.use_tread = True
            self.dropout_ratio = tread_config["dropout_ratio"]
            self.prev_tread_trns = TBackBone(
                dim,
                None if concat_ctx else ctx_dim,
                heads,
                dim_head,
                mlp_dim,
                2,
                tread_config["prev_trns_depth"],
                use_adaln=True,
                use_shared_adaln=shared_adaln,
                use_dyt=use_dyt,
            )
            self.post_tread_trns = TBackBone(
                dim,
                None if concat_ctx else ctx_dim,
                heads,
                dim_head,
                mlp_dim,
                2,
                tread_config["post_trns_depth"],
                use_adaln=True,
                use_shared_adaln=shared_adaln,
                use_dyt=use_dyt,
            )

        self.patch_size = patch_size
        self.in_patch = PatchEmbed(patch_size, input_dim, dim)
        self.out_patch = UnPatch(patch_size, dim, input_dim)
        self.time_emb = TimestepEmbedding(dim)
        if double_t:
            self.r_emb = TimestepEmbedding(dim)
        if shared_adaln:
            self.shared_adaln_attn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim * 3),
            )
            nn.init.constant_(self.shared_adaln_attn[-1].bias, 0)
            nn.init.constant_(self.shared_adaln_attn[-1].weight, 0)
            self.shared_adaln_xattn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim * 3),
            )
            nn.init.constant_(self.shared_adaln_xattn[-1].bias, 0)
            nn.init.constant_(self.shared_adaln_xattn[-1].weight, 0)
            self.shared_adaln_ffw = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim * 3),
            )
            nn.init.constant_(self.shared_adaln_ffw[-1].bias, 0)
            nn.init.constant_(self.shared_adaln_ffw[-1].weight, 0)
        if class_cond > 0:
            self.class_token = nn.Embedding(class_cond, dim)
        else:
            self.class_token = None
        if concat_ctx and ctx_dim is not None:
            self.ctx_proj = nn.Linear(ctx_dim, dim)
        else:
            self.ctx_proj = None
        if addon_info_embs_dim is not None:
            self.addon_info_embs_proj = nn.Sequential(
                nn.Linear(addon_info_embs_dim, dim), nn.Mish(), nn.Linear(dim, dim)
            )
            nn.init.constant_(self.addon_info_embs_proj[-1].bias, 0)
            nn.init.constant_(self.addon_info_embs_proj[-1].weight, 0)

        self.concat_ctx = concat_ctx
        self.shared_adaln = shared_adaln
        self.need_ctx = ctx_dim is not None
        self.ctx_dim = ctx_dim
        self.ctx_size = ctx_size
        self.grad_ckpt = False
        self.init_weight()

    def init_weight(self):
        if isinstance(self.out_patch.proj, nn.Linear):
            nn.init.normal_(
                self.out_patch.proj.weight,
                mean=0.0,
                std=1 / self.out_patch.proj.in_features**2,
            )

    def set_grad_ckpt(self, grad_ckpt):
        self.backbone.grad_ckpt = grad_ckpt
        self.grad_ckpt = grad_ckpt
        if self.use_tread:
            self.prev_tread_trns.grad_ckpt = grad_ckpt
            self.post_tread_trns.grad_ckpt = grad_ckpt

    def forward(
        self,
        x,
        t,
        ctx=None,
        pos_map=None,
        r=None,
        addon_info=None,
        tread_rate=None,
        return_enc_out=False,
    ):
        n, c, h, w = x.size()
        t = t.reshape(n, -1)
        x, pos_map = self.in_patch(x, pos_map)
        x = x.contiguous()
        if pos_map is None:
            pos_map = (
                make_axial_pos(
                    h // self.patch_size,
                    w // self.patch_size,
                    dtype=x.dtype,
                    device=x.device,
                )
                .unsqueeze(0)
                .expand(n, -1, -1)
            )
        t_emb = self.time_emb(t)
        if r is not None:
            t_emb = t_emb + self.r_emb((t - r.reshape(n, -1)))
        if self.class_token is not None and ctx is not None:
            if ctx.ndim == 1:
                ctx = ctx[:, None]
            t_emb = t_emb + self.class_token(ctx)
            ctx = None
        if addon_info is not None:
            if addon_info.ndim == 1:
                # [B] -> [B, 1] for single value info
                addon_info = addon_info[:, None]
            # [B, D] -> [B, 1, D] for t_emb shape
            addon_embs = self.addon_info_embs_proj(addon_info)[:, None]
            t_emb = t_emb + addon_embs
        if ctx == None and self.need_ctx:
            ctx = torch.zeros(n, self.ctx_size, self.ctx_dim, device=x.device)

        if self.shared_adaln:
            shared_adaln_state = [
                self.shared_adaln_attn(t_emb).chunk(3, dim=-1),
                self.shared_adaln_xattn(t_emb).chunk(3, dim=-1),
                self.shared_adaln_ffw(t_emb).chunk(3, dim=-1),
            ]
        else:
            shared_adaln_state = None

        length = x.size(1)
        if self.ctx_proj is not None:
            ctx = self.ctx_proj(ctx)
            x = torch.cat([x, ctx], dim=1)
            if pos_map is not None:
                pos_map = torch.cat(
                    [
                        pos_map,
                        torch.zeros(n, ctx.size(1), pos_map.size(2), device=x.device),
                    ],
                    dim=1,
                )
            ctx = None

        if self.use_tread:
            x = self.prev_tread_trns(
                x,
                ctx=ctx,
                pos_map=pos_map,
                y=t_emb,
                shared_adaln=shared_adaln_state,
            )
            if self.training or tread_rate is not None:
                xt_selection_length = selection_length = length - int(
                    length * (tread_rate or self.dropout_ratio)
                )
                selection = torch.stack(
                    [
                        torch.randperm(length, device=x.device) < selection_length
                        for _ in range(n)
                    ]
                )
                if self.ctx_proj is not None:
                    ctx_length = x.size(1) - length
                    selection = torch.concat(
                        [
                            selection,
                            torch.ones(
                                n, ctx_length, device=x.device, dtype=torch.bool
                            ),
                        ],
                        dim=1,
                    )
                    selection_length += ctx_length
                full_length = x.size(1)
                not_masked_part = x[~selection, :]
                masked_part = x[selection, :].unflatten(0, (n, selection_length))
                x = masked_part
                raw_pos_map = pos_map
                pos_map = pos_map[selection, :].unflatten(0, (n, selection_length))
        backbone_out = self.backbone(
            x,
            ctx=ctx,
            pos_map=pos_map,
            y=t_emb,
            shared_adaln=shared_adaln_state,
            return_enc_out=return_enc_out,
        )
        if return_enc_out:
            backbone_out, enc_out = backbone_out
        if self.use_tread:
            if self.training or tread_rate is not None:
                out = torch.empty(
                    n, full_length, x.size(2), device=x.device, dtype=x.dtype
                )
                out[~selection, :] = not_masked_part
                out[selection, :] = backbone_out.flatten(0, 1)
                pos_map = raw_pos_map
            else:
                out = backbone_out
            out = self.post_tread_trns(
                out,
                ctx=ctx,
                pos_map=pos_map,
                y=t_emb,
                shared_adaln=shared_adaln_state,
            )
        else:
            out = backbone_out
        out = out[:, :length]
        out = self.out_patch(out, h, w)

        if return_enc_out:
            length = (
                xt_selection_length if self.use_tread and self.training else full_length
            )
            return out, enc_out[:, :length]
        return out
