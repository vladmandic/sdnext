import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_channels=3,
        embed_dim=512,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=bias)
        self.norm = nn.Identity() if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x, pos_map=None):
        b, _, h, w = x.shape
        x = self.proj(x)
        b, _, new_h, new_w = x.shape
        if pos_map is not None:
            pos_map = (
                F.interpolate(
                    pos_map.reshape(b, h, w, -1).permute(0, 3, 1, 2),
                    (new_h, new_w),
                    mode="bilinear",
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .flatten(1, 2)
            )
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, pos_map


class UnPatch(nn.Module):
    def __init__(self, patch_size=4, input_dim=512, out_channel=3, proj=True):
        super().__init__()
        self.patch_size = patch_size
        self.c = out_channel

        if proj:
            self.proj = nn.Linear(input_dim, patch_size**2 * out_channel)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, axis1=None, axis2=None, loss_mask=None):
        b, n, _ = x.shape
        p = q = self.patch_size
        if axis1 is None and axis2 is None:
            w = h = int(n**0.5)
            assert h * w == n
        else:
            h = axis1 // p if axis1 else n // (axis2 // p)
            w = axis2 // p if axis2 else n // h
            assert h * w == n

        x = self.proj(x)
        if loss_mask is not None:
            x = torch.where(loss_mask[..., None], x, x.detach())
        x = (
            x.reshape(b, h, w, p, q, self.c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, self.c, h * p, w * q)
        )
        return x
