import torch
import torch.nn as nn


class PostfuseModule(nn.Module):
    def __init__(self, embed_dim, embed_dim_img):
        super().__init__()
        self.mlp1 = MLP(embed_dim_img, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def fuse_fn(self, object_embeds):
        text_object_embeds = self.mlp1(object_embeds)
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        fuse_index,
    ) -> torch.Tensor:
        text_object_embed = self.fuse_fn(object_embeds)
        text_embeds_new = text_embeds.clone()
        text_embeds_new[:, fuse_index, :] = text_object_embed.squeeze(1)

        return text_embeds_new

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x
