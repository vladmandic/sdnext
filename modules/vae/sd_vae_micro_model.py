import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# MicroDecoder Architecture
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, 1) in [0, 1]
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = (t * 1000.0) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2:
            t = t.view(-1, 1)
        return self.mlp(self.sin_emb(t))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class FiLMSmoothResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.film_proj = nn.Linear(emb_dim, out_channels * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        scale_shift = self.film_proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.se(h)
        return F.silu(self.skip(x) + h)


class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.gn = nn.GroupNorm(min(4, out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.shuffle(h)
        h = self.gn(h)
        return F.silu(h)


class MicroDecoder(nn.Module):
    def __init__(self, in_channels: int = 64, hidden_dim: int = 256, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU()
        )
        self.res1 = FiLMSmoothResBlock(hidden_dim, hidden_dim, emb_dim=hidden_dim)
        self.res2 = FiLMSmoothResBlock(hidden_dim, hidden_dim, emb_dim=hidden_dim)

        # Multi-stage progressive upsampling
        curr_scale = scale_factor
        curr_dim = hidden_dim
        self.up_blocks = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()
        while curr_scale > 1:
            step_scale = 2 if (curr_scale % 2 == 0) else curr_scale
            next_dim = max(curr_dim // 2, 32)
            self.up_blocks.append(PixelShuffleUpsampleBlock(curr_dim, next_dim, scale_factor=step_scale))
            self.up_res_blocks.append(FiLMSmoothResBlock(next_dim, next_dim, emb_dim=hidden_dim))
            curr_dim = next_dim
            curr_scale //= step_scale

        self.out_proj = nn.Sequential(
            nn.Conv2d(curr_dim, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.out_proj[2].bias)

    def forward(self, latents: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        if latents.ndim == 5:
            latents = latents[:, :, 0] # [B, C, T, H, W] -> [B, C, H, W]
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if t is None:
            t = torch.zeros((latents.shape[0], 1), device=latents.device, dtype=latents.dtype)
        t_emb = self.t_embedder(t)
        x = self.in_proj(latents)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        for up_block, res_block in zip(self.up_blocks, self.up_res_blocks):
            x = up_block(x)
            x = res_block(x, t_emb)
        return self.out_proj(x)
