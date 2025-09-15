import math

import torch
import torch.nn as nn

from ..utils import compile_wrapper


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000, time_factor: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.time_factor = time_factor
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=dim // 2, dtype=torch.float32)
                / (dim // 2)
            )[None],
        )
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Mish())

    @compile_wrapper
    def forward(self, t):
        t = self.time_factor * t
        args = t[:, None] * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.proj(embedding)
