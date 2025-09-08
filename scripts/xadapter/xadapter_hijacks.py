import torch
from torch import nn


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 1, 3, 4, 2).reshape(*x.shape[:2], -1)


class PositionNet(nn.Module):
    def __init__(self, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]
        self.linears = nn.Sequential(
            nn.Linear(self.positive_len + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings):
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 -> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        return objs
