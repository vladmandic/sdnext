from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LEAK = 0.01


def _act(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, LEAK), -6.0, 6.0)


def _conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, w, b, padding=w.shape[-1] // 2)


def _pad_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.shape[1] == channels:
        return x
    return F.pad(x, (0, 0, 0, 0, 0, channels - x.shape[1]))


def _up2(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class SynthesisBlock(nn.Module):
    """One synthesis block: stem convs (act + 2x2 avg pool each), residual pairs, three
    activated heads on the x2-upsampled trunk, one linear 8-channel head on the x2-upsampled heads."""

    def __init__(self, prefix: str, stems: int, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.stems = stems
        for i in range(stems):
            self.register_buffer(f"stem_{i}_w", weights[f"{prefix}.stem{i}.weight"])
            self.register_buffer(f"stem_{i}_b", weights[f"{prefix}.stem{i}.bias"])

        for i in range(8):
            self.register_buffer(f"res_{i}_w", weights[f"{prefix}.res{i}.weight"])
            self.register_buffer(f"res_{i}_b", weights[f"{prefix}.res{i}.bias"])

        for i in range(3):
            self.register_buffer(f"head_{i}_w", weights[f"{prefix}.bot0.head{i}.weight"])
            self.register_buffer(f"head_{i}_b", weights[f"{prefix}.bot0.head{i}.bias"])

        self.register_buffer("out_w", weights[f"{prefix}.bot1.weight"])
        self.register_buffer("out_b", weights[f"{prefix}.bot1.bias"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(self.stems):
            w = getattr(self, f"stem_{i}_w")
            b = getattr(self, f"stem_{i}_b")
            x = F.avg_pool2d(_act(_conv(_pad_channels(x, w.shape[1]), w, b)), 2)

        skip = x
        for i in range(8):
            w = getattr(self, f"res_{i}_w")
            b = getattr(self, f"res_{i}_b")
            y = _act(_conv(x, w, b))
            if i % 2 == 0:
                skip, x = x, y
            else:
                x = y + skip

        up_x = _up2(x)
        heads = [
            _act(_conv(up_x, getattr(self, f"head_{i}_w"), getattr(self, f"head_{i}_b")))
            for i in range(3)
        ]
        spans = [(0, 4), (4, 5), (5, 8)]
        outs = [
            _conv(_up2(heads[i]), self.out_w[lo:hi], self.out_b[lo:hi])
            for i, (lo, hi) in enumerate(spans)
        ]
        return outs[0], outs[1], outs[2]  # flow (4), mask (1), residual (3)


class NeuralFrameGenModel(nn.Module):
    """Neural Frame Generation network composed of Block0 and Block1."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.block0 = SynthesisBlock("block0", 3, weights)
        self.block1 = SynthesisBlock("block1", 2, weights)

    def forward_block0(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.block0(x)

    def forward_block1(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.block1(x)
