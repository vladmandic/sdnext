from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LEAK = 0.1


class ResidualBlock(nn.Module):
    def __init__(self, prefix: str, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.register_buffer("conv0_w", weights[f"{prefix}.conv0.weight"])
        self.register_buffer("conv0_b", weights[f"{prefix}.conv0.bias"])
        self.register_buffer("conv1_w", weights[f"{prefix}.conv1.weight"])
        self.register_buffer("conv1_b", weights[f"{prefix}.conv1.bias"])
        self.register_buffer("shortcut_w", weights[f"{prefix}.shortcut.weight"])
        self.register_buffer("shortcut_b", weights[f"{prefix}.shortcut.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(F.conv2d(x, self.conv0_w, self.conv0_b, padding=1), LEAK)
        h = F.conv2d(h, self.conv1_w, self.conv1_b, padding=1)
        sc = F.conv2d(x, self.shortcut_w, self.shortcut_b, padding=0)
        return F.leaky_relu(h + sc, LEAK)


class NeuralVSRModel(nn.Module):
    """RTX Video Super Resolution U-Net network."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        for i in range(5):
            setattr(self, f"encoder{i}", ResidualBlock(f"encoder{i}", weights))

        for i in range(4):
            self.register_buffer(f"decoder{i}_up_w", weights[f"decoder{i}.upsample.weight"])
            self.register_buffer(f"decoder{i}_up_b", weights[f"decoder{i}.upsample.bias"])
            setattr(self, f"decoder{i}", ResidualBlock(f"decoder{i}", weights))

        self.register_buffer("out_conv_w", weights["output.conv.weight"])
        self.register_buffer("out_conv_b", weights["output.conv.bias"])
        self.register_buffer("out_proj_w", weights["output.project.weight"])
        self.register_buffer("out_proj_b", weights["output.project.bias"])

    def forward(self, x_unshuffle: torch.Tensor) -> torch.Tensor:
        """Forward pass taking 16-channel folded features and producing 48-channel subpixel projection."""
        e0 = self.encoder0(x_unshuffle)
        e1_in = F.avg_pool2d(e0, 2)
        e1 = self.encoder1(e1_in)
        e2 = self.encoder2(e1)
        e3_in = F.avg_pool2d(e2, 2)
        e3 = self.encoder3(e3_in)
        e4 = self.encoder4(e3)

        d0_in = F.leaky_relu(F.conv2d(e4, self.decoder0_up_w, self.decoder0_up_b, padding=1), LEAK)
        d0 = self.decoder0(torch.cat([d0_in, e3], dim=1))

        d1_in = F.interpolate(d0, scale_factor=2, mode="bilinear", align_corners=False)
        d1_in = F.leaky_relu(F.conv2d(d1_in, self.decoder1_up_w, self.decoder1_up_b, padding=1), LEAK)
        d1 = self.decoder1(torch.cat([d1_in, e2], dim=1))

        d2_in = F.leaky_relu(F.conv2d(d1, self.decoder2_up_w, self.decoder2_up_b, padding=1), LEAK)
        d2 = self.decoder2(torch.cat([d2_in, e1], dim=1))

        d3_in = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d3_in = F.leaky_relu(F.conv2d(d3_in, self.decoder3_up_w, self.decoder3_up_b, padding=1), LEAK)
        d3 = self.decoder3(torch.cat([d3_in, e0], dim=1))

        out_conv = F.leaky_relu(F.conv2d(d3, self.out_conv_w, self.out_conv_b, padding=1), LEAK)
        out_proj = F.conv2d(out_conv, self.out_proj_w, self.out_proj_b, padding=0)
        return out_proj
