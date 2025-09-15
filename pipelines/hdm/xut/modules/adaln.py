import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .norm import RMSNorm


class AdaLN(nn.Module):
    def __init__(self, dim, y_dim, gate=True, norm_layer=RMSNorm, shared=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.gate = gate
        if shared:
            self.adaln = None
        else:
            self.adaln = nn.Linear(y_dim, dim * (2 + bool(gate)))
            nn.init.constant_(self.adaln.bias, 0)
            nn.init.constant_(self.adaln.weight, 0)

    def forward(self, x, y, shared_adaln=None):
        if shared_adaln is None:
            scale, shift, *gate = self.adaln(y).chunk(2 + bool(self.gate), dim=-1)
        else:
            scale, shift, *gate = shared_adaln
        normed_x, _ = self.norm(x)
        result = normed_x * (scale + 1.0) + shift
        return result, (gate[0] + 1) if self.gate else 1
