import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from .. import env
from ..utils import compile_wrapper


class SwiGLUTorch(nn.Module):
    def __init__(
        self, in_features, hidden_features, out_features, bias=True, _pack_weights=True
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        if _pack_weights:
            self.w12 = torch.nn.Linear(in_features, 2 * hidden_features, bias=bias)
        else:
            self.w1 = torch.nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = torch.nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = torch.nn.Linear(hidden_features, out_features, bias=bias)

    @compile_wrapper
    def forward(self, x):
        if self.w12 is not None:
            x1, x2 = self.w12(x).chunk(2, dim=-1)
        else:
            x1 = self.w1(x)
            x2 = self.w2(x)
        return self.w3(F.silu(x1) * x2)


if XFORMERS_AVAILABLE:
    from xformers.ops import SwiGLU
else:
    SwiGLU = SwiGLUTorch
if not env.USE_XFORMERS_LAYERS:
    SwiGLU = SwiGLUTorch


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)
    model1 = SwiGLU(128, 256, 128)
    model2 = SwiGLUTorch(128, 256, 128)

    model1.load_state_dict(model2.state_dict())

    print(F.mse_loss(model1(x), model2(x)), torch.norm(model1(x)))
