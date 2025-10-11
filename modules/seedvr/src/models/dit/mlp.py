# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


def get_mlp(mlp_type: Optional[str] = "normal"):
    if mlp_type == "normal":
        return MLP
    elif mlp_type == "swiglu":
        return SwiGLUMLP


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
    ):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))
        return x
