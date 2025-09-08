# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from modules import devices


# Omnigen uses x.shape[-1] // 2 instead of -1
# Functionally the same but -1 does fail with when the shape becomes 0
if devices.backend != "ipex":
    def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
        use_real: bool = True,
        use_real_unbind_dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        if use_real:
            cos, sin = freqs_cis  # [S, D]
            cos = cos[None, None]
            sin = sin[None, None]
            cos, sin = cos.to(x.device), sin.to(x.device)

            if use_real_unbind_dim == -1:
                # Used for flux, cogvideox, hunyuan-dit
                x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
                x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            elif use_real_unbind_dim == -2:
                # Used for Stable Audio, OmniGen and CogView4
                x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
                x_rotated = torch.cat([-x_imag, x_real], dim=-1)
            else:
                raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

            out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

            return out
        else:
            # used for lumina
            # x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

            return x_out.type_as(x)
else:
    def apply_rotary_emb(x, freqs_cis, use_real: bool = True, use_real_unbind_dim: int = -1):
        if use_real:
            cos, sin = freqs_cis  # [S, D]
            cos = cos[None, None]
            sin = sin[None, None]
            cos, sin = cos.to(x.device), sin.to(x.device)

            if use_real_unbind_dim == -1:
                # Used for flux, cogvideox, hunyuan-dit
                x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
                x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            elif use_real_unbind_dim == -2:
                # Used for Stable Audio, OmniGen, CogView4 and Cosmos
                x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
                x_rotated = torch.cat([-x_imag, x_real], dim=-1)
            else:
                raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

            out = (x.to(dtype=torch.float32) * cos + x_rotated.to(dtype=torch.float32) * sin).to(x.dtype)
            return out
        else:
            # used for lumina
            # force cpu with Alchemist
            x_rotated = torch.view_as_complex(x.to("cpu").to(dtype=torch.float32).reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
            freqs_cis = freqs_cis.to("cpu").unsqueeze(2)
            x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
            return x_out.type_as(x).to(x.device)
