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

import torch
import torch.nn.functional as F

#from flash_attn import flash_attn_varlen_func

from torch import nn


def pytorch_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False):
    """
    A PyTorch-based implementation of variable-length attention to replace flash_attn_varlen_func.
    It processes each sequence in the batch individually.
    """
    # Create an empty tensor to store the output.
    output = torch.empty_like(q)

    # Iterate over each sequence in the batch. The batch size is the number of sequences.
    for i in range(len(cu_seqlens_q) - 1):
        # Determine the start and end indices for the current sequence.
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]

        # Slice the q, k, and v tensors to get the data for the current sequence.
        # The shape is (seq_len, heads, head_dim).
        q_i = q[start_q:end_q]
        k_i = k[start_k:end_k]
        v_i = v[start_k:end_k]

        # Reshape for torch's scaled_dot_product_attention which expects (batch, heads, seq, dim).
        # Here, we treat each sequence as a batch of 1.
        q_i = q_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_q, head_dim)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_k, head_dim)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_k, head_dim)

        # Use PyTorch's built-in scaled dot-product attention.
        output_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i,
            dropout_p=dropout_p if not deterministic else 0.0,
            is_causal=causal
        )

        # Reshape the output back to the original format (seq_len, heads, head_dim)
        output_i = output_i.squeeze(0).permute(1, 0, 2)

        # Place the result for the current sequence into the main output tensor.
        output[start_q:end_q] = output_i

    return output


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        #return pytorch_varlen_attention(*args, **kwargs)
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        try:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(*args, **kwargs)
        except ImportError:
            return pytorch_varlen_attention(*args, **kwargs)
