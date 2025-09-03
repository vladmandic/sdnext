import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
if XFORMERS_AVAILABLE:
    from xformers.ops import memory_efficient_attention
else:
    memory_efficient_attention = None

from .. import env
from ..utils import compile_wrapper
from .axial_rope import AxialRoPE


if not env.USE_XFORMERS:
    memory_efficient_attention = None
if env.USE_VANILLA:

    @compile_wrapper
    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0):
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        attn = F.dropout(attn, p)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=-1, pos_dim=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim > 0 else dim // n_heads
        self.n_heads = dim // self.head_dim
        assert (
            self.n_heads * self.head_dim == dim
        ), "dim must be divisible by n_heads or head_dim"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.rope = AxialRoPE(self.head_dim, self.n_heads, pos_dim)
        self.attn = memory_efficient_attention or F.scaled_dot_product_attention
        self.xformers = memory_efficient_attention is not None

    def forward(self, x, pos_map=None, mask=None):
        b, n, _, h = *x.shape, self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        if pos_map is not None:
            q = self.rope(q.reshape(b, n, h, -1).transpose(1, 2), pos_map)
            k = self.rope(k.reshape(b, n, h, -1).transpose(1, 2), pos_map)
            v = v.reshape(b, n, h, -1)
            if self.xformers:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
            else:
                v = v.transpose(1, 2)
        else:
            q, k, v = map(lambda t: t.reshape(b, n, h, -1), (q, k, v))
            if not self.xformers:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if n % 8 and self.xformers:
                align_n = math.ceil(n / 8) * 8
                mask_align = torch.empty(
                    *mask.shape[:3], align_n, device=mask.device, dtype=mask.dtype
                )
                mask_align[..., :n] = mask
                mask = mask_align.to(q).expand(b, h, n, align_n)[..., :n]
            else:
                mask = mask.to(q).expand(b, h, n, n)

        attn = self.attn(q, k, v, mask)
        if not self.xformers:
            attn = attn.transpose(1, 2)
        attn = attn.reshape(b, n, h * self.head_dim)
        attn = self.out(attn)
        return attn


class CrossAttention(nn.Module):
    def __init__(self, dim, ctx_dim, n_heads=8, head_dim=-1, pos_dim=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim > 0 else dim // n_heads
        self.n_heads = dim // self.head_dim
        assert (
            self.n_heads * self.head_dim == dim
        ), "dim must be divisible by n_heads or head_dim"

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim)
        self.rope = AxialRoPE(self.head_dim, self.n_heads, pos_dim)
        self.attn = memory_efficient_attention or F.scaled_dot_product_attention
        self.xformers = memory_efficient_attention is not None

    def forward(self, x, ctx, pos_map=None, ctx_pos_map=None, mask=None):
        b, n, _, h = *x.shape, self.n_heads
        ctx_n = ctx.shape[1]
        q = self.q(x)
        k, v = self.kv(ctx).chunk(2, dim=-1)

        if pos_map is not None:
            q = self.rope(q.reshape(b, n, h, -1).transpose(1, 2), pos_map)
            q = q if not self.xformers else q.transpose(1, 2)
        else:
            q = q.reshape(b, n, h, -1)
            q = q if self.xformers else q.transpose(1, 2)
        if ctx_pos_map is not None:
            k = self.rope(k.reshape(b, ctx_n, h, -1).transpose(1, 2), ctx_pos_map)
            k = k if not self.xformers else k.transpose(1, 2)
        else:
            k = k.reshape(b, ctx_n, h, -1)
            k = k if self.xformers else k.transpose(1, 2)
        v = v.reshape(b, ctx_n, h, -1)
        v = v if self.xformers else v.transpose(1, 2)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if ctx_n % 8 and self.xformers:
                align_n = math.ceil(ctx_n / 8) * 8
                mask_align = torch.empty(
                    *mask.shape[:3], align_n, device=mask.device, dtype=mask.dtype
                )
                mask_align[..., :ctx_n] = mask
                mask = mask_align.to(q).expand(b, h, n, align_n)[..., :ctx_n]
            else:
                mask = mask.to(q).expand(b, h, n, ctx_n)

        attn = self.attn(q, k, v, mask)
        if not self.xformers:
            attn = attn.transpose(1, 2)
        attn = attn.reshape(b, n, h * self.head_dim)
        attn = self.out(attn)
        return attn


class AttentionPooling(CrossAttention):
    def __init__(self, dim, n_heads=8, head_dim=-1, pos_dim=2):
        super().__init__(dim, dim, n_heads, head_dim, pos_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, dim) * 1 / dim**0.5)

    def forward(self, x, pos_map=None, mask=None):
        query = self.query_token.expand(x.shape[0], -1, -1)
        return super().forward(query, x, None, pos_map, mask).squeeze(1)


class AttentiveProbe(CrossAttention):
    def __init__(self, dim, out_dim, n_heads=8, head_dim=-1, pos_dim=2, n_probes=1):
        super().__init__(dim, dim, n_heads, head_dim, pos_dim)
        self.query_token = nn.Parameter(torch.randn(1, n_probes, dim) * 1 / dim**0.5)
        self.token_proj = nn.Linear(dim * n_probes, out_dim)

    def forward(self, x, pos_map=None, mask=None):
        query = self.query_token.expand(x.shape[0], -1, -1)
        output_embedding = super().forward(query, x, None, pos_map, mask)
        output_embedding = output_embedding.flatten(-2, -1)
        return self.token_proj(output_embedding)


@cache
def prefix_causal_attention_mask(
    q_len, kv_len, prefix_len=0, is_self_attn=False, dtype=None, device=None
):
    """
    **Made by claude 3.7 sonnet without thinking**
    Generate attention masks and biases for transformer models.

    Parameters:
    -----------
    q_len : int
        Length of the query sequence
    kv_len : int
        Length of the key/value sequence
    prefix_len : int, optional
        Length of the prefix for which we allow full attention (no causal masking)
        Default: 0 (standard causal mask)
    is_self_attn : bool, optional
        Whether this is for self-attention (q_len == kv_len and they represent the same sequence)
        Enables faster mask generation
        Default: False
    dtype : torch.dtype, optional
        Data type for the output tensors
        Default: None (will use torch.bool for mask, torch.float for bias)
    device : torch.device, optional
        Device on which to create the tensors
        Default: None (will use the default torch device)

    Returns:
    --------
    tuple: (attention_mask, attention_bias)
        - attention_mask: Boolean tensor of shape (q_len, kv_len) where True values indicate
          positions that should be attended to
        - attention_bias: Tensor of same shape with dtype specified (or float), containing
          0.0 for positions to attend to and -float('inf') for positions to mask out
    """
    # Fast path for self-attention with no prefix
    if is_self_attn and prefix_len == 0:
        # Simple lower triangular matrix for standard causal self-attention
        attention_mask = torch.tril(
            torch.ones(q_len, q_len, dtype=torch.bool, device=device)
        )

    # Fast path for self-attention with prefix
    elif is_self_attn and prefix_len > 0:
        attention_mask = torch.tril(
            torch.ones(q_len, q_len, dtype=torch.bool, device=device)
        )

        # Add the prefix part (allow full attention to the prefix)
        if prefix_len < q_len:
            # Set the prefix columns to all True (we use indexing which is faster than cat)
            attention_mask[:, :prefix_len] = True

    # General case for cross-attention or when fast path is not used
    else:
        # Create base causal mask (lower triangular)
        # Each query position i can attend to key positions j where j <= i
        causal_mask = torch.tril(
            torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        )

        # If there's a prefix, allow full attention within that prefix
        if prefix_len > 0:
            # Combine masks:
            # - For the prefix part of kv, use all True
            # - For the rest, use causal mask
            if prefix_len < kv_len:
                attention_mask = torch.cat(
                    [
                        torch.ones(q_len, prefix_len, dtype=torch.bool, device=device),
                        causal_mask[:, prefix_len:],
                    ],
                    dim=1,
                )
            else:
                # If prefix_len >= kv_len, the entire sequence gets full attention
                attention_mask = torch.ones(
                    q_len, kv_len, dtype=torch.bool, device=device
                )
        else:
            # Without prefix, just use the causal mask
            attention_mask = causal_mask

    # Convert boolean mask to attention bias
    # True -> 0.0, False -> -inf
    float_dtype = torch.float if dtype is None else dtype
    attention_bias = torch.zeros_like(attention_mask, dtype=float_dtype, device=device)
    attention_bias = attention_bias.masked_fill(~attention_mask, float("-inf"))

    return attention_mask, attention_bias


# Example usage:
if __name__ == "__main__":
    # Standard causal mask for sequence length 6
    mask, bias = prefix_causal_attention_mask(q_len=6, kv_len=6)
    print("Standard causal mask:")
    print(mask)
    print("\nStandard causal bias:")
    print(bias)

    # Same with self-attention flag
    mask_self, bias_self = prefix_causal_attention_mask(
        q_len=6, kv_len=6, is_self_attn=True
    )
    print("\nSelf-attention causal mask (should be identical):")
    print(mask_self)
    print("Masks are identical:", torch.all(mask == mask_self).item())

    # Causal mask with prefix_len=3 (first 2 tokens get full attention)
    mask, bias = prefix_causal_attention_mask(q_len=6, kv_len=6, prefix_len=3)
    print("\nCausal mask with prefix_len=3:")
    print(mask)
    print("\nCausal bias with prefix_len=3:")
    print(bias)

    # Same with self-attention flag
    mask_self, bias_self = prefix_causal_attention_mask(
        q_len=6, kv_len=6, prefix_len=3, is_self_attn=True
    )
    print("\nSelf-attention mask with prefix_len=3 (should be identical):")
    print(mask_self)
    print("Masks are identical:", torch.all(mask == mask_self).item())

    # Handling different q_len and kv_len (for cross-attention)
    mask, bias = prefix_causal_attention_mask(q_len=4, kv_len=6, prefix_len=3)
    print("\nCross-attention mask with q_len=4, kv_len=6, prefix_len=3:")
    print(mask)
    print("\nCross-attention bias:")
    print(bias)

    self_attn = SelfAttention(64, 8).cuda().half()
    x = torch.randn(1, 16, 64).cuda().half()
    mask, bias = prefix_causal_attention_mask(
        16, 16, is_self_attn=True, device=x.device, dtype=x.dtype
    )
    test_out = self_attn(x, mask=bias)
    torch.sum(test_out).backward()

    print(x.shape, mask.shape, bias.shape)
    print(test_out.shape)
    print(torch.isnan(test_out).any())
    print(torch.norm(next(self_attn.parameters()).grad))
