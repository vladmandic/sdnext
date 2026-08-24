from contextlib import contextmanager
import torch
import torch.nn.functional as F


orig_sdpa = F.scaled_dot_product_attention


def safe_slice_mask(mask, start, end):
    if mask is None:
        return None
    ndim = mask.ndim
    if ndim == 2:
        return mask[start:end, :]
    elif ndim == 3:
        return mask[:, start:end, :]
    elif ndim >= 4:
        return mask[..., start:end, :]
    return mask


def chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, chunk_size=512, **kwargs):
    """
    Drop-in replacement for F.scaled_dot_product_attention
    """
    N_q = query.shape[2]
    if (chunk_size == 0) or (N_q <= chunk_size): # fallback if disabled or sequence length is already smaller than chunk size
        return orig_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )
    out_chunks = []
    for start in range(0, N_q, chunk_size): # process query sequence in chunks against all Keys/Values
        end = min(start + chunk_size, N_q)
        q_chunk = query[:, :, start:end, :]
        chunk_mask = safe_slice_mask(attn_mask, start, end)
        out_chunk = orig_sdpa(
            q_chunk, key, value,
            attn_mask=chunk_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )
        out_chunks.append(out_chunk)
    return torch.cat(out_chunks, dim=2)


@contextmanager
def minimax_attention(chunk_size=0):
    """
    Context manager to safely monkey-patch PyTorch's SDPA function
    Value 0 bypasses the patch entirely and uses native SDPA
    Values are 64-2048 aligned to multiples of 64. Lower values reduce VRAM usage at the cost of speed
    """
    global orig_sdpa # pylint: disable=global-statement
    orig_sdpa = F.scaled_dot_product_attention
    chunk_size = max(64, (chunk_size // 64) * 64) # sanitize and align chunk_size to a multiple of 64 (minimum 64)

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        return chunked_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            chunk_size=chunk_size,
            **kwargs,
        )
    F.scaled_dot_product_attention = patched_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa
