import torch
from modules.logger import log
from modules.attention.registry import AttentionBackend, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    def causal_mask(b, h, q_idx, kv_idx): # pylint: disable=unused-argument
        return q_idx >= kv_idx

    def call(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs): # pylint: disable=unused-argument
        score_mod = None
        block_mask = None
        if attn_mask is not None:
            batch_size, num_heads = query.shape[:2]
            seq_len_q = query.shape[-2]
            seq_len_kv = key.shape[-2]
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.view(attn_mask.shape[0], 1, attn_mask.size[1], 1)
            attn_mask = attn_mask.expand(batch_size, num_heads, seq_len_q, seq_len_kv)
            if attn_mask.dtype == torch.bool:
                def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                    return attn_mask[batch_idx, head_idx, q_idx, kv_idx]
                block_mask = create_block_mask(mask_mod, batch_size, None, seq_len_q, seq_len_kv, device=query.device)
            else:
                def score_mod_fn(score, batch_idx, head_idx, q_idx, kv_idx):
                    return score + attn_mask[batch_idx, head_idx, q_idx, kv_idx]
                score_mod = score_mod_fn
        elif is_causal:
            block_mask = create_block_mask(causal_mask, query.shape[0], query.shape[1], query.shape[-2], key.shape[-2], device=query.device)
        return flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa)

    log.debug('Torch attention: type="Flex attention"')
    return call


backend = AttentionBackend(name='flex', label='Flex attention', priority=20, prepare=prepare, terminal=True)
