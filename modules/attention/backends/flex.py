import torch
from modules.logger import log
from modules.attention.registry import AttentionBackend, Constraints, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    def causal_mask(b, h, q_idx, kv_idx): # pylint: disable=unused-argument
        return q_idx >= kv_idx

    def call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, selection=None): # pylint: disable=unused-argument
        if selection is not None:
            from modules.attention.sparse import flex as sparse_flex
            return sparse_flex.attend(query, key, value, selection, scale=scale, enable_gqa=enable_gqa)
        score_mod = None
        block_mask = None
        if attn_mask is not None:
            batch_size, num_heads = query.shape[:2]
            seq_len_q = query.shape[-2]
            seq_len_kv = key.shape[-2]
            attn_mask = attn_mask.expand(batch_size, num_heads, seq_len_q, seq_len_kv) # sdpa masks broadcast over the trailing dims
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


backend = AttentionBackend(
    name='flex', label='Flex attention', priority=20, prepare=prepare,
    constraints=Constraints(min_ndim=4, same_device=True), # flex_attention takes 4d tensors on one device and compiles on cpu
    caps=frozenset({'block_mask'}),
)
