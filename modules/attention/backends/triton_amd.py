import torch
from modules.logger import log
from modules.attention.registry import AttentionBackend, Constraints, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    from modules.flash_attn_triton_amd import interface_fa

    def call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa): # pylint: disable=unused-argument
        if scale is None:
            scale = query.shape[-1] ** (-0.5)
        head_size_og = query.size(3)
        if head_size_og % 8 != 0:
            query = torch.nn.functional.pad(query, [0, 8 - head_size_og % 8])
            key = torch.nn.functional.pad(key, [0, 8 - head_size_og % 8])
            value = torch.nn.functional.pad(value, [0, 8 - head_size_og % 8])
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out_padded = torch.zeros_like(query)
        interface_fa.fwd(query, key, value, out_padded, dropout_p, scale, is_causal)
        return out_padded[..., :head_size_og].transpose(1, 2)

    log.debug('Attention: type="Triton AMD Flash attention"')
    return call


backend = AttentionBackend(
    name='triton', label='Triton AMD Flash attention', priority=30, prepare=prepare,
    constraints=Constraints(max_head_dim=128, allow_mask=False, same_device=True),
    platforms=frozenset({'rocm', 'zluda'}),
)
