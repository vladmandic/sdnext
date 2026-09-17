from installer import install, installed
from modules import rocm
from modules.logger import log
from modules.attention.registry import AttentionBackend, Constraints, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    try:
        import flash_attn # pylint: disable=unused-import
    except ImportError:
        log.warning('Attention: type="Flash attention" not installed: starting build, this may take a while...')
    if platform.backend == 'rocm':
        if not installed('flash-attn'):
            log.info('Attention: type="Flash attention" building...')
            agent = rocm.Agent(platform.device)
            install(rocm.get_flash_attention_command(agent), reinstall=True)
    else:
        install('--no-build-isolation flash-attn')
    from flash_attn import flash_attn_func

    def call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa): # pylint: disable=unused-argument
        is_unsqueezed = False
        if query.dim() == 3:
            query = query.unsqueeze(0)
            is_unsqueezed = True
            if key.dim() == 3:
                key = key.unsqueeze(0)
            if value.dim() == 3:
                value = value.unsqueeze(0)
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_output = flash_attn_func(q=query, k=key, v=value, dropout_p=dropout_p, causal=is_causal, softmax_scale=scale).transpose(1, 2)
        if is_unsqueezed:
            attn_output = attn_output.squeeze(0)
        return attn_output

    log.debug('Attention: type="Flash attention"')
    return call


backend = AttentionBackend(
    name='flash', label='Flash attention', priority=40, prepare=prepare,
    constraints=Constraints(max_head_dim=128, allow_mask=False, allow_float32=False, same_device=True),
)
