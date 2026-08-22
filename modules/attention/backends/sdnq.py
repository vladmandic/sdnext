from modules.logger import log
from modules.attention.registry import AttentionBackend, Constraints, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    from modules import shared
    from sdnq.kernels.triton_atten import sdnq_triton_atten
    options = {
        'matmul_dtype': shared.opts.sdnq_attention_matmul_type,
        'pv_matmul_dtype': shared.opts.sdnq_attention_pv_matmul_type,
        'smooth_k': shared.opts.sdnq_attention_smooth_k,
        'use_hadamard': shared.opts.sdnq_attention_use_hadamard,
        'hadamard_group_size': shared.opts.sdnq_attention_hadamard_group_size,
        'use_fp16_accum': shared.opts.sdnq_attention_use_fp16_accum,
    }

    def call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa): # pylint: disable=unused-argument
        return sdnq_triton_atten(query=query, key=key, value=value, attn_mask=attn_mask, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **options)

    log.debug(f'Torch attention: type="SDNQ attention" matmul={options["matmul_dtype"]}:{options["pv_matmul_dtype"]} smooth={options["smooth_k"]} hadamard={options["use_hadamard"]} fp16_accum={options["use_fp16_accum"]}')
    return call


backend = AttentionBackend(
    name='sdnq', label='SDNQ attention', priority=60, prepare=prepare,
    constraints=Constraints(min_tokens=32, min_long_side=512, min_heads=2), # sequences of 512 or fewer are text encoders, single-head calls the vae
    options=('sdnq_attention_matmul_type', 'sdnq_attention_pv_matmul_type', 'sdnq_attention_smooth_k', 'sdnq_attention_use_hadamard', 'sdnq_attention_hadamard_group_size', 'sdnq_attention_use_fp16_accum'),
)
