import torch
from installer import install
from modules.logger import log
from modules.attention.registry import AttentionBackend, Constraints, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    install('sageattention')

    use_cuda_backend = False
    if platform.backend == 'cuda' and torch.cuda.get_device_capability(platform.device) == (8, 6):
        use_cuda_backend = True # sm86 needs the cuda backend, sage attention over triton produces NaNs there
        try:
            from sageattention import sageattn_qk_int8_pv_fp16_cuda
        except Exception:
            use_cuda_backend = False

    if use_cuda_backend:
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_attn_impl(query, key, value, is_causal, scale):
            return sageattn_qk_int8_pv_fp16_cuda(
                q=query, k=key, v=value,
                tensor_layout="HND",
                is_causal=is_causal,
                sm_scale=scale,
                return_lse=False,
                pv_accum_dtype="fp32",
            )
    else:
        from sageattention import sageattn
        def sage_attn_impl(query, key, value, is_causal, scale):
            return sageattn(
                q=query, k=key, v=value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=scale,
            )

    def call(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa): # pylint: disable=unused-argument
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
        return sage_attn_impl(query, key, value, is_causal, scale)

    log.debug(f'Torch attention: type="Sage attention" backend={"cuda" if use_cuda_backend else "auto"}')
    return call


backend = AttentionBackend(
    name='sage', label='Sage attention', priority=50, prepare=prepare,
    constraints=Constraints(head_dims=frozenset({64, 96, 128}), allow_mask=False, same_device=True),
)
