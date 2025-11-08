from typing import Optional
from functools import wraps
import torch
from modules import rocm
from modules.errors import log
from installer import install, installed


def set_dynamic_attention():
    try:
        sdpa_pre_dyanmic_atten = torch.nn.functional.scaled_dot_product_attention
        from modules.sd_hijack_dynamic_atten import dynamic_scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = dynamic_scaled_dot_product_attention
        return sdpa_pre_dyanmic_atten
    except Exception as err:
        log.error(f'Torch attention: type="dynamic attention" {err}')
        return None

def set_triton_flash_attention(backend: str):
    try:
        if backend in {"zluda", "rocm"}:
            from modules.flash_attn_triton_amd import interface_fa
            sdpa_pre_triton_flash_atten = torch.nn.functional.scaled_dot_product_attention
            @wraps(sdpa_pre_triton_flash_atten)
            def sdpa_triton_flash_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: Optional[torch.FloatTensor] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
                if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
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
                else:
                    if enable_gqa:
                        kwargs["enable_gqa"] = enable_gqa
                    return sdpa_pre_triton_flash_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
            torch.nn.functional.scaled_dot_product_attention = sdpa_triton_flash_atten
            log.debug('Torch attention: type="triton flash attention"')
    except Exception as err:
        log.error(f'Torch attention: type="triton flash attention" {err}')

def set_ck_flash_attention(backend: str, device: torch.device):
    try:
        if backend == "rocm":
            if not installed('flash-attn'):
                log.info('Building CK Flash attention...')
                agent = rocm.Agent(getattr(torch.cuda.get_device_properties(device), "gcnArchName", "gfx0000"))
                install(rocm.get_flash_attention_command(agent), reinstall=True)
        else:
            install('flash-attn')
        from flash_attn import flash_attn_func
        sdpa_pre_flash_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_flash_atten)
        def sdpa_flash_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: Optional[torch.FloatTensor] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
            if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
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
            else:
                if enable_gqa:
                    kwargs["enable_gqa"] = enable_gqa
                return sdpa_pre_flash_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = sdpa_flash_atten
        log.debug('Torch attention: type="ck flash attention"')
    except Exception as err:
        log.error(f'Torch attention: type="ck flash attention" {err}')

def set_sage_attention(backend: str, device: torch.device):
    try:
        install('sageattention')

        use_cuda_backend = False
        if (backend == "cuda") and (torch.cuda.get_device_capability(device) == (8, 6)):
            use_cuda_backend = True # Detect GPU architecture - sm86 confirmed to need CUDA backend workaround as Sage Attention + Triton causes NaNs
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_cuda
            except:
                use_cuda_backend = False

        if use_cuda_backend:
            log.debug('Torch attention: type=SageAttention backend=cuda')
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
            log.debug('Torch attention: type=SageAttention backend=auto')
            from sageattention import sageattn
            def sage_attn_impl(query, key, value, is_causal, scale):
                return sageattn(
                    q=query, k=key, v=value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=scale,
                )

        sdpa_pre_sage_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_sage_atten)
        def sdpa_sage_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: Optional[torch.FloatTensor] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
            if (query.shape[-1] in {128, 96, 64}) and (attn_mask is None) and (query.dtype != torch.float32):
                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

                # Call pre-selected sage attention implementation
                return sage_attn_impl(query, key, value, is_causal, scale)
            else:
                if enable_gqa:
                    kwargs["enable_gqa"] = enable_gqa
                return sdpa_pre_sage_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = sdpa_sage_atten
        log.debug('Torch attention: type="sage attention"')
    except Exception as err:
        log.error(f'Torch attention: type="sage attention" {err}')
