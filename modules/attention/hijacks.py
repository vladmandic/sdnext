from functools import wraps
import torch
from modules import rocm
from modules.logger import log
from installer import install, installed, torch_info


def set_dynamic_attention():
    try:
        sdpa_pre_dyanmic_atten = torch.nn.functional.scaled_dot_product_attention
        from modules.sd_hijack_dynamic_atten import dynamic_scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = dynamic_scaled_dot_product_attention
        torch_info.set(attention='dynamic')
        return sdpa_pre_dyanmic_atten
    except Exception as err:
        log.error(f'Torch attention: type="dynamic attention" {err}')
        return None


def set_sdnq_attention():
    try:
        from modules import shared
        from sdnq.kernels.triton_atten import sdnq_triton_atten
        sdpa_pre_sdnq_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_sdnq_atten)
        def sdpa_sdnq_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.Tensor:
            if (
                query.device.type != "cpu"
                and (query.shape[-2] >= 32 and key.shape[-2] >= 32)
                and (query.shape[-2] > 512 or key.shape[-2] > 512) # Skip TE
                and query.shape[-3] > 1 # Skip VAE
            ):
                return sdnq_triton_atten(
                    query=query, key=key, value=value, attn_mask=attn_mask,
                    is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
                    matmul_dtype=shared.opts.sdnq_attention_matmul_type,
                    pv_matmul_dtype=shared.opts.sdnq_attention_pv_matmul_type,
                    smooth_k=shared.opts.sdnq_attention_smooth_k,
                    use_hadamard=shared.opts.sdnq_attention_use_hadamard,
                    hadamard_group_size=shared.opts.sdnq_attention_hadamard_group_size,
                    use_fp16_accum=shared.opts.sdnq_attention_use_fp16_accum,
                )
            else:
                if enable_gqa:
                    kwargs["enable_gqa"] = enable_gqa
                return sdpa_pre_sdnq_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = sdpa_sdnq_atten
        torch_info.set(attention='sdnq')
        log.debug(f'Torch attention: type="SDNQ attention" matmul={shared.opts.sdnq_attention_matmul_type}:{shared.opts.sdnq_attention_pv_matmul_type} smooth={shared.opts.sdnq_attention_smooth_k} hadamard={shared.opts.sdnq_attention_use_hadamard} fp16_accum={shared.opts.sdnq_attention_use_fp16_accum}')
    except Exception as err:
        log.error(f'Torch attention: type="SDNQ attention" {err}')


def set_triton_flash_attention(backend: str):
    try:
        if backend in {"rocm", "zluda"}: # flash_attn_triton_amd only works with AMD
            from modules.flash_attn_triton_amd import interface_fa

            sdpa_pre_triton_flash_atten = torch.nn.functional.scaled_dot_product_attention
            @wraps(sdpa_pre_triton_flash_atten)
            def sdpa_triton_flash_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.Tensor:
                use_triton = (
                    query.shape[-1] <= 128
                    and attn_mask is None
                    and query.device.type != "cpu"
                    and key.device == query.device
                    and value.device == query.device
                )
                if use_triton:
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
            torch_info.set(attention='triton')
            log.debug('Torch attention: type="Triton Flash attention"')
    except Exception as err:
        log.error(f'Torch attention: type="Triton Flash attention" {err}')


def set_flex_attention():
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        def flex_attention_causal_mask(b, h, q_idx, kv_idx): # pylint: disable=unused-argument
            return q_idx >= kv_idx

        sdpa_pre_flex_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_flex_atten)
        def sdpa_flex_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.Tensor: # pylint: disable=unused-argument
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
                block_mask = create_block_mask(flex_attention_causal_mask, query.shape[0], query.shape[1], query.shape[-2], key.shape[-2], device=query.device)
            return flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = sdpa_flex_atten
        torch_info.set(attention="flex")
        log.debug('Torch attention: type="Flex attention"')
    except Exception as err:
        log.error(f'Torch attention: type="Flex attention" {err}')


def set_ck_flash_attention(backend: str, device: torch.device):
    try:
        if backend == "rocm":
            if not installed('flash-attn'):
                log.info('Torch attention: type="Flash attention" building...')
                agent = rocm.Agent(device)
                install(rocm.get_flash_attention_command(agent), reinstall=True)
        else:
            install('flash-attn')
        from flash_attn import flash_attn_func

        sdpa_pre_flash_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_flash_atten)
        def sdpa_flash_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.Tensor:
            use_flash = (
                query.shape[-1] <= 128
                and attn_mask is None
                and query.dtype != torch.float32
                and query.device.type != "cpu"
                and key.device == query.device
                and value.device == query.device
            )
            if use_flash:
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
        torch_info.set(attention="flash")
        log.debug('Torch attention: type="Flash attention"')
    except Exception as err:
        log.error(f'Torch attention: type="Flash attention" {err}')


def set_sage_attention(backend: str, device: torch.device):
    try:
        install('sageattention')

        use_cuda_backend = False
        if (backend == "cuda") and (torch.cuda.get_device_capability(device) == (8, 6)):
            use_cuda_backend = True # Detect GPU architecture - sm86 confirmed to need CUDA backend workaround as Sage Attention + Triton causes NaNs
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

        sdpa_pre_sage_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_sage_atten)
        def sdpa_sage_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.Tensor:
            use_sage = (
                query.shape[-1] in {128, 96, 64}
                and attn_mask is None
                and query.device.type != "cpu"
                and key.device == query.device
                and value.device == query.device
            )
            if use_sage:
                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

                # Call preselected sage attention implementation
                return sage_attn_impl(query, key, value, is_causal, scale)
            else:
                if enable_gqa:
                    kwargs["enable_gqa"] = enable_gqa
                return sdpa_pre_sage_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = sdpa_sage_atten
        torch_info.set(attention="sage")
        log.debug(f'Torch attention: type="Sage attention" backend={"cuda" if use_cuda_backend else "auto"}')
    except Exception as err:
        log.error(f'Torch attention: type="Sage attention" {err}')
