"""Attention backends: the SDPA hijacks stacked by devices.set_sdpa_params, plus the diffusers-side processor and dispatcher setup."""
from modules.attention.hijacks import set_dynamic_attention, set_sdnq_attention, set_triton_flash_attention, set_flex_attention, set_ck_flash_attention, set_sage_attention
from modules.attention.dispatcher import set_diffusers_attention, set_attention_dispatcher, hijack_kernels, get_kernel_hijack, get_hf_api_hijack

__all__ = [
    'set_dynamic_attention', 'set_sdnq_attention', 'set_triton_flash_attention', 'set_flex_attention', 'set_ck_flash_attention', 'set_sage_attention',
    'set_diffusers_attention', 'set_attention_dispatcher', 'hijack_kernels', 'get_kernel_hijack', 'get_hf_api_hijack',
]
