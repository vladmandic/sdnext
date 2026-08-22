"""Attention backends: one scaled_dot_product_attention router over the registered backends, the per-generation context, and the diffusers-side processor and dispatcher setup."""
from modules.attention.registry import AttentionBackend, AttentionCall, Constraints, Platform, Registry, registry
from modules.attention.router import Plan, PlanEntry, build_plan, get_plan, install_router
from modules.attention.dispatcher import set_diffusers_attention, set_attention_dispatcher, hijack_kernels, get_kernel_hijack, get_hf_api_hijack
from modules.attention import backends, context

__all__ = [
    'AttentionBackend', 'AttentionCall', 'Constraints', 'Platform', 'Registry', 'registry',
    'Plan', 'PlanEntry', 'build_plan', 'get_plan', 'install_router',
    'set_diffusers_attention', 'set_attention_dispatcher', 'hijack_kernels', 'get_kernel_hijack', 'get_hf_api_hijack',
    'backends', 'context',
]
