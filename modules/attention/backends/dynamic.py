from modules.attention.registry import AttentionBackend, Platform


def prepare(platform: Platform, original): # pylint: disable=unused-argument
    from modules import devices
    devices.sdpa_pre_dyanmic_atten = original # the sliced path calls this pin for every slice
    from modules.sd_hijack_dynamic_atten import dynamic_scaled_dot_product_attention
    return dynamic_scaled_dot_product_attention


backend = AttentionBackend(name='dynamic', label='Dynamic attention', priority=10, prepare=prepare, terminal=True)
