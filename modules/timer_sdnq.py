def update_sdnq_attention_timers():
    from modules.timer import autotune
    autotune.reset()
    from modules.sdnq.kernels import triton_atten, triton_mm, triton_scaled_mm
    if getattr(triton_atten.sdnq_attn_kernel, 'bench_time', None) is not None:
        autotune.add('sdnq_attn_kernel', getattr(triton_atten.sdnq_attn_kernel, 'bench_time', 0))
        triton_atten.sdnq_attn_kernel.bench_time = 0
    if getattr(triton_mm.sdnq_triton_mm, 'bench_time', None) is not None:
        autotune.add('sdnq_triton_mm', getattr(triton_mm.sdnq_triton_mm, 'bench_time', 0))
        triton_mm.sdnq_triton_mm.bench_time = 0
    if getattr(triton_scaled_mm.sdnq_scaled_mm, 'bench_time', None) is not None:
        autotune.add('sdnq_scaled_mm', getattr(triton_scaled_mm.sdnq_scaled_mm, 'bench_time', 0))
        triton_scaled_mm.sdnq_scaled_mm.bench_time = 0
