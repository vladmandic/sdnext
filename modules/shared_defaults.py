from installer import log
from modules import devices


def get_default_modes(cmd_opts, mem_stat):
    default_offload_mode = "none"
    default_diffusers_offload_min_gpu_memory = 0.2
    default_diffusers_offload_max_gpu_memory = 0.6
    default_diffusers_offload_always = ''
    default_diffusers_offload_never = ''
    gpu_memory = round(mem_stat['gpu']['total'] if "gpu" in mem_stat else 0)
    if not (cmd_opts.lowvram or cmd_opts.medvram):
        if "gpu" in mem_stat and gpu_memory != 0:
            if gpu_memory <= 4:
                cmd_opts.lowvram = True
                default_offload_mode = "sequential"
                default_diffusers_offload_min_gpu_memory = 0
                log.info(f"Device detect: memory={gpu_memory:.1f} default=sequential optimization=lowvram")
            elif gpu_memory <= 12:
                cmd_opts.medvram = True # VAE Tiling and other stuff
                default_offload_mode = "balanced"
                default_diffusers_offload_min_gpu_memory = 0
                log.info(f"Device detect: memory={gpu_memory:.1f} default=balanced optimization=medvram")
            elif gpu_memory >= 24:
                default_offload_mode = "balanced"
                default_diffusers_offload_max_gpu_memory = 0.8
                default_diffusers_offload_never = ', '.join(['CLIPTextModel', 'CLIPTextModelWithProjection', 'AutoencoderKL'])
                log.info(f"Device detect: memory={gpu_memory:.1f} default=balanced optimization=highvram")
            else:
                default_offload_mode = "balanced"
                log.info(f"Device detect: memory={gpu_memory:.1f} default=balanced")
    elif cmd_opts.medvram:
        default_offload_mode = "balanced"
        default_diffusers_offload_min_gpu_memory = 0
    elif cmd_opts.lowvram:
        default_offload_mode = "sequential"
        default_diffusers_offload_min_gpu_memory = 0

    default_cross_attention = "Scaled-Dot-Product"

    if devices.backend == "zluda":
        default_sdp_options = ['Flash attention', 'Math attention', 'Dynamic attention']
    elif devices.backend in {"rocm", "directml", "cpu", "mps"}:
        default_sdp_options = ['Flash attention', 'Memory attention', 'Math attention', 'Dynamic attention']
    else:
        default_sdp_options = ['Flash attention', 'Memory attention', 'Math attention']

    return (
        default_offload_mode,
        default_diffusers_offload_min_gpu_memory,
        default_diffusers_offload_max_gpu_memory,
        default_cross_attention,
        default_sdp_options,
        default_diffusers_offload_always,
        default_diffusers_offload_never
    )
