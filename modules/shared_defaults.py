from modules.logger import log
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
                default_offload_mode = "balanced"
                default_diffusers_offload_min_gpu_memory = 0
                log.info(f"Device detect: memory={gpu_memory:.1f} default=balanced optimization=lowvram")
            elif gpu_memory <= 12:
                cmd_opts.medvram = True # VAE Tiling and other stuff
                default_offload_mode = "balanced"
                default_diffusers_offload_min_gpu_memory = 0
                default_diffusers_offload_always = ', '.join(['T5EncoderModel', 'UMT5EncoderModel'])
                log.info(f"Device detect: memory={gpu_memory:.1f} default=balanced optimization=medvram")
            elif gpu_memory >= 24:
                default_offload_mode = "balanced"
                default_diffusers_offload_max_gpu_memory = 0.8
                default_diffusers_offload_always = ', '.join(['T5EncoderModel', 'UMT5EncoderModel'])
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

    default_sdp_choices = ['Flash', 'Memory', 'Math']
    default_sdp_options = ['Flash', 'Memory', 'Math']

    default_sdp_override_choices = ['Dynamic attention', 'Flex attention', 'Flash attention', 'Sage attention']
    default_sdp_override_options = []

    if devices.backend == "zluda":
        default_sdp_options = ['Math']
        default_sdp_override_options = ['Dynamic attention']
        default_sdp_override_choices.append('Triton Flash attention')
    elif devices.backend == "rocm":
        default_sdp_override_choices.append('Triton Flash attention')
        agent = devices.get_hip_agent()
        if agent.gfx_version < 0x1100:
            default_sdp_override_options = ['Dynamic attention'] # only RDNA2 and older GPUs needs this
    elif devices.backend in {"directml", "cpu", "mps"}:
        default_sdp_override_options = ['Dynamic attention']

    return (
        default_offload_mode,
        default_diffusers_offload_min_gpu_memory,
        default_diffusers_offload_max_gpu_memory,
        default_cross_attention,
        default_sdp_options,
        default_sdp_choices,
        default_sdp_override_options,
        default_sdp_override_choices,
        default_diffusers_offload_always,
        default_diffusers_offload_never,
    )
