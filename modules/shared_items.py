import diffusers


class OnlinePipeline(diffusers.DiffusionPipeline):
    pass


pipelines = {
    'Autodetect': None,
    'AutoPipeline': diffusers.AutoPipelineForText2Image,
    'Diffusion': diffusers.DiffusionPipeline,

    # standard diffusers pipelines
    'Stable Diffusion': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion Inpaint': getattr(diffusers, 'StableDiffusionInpaintPipeline', None),
    'Stable Diffusion Instruct': getattr(diffusers, 'StableDiffusionInstructPix2PixPipeline', None),
    'Stable Diffusion 1.5': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion 2': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion 2.x': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion Upscale': getattr(diffusers, 'StableDiffusionUpscalePipeline', None),
    'Stable Diffusion XL': getattr(diffusers, 'StableDiffusionXLPipeline', None),
    'Stable Diffusion XL Inpaint': getattr(diffusers, 'StableDiffusionXLInpaintPipeline', None),
    'Stable Diffusion XL Instruct': getattr(diffusers, 'StableDiffusionXLInstructPix2PixPipeline', None),
    'Stable Diffusion XL Refiner': getattr(diffusers, 'StableDiffusionXLImg2ImgPipeline', None),
    'Stable Cascade': getattr(diffusers, 'StableCascadeCombinedPipeline', None),
    'Stable Diffusion 3': getattr(diffusers, 'StableDiffusion3Pipeline', None),
    'Latent Consistency Model': getattr(diffusers, 'LatentConsistencyModelPipeline', None),
    'AuraFlow': getattr(diffusers, 'AuraFlowPipeline', None),
    'Chroma': getattr(diffusers, 'ChromaPipeline', None),
    'ChronoEdit': getattr(diffusers, 'ChronoEditPipeline', None),
    'CogView3': getattr(diffusers, 'CogView3PlusPipeline', None),
    'CogView4': getattr(diffusers, 'CogView4Pipeline', None),
    'Cosmos': getattr(diffusers, 'Cosmos2TextToImagePipeline', None),
    'DeepFloydIF': getattr(diffusers, 'IFPipeline', None),
    'ERNIEImage': getattr(diffusers, 'ErnieImagePipeline', None),
    'FLUX': getattr(diffusers, 'FluxPipeline', None),
    'FLUX2Klein': getattr(diffusers, 'Flux2KleinPipeline', None),
    'FLUX2': getattr(diffusers, 'Flux2Pipeline', None),
    'GLMImage': getattr(diffusers, 'GlmImagePipeline', None),
    'HiDream': getattr(diffusers, 'HiDreamImagePipeline', None),
    'HunyuanDiT': getattr(diffusers, 'HunyuanDiTPipeline', None),
    'HunyuanImage': getattr(diffusers, 'HunyuanImagePipeline', None),
    'JoyEdit': getattr(diffusers, 'JoyImageEditPipeline', None),
    'Kandinsky21': getattr(diffusers, 'KandinskyCombinedPipeline', None),
    'Kandinsky22': getattr(diffusers, 'KandinskyV22CombinedPipeline', None),
    'Kandinsky30': getattr(diffusers, 'Kandinsky3Pipeline', None),
    'Kandinsky50': getattr(diffusers, 'Kandinsky5T2IPipeline', None),
    'Kolors': getattr(diffusers, 'KolorsPipeline', None),
    'LongCat': getattr(diffusers, 'LongCatImagePipeline', None),
    'Lumina2': getattr(diffusers, 'Lumina2Pipeline', None),
    'LuminaNext': getattr(diffusers, 'LuminaText2ImgPipeline', None),
    'NucleusImage': getattr(diffusers, 'NucleusMoEImagePipeline', None),
    'OvisImage': getattr(diffusers, 'OvisImagePipeline', None),
    'OmniGen': getattr(diffusers, 'OmniGenPipeline', None),
    'PixArtAlpha': getattr(diffusers, 'PixArtAlphaPipeline', None),
    'PixArtSigma': getattr(diffusers, 'PixArtSigmaPipeline', None),
    'Qwen': getattr(diffusers, 'QwenImagePipeline', None),
    'Sana': getattr(diffusers, 'SanaPipeline', None),
    'WanAI': getattr(diffusers, 'WanPipeline', None),
    'ZImage': getattr(diffusers, 'ZImagePipeline', None),

    # pipelines with custom code that is fetched online
    'InstaFlow': OnlinePipeline,
    'Anima': OnlinePipeline,

    # sdnext custom pipelines are dynamically imported and redefined later
    'Bria': None,
    'FLite': None,
    'FLEX': None,
    'HiDreamO1': None,
    'HunyuanImage3': None,
    'Lens': None,
    'LuminaDiMOO': None,
    'Meissonic': None,
    'OmniGen2': None,
    'SDXS': None,
    'SegMoE': None,
    'Step1XEdit': None,
    'UltraFlux': None,
    'VIBE': None,
    'XOmni': None,
    'ZetaChroma': None,
}


def postprocessing_scripts():
    import modules.scripts_manager
    return modules.scripts_manager.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae
    return ["Automatic", "Default"] + list(modules.sd_vae.vae_dict)


def sd_taesd_items():
    import modules.vae.sd_vae_taesd
    return list(modules.vae.sd_vae_taesd.TAESD_MODELS.keys()) + list(modules.vae.sd_vae_taesd.CQYAN_MODELS.keys())

def refresh_vae_list():
    import modules.sd_vae
    modules.sd_vae.refresh_vae_list()


def sd_unet_items():
    import modules.sd_unet
    return ['Default'] + list(modules.sd_unet.unet_dict)


def refresh_unet_list():
    import modules.sd_unet
    modules.sd_unet.refresh_unet_list()


def sd_te_items():
    import modules.model_te
    predefined = ['Default']
    return predefined + list(modules.model_te.te_dict)


def refresh_te_list():
    import modules.model_te
    modules.model_te.refresh_te_list()


def list_crossattention():
    return [
        "Disabled",
        "Scaled-Dot-Product",
        "xFormers",
        "Batch matrix-matrix",
        "Dynamic Attention BMM"
    ]


def get_pipelines():
    from modules.logger import log
    if hasattr(diffusers, 'OnnxStableDiffusionPipeline') and 'ONNX Stable Diffusion' not in list(pipelines):
        try:
            from modules.onnx_impl import initialize_onnx
            initialize_onnx()
            onnx_pipelines = {
                'ONNX Stable Diffusion': getattr(diffusers, 'OnnxStableDiffusionPipeline', None),
                'ONNX Stable Diffusion Img2Img': getattr(diffusers, 'OnnxStableDiffusionImg2ImgPipeline', None),
                'ONNX Stable Diffusion Inpaint': getattr(diffusers, 'OnnxStableDiffusionInpaintPipeline', None),
                'ONNX Stable Diffusion Upscale': getattr(diffusers, 'OnnxStableDiffusionUpscalePipeline', None),
            }
        except Exception as e:
            log.error(f'ONNX initialization error: {e}')
            onnx_pipelines = {}
        pipelines.update(onnx_pipelines)
    stats_builtin = 0
    stats_custom = 0
    for k, v in pipelines.copy().items():
        if k != 'Autodetect' and v is None:
            stats_custom += 1
            pipelines[k] = diffusers.DiffusionPipeline
        else:
            stats_builtin += 1
    if stats_custom > 0:
        log.debug(f'Pipelines init: diffusers={stats_builtin} custom={stats_custom}')
    else:
        log.debug(f'Pipelines init: verified={stats_builtin}')
    return pipelines


def get_repo(model):
    if model == 'StableDiffusionPipeline' or model == 'Stable Diffusion 1.5':
        return 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    elif model == 'StableDiffusionXLPipeline' or model == 'Stable Diffusion XL':
        return 'stabilityai/stable-diffusion-xl-base-1.0'
    elif model == 'StableDiffusion3Pipeline' or model == 'Stable Diffusion 3':
        return 'stabilityai/stable-diffusion-3.5-medium'
    else:
        return None


sdnq_quant_modes = ["int8", "int7", "int6", "uint5", "uint4", "uint3", "uint2", "float8_e4m3fn", "float8_e3m4fn", "float7_e3m3fn", "float6_e3m2fn", "float5_e2m2fn", "float4_e2m1fn", "float3_e1m1fn", "float2_e1m0fn"]
sdnq_matmul_modes = ["auto", "int8", "float8_e4m3fn", "float16"]
