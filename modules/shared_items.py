import diffusers


pipelines = {
    # note: not all pipelines can be used manually as they require prior pipeline next to decoder pipeline
    'Autodetect': None,
    'Custom Diffusers Pipeline': getattr(diffusers, 'DiffusionPipeline', None),

    # standard pipelines
    'Stable Diffusion': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion Inpaint': getattr(diffusers, 'StableDiffusionInpaintPipeline', None),
    'Stable Diffusion Instruct': getattr(diffusers, 'StableDiffusionInstructPix2PixPipeline', None),
    'Stable Diffusion 1.5': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion 2.x': getattr(diffusers, 'StableDiffusionPipeline', None),
    'Stable Diffusion Upscale': getattr(diffusers, 'StableDiffusionUpscalePipeline', None),
    'Stable Diffusion XL': getattr(diffusers, 'StableDiffusionXLPipeline', None),
    'Stable Diffusion XL Inpaint': getattr(diffusers, 'StableDiffusionXLInpaintPipeline', None),
    'Stable Diffusion XL Instruct': getattr(diffusers, 'StableDiffusionXLInstructPix2PixPipeline', None),
    'Stable Diffusion XL Refiner': getattr(diffusers, 'StableDiffusionXLImg2ImgPipeline', None),
    'Stable Cascade': getattr(diffusers, 'StableCascadeCombinedPipeline', None),
    'Stable Diffusion 3': getattr(diffusers, 'StableDiffusion3Pipeline', None),
    'Latent Consistency Model': getattr(diffusers, 'LatentConsistencyModelPipeline', None),
    'PixArt Alpha': getattr(diffusers, 'PixArtAlphaPipeline', None),
    'PixArt Sigma': getattr(diffusers, 'PixArtSigmaPipeline', None),
    'HunyuanDiT': getattr(diffusers, 'HunyuanDiTPipeline', None),
    'DeepFloyd IF': getattr(diffusers, 'IFPipeline', None),
    'FLUX': getattr(diffusers, 'FluxPipeline', None),
    'FLEX': getattr(diffusers, 'AutoPipelineForText2Image', None),
    'Chroma': getattr(diffusers, 'ChromaPipeline', None),
    'Sana': getattr(diffusers, 'SanaPipeline', None),
    'Lumina-Next': getattr(diffusers, 'LuminaText2ImgPipeline', None),
    'Lumina 2': getattr(diffusers, 'Lumina2Pipeline', None),
    'AuraFlow': getattr(diffusers, 'AuraFlowPipeline', None),
    'Kandinsky 2.1': getattr(diffusers, 'KandinskyCombinedPipeline', None),
    'Kandinsky 2.2': getattr(diffusers, 'KandinskyV22CombinedPipeline', None),
    'Kandinsky 3.0': getattr(diffusers, 'Kandinsky3Pipeline', None),
    'Wuerstchen': getattr(diffusers, 'WuerstchenCombinedPipeline', None),
    'Kolors': getattr(diffusers, 'KolorsPipeline', None),
    'CogView 3': getattr(diffusers, 'CogView3PlusPipeline', None),
    'CogView 4': getattr(diffusers, 'CogView4Pipeline', None),
    'UniDiffuser': getattr(diffusers, 'UniDiffuserPipeline', None),
    'Amused': getattr(diffusers, 'AmusedPipeline', None),
    'HiDream': getattr(diffusers, 'HiDreamImagePipeline', None),
    'OmniGen': getattr(diffusers, 'OmniGenPipeline', None),
    'Cosmos': getattr(diffusers, 'Cosmos2TextToImagePipeline', None),
    'WanAI': getattr(diffusers, 'WanPipeline', None),
    'Qwen': getattr(diffusers, 'QwenImagePipeline', None),

    # dynamically imported and redefined later
    'Meissonic': getattr(diffusers, 'DiffusionPipeline', None),
    'Monetico': getattr(diffusers, 'DiffusionPipeline', None),
    'OmniGen2': getattr(diffusers, 'DiffusionPipeline', None),
    'InstaFlow': getattr(diffusers, 'DiffusionPipeline', None),
    'SegMoE': getattr(diffusers, 'DiffusionPipeline', None),
    'FLite': getattr(diffusers, 'DiffusionPipeline', None),
    'Bria': getattr(diffusers, 'DiffusionPipeline', None),
}


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
    from installer import log
    log.error(f'ONNX initialization error: {e}')
    onnx_pipelines = {}


def postprocessing_scripts():
    import modules.scripts_manager
    return modules.scripts_manager.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae
    return ["Automatic", "Default"] + list(modules.sd_vae.vae_dict)


def sd_taesd_items():
    import modules.sd_vae_taesd
    return list(modules.sd_vae_taesd.TAESD_MODELS.keys()) + list(modules.sd_vae_taesd.CQYAN_MODELS.keys())

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
        "Split attention",
        "Dynamic Attention BMM"
    ]

def get_pipelines():
    if hasattr(diffusers, 'OnnxStableDiffusionPipeline') and 'ONNX Stable Diffusion' not in list(pipelines):
        pipelines.update(onnx_pipelines)
    for k, v in pipelines.items():
        if k != 'Autodetect' and v is None:
            from installer import log # pylint: disable=redefined-outer-name
            log.error(f'Not available: pipeline={k} diffusers={diffusers.__version__} path={diffusers.__file__}')
    return pipelines


def get_repo(model):
    if model == 'StableDiffusionPipeline' or model == 'Stable Diffusion 1.5':
        return 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    elif model == 'StableDiffusionXLPipeline' or model == 'Stable Diffusion XL':
        return 'stabilityai/stable-diffusion-xl-base-1.0'
    elif model == 'StableDiffusion3Pipeline' or model == 'Stable Diffusion 3':
        return 'stabilityai/stable-diffusion-3.5-medium'
    elif model == 'FluxPipeline' or model == 'FLUX':
        return 'black-forest-labs/FLUX.1-dev'
    else:
        return None
