import diffusers


pipelines = {
    # note: not all pipelines can be used manually as they require prior pipeline next to decoder pipeline
    'Autodetect': None,
    'Custom Diffusers Pipeline': getattr(diffusers, 'DiffusionPipeline', None),

    # standard pipelines
    'Diffusion': getattr(diffusers, 'DiffusionPipeline', None),
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
    'HunyuanImage': getattr(diffusers, 'HunyuanImagePipeline', None),
    'Z-Image': getattr(diffusers, 'ZImagePipeline', None),
    'FLUX2': getattr(diffusers, 'Flux2Pipeline', None),
    'FLUX2 Klein': getattr(diffusers, 'Flux2KleinPipeline', None),
    'LongCat': getattr(diffusers, 'LongCatImagePipeline', None),
    'GLM-Image': getattr(diffusers, 'GlmImagePipeline', None),
    # dynamically imported and redefined later
    'Meissonic': getattr(diffusers, 'DiffusionPipeline', None),
    'Monetico': getattr(diffusers, 'DiffusionPipeline', None),
    'OmniGen2': getattr(diffusers, 'DiffusionPipeline', None),
    'InstaFlow': getattr(diffusers, 'DiffusionPipeline', None),
    'SegMoE': getattr(diffusers, 'DiffusionPipeline', None),
    'FLite': getattr(diffusers, 'DiffusionPipeline', None),
    'Bria': getattr(diffusers, 'DiffusionPipeline', None),
    'hdm': getattr(diffusers, 'DiffusionPipeline', None),
    'X-Omni': getattr(diffusers, 'DiffusionPipeline', None),
    'HunyuanImage3': getattr(diffusers, 'DiffusionPipeline', None),
    'ChronoEdit': getattr(diffusers, 'DiffusionPipeline', None),
    'Anima': getattr(diffusers, 'DiffusionPipeline', None),
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
            from modules.logger import log
            log.error(f'ONNX initialization error: {e}')
            onnx_pipelines = {}
        pipelines.update(onnx_pipelines)
    for k, v in pipelines.items():
        if k != 'Autodetect' and v is None:
            log.error(f'Model="{k}" diffusers={diffusers.__version__} path={diffusers.__file__} pipeline not available')
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


sdnq_quant_modes = ["int8", "int7", "int6", "uint5", "uint4", "uint3", "uint2", "float8_e4m3fn", "float7_e3m3fn", "float6_e3m2fn", "float5_e2m2fn", "float4_e2m1fn", "float3_e1m1fn", "float2_e1m0fn"]
sdnq_matmul_modes = ["auto", "int8", "float8_e4m3fn", "float16"]
