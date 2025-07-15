import diffusers.pipelines as p


def is_compatible(model, compatible):
    if model is None:
        return False
    if hasattr(model, '__class__'):
        return any(model.__class__.__name__ == c.__name__ for c in compatible)
    return any(isinstance(model, c) for c in compatible)


def is_sd15(model):
    compatible = [
        p.StableDiffusionPipeline,
        p.StableDiffusionImg2ImgPipeline,
        p.StableDiffusionInpaintPipeline,
        p.StableDiffusionControlNetPipeline,
    ]
    return is_compatible(model, compatible)


def is_sdxl(model):
    compatible = [
        p.StableDiffusionXLPipeline,
        p.StableDiffusionXLImg2ImgPipeline,
        p.StableDiffusionXLInpaintPipeline,
        p.StableDiffusionXLControlNetPipeline,
        p.StableDiffusionXLControlNetImg2ImgPipeline,
        p.StableDiffusionXLControlNetUnionPipeline,
    ]
    return is_compatible(model, compatible)


def is_f1(model):
    compatible = [
        p.FluxPipeline,
        p.FluxImg2ImgPipeline,
        p.FluxInpaintPipeline,
        p.FluxControlNetPipeline,
    ]
    return is_compatible(model, compatible)


def is_sd3(model):
    compatible = [
        p.StableDiffusion3Pipeline,
        p.StableDiffusion3Img2ImgPipeline,
        p.StableDiffusion3InpaintPipeline,
        p.StableDiffusion3ControlNetPipeline,
    ]
    return is_compatible(model, compatible)
