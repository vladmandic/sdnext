def is_compatible(model, pattern='None'):
    if model is None:
        return False
    if hasattr(model, '__class__'):
        return model.__class__.__name__.startswith(pattern)
    return False


def is_sd15(model):
    return is_compatible(model, pattern='StableDiffusion')


def is_sdxl(model):
    return is_compatible(model, pattern='StableDiffusionXL')


def is_f1(model):
    return is_compatible(model, pattern='Flux')


def is_sd3(model):
    return is_compatible(model, pattern='StableDiffusion3Pipeline')

def is_qwen(model):
    return is_compatible(model, pattern='Qwen')
