import numpy as np
import torch
from modules import images_sharpfin
import PIL


JPEG_QUALITY = 95


def preprocess(image, processor, **kwargs):
    if isinstance(image, PIL.Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = images_sharpfin.to_pil(image)
    else:
        raise TypeError(f"Image must be of type PIL.Image, np.ndarray, or torch.Tensor, got {type(image)} instead.")

    image = processor.preprocess(image, **kwargs)
    return image
