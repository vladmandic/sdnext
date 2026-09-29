import numpy as np
from PIL import Image


def pad_to_multiple(image: np.ndarray, multiple: int = 8):
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, h, w


def crop_to_original(image: np.ndarray, h: int, w: int):
    return image[:h, :w]


def resize_by_short_side(image, target_short=512, resample=Image.Resampling.BICUBIC):
    w, h = image.size
    if min(w, h) < target_short:
        new_w = (w + 15) // 16 * 16
        new_h = (h + 15) // 16 * 16
    else:
        if w < h:
            new_w = target_short
            new_h = int(h * target_short / w)
        else:
            new_h = target_short
            new_w = int(w * target_short / h)
        new_w = (new_w + 15) // 16 * 16
        new_h = (new_h + 15) // 16 * 16

    return image.resize((new_w, new_h), resample=resample)
