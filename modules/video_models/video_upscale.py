import torch
from modules.logger import log


def upscale_video(pixels: torch.Tensor, scale: float = 1.0, upscaler: str = ""):
    log.debug(f'Upscale video: scale={scale} upscaler="{upscaler}" shape={list(pixels.shape)} TODO')
    return pixels
