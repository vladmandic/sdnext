import sys
import torch
import numpy as np
from PIL import Image
from modules.logger import log


def to_tensor(image: Image.Image | np.ndarray):
    """PIL Image -> float32 CHW tensor [0,1]. Pure torch, no torchvision."""
    if isinstance(image, Image.Image):
        pic = np.array(image, copy=True)
    elif isinstance(image, np.ndarray):
        pic = image.copy()
    else:
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        raise TypeError(f"convert: target=tensor type={type(image)} fn={fn} unsupported")
    if pic.ndim == 2:
        pic = pic[:, :, np.newaxis]
    tensor = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
    # log.debug(f'Convert: source={type(image)} target={tensor.shape} fn={fn}')
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div_(255.0)
    return tensor.to(torch.float32)


def to_pil(tensor: torch.Tensor | np.ndarray):
    """Float CHW/HWC or BCHW/BHWC tensor [0,1] -> PIL Image. Pure torch, no torchvision."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    elif isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    else:
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        raise TypeError(f"convert: target=image type={type(tensor)} fn={fn} unsupported")
    try:
        if tensor.dim() == 4:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[-1] < tensor.shape[-2]:  # BHWC
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor[0]
        elif tensor.dim() == 3:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[-1] < tensor.shape[-2] and tensor.shape[-1] < tensor.shape[-3]:  # HWC
                tensor = tensor.permute(2, 0, 1)
        if tensor.dtype != torch.uint8:
            tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
        ndarr = tensor.permute(1, 2, 0).numpy()
        if ndarr.shape[2] == 1:
            ndarr = ndarr[:, :, 0]
            mode = 'L'
        elif ndarr.shape[2] == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'
        image = Image.fromarray(ndarr, mode=mode)
    except Exception as e:
        image = Image.new('RGB', (tensor.shape[-1], tensor.shape[-2]), color=(152, 32, 48))
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.error(f'Convert: source={type(tensor)} target={image} fn={fn} {e}')
    return image


def pil_to_tensor(image):
    """PIL Image -> uint8 CHW tensor (no float scaling). Replaces TF.pil_to_tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    pic = np.array(image, copy=True)
    if pic.ndim == 2:
        pic = pic[:, :, np.newaxis]
    return torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()


def normalize(tensor, mean, std, inplace=False):
    """Tensor normalization. Replaces TF.normalize."""
    if not inplace:
        tensor = tensor.clone()
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean_t.ndim == 1:
        mean_t = mean_t[:, None, None]
    if std_t.ndim == 1:
        std_t = std_t[:, None, None]
    tensor.sub_(mean_t).div_(std_t)
    return tensor
