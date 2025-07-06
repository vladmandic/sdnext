from typing import List

import math
import torch
import torchvision
import numpy as np

from PIL import Image
from diffusers.utils import CONFIG_NAME
from diffusers.image_processor import PipelineImageInput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from transformers import ImageProcessingMixin


def img_to_pixelart(image: PipelineImageInput, sharpen: float = 0, block_size: int = 8, return_type: str = "pil", device: torch.device = "cpu") -> PipelineImageInput:
    block_size_sq = block_size * block_size
    processor = JPEGEncoder(block_size=block_size, cbcr_downscale=1)
    new_image = processor.encode(image, device=device)
    y = new_image[:,0,:,:].unsqueeze(1)
    cb = new_image[:,block_size_sq,:,:].unsqueeze(1)
    cr = new_image[:,block_size_sq*2,:,:].unsqueeze(1)

    if sharpen > 0:
        ycbcr = torch.cat([y,cb,cr], dim=1)
        laplacian_kernel = torch.tensor(
            [
                [[[ 0, 1, 0], [1, -4, 1], [ 0, 1, 0]]],
                [[[ 0, 1, 0], [1, -4, 1], [ 0, 1, 0]]],
                [[[ 0, 1, 0], [1, -4, 1], [ 0, 1, 0]]],
            ],
            dtype=torch.float32,
        ).to(device)
        ycbcr = ycbcr - (sharpen * torch.nn.functional.conv2d(ycbcr, laplacian_kernel, padding=1, groups=3))
        y = ycbcr[:,0,:,:].unsqueeze(1)
        cb = ycbcr[:,1,:,:].unsqueeze(1)
        cr = ycbcr[:,2,:,:].unsqueeze(1)

    new_image = torch.zeros_like(new_image)
    new_image[:,0,:,:] = y
    new_image[:,block_size_sq,:,:] = cb
    new_image[:,block_size_sq*2,:,:] = cr
    new_image = processor.decode(new_image, return_type=return_type)
    return new_image


def edge_detect_for_pixelart(image: PipelineImageInput, image_weight: float = 1.0, block_size: int = 8, device: torch.device = "cpu") -> torch.Tensor:
    block_size_sq = block_size * block_size
    new_image = process_image_input(image).to(device, dtype=torch.float32) / 255
    new_image = new_image.permute(0,3,1,2)
    batch_size, channels, height, width = new_image.shape

    min_pool = -torch.nn.functional.max_pool2d(-new_image, block_size, 1, block_size//2, 1, False, False)
    min_pool = min_pool[:, :, :height, :width]

    greyscale = (new_image[:,0,:,:] * 0.299) + (new_image[:,1,:,:] * 0.587) + (new_image[:,2,:,:] * 0.114)
    greyscale = greyscale[:, :(new_image.shape[-2]//block_size)*block_size, :(new_image.shape[-1]//block_size)*block_size] # crop to a multiple of block_size
    greyscale_reshaped = greyscale.reshape(batch_size, block_size, height // block_size, block_size, width // block_size)
    greyscale_reshaped = greyscale_reshaped.permute(0,1,3,2,4)
    greyscale_reshaped = greyscale_reshaped.reshape(batch_size, block_size_sq, height // block_size, width // block_size)

    greyscale_median = greyscale.median()
    greyscale_max = greyscale_reshaped.amax(dim=1, keepdim=True)
    greyscale_min = greyscale_reshaped.amin(dim=1, keepdim=True)

    upsample = torchvision.transforms.Resize((height, width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    range_weight = upsample(greyscale_max - greyscale_min)
    range_weight = range_weight / range_weight.max()
    weight_map = upsample((greyscale > greyscale_median).to(dtype=torch.float32))
    weight_map = (weight_map / 2) + (range_weight / 2)
    weight_map = weight_map * image_weight

    new_image = (new_image * weight_map) + (min_pool * (1-weight_map))
    new_image = new_image.permute(0,2,3,1).clamp(0, 1) * 255
    return new_image


def rgb_to_ycbcr_tensor(image: torch.ByteTensor) -> torch.FloatTensor:
    img = image.float() / 255
    y = (img[:,:,:,0] * 0.299) + (img[:,:,:,1] * 0.587) + (img[:,:,:,2] * 0.114)
    cb = 0.5 + (img[:,:,:,0] * -0.168935) + (img[:,:,:,1] * -0.331665) + (img[:,:,:,2] * 0.50059)
    cr = 0.5 + (img[:,:,:,0] * 0.499813) + (img[:,:,:,1] * -0.418531) + (img[:,:,:,2] * -0.081282)
    ycbcr = torch.stack([y,cb,cr], dim=1)
    ycbcr = (ycbcr - 0.5) * 2
    return ycbcr


def ycbcr_tensor_to_rgb(ycbcr: torch.FloatTensor) -> torch.ByteTensor:
    ycbcr_img = (ycbcr / 2) + 0.5
    y = ycbcr_img[:,0,:,:]
    cb = ycbcr_img[:,1,:,:] - 0.5
    cr = ycbcr_img[:,2,:,:] - 0.5

    r = y + (cr * 1.402525)
    g = y + (cb * -0.343730) + (cr * -0.714401)
    b = y + (cb * 1.769905) + (cr * 0.000013)
    rgb = torch.stack([r,g,b], dim=-1).clamp(0,1)
    rgb = (rgb*255).to(torch.uint8)
    return rgb


def encode_single_channel_dct_2d(img: torch.FloatTensor, block_size: int=16, norm: str='ortho') -> torch.FloatTensor:
    batch_size, height, width = img.shape
    h_blocks = int(height//block_size)
    w_blocks = int(width//block_size)

    # batch_size, h_blocks, w_blocks, block_size_h, block_size_w
    dct_tensor = img.view(batch_size, h_blocks, block_size, w_blocks, block_size).transpose(2,3).float()
    dct_tensor = dct_2d(dct_tensor, norm=norm)

    # batch_size, combined_block_size, h_blocks, w_blocks
    dct_tensor = dct_tensor.reshape(batch_size, h_blocks, w_blocks, block_size*block_size).permute(0,3,1,2)
    return dct_tensor


def decode_single_channel_dct_2d(img: torch.FloatTensor, norm: str='ortho') -> torch.FloatTensor:
    batch_size, combined_block_size, h_blocks, w_blocks = img.shape
    block_size = int(math.sqrt(combined_block_size))
    height = int(h_blocks*block_size)
    width = int(w_blocks*block_size)

    img_tensor = img.permute(0,2,3,1).view(batch_size, h_blocks, w_blocks, block_size, block_size)
    img_tensor = idct_2d(img_tensor, norm=norm)
    img_tensor = img_tensor.permute(0,1,3,2,4).reshape(batch_size, height, width)
    return img_tensor


def encode_jpeg_tensor(img: torch.FloatTensor, block_size: int=16, cbcr_downscale: int=2, norm: str='ortho') -> torch.FloatTensor:
    img = img[:, :, :(img.shape[-2]//block_size)*block_size, :(img.shape[-1]//block_size)*block_size] # crop to a multiply of block_size
    _, _, height, width = img.shape
    downsample = torchvision.transforms.Resize((height//cbcr_downscale, width//cbcr_downscale), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    down_img = downsample(img[:, 1:,:,:])
    y = encode_single_channel_dct_2d(img[:, 0, :,:], block_size=block_size, norm=norm)
    cb = encode_single_channel_dct_2d(down_img[:, 0, :,:], block_size=block_size//cbcr_downscale, norm=norm)
    cr = encode_single_channel_dct_2d(down_img[:, 1, :,:], block_size=block_size//cbcr_downscale, norm=norm)
    return torch.cat([y,cb,cr], dim=1)


def decode_jpeg_tensor(jpeg_img: torch.FloatTensor, block_size: int=16, cbcr_downscale: int=2, norm: str='ortho') -> torch.FloatTensor:
    _, _, h_blocks, w_blocks = jpeg_img.shape
    y_block_size = block_size*block_size
    cbcr_block_size = int((block_size//cbcr_downscale)*(block_size//cbcr_downscale))
    y = jpeg_img[:, :y_block_size]
    cb = jpeg_img[:, y_block_size:y_block_size+cbcr_block_size]
    cr = jpeg_img[:, y_block_size+cbcr_block_size:]
    y = decode_single_channel_dct_2d(y, norm=norm)
    cb = decode_single_channel_dct_2d(cb, norm=norm)
    cr = decode_single_channel_dct_2d(cr, norm=norm)
    upsample = torchvision.transforms.Resize((h_blocks*block_size, w_blocks*block_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    cb = upsample(cb)
    cr = upsample(cr)
    return torch.stack([y,cb,cr], dim=1)


def process_image_input(images: PipelineImageInput) -> torch.ByteTensor:
    if isinstance(images, list):
        combined_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img = torch.from_numpy(np.asarray(img).copy()).unsqueeze(0)
                combined_images.append(img)
            elif isinstance(img, np.ndarray):
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                img = torch.from_numpy(img)
                combined_images.append(img)
            elif isinstance(img, torch.Tensor):
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                combined_images.append(img)
            else:
                raise RuntimeError(f"Invalid input! Given: {type(img)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")
        combined_images = torch.cat(combined_images, dim=0)
    elif isinstance(images, Image.Image):
        combined_images = torch.from_numpy(np.asarray(images).copy()).unsqueeze(0)
    elif isinstance(images, np.ndarray):
        combined_images = torch.from_numpy(images)
        if len(combined_images.shape) == 3:
            combined_images = combined_images.unsqueeze(0)
    elif isinstance(images, torch.Tensor):
        combined_images = images
        if len(combined_images.shape) == 3:
            combined_images = combined_images.unsqueeze(0)
    else:
        raise RuntimeError(f"Invalid input! Given: {type(images)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")
    return combined_images


class JPEGEncoder(ImageProcessingMixin, ConfigMixin):

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        block_size: int = 16,
        cbcr_downscale: int = 2,
        norm: str = "ortho",
        latents_std: List[float] = None,
        latents_mean: List[float] = None,
    ):
        self.block_size = block_size
        self.cbcr_downscale = cbcr_downscale
        self.norm = norm
        self.latents_std = latents_std
        self.latents_mean = latents_mean


    def encode(self, images: PipelineImageInput, device: str="cpu") -> torch.FloatTensor:
        """
        Encode RGB 0-255 image to JPEG Latents.

        Args:
            image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
                The image input, can be a PIL image, numpy array or pytorch tensor.
                Must be an RGB image or a list of RGB images with 0-255 range and (batch_size, height, width, channels) shape.

        Returns:
            `torch.Tensor`:
                The encoded JPEG Latents.
        """

        combined_images = process_image_input(images).to(device)
        latents = rgb_to_ycbcr_tensor(combined_images)
        latents = encode_jpeg_tensor(latents, block_size=self.block_size, cbcr_downscale=self.cbcr_downscale, norm=self.norm)

        if self.latents_mean is not None:
            latents = latents - torch.tensor(self.latents_mean, device=device, dtype=torch.float32).view(1,-1,1,1)
        if self.latents_std is not None:
            latents = latents / torch.tensor(self.latents_std, device=device, dtype=torch.float32).view(1,-1,1,1)

        return latents

    def decode(self, latents: torch.FloatTensor, return_type: str="pil") -> PipelineImageInput:
        latents = latents.to(dtype=torch.float32)
        if self.latents_std is not None:
            latents = latents * torch.tensor(self.latents_std, device=latents.device, dtype=torch.float32).view(1,-1,1,1)
        if self.latents_mean is not None:
            latents = latents + torch.tensor(self.latents_mean, device=latents.device, dtype=torch.float32).view(1,-1,1,1)

        images = decode_jpeg_tensor(latents, block_size=self.block_size, cbcr_downscale=self.cbcr_downscale, norm=self.norm)
        images = ycbcr_tensor_to_rgb(images)

        if return_type == "pt":
            return images
        elif return_type == "np":
            return images.detach().cpu().numpy()
        elif return_type == "pil":
            image_list = []
            for i in range(images.shape[0]):
                image_list.append(Image.fromarray(images[i].detach().cpu().numpy()))
            return image_list
        else:
            raise RuntimeError(f"Invalid return_type! Given: {return_type} should be in ('pt', 'np', 'pil')")


# dct functions are copied from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py (MIT license)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)
