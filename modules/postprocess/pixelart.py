from typing import List

import math
import torch
import numpy as np
from modules import images_sharpfin

from PIL import Image
from diffusers.utils import CONFIG_NAME
from diffusers.image_processor import PipelineImageInput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from transformers import ImageProcessingMixin

from modules import devices


@devices.inference_context()
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
        ycbcr = ycbcr.sub_(torch.nn.functional.conv2d(ycbcr, laplacian_kernel, padding=1, groups=3), alpha=sharpen)
        y = ycbcr[:,0,:,:].unsqueeze(1)
        cb = ycbcr[:,1,:,:].unsqueeze(1)
        cr = ycbcr[:,2,:,:].unsqueeze(1)

    new_image = torch.zeros_like(new_image)
    new_image[:,0,:,:] = y
    new_image[:,block_size_sq,:,:] = cb
    new_image[:,block_size_sq*2,:,:] = cr
    new_image = processor.decode(new_image, return_type=return_type)
    return new_image


@devices.inference_context()
def edge_detect_for_pixelart(image: PipelineImageInput, image_weight: float = 1.0, block_size: int = 8, device: torch.device = "cpu") -> torch.Tensor:
    block_size_sq = block_size * block_size
    new_image = process_image_input(image).to(device).to(dtype=torch.float32) / 255
    new_image = new_image.permute(0,3,1,2)
    batch_size, _channels, height, width = new_image.shape
    block_height = height // block_size
    block_width = width // block_size

    min_pool = 0 - torch.nn.functional.max_pool2d(-new_image, block_size, 1, block_size//2, 1, False, False)
    min_pool = min_pool[:, :, :height, :width]

    greyscale = (new_image[:,0,:,:] * 0.299).add_(new_image[:,1,:,:], alpha=0.587).add_(new_image[:,2,:,:], alpha=0.114)
    greyscale = greyscale[:, :(new_image.shape[-2]//block_size)*block_size, :(new_image.shape[-1]//block_size)*block_size] # crop to a multiple of block_size
    greyscale_reshaped = greyscale.reshape(batch_size, block_size, block_height, block_size, block_width)
    greyscale_reshaped = greyscale_reshaped.permute(0,1,3,2,4)
    greyscale_reshaped = greyscale_reshaped.reshape(batch_size, block_size_sq, block_height, block_width)

    greyscale_range = greyscale_reshaped.amax(dim=1, keepdim=True).sub_(greyscale_reshaped.amin(dim=1, keepdim=True))
    range_weight = images_sharpfin.resize_tensor(greyscale_range, (height, width), linearize=False)
    range_weight = range_weight.div_(range_weight.max())
    weight_map = images_sharpfin.resize_tensor((greyscale > greyscale.median()).to(dtype=torch.float32), (height, width), linearize=False)
    weight_map = weight_map.unsqueeze(0).add_(range_weight).mul_(image_weight / 2)

    new_image = new_image.mul_(weight_map).addcmul_(min_pool, (1-weight_map))
    new_image = new_image.permute(0,2,3,1).mul_(255).clamp_(0, 255)
    return new_image


def get_dct_harmonics(N: int, device: torch.device) -> torch.FloatTensor:
    k = torch.arange(N, dtype=torch.float32, device=device)
    spatial = torch.add(1, k.unsqueeze(1), alpha=2)
    spectral = k.unsqueeze(0) * (torch.pi / (2 * N))
    return torch.cos(torch.mm(spatial, spectral))


def get_dct_norm(N: int, device: torch.device) -> torch.FloatTensor:
    n = torch.ones((N, 1), dtype=torch.float32, device=device)
    n[0, 0] = 1 / math.sqrt(2)
    n = torch.mm(n, n.t())
    return n


def dct_2d(x: torch.FloatTensor, norm: str="ortho") -> torch.FloatTensor:
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N, N)

    h = get_dct_harmonics(N, x.device)
    coeff = torch.matmul(torch.matmul(h.t(), x), (h * (2 / N)))
    if norm == "ortho":
        coeff = torch.mul(coeff, get_dct_norm(N, x.device))

    coeff = coeff.view(x_shape)
    return coeff


def idct_2d(coeff: torch.FloatTensor, norm: str="ortho") -> torch.FloatTensor:
    x_shape = coeff.shape
    N = x_shape[-1]
    coeff = coeff.contiguous().view(-1, N, N)

    h = get_dct_harmonics(N, coeff.device)
    if norm == "ortho":
        coeff = torch.mul(coeff, get_dct_norm(N, coeff.device))
    x = torch.matmul(torch.matmul((h * (2 / N)), coeff), h.t())

    x = x.view(x_shape)
    return x


def encode_single_channel_dct_2d(img: torch.FloatTensor, block_size: int=16, norm: str="ortho") -> torch.FloatTensor:
    batch_size, height, width = img.shape
    h_blocks = int(height//block_size)
    w_blocks = int(width//block_size)

    # batch_size, h_blocks, w_blocks, block_size_h, block_size_w
    dct_tensor = img.view(batch_size, h_blocks, block_size, w_blocks, block_size).transpose(2,3).to(dtype=torch.float32)
    dct_tensor = dct_2d(dct_tensor, norm=norm)

    # batch_size, combined_block_size, h_blocks, w_blocks
    dct_tensor = dct_tensor.reshape(batch_size, h_blocks, w_blocks, block_size*block_size).permute(0,3,1,2)
    return dct_tensor


def decode_single_channel_dct_2d(img: torch.FloatTensor, norm: str="ortho") -> torch.FloatTensor:
    batch_size, combined_block_size, h_blocks, w_blocks = img.shape
    block_size = int(math.sqrt(combined_block_size))
    height = int(h_blocks*block_size)
    width = int(w_blocks*block_size)

    img_tensor = img.permute(0,2,3,1).view(batch_size, h_blocks, w_blocks, block_size, block_size)
    img_tensor = idct_2d(img_tensor, norm=norm)
    img_tensor = img_tensor.permute(0,1,3,2,4).reshape(batch_size, height, width)
    return img_tensor


def rgb_to_ycbcr_tensor(image: torch.ByteTensor) -> torch.FloatTensor:
    rgb_weights = torch.tensor([[0.002345098, -0.001323419, 0.003921569], [0.004603922, -0.00259815, -0.003283824], [0.000894118, 0.003921569, -0.000637744]], device=image.device)
    ycbcr = torch.einsum("cv,...chw->...vhw", [rgb_weights, image.permute(0,3,1,2).to(dtype=torch.float32)])
    ycbcr[:,0,:,:] = ycbcr[:,0,:,:].add(-1)
    return ycbcr


def ycbcr_tensor_to_rgb(ycbcr: torch.FloatTensor) -> torch.ByteTensor:
    ycbcr_weights = torch.tensor([[127.5, 127.5, 127.5], [0, -43.877376465, 225.93], [178.755, -91.052376465, 0]], device=ycbcr.device)
    return torch.einsum("cv,...chw->...vhw", [ycbcr_weights, ycbcr]).add(127.5).round().clamp(0,255).permute(0,2,3,1).to(dtype=torch.uint8)


def encode_jpeg_tensor(img: torch.FloatTensor, block_size: int=16, cbcr_downscale: int=2, norm: str="ortho") -> torch.FloatTensor:
    img = img[:, :, :(img.shape[-2]//block_size)*block_size, :(img.shape[-1]//block_size)*block_size] # crop to a multiply of block_size
    cbcr_block_size = block_size//cbcr_downscale
    _, _, height, width = img.shape
    down_img = images_sharpfin.resize_tensor(img[:, 1:,:,:], (height//cbcr_downscale, width//cbcr_downscale), linearize=False)
    y = encode_single_channel_dct_2d(img[:, 0, :,:], block_size=block_size, norm=norm)
    cb = encode_single_channel_dct_2d(down_img[:, 0, :,:], block_size=cbcr_block_size, norm=norm)
    cr = encode_single_channel_dct_2d(down_img[:, 1, :,:], block_size=cbcr_block_size, norm=norm)
    return torch.cat([y,cb,cr], dim=1)


def decode_jpeg_tensor(jpeg_img: torch.FloatTensor, block_size: int=16, cbcr_downscale: int=2, norm: str="ortho") -> torch.FloatTensor:
    _, _, h_blocks, w_blocks = jpeg_img.shape
    y_block_size = block_size*block_size
    cbcr_block_size = int((block_size//cbcr_downscale) ** 2)
    cr_start = y_block_size + cbcr_block_size
    y = jpeg_img[:, :y_block_size]
    cb = jpeg_img[:, y_block_size:cr_start]
    cr = jpeg_img[:, cr_start:]
    y = decode_single_channel_dct_2d(y, norm=norm)
    cb = decode_single_channel_dct_2d(cb, norm=norm)
    cr = decode_single_channel_dct_2d(cr, norm=norm)
    cb = images_sharpfin.resize_tensor(cb, (h_blocks*block_size, w_blocks*block_size), linearize=False)
    cr = images_sharpfin.resize_tensor(cr, (h_blocks*block_size, w_blocks*block_size), linearize=False)
    return torch.stack([y,cb,cr], dim=1)


def process_image_input(images: PipelineImageInput) -> torch.ByteTensor:
    if isinstance(images, list):
        combined_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img = torch.from_numpy(np.asarray(img).copy()).unsqueeze(0)
                combined_images.append(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                combined_images.append(img)
            elif isinstance(img, torch.Tensor):
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                combined_images.append(img)
            else:
                raise RuntimeError(f"Invalid input! Given: {type(img)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")
        combined_images = torch.cat(combined_images, dim=0)
    elif isinstance(images, Image.Image):
        combined_images = torch.from_numpy(np.asarray(images).copy()).unsqueeze(0)
    elif isinstance(images, np.ndarray):
        combined_images = torch.from_numpy(images)
        if combined_images.ndim == 3:
            combined_images = combined_images.unsqueeze(0)
    elif isinstance(images, torch.Tensor):
        combined_images = images
        if combined_images.ndim == 3:
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
        super().__init__()

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
            latents_std = torch.tensor(self.latents_std, device=latents.device, dtype=torch.float32).view(1,-1,1,1)
            if self.latents_mean is not None:
                latents_mean = torch.tensor(self.latents_mean, device=latents.device, dtype=torch.float32).view(1,-1,1,1)
                latents = torch.addcmul(latents_mean, latents, latents_std)
            else:
                latents = latents * latents_std
        elif self.latents_mean is not None:
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
