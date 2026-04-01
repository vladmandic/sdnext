from functools import wraps
import torch
import diffusers
from modules import devices # pylint: disable=ungrouped-imports


def model_hijack(): # a111 compatibility item
    pass


def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    if type(attr) == torch.Tensor:
        if attr.device != devices.device:
            attr = attr.to(device=devices.device, dtype=(torch.float32 if devices.device.type == 'mps' else None))

    setattr(self, name, attr)


# Upcast BF16 to FP32
original_fft_fftn = torch.fft.fftn
@wraps(torch.fft.fftn)
def fft_fftn(input, s=None, dim=None, norm=None, *, out=None): # pylint: disable=redefined-builtin
    return_dtype = input.dtype
    if input.dtype == torch.bfloat16:
        input = input.to(dtype=torch.float32)
    return original_fft_fftn(input, s=s, dim=dim, norm=norm, out=out).to(dtype=return_dtype)


# Upcast BF16 to FP32
original_fft_ifftn = torch.fft.ifftn
@wraps(torch.fft.ifftn)
def fft_ifftn(input, s=None, dim=None, norm=None, *, out=None): # pylint: disable=redefined-builtin
    return_dtype = input.dtype
    if input.dtype == torch.bfloat16:
        input = input.to(dtype=torch.float32)
    return original_fft_ifftn(input, s=s, dim=dim, norm=norm, out=out).to(dtype=return_dtype)


# Diffusers FreeU
# Diffusers is imported before sd_hijacks so fourier_filter needs hijacking too
original_fourier_filter = diffusers.utils.torch_utils.fourier_filter
@wraps(diffusers.utils.torch_utils.fourier_filter)
def fourier_filter(x_in, threshold, scale):
    return_dtype = x_in.dtype
    if x_in.dtype == torch.bfloat16:
        x_in = x_in.to(dtype=torch.float32)
    return original_fourier_filter(x_in, threshold, scale).to(dtype=return_dtype)


# IPEX always upcasts
if devices.backend != "ipex":
    torch.fft.fftn = fft_fftn
    torch.fft.ifftn = fft_ifftn
    diffusers.utils.torch_utils.fourier_filter = fourier_filter


# Fix "torch is not defined" error on img2img pipelines when torch.compile for vae.encode is enabled:
# disable_compile for AutoencoderKLOutput is the only change
if torch.__version__.startswith("2.6"):
    from dataclasses import dataclass
    from torch.compiler import disable as disable_compile # pylint: disable=ungrouped-imports
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution # pylint: disable=ungrouped-imports

    @dataclass
    @disable_compile
    class AutoencoderKLOutput(diffusers.utils.BaseOutput):
        latent_dist: DiagonalGaussianDistribution
    diffusers.models.autoencoders.autoencoder_kl.AutoencoderKLOutput = AutoencoderKLOutput
