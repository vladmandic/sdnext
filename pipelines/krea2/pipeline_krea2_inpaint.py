"""Krea 2 inpainting pipeline.

This module adds inpainting support on top of the existing Krea2 image-to-image
variant without modifying the upstream Krea2 denoising path.
"""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor

from modules import devices
from .pipeline_krea2 import Krea2Pipeline, Krea2Img2ImgPipeline


def _setup_img2img_schedule(scheduler, strength, num_inference_steps, device, mu=None):
    """Set custom sigma schedule, return first sigma after scheduler shift."""
    min_sigma = 1e-8
    custom_sigmas = torch.linspace(max(strength, 0.01), min_sigma, num_inference_steps).tolist()
    scheduler.set_timesteps(sigmas=custom_sigmas, device=device, mu=mu)
    return scheduler.sigmas[0].item()


def _prepare_mask(pipe, mask_image, height, width, device):
    if isinstance(mask_image, list):
        mask_image = mask_image[0]
    if isinstance(mask_image, Image.Image):
        mask_image = mask_image.convert("L")
    if isinstance(mask_image, Image.Image):
        import torchvision.transforms.functional as TF

        mask_tensor = TF.to_tensor(mask_image).unsqueeze(0).to(device=device, dtype=torch.float32)
    elif isinstance(mask_image, torch.Tensor):
        mask_tensor = mask_image.to(device=device, dtype=torch.float32)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
    else:
        mask_tensor = torch.ones(1, 1, height, width, device=device, dtype=torch.float32)

    latent_h = height // pipe.vae_compression
    latent_w = width // pipe.vae_compression
    mask_latent = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
    return mask_latent[:, :1, :, :]


class Krea2InpaintPipeline(Krea2Img2ImgPipeline):
    """Krea 2 inpainting pipeline."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        strength: float = 0.8,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        align = self.vae_compression * self.patch
        height = (height // align) * align
        width = (width // align) * align

        device = devices.device
        dtype = self.transformer.dtype

        cfg = self.scheduler.config
        grid_h = height // (self.vae_compression * self.patch)
        grid_w = width // (self.vae_compression * self.patch)
        mu = self.calculate_shift(
            grid_h * grid_w,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 6400),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        actual_sigma = _setup_img2img_schedule(self.scheduler, strength, num_inference_steps, device, mu=mu)
        init_latents = self.encode_image(image, height, width, dtype, device)
        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=devices.dtype)
        noised = (actual_sigma * noise + (1.0 - actual_sigma) * init_latents).to(torch.float32)
        mask_latent = _prepare_mask(self, mask_image, height, width, device)

        orig_set_timesteps = self.scheduler.set_timesteps
        self.scheduler.set_timesteps = lambda *args, **kwargs: None

        user_callback = callback_on_step_end

        def blend_callback(pipe, i, t, callback_kwargs):
            cur_latents = callback_kwargs.get("latents")
            if cur_latents is not None:
                sigma_next = pipe.scheduler.sigmas[i + 1].item() if i + 1 < len(pipe.scheduler.sigmas) else 0.0
                init_at_t = sigma_next * noise + (1.0 - sigma_next) * init_latents
                blended = mask_latent * cur_latents + (1.0 - mask_latent) * init_at_t.to(cur_latents.dtype)
                callback_kwargs["latents"] = blended
            if user_callback is not None:
                callback_kwargs = user_callback(pipe, i, t, callback_kwargs)
            return callback_kwargs

        try:
            return Krea2Pipeline.__call__(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=noised,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=blend_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        finally:
            self.scheduler.set_timesteps = orig_set_timesteps
