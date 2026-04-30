"""Anima img2img and inpainting pipelines (built dynamically from the runtime-imported base class)."""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor

from modules import devices


def _encode_image(pipe, image, dtype, device, generator):
    """VAE-encode an image and normalize to denoiser latent space."""
    if isinstance(image, list):
        image = image[0]
    image_tensor = pipe.video_processor.preprocess(image, None, None)
    image_tensor = image_tensor.squeeze(0).to(device=device, dtype=pipe.vae.dtype)
    image_tensor = image_tensor[None, :, None, :, :]
    init_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=torch.float32).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents_std_inv = (1.0 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=torch.float32)).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    return ((init_latents.float() - latents_mean) * latents_std_inv).to(dtype)


def _setup_img2img_schedule(scheduler, strength, num_inference_steps, device):
    """Set custom sigma schedule, return first sigma after scheduler shift."""
    custom_sigmas = torch.linspace(max(strength, 0.01), 0.0, num_inference_steps).tolist()
    scheduler.set_timesteps(sigmas=custom_sigmas, device=device)
    return scheduler.sigmas[0].item()


def build_anima_pipeline_classes(base_cls):
    """Return (AnimaImageToImagePipeline, AnimaInpaintPipeline) inheriting from base_cls."""

    class AnimaImageToImagePipeline(base_cls):
        """Anima img2img pipeline."""

        @torch.no_grad()
        def __call__(
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            image: Optional[PipelineImageInput] = None,
            strength: float = 0.8,
            height: int = 768,
            width: int = 1360,
            num_inference_steps: int = 35,
            guidance_scale: float = 7.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
        ):
            actual_sigma = _setup_img2img_schedule(self.scheduler, strength, num_inference_steps, devices.device)
            init_latents = _encode_image(self, image, devices.dtype, devices.device, generator)
            noise = randn_tensor(init_latents.shape, generator=generator, device=devices.device, dtype=devices.dtype)
            noised = (actual_sigma * noise + (1.0 - actual_sigma) * init_latents).to(torch.float32)

            orig_set_timesteps = self.scheduler.set_timesteps
            self.scheduler.set_timesteps = lambda *args, **kwargs: None
            try:
                return super().__call__(
                    prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt, generator=generator, latents=noised,
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                    output_type=output_type, return_dict=return_dict,
                    callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    max_sequence_length=max_sequence_length,
                )
            finally:
                self.scheduler.set_timesteps = orig_set_timesteps

    class AnimaInpaintPipeline(AnimaImageToImagePipeline):
        """Anima inpainting pipeline."""

        @torch.no_grad()
        def __call__(
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            image: Optional[PipelineImageInput] = None,
            mask_image: Optional[PipelineImageInput] = None,
            strength: float = 0.8,
            height: int = 768,
            width: int = 1360,
            num_inference_steps: int = 35,
            guidance_scale: float = 7.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
        ):
            actual_sigma = _setup_img2img_schedule(self.scheduler, strength, num_inference_steps, devices.device)
            init_latents = _encode_image(self, image, devices.dtype, devices.device, generator)
            noise = randn_tensor(init_latents.shape, generator=generator, device=devices.device, dtype=devices.dtype)
            noised = (actual_sigma * noise + (1.0 - actual_sigma) * init_latents).to(torch.float32)
            mask_latent = _prepare_mask(self, mask_image, height, width, devices.device)

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
                return base_cls.__call__(
                    self,
                    prompt=prompt, negative_prompt=negative_prompt,
                    height=height, width=width, num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt,
                    generator=generator, latents=noised, prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds, output_type=output_type,
                    return_dict=return_dict, callback_on_step_end=blend_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                    max_sequence_length=max_sequence_length,
                )
            finally:
                self.scheduler.set_timesteps = orig_set_timesteps

    return AnimaImageToImagePipeline, AnimaInpaintPipeline


def _prepare_mask(pipe, mask_image, height, width, device):
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
    latent_h = height // pipe.vae_scale_factor_spatial
    latent_w = width // pipe.vae_scale_factor_spatial
    mask_latent = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
    mask_latent = mask_latent[:, :1, :, :]
    mask_latent = mask_latent.unsqueeze(2)
    return mask_latent
