"""Lens img2img and inpainting extensions.

This module adds image-to-image and inpainting support on top of the
existing Lens text-to-image :class:`LensPipeline` without modifying the
upstream denoising path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from modules import devices
from .pipeline import LensPipeline, compute_empirical_mu

if TYPE_CHECKING:
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback


def _get_image_processor(pipe: LensPipeline) -> VaeImageProcessor:
    processor = getattr(pipe, "lens_image_processor", None)
    if processor is None:
        processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor // 2)
        pipe.lens_image_processor = processor
    return processor


def _encode_image(
    pipe: LensPipeline,
    image: PipelineImageInput,
    height: int,
    width: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if isinstance(image, list):
        image = image[0]

    processor = _get_image_processor(pipe)
    image_tensor = processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=devices.device, dtype=pipe.vae.dtype)

    latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    patched = pipe._patchify_latents(latents) # pylint: disable=protected-access

    bn = pipe.vae.bn
    mean = bn.running_mean.to(device=devices.device).view(1, -1, 1, 1)
    std = torch.sqrt(
        bn.running_var.to(device=devices.device).view(1, -1, 1, 1)
        + pipe.vae.config.batch_norm_eps
    )
    normalized = (patched - mean) / std
    return normalized.flatten(2).transpose(1, 2).to(devices.dtype)


def _setup_img2img_schedule(
    scheduler,
    strength: float,
    num_inference_steps: int,
    device: torch.device,
    mu: Optional[float] = None,
) -> float:
    # Avoid passing an exact zero sigma to FlowMatchEulerDiscreteScheduler,
    # which can trigger a divide-by-zero warning during dynamic shifting.
    min_sigma = 1e-8
    custom_sigmas = torch.linspace(max(strength, 0.01), min_sigma, num_inference_steps).tolist()
    scheduler.set_timesteps(sigmas=custom_sigmas, device=device, mu=mu)
    return scheduler.sigmas[0].item()


def _prepare_mask(
    pipe: LensPipeline,
    mask_image: Optional[PipelineImageInput],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
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
        mask_tensor = torch.ones(
            1, 1, height, width, device=device, dtype=torch.float32
        )

    latent_h = height // pipe.vae_scale_factor
    latent_w = width // pipe.vae_scale_factor
    mask_latent = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
    mask_latent = mask_latent[:, :1, :, :]
    return mask_latent.flatten(2).transpose(1, 2).to(device=device, dtype=pipe.vae.dtype)


class LensImg2ImgPipeline(LensPipeline):
    """Lens image-to-image pipeline."""

    @torch.no_grad()
    def __call__( # pylint: disable=signature-differs
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        image: Optional[PipelineImageInput] = None,
        strength: float = 0.8,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[List[torch.Tensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_reasoner: bool = False,
    ):
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        actual_sigma = _setup_img2img_schedule(
            self.scheduler,
            strength,
            num_inference_steps,
            devices.device,
            mu=compute_empirical_mu(latent_h * latent_w, num_inference_steps),
        )
        init_latents = _encode_image(self, image, height, width, generator)
        noise = randn_tensor(
            init_latents.shape,
            generator=generator,
            device=devices.device,
            dtype=devices.dtype,
        )
        noised = actual_sigma * noise + (1.0 - actual_sigma) * init_latents

        orig_set_timesteps = self.scheduler.set_timesteps
        self.scheduler.set_timesteps = lambda *args, **kwargs: None
        try:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=noised,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                enable_reasoner=enable_reasoner,
            )
        finally:
            self.scheduler.set_timesteps = orig_set_timesteps


class LensInpaintPipeline(LensImg2ImgPipeline):
    """Lens inpainting pipeline."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        strength: float = 0.8,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[List[torch.Tensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_reasoner: bool = False,
    ):
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        actual_sigma = _setup_img2img_schedule(
            self.scheduler,
            strength,
            num_inference_steps,
            devices.device,
            mu=compute_empirical_mu(latent_h * latent_w, num_inference_steps),
        )
        init_latents = _encode_image(self, image, height, width, generator)
        noise = randn_tensor(
            init_latents.shape,
            generator=generator,
            device=devices.device,
            dtype=devices.dtype,
        )
        noised = actual_sigma * noise + (1.0 - actual_sigma) * init_latents
        mask_latent = _prepare_mask(self, mask_image, height, width, devices.device)

        orig_set_timesteps = self.scheduler.set_timesteps
        self.scheduler.set_timesteps = lambda *args, **kwargs: None

        user_callback = callback_on_step_end

        def blend_callback(pipe, i, t, callback_kwargs):
            cur_latents = callback_kwargs.get("latents")
            if cur_latents is not None:
                sigma_next = (
                    pipe.scheduler.sigmas[i + 1].item()
                    if i + 1 < len(pipe.scheduler.sigmas)
                    else 0.0
                )
                init_at_t = sigma_next * noise + (1.0 - sigma_next) * init_latents
                blended = mask_latent * cur_latents + (1.0 - mask_latent) * init_at_t.to(cur_latents.dtype)
                callback_kwargs["latents"] = blended
            if user_callback is not None:
                callback_kwargs = user_callback(pipe, i, t, callback_kwargs)
            return callback_kwargs

        try:
            return LensPipeline.__call__(
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
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=blend_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                max_sequence_length=max_sequence_length,
                enable_reasoner=enable_reasoner,
            )
        finally:
            self.scheduler.set_timesteps = orig_set_timesteps
