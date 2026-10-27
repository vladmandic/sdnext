"""MOSS-Image img2img and shared inpainting helpers."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from pipelines.moss.moss_image import MOSSImagePipeline


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, 32, H, W] VAE latents into the transformer's [B, 128, H/2, W/2] layout."""
    batch, channels, height, width = latents.shape
    if channels != 32 or height % 2 or width % 2:
        raise ValueError("MOSS image latents must have 32 channels and even spatial dimensions.")
    return latents.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def _expand_batch(tensor: torch.Tensor, prompt_batch: int, num_images_per_prompt: int) -> torch.Tensor:
    target_batch = prompt_batch * num_images_per_prompt
    if tensor.shape[0] == target_batch:
        return tensor
    if tensor.shape[0] == prompt_batch:
        return tensor.repeat_interleave(num_images_per_prompt, dim=0)
    if tensor.shape[0] == 1:
        return tensor.repeat(target_batch, *([1] * (tensor.ndim - 1)))
    raise ValueError(f"Condition batch size {tensor.shape[0]} does not match prompt batch size {prompt_batch}.")


def _encode_image(pipe, image, height, width, device, generator) -> torch.Tensor:
    if pipe.vae is None:
        raise ValueError("A FLUX.2 VAE is required for image-to-image workflows.")
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor // pipe.patch_size)
    pixels = image_processor.preprocess(image, height=height, width=width).to(device=device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(pixels).latent_dist.sample(generator=generator)
    packed = _pack_latents(latents)
    bn = pipe.vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=torch.float32)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1).to(device=device, dtype=torch.float32) + pipe.vae.config.batch_norm_eps)
    normalized = ((packed.float() - mean) / std).to(dtype=pipe.transformer.dtype)
    latent_channels = getattr(pipe.transformer.config, "latent_channels", None)
    if latent_channels is None:
        latent_channels = pipe.transformer.config.in_channels
    if normalized.shape[1] != latent_channels:
        raise ValueError(f"Expected {latent_channels} packed latent channels, got {normalized.shape[1]}.")
    return normalized


def _prepare_mask(pipe, mask_image, height, width, device) -> torch.Tensor:
    if mask_image is None:
        raise ValueError("mask_image is required for inpainting.")
    processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor, do_convert_grayscale=True, do_normalize=False)
    mask = processor.preprocess(mask_image, height=height, width=width).to(device=device, dtype=torch.float32)
    latent_scale_factor = pipe.vae_scale_factor * pipe.patch_size
    mask = F.interpolate(mask[:, :1], size=(height // latent_scale_factor, width // latent_scale_factor), mode="nearest")
    return mask


def _prepare_schedule(scheduler, num_inference_steps, strength, device):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1.")
    scheduler.set_timesteps(num_inference_steps, device=device)
    original = (scheduler.timesteps, scheduler.sigmas, scheduler.num_inference_steps)
    sigma_grid = scheduler.sigmas[:-1]
    denoising_steps = max(1, int(len(sigma_grid) * math.sqrt(strength)))
    start_index = len(sigma_grid) - denoising_steps
    actual_sigma = sigma_grid[start_index].item()
    scheduler.timesteps = scheduler.timesteps[start_index:]
    scheduler.sigmas = scheduler.sigmas[start_index:]
    scheduler.num_inference_steps = len(scheduler.timesteps)
    return actual_sigma, original


def _run_img2img(
    pipe,
    *,
    prompt,
    negative_prompt,
    image,
    mask_image,
    strength,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    num_images_per_prompt,
    generator,
    latents,
    prompt_embeds,
    prompt_attention_mask,
    negative_prompt_embeds,
    negative_prompt_attention_mask,
    max_sequence_length,
    output_type,
    return_dict,
    uncond_skip_layers,
    slg_layers,
    slg_scale,
    slg_start,
    slg_stop,
    callback_on_step_end,
    callback_on_step_end_tensor_inputs,
):
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("height and width must be positive multiples of 16.")
    if num_images_per_prompt < 1:
        raise ValueError("num_images_per_prompt must be positive.")
    device = pipe._execution_device # pylint: disable=protected-access
    if prompt_embeds is not None:
        prompt_batch = prompt_embeds.shape[0]
    elif isinstance(prompt, str):
        prompt_batch = 1
    elif isinstance(prompt, list):
        prompt_batch = len(prompt)
    else:
        raise ValueError("Provide prompt or prompt_embeds for image-to-image workflows.")

    init_latents = _encode_image(pipe, image, height, width, device, generator)
    init_latents = _expand_batch(init_latents, prompt_batch, num_images_per_prompt).to(dtype=torch.float32)
    noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=torch.float32) if latents is None else latents.to(device=device, dtype=torch.float32)
    if noise.shape != init_latents.shape:
        raise ValueError(f"latents must have shape {tuple(init_latents.shape)}.")

    mask_latent = None
    if mask_image is not None:
        mask_latent = _prepare_mask(pipe, mask_image, height, width, device)
        mask_latent = _expand_batch(mask_latent, prompt_batch, num_images_per_prompt)

    scheduler = pipe.scheduler
    original_set_timesteps = scheduler.set_timesteps
    actual_sigma, _ = _prepare_schedule(scheduler, num_inference_steps, strength, device)
    noised = actual_sigma * noise + (1.0 - actual_sigma) * init_latents
    user_callback = callback_on_step_end

    def blend_callback(current_pipe, step_index, timestep, callback_kwargs):
        current_latents = callback_kwargs.get("latents")
        if mask_latent is not None and current_latents is not None:
            sigma_next = current_pipe.scheduler.sigmas[step_index + 1].item()
            init_at_t = sigma_next * noise + (1.0 - sigma_next) * init_latents
            mask = mask_latent.to(dtype=current_latents.dtype)
            callback_kwargs["latents"] = mask * current_latents + (1.0 - mask) * init_at_t.to(current_latents.dtype)
        if user_callback is not None:
            callback_kwargs = user_callback(current_pipe, step_index, timestep, callback_kwargs)
        return callback_kwargs

    callback_inputs = list(callback_on_step_end_tensor_inputs or ["latents"])
    if mask_latent is not None and "latents" not in callback_inputs:
        callback_inputs.append("latents")
    try:
        scheduler.set_timesteps = lambda *args, **kwargs: None
        return MOSSImagePipeline.__call__(
            pipe,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=noised,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            return_dict=return_dict,
            uncond_skip_layers=uncond_skip_layers,
            slg_layers=slg_layers,
            slg_scale=slg_scale,
            slg_start=slg_start,
            slg_stop=slg_stop,
            callback_on_step_end=blend_callback if mask_latent is not None else user_callback,
            callback_on_step_end_tensor_inputs=callback_inputs,
        )
    finally:
        scheduler.set_timesteps = original_set_timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)


class MOSSImageToImagePipeline(MOSSImagePipeline):
    """MOSS-Image image-to-image pipeline."""

    @torch.no_grad()
    def __call__( # pylint: disable=arguments-differ
        self,
        prompt=None,
        *,
        image: PipelineImageInput,
        strength: float = 0.8,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt="",
        num_images_per_prompt: int = 1,
        generator=None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 4096,
        output_type: str = "pil",
        return_dict: bool = True,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale: float = 0.0,
        slg_start: float = 0.0,
        slg_stop: float = 1.0,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ):
        return _run_img2img(
            self,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=None,
            strength=strength,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            return_dict=return_dict,
            uncond_skip_layers=uncond_skip_layers,
            slg_layers=slg_layers,
            slg_scale=slg_scale,
            slg_start=slg_start,
            slg_stop=slg_stop,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
