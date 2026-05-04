"""ERNIE-Image img2img and inpainting pipelines.

Diffusers only ships ``ErnieImagePipeline`` (text-to-image). These two
subclasses add img2img and inpainting on top of it without touching the
upstream pipeline.

ERNIE-specific subtleties relative to the Anima equivalent:

- Latent rank is 4-D ``[B, C, H, W]`` (image, not video). No ``unsqueeze(2)``
  on mask or init latent.
- The transformer takes a *patched* 128-channel latent at H/16 resolution.
  The mirror of the upstream decode (BN-unnormalize -> unpatchify -> vae.decode)
  is ``vae.encode -> patchify -> BN-normalize`` here.
- Latent normalization uses ``vae.bn.running_mean``/``running_var``
  applied to the patched 128-channel latent, not scalar
  ``vae.config.latents_mean``/``latents_std``.
- ``ErnieImagePipeline`` has no ``image_processor`` instance attribute, so we
  build a ``VaeImageProcessor`` ad-hoc and cache it on the pipeline.
"""

from typing import Callable, Dict, List, Optional, Union

import diffusers
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from modules import devices


def _get_image_processor(pipe):
    """Build (and cache) a VaeImageProcessor for the pipe.

    pipe.vae_scale_factor=16 already accounts for the 2x2 patchify; the
    underlying VAE downsamples by 8. The processor needs the VAE-only factor.
    """
    proc = getattr(pipe, 'ernie_image_processor', None)
    if proc is None:
        proc = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor // 2)
        pipe.ernie_image_processor = proc
    return proc


def _encode_image(pipe, image, dtype, device, generator):
    """VAE-encode an image and produce the patched, BN-normalized 128-channel latent."""
    if isinstance(image, list):
        image = image[0]
    proc = _get_image_processor(pipe)
    image_tensor = proc.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
    raw_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    patched = pipe._patchify_latents(raw_latents)  # pylint: disable=protected-access
    bn_mean = pipe.vae.bn.running_mean.to(device=device).view(1, -1, 1, 1)
    bn_std = torch.sqrt(pipe.vae.bn.running_var.to(device=device).view(1, -1, 1, 1) + 1e-5)
    return ((patched.float() - bn_mean) / bn_std).to(dtype)


def _setup_img2img_schedule(scheduler, strength, num_inference_steps, device):
    """Set custom sigma schedule, return first sigma after scheduler shift."""
    custom_sigmas = torch.linspace(max(strength, 0.01), 0.0, num_inference_steps).tolist()
    scheduler.set_timesteps(sigmas=custom_sigmas, device=device)
    return scheduler.sigmas[0].item()


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
    latent_h = height // pipe.vae_scale_factor
    latent_w = width // pipe.vae_scale_factor
    mask_latent = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
    return mask_latent[:, :1, :, :]


class ErnieImageImg2ImgPipeline(diffusers.ErnieImagePipeline):
    """ERNIE-Image image-to-image pipeline."""

    @torch.no_grad()
    def __call__(
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
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,
    ):
        actual_sigma = _setup_img2img_schedule(self.scheduler, strength, num_inference_steps, devices.device)
        init_latents = _encode_image(self, image, devices.dtype, devices.device, generator)
        noise = randn_tensor(init_latents.shape, generator=generator, device=devices.device, dtype=devices.dtype)
        noised = actual_sigma * noise + (1.0 - actual_sigma) * init_latents

        orig_set_timesteps = self.scheduler.set_timesteps
        self.scheduler.set_timesteps = lambda *args, **kwargs: None
        try:
            return super().__call__(
                prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt, generator=generator, latents=noised,
                prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type, return_dict=return_dict,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                use_pe=use_pe,
            )
        finally:
            self.scheduler.set_timesteps = orig_set_timesteps


class ErnieImageInpaintPipeline(ErnieImageImg2ImgPipeline):
    """ERNIE-Image inpainting pipeline."""

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
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,
    ):
        actual_sigma = _setup_img2img_schedule(self.scheduler, strength, num_inference_steps, devices.device)
        init_latents = _encode_image(self, image, devices.dtype, devices.device, generator)
        noise = randn_tensor(init_latents.shape, generator=generator, device=devices.device, dtype=devices.dtype)
        noised = actual_sigma * noise + (1.0 - actual_sigma) * init_latents
        mask_latent = _prepare_mask(self, mask_image, height, width, devices.device)

        orig_set_timesteps = self.scheduler.set_timesteps
        self.scheduler.set_timesteps = lambda *args, **kwargs: None

        user_callback = callback_on_step_end

        def blend_callback(pipe, i, t, callback_kwargs):
            cur_latents = callback_kwargs.get("latents")
            if cur_latents is not None:
                sigma_next = pipe.scheduler.sigmas[i + 1].item() if i + 1 < len(pipe.scheduler.sigmas) else 0.0
                init_at_t = sigma_next * noise + (1.0 - sigma_next) * init_latents
                mask_dt = mask_latent.to(cur_latents.dtype)
                blended = mask_dt * cur_latents + (1.0 - mask_dt) * init_at_t.to(cur_latents.dtype)
                callback_kwargs["latents"] = blended.to(cur_latents.dtype)
            if user_callback is not None:
                callback_kwargs = user_callback(pipe, i, t, callback_kwargs)
            return callback_kwargs

        try:
            return diffusers.ErnieImagePipeline.__call__(
                self,
                prompt=prompt, negative_prompt=negative_prompt,
                height=height, width=width, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt,
                generator=generator, latents=noised, prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds, output_type=output_type,
                return_dict=return_dict, callback_on_step_end=blend_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                use_pe=use_pe,
            )
        finally:
            self.scheduler.set_timesteps = orig_set_timesteps
