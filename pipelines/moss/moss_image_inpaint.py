"""MOSS-Image latent-blend inpainting pipeline."""

from collections.abc import Callable

import torch
from diffusers.image_processor import PipelineImageInput

from pipelines.moss.moss_image import MOSSImagePipeline
from pipelines.moss.moss_image_i2i import _run_img2img


class MOSSImageInpaintPipeline(MOSSImagePipeline):
    """MOSS-Image inpainting using per-step latent preservation outside the mask."""

    @torch.no_grad()
    def __call__(  # pylint: disable=arguments-differ
        self,
        prompt=None,
        *,
        image: PipelineImageInput,
        mask_image: PipelineImageInput,
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
            mask_image=mask_image,
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
