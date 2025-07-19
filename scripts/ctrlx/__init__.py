from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import rescale_noise_cfg, retrieve_latents, retrieve_timesteps
from diffusers.utils import BaseOutput, deprecate
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import PIL
import torch
from .sdxl import register_attr
from .media import preprocess
from .utils import batch_dict_to_tensor, batch_tensor_to_dict, noise_prev, noise_t2t


BATCH_ORDER = [
    "structure_uncond", "appearance_uncond", "uncond", "structure_cond", "appearance_cond", "cond",
]


def get_last_control_i(control_schedule, num_inference_steps):
    if control_schedule is None:
        return num_inference_steps, num_inference_steps

    def max_(l):
        if len(l) == 0:
            return 0.0
        return max(l)

    structure_max = 0.0
    appearance_max = 0.0
    for block in control_schedule.values():
        if isinstance(block, list):  # Handling mid_block
            block = {0: block}
        for layer in block.values():
            structure_max = max(structure_max, max_(layer[0] + layer[1]))
            appearance_max = max(appearance_max, max_(layer[2]))

    structure_i = round(num_inference_steps * structure_max)
    appearance_i = round(num_inference_steps * appearance_max)
    return structure_i, appearance_i


@dataclass
class CtrlXStableDiffusionXLPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray] = None
    structures: Union[List[PIL.Image.Image], np.ndarray] = None
    appearances: Union[List[PIL.Image.Image], np.ndarray] = None


class CtrlXStableDiffusionXLPipeline(StableDiffusionXLPipeline):  # diffusers==0.28.0

    def prepare_latents(
        self, image, batch_size, num_images_per_prompt, num_channels_latents, height, width,
        dtype, device, generator=None, noise=None,
    ):
        batch_size = batch_size * num_images_per_prompt
        if noise is None:
            shape = (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor
            )
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            noise = noise * self.scheduler.init_noise_sigma  # Starting noise, need to scale
        else:
            noise = noise.to(device)

        if image is None:
            return noise, None

        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:  # Image already in latents form
            init_latents = image

        else:
            # Make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.to(torch.float32)
                self.vae.to(torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # Expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        return noise, init_latents

    @property
    def structure_guidance_scale(self):
        return self._guidance_scale if self._structure_guidance_scale is None else self._structure_guidance_scale

    @property
    def appearance_guidance_scale(self):
        return self._guidance_scale if self._appearance_guidance_scale is None else self._appearance_guidance_scale

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        structure_prompt: Optional[Union[str, List[str]]] = None,
        appearance_prompt: Optional[Union[str, List[str]]] = None,
        structure_image: Optional[PipelineImageInput] = None,
        appearance_image: Optional[PipelineImageInput] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        positive_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 5.0,
        structure_guidance_scale: Optional[float] = None,
        appearance_guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        structure_latents: Optional[torch.Tensor] = None,
        appearance_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,  # Positive prompt is concatenated with prompt, so no embeddings
        structure_prompt_embeds: Optional[torch.Tensor] = None,
        appearance_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        structure_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        appearance_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        control_schedule: Optional[Dict] = None,
        self_recurrence_schedule: Optional[List[int]] = [],  # Format: [(start, end, num_repeat)]
        decode_structure: Optional[bool] = True,
        decode_appearance: Optional[bool] = True,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to U-Net
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            None,  # prompt_2
            height,
            width,
            callback_steps,
            negative_prompt = negative_prompt,
            negative_prompt_2 = None,  # negative_prompt_2
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._structure_guidance_scale = structure_guidance_scale
        self._appearance_guidance_scale = appearance_guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = None  # denoising_end
        self._denoising_start = None  # denoising_start
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if batch_size * num_images_per_prompt != 1:
            raise ValueError(
                f"Pipeline currently does not support batch_size={batch_size} and num_images_per_prompt=1. "
                "Effective batch size (batch_size * num_images_per_prompt) must be 1."
            )

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        if positive_prompt is not None and positive_prompt != "":
            prompt = prompt + ", " + positive_prompt  # Add positive prompt with comma
            # By default, only add positive prompt to the appearance prompt and not the structure prompt
            if appearance_prompt is not None and appearance_prompt != "":
                appearance_prompt = appearance_prompt + ", " + positive_prompt

        (
            prompt_embeds_,
            negative_prompt_embeds,
            pooled_prompt_embeds_,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt = prompt,
            prompt_2 = None,  # prompt_2
            device = device,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = True,  # self.do_classifier_free_guidance, TODO: Support no CFG
            negative_prompt = negative_prompt,
            negative_prompt_2 = None,  # negative_prompt_2
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
            lora_scale = text_encoder_lora_scale,
            clip_skip = self.clip_skip,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds_], dim=0).to(device)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds_], dim=0).to(device)

        # 3.1. Structure prompt embeddings
        if structure_prompt is not None and structure_prompt != "":
            (
                structure_prompt_embeds,
                negative_structure_prompt_embeds,
                structure_pooled_prompt_embeds,
                negative_structure_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt = structure_prompt,
                prompt_2 = None,  # prompt_2
                device = device,
                num_images_per_prompt = num_images_per_prompt,
                do_classifier_free_guidance = True,  # self.do_classifier_free_guidance, TODO: Support no CFG
                negative_prompt = negative_prompt if structure_image is None else "",
                negative_prompt_2 = None,  # negative_prompt_2
                prompt_embeds = structure_prompt_embeds,
                negative_prompt_embeds = None,  # negative_prompt_embeds
                pooled_prompt_embeds = structure_pooled_prompt_embeds,
                negative_pooled_prompt_embeds = None,  # negative_pooled_prompt_embeds
                lora_scale = text_encoder_lora_scale,
                clip_skip = self.clip_skip,
            )
            structure_prompt_embeds = torch.cat(
                [negative_structure_prompt_embeds, structure_prompt_embeds], dim=0
            ).to(device)
            structure_add_text_embeds = torch.cat(
                [negative_structure_pooled_prompt_embeds, structure_pooled_prompt_embeds], dim=0
            ).to(device)
        else:
            structure_prompt_embeds = prompt_embeds
            structure_add_text_embeds = add_text_embeds

        # 3.2. Appearance prompt embeddings
        if appearance_prompt is not None and appearance_prompt != "":
            (
                appearance_prompt_embeds,
                negative_appearance_prompt_embeds,
                appearance_pooled_prompt_embeds,
                negative_appearance_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt = appearance_prompt,
                prompt_2 = None,  # prompt_2
                device = device,
                num_images_per_prompt = num_images_per_prompt,
                do_classifier_free_guidance = True,  # self.do_classifier_free_guidance, TODO: Support no CFG
                negative_prompt = negative_prompt if appearance_image is None else "",
                negative_prompt_2 = None,  # negative_prompt_2
                prompt_embeds = appearance_prompt_embeds,
                negative_prompt_embeds = None,  # negative_prompt_embeds
                pooled_prompt_embeds = appearance_pooled_prompt_embeds,  # pooled_prompt_embeds
                negative_pooled_prompt_embeds = None,  # negative_pooled_prompt_embeds
                lora_scale = text_encoder_lora_scale,
                clip_skip = self.clip_skip,
            )
            appearance_prompt_embeds = torch.cat(
                [negative_appearance_prompt_embeds, appearance_prompt_embeds], dim=0
            ).to(device)
            appearance_add_text_embeds = torch.cat(
                [negative_appearance_pooled_prompt_embeds, appearance_pooled_prompt_embeds], dim=0
            ).to(device)
        else:
            appearance_prompt_embeds = prompt_embeds
            appearance_add_text_embeds = add_text_embeds

        # 3.3. Prepare added time ids & embeddings, TODO: Support no CFG
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype = prompt_embeds.dtype,
            text_encoder_projection_dim = text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(device)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        latents, _ = self.prepare_latents(
            None, batch_size, num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents
        )

        if structure_image is not None:
            structure_image = preprocess(  # Center crop + resize
                structure_image, self.image_processor, height=height, width=width, resize_mode="crop"
            )
            _, clean_structure_latents = self.prepare_latents(
                structure_image, batch_size, num_images_per_prompt, num_channels_latents, height, width,
                prompt_embeds.dtype, device, generator, structure_latents,
            )
        else:
            clean_structure_latents = None
        structure_latents = latents if structure_latents is None else structure_latents

        if appearance_image is not None:
            appearance_image = preprocess(  # Center crop + resize
                appearance_image, self.image_processor, height=height, width=width, resize_mode="crop"
            )
            _, clean_appearance_latents = self.prepare_latents(
                appearance_image, batch_size, num_images_per_prompt, num_channels_latents, height, width,
                prompt_embeds.dtype, device, generator, appearance_latents,
            )
        else:
            clean_appearance_latents = None
        appearance_latents = latents if appearance_latents is None else appearance_latents

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        def denoising_value_valid(dnv):
            return isinstance(self.denoising_end, float) and 0 < dnv < 1

        if (
            self.denoising_end is not None
            and self.denoising_start is not None
            and denoising_value_valid(self.denoising_end)
            and denoising_value_valid(self.denoising_start)
            and self.denoising_start >= self.denoising_end
        ):
            raise ValueError(f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: {self.denoising_end} when using type float.")
        elif self.denoising_end is not None and denoising_value_valid(self.denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 7.2 Optionally get guidance scale embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7.3 Get batch order
        batch_order = deepcopy(BATCH_ORDER)
        if structure_image is not None:  # If image is provided, not generating, so no CFG needed
            batch_order.remove("structure_uncond")
        if appearance_image is not None:
            batch_order.remove("appearance_uncond")

        structure_control_stop_i, appearance_control_stop_i = get_last_control_i(control_schedule, num_inference_steps)
        if self_recurrence_schedule is None or len(self_recurrence_schedule) == 0:
            self_recurrence_schedule = [0] * num_inference_steps

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if i == structure_control_stop_i:  # If not generating structure/appearance, drop after last control
                    if "structure_uncond" not in batch_order:
                        batch_order.remove("structure_cond")
                if i == appearance_control_stop_i:
                    if "appearance_uncond" not in batch_order:
                        batch_order.remove("appearance_cond")

                register_attr(self, t=t.item(), do_control=True, batch_order=batch_order)

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                structure_latent_model_input = self.scheduler.scale_model_input(structure_latents, t)
                appearance_latent_model_input = self.scheduler.scale_model_input(appearance_latents, t)

                all_latent_model_input = {
                    "structure_uncond": structure_latent_model_input[0:1],
                    "appearance_uncond": appearance_latent_model_input[0:1],
                    "uncond": latent_model_input[0:1],
                    "structure_cond": structure_latent_model_input[0:1],
                    "appearance_cond": appearance_latent_model_input[0:1],
                    "cond": latent_model_input[0:1],
                }
                all_prompt_embeds = {
                    "structure_uncond": structure_prompt_embeds[0:1],
                    "appearance_uncond": appearance_prompt_embeds[0:1],
                    "uncond": prompt_embeds[0:1],
                    "structure_cond": structure_prompt_embeds[1:2],
                    "appearance_cond": appearance_prompt_embeds[1:2],
                    "cond": prompt_embeds[1:2],
                }
                all_add_text_embeds = {
                    "structure_uncond": structure_add_text_embeds[0:1],
                    "appearance_uncond": appearance_add_text_embeds[0:1],
                    "uncond": add_text_embeds[0:1],
                    "structure_cond": structure_add_text_embeds[1:2],
                    "appearance_cond": appearance_add_text_embeds[1:2],
                    "cond": add_text_embeds[1:2],
                }
                all_time_ids = {
                    "structure_uncond": add_time_ids[0:1],
                    "appearance_uncond": add_time_ids[0:1],
                    "uncond": add_time_ids[0:1],
                    "structure_cond": add_time_ids[1:2],
                    "appearance_cond": add_time_ids[1:2],
                    "cond": add_time_ids[1:2],
                }

                concat_latent_model_input = batch_dict_to_tensor(all_latent_model_input, batch_order)
                concat_prompt_embeds = batch_dict_to_tensor(all_prompt_embeds, batch_order)
                concat_add_text_embeds = batch_dict_to_tensor(all_add_text_embeds, batch_order)
                concat_add_time_ids = batch_dict_to_tensor(all_time_ids, batch_order)

                # Predict the noise residual
                added_cond_kwargs = {"text_embeds": concat_add_text_embeds, "time_ids": concat_add_time_ids}

                concat_noise_pred = self.unet(
                    concat_latent_model_input,
                    t,
                    encoder_hidden_states = concat_prompt_embeds,
                    timestep_cond = timestep_cond,
                    cross_attention_kwargs = self.cross_attention_kwargs,
                    added_cond_kwargs = added_cond_kwargs,
                ).sample
                all_noise_pred = batch_tensor_to_dict(concat_noise_pred, batch_order)

                # Classifier-free guidance, TODO: Support no CFG
                noise_pred = all_noise_pred["uncond"] +\
                    self.guidance_scale * (all_noise_pred["cond"] - all_noise_pred["uncond"])

                structure_noise_pred = all_noise_pred["structure_cond"]\
                    if "structure_cond" in batch_order else noise_pred
                if "structure_uncond" in all_noise_pred:
                    structure_noise_pred = all_noise_pred["structure_uncond"] +\
                        self.structure_guidance_scale * (structure_noise_pred - all_noise_pred["structure_uncond"])

                appearance_noise_pred = all_noise_pred["appearance_cond"]\
                    if "appearance_cond" in batch_order else noise_pred
                if "appearance_uncond" in all_noise_pred:
                    appearance_noise_pred = all_noise_pred["appearance_uncond"] +\
                        self.appearance_guidance_scale * (appearance_noise_pred - all_noise_pred["appearance_uncond"])

                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, all_noise_pred["cond"], guidance_rescale=self.guidance_rescale
                    )
                    if "structure_uncond" in all_noise_pred:
                        structure_noise_pred = rescale_noise_cfg(
                            structure_noise_pred, all_noise_pred["structure_cond"],
                            guidance_rescale=self.guidance_rescale
                        )
                    if "appearance_uncond" in all_noise_pred:
                        appearance_noise_pred = rescale_noise_cfg(
                            appearance_noise_pred, all_noise_pred["appearance_cond"],
                            guidance_rescale=self.guidance_rescale
                        )

                # Compute the previous noisy sample x_t -> x_t-1
                concat_noise_pred = torch.cat(
                    [structure_noise_pred, appearance_noise_pred, noise_pred], dim=0,
                )
                concat_latents = torch.cat(
                    [structure_latents, appearance_latents, latents], dim=0,
                )
                structure_latents, appearance_latents, latents = self.scheduler.step(
                    concat_noise_pred, t, concat_latents, **extra_step_kwargs,
                ).prev_sample.chunk(3)

                if clean_structure_latents is not None:
                    structure_latents = noise_prev(self.scheduler, t, clean_structure_latents)
                if clean_appearance_latents is not None:
                    appearance_latents = noise_prev(self.scheduler, t, clean_appearance_latents)

                # Self-recurrence
                for _ in range(self_recurrence_schedule[i]):
                    if hasattr(self.scheduler, "_step_index"):  # For fancier schedulers
                        self.scheduler._step_index -= 1

                    t_prev = 0 if i + 1 >= num_inference_steps else timesteps[i + 1]
                    latents = noise_t2t(self.scheduler, t_prev, t, latents)
                    latent_model_input = torch.cat([latents] * 2)

                    register_attr(self, t=t.item(), do_control=False, batch_order=["uncond", "cond"])

                    # Predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred_uncond, noise_pred_ = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states = prompt_embeds,
                        timestep_cond = timestep_cond,
                        cross_attention_kwargs = self.cross_attention_kwargs,
                        added_cond_kwargs = added_cond_kwargs,
                    ).sample.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_ - noise_pred_uncond)

                    if self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_, guidance_rescale=self.guidance_rescale)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # Callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds)
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    # add_neg_time_ids = callback_outputs.pop("add_neg_time_ids", add_neg_time_ids)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # "Reconstruction"
        if clean_structure_latents is not None:
            structure_latents = clean_structure_latents
        if clean_appearance_latents is not None:
            appearance_latents = clean_appearance_latents

        # For passing important information onto the refiner
        self.refiner_args = {"latents": latents.detach(), "prompt": prompt, "negative_prompt": negative_prompt}

        if output_type != "latent":
            # Make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                self.upcast_vae()
                vae_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                latents = latents.to(vae_dtype)
                structure_latents = structure_latents.to(vae_dtype)
                appearance_latents = appearance_latents.to(vae_dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            if decode_structure:
                structure = self.vae.decode(structure_latents / self.vae.config.scaling_factor, return_dict=False)[0]
                structure = self.image_processor.postprocess(structure, output_type=output_type)
            else:
                structure = structure_latents
            if decode_appearance:
                appearance = self.vae.decode(appearance_latents / self.vae.config.scaling_factor, return_dict=False)[0]
                appearance = self.image_processor.postprocess(appearance, output_type=output_type)
            else:
                appearance = appearance_latents

            # Cast back to fp16 if needed
            if self.vae.config.force_upcast:
                self.vae.to(dtype=torch.float16)

        else:
            # combined = torch.cat([latents, structure_latents, appearance_latents], dim=0)
            # return CtrlXStableDiffusionXLPipelineOutput(images=combined)
            return CtrlXStableDiffusionXLPipelineOutput(images=latents, structures=structure_latents, appearances=appearance_latents)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, structure, appearance)

        return CtrlXStableDiffusionXLPipelineOutput(images=image, structures=structure, appearances=appearance)
