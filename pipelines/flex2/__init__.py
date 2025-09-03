from diffusers import FluxControlPipeline, FluxTransformer2DModel
from typing import Any, Callable, Dict, List, Optional, Union
import torch

from diffusers.image_processor import PipelineImageInput
import numpy as np
import torch.nn.functional as F
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps, XLA_AVAILABLE


class Flex2Pipeline(FluxControlPipeline):
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
    ):
        super().__init__(scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer)

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        inpaint_image=None,
        inpaint_mask=None,
        control_image=None,
    ):
        super().check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        if inpaint_image is not None and inpaint_mask is None:
            raise ValueError(
                "If `inpaint_image` is passed, `inpaint_mask` must be passed as well. "
                "Please make sure to pass both `inpaint_image` and `inpaint_mask`."
            )
        if inpaint_mask is not None and inpaint_image is None:
            raise ValueError(
                "If `inpaint_mask` is passed, `inpaint_image` must be passed as well. "
                "Please make sure to pass both `inpaint_image` and `inpaint_mask`."
            )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        inpaint_image: Optional[PipelineImageInput] = None,
        inpaint_mask: Optional[PipelineImageInput] = None,
        control_image: Optional[PipelineImageInput] = None,
        control_strength: Optional[float] = 1.0,
        control_stop: Optional[float] = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            inpaint_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The image to be inpainted.
            inpaint_mask (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                A black and white mask to be used for inpainting. The white pixels are the areas to be inpainted, while the
                black pixels are the areas to be kept.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The control image (line, depth, pose, etc.) to be used for the generation. The control image
            control_strength (`float`, *optional*, defaults to 1.0):
                The strength of the control image. The higher the value, the more the control image will be used to
                guide the generation. The lower the value, the less the control image will be used to guide the
                generation.
            control_stop (`float`, *optional*, defaults to 1.0):
                The percentage of the generation to drop out the control. 0.0 to 1.0.  0.5 mean the control will be dropped
                out at 50% of the generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare text embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        # only prepare latents for non controls
        # (16 + 1 + 16 )
        num_control_channels = 33
        num_channels_latents = num_channels_latents - num_control_channels

        control_latents = None
        inpaint_latents = None
        inpaint_latents_mask = None

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # process the control and inpaint channels

        if control_image is None:
            control_latents = torch.zeros(
                batch_size * num_images_per_prompt,
                16,
                latent_height,
                latent_width,
                device=device,
                dtype=self.vae.dtype,
            )
        else:
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )
            control_image = self.vae.encode(control_image).latent_dist.sample(generator=generator)
            control_latents = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # apply control strength
        control_latents = control_latents * control_strength

        if inpaint_image is None and inpaint_mask is None:
            inpaint_latents = torch.zeros(
                batch_size * num_images_per_prompt,
                16,
                latent_height,
                latent_width,
                device=device,
                dtype=self.vae.dtype,
            )
            inpaint_latents_mask = torch.ones(
                batch_size * num_images_per_prompt,
                1,
                latent_height,
                latent_width,
                device=device,
                dtype=self.vae.dtype,
            )
        else:
            inpaint_image = self.prepare_image(
                image=inpaint_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )
            inpaint_image = self.vae.encode(inpaint_image).latent_dist.sample(generator=generator)
            inpaint_latents = (inpaint_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            height_inpaint_image, width_inpaint_image = inpaint_image.shape[2:]

            inpaint_mask = self.prepare_image(
                image=inpaint_mask,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )
            # mask is 3 ch -1 to 1. make it 1ch, 0 to 1
            inpaint_mask = inpaint_mask[:, 0:1, :, :] * 0.5 + 0.5
            # resize to match height_inpaint_image and width_inpaint_image
            inpaint_latents_mask = F.interpolate(inpaint_mask, size=(height_inpaint_image, width_inpaint_image), mode="bilinear", align_corners=False)

        # apply inverted mask to inpaint latents
        inpaint_latents = inpaint_latents * (1 - inpaint_latents_mask)

        # concat the latent controls on the channel dimension every step
        latent_controls = torch.cat([inpaint_latents, inpaint_latents_mask, control_latents], dim=1)
        latent_no_controls = torch.cat([inpaint_latents, inpaint_latents_mask, torch.zeros_like(control_latents)], dim=1)

        # pack the controls
        height_latent_controls, width_latent_controls = latent_controls.shape[2:]
        packed_latent_controls = self._pack_latents(
            latent_controls,
            batch_size * num_images_per_prompt,
            num_control_channels,
            height_latent_controls,
            width_latent_controls,
        )
        packed_latent_no_controls = self._pack_latents(
            latent_no_controls,
            batch_size * num_images_per_prompt,
            num_control_channels,
            height_latent_controls,
            width_latent_controls,
        )

        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        control_cutoff = int(len(timesteps) * control_stop)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                control_latents = packed_latent_controls if i < control_cutoff else packed_latent_no_controls

                latent_model_input = torch.cat([latents, control_latents], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
