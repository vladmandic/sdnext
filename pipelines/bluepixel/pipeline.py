import math

import numpy as np
import torch
from PIL import Image
from diffusers.pipelines.flux2.pipeline_flux2 import (
    Flux2ImageProcessor,
    Flux2Pipeline,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class BluePixelPipeline(Flux2Pipeline):
    def __init__(self, transformer, vae, scheduler, text_encoder, tokenizer, vae_scale_factor=64):
        super().__init__(
            scheduler=scheduler,
            vae=None,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        self.register_modules(vae=vae)
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128
        self._guidance_scale = None
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self._num_timesteps = 0
        self.last_mu = None
        self.last_mu_mode = None
        self.last_image_seq_len = 0

    @staticmethod
    @torch.no_grad()
    def _get_qwen3_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt,
        dtype=None,
        device=None,
        max_sequence_length=512,
        hidden_states_layers=(9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device
        prompts = [prompt] if isinstance(prompt, str) else prompt
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": single_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for single_prompt in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        output = text_encoder(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = torch.stack([output.hidden_states[layer] for layer in hidden_states_layers], dim=1)
        hidden_states = hidden_states.to(dtype=dtype, device=device)
        batch_size, num_layers, sequence_length, hidden_size = hidden_states.shape
        return hidden_states.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, num_layers * hidden_size)

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt=1,
        prompt_embeds=None,
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    ):
        device = device or self._execution_device
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompts = [""] if prompts is None else prompts
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompts,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_ids = self._prepare_text_ids(prompt_embeds).to(device)
        return prompt_embeds, text_ids

    def _encode_vae_image(self, image, generator):
        latents = self.vae.encode(image)[0]
        return latents * self.vae.config.scaling_factor

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_latents_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"generator list length {len(generator)} != batch_size {batch_size}")
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        latent_ids = self._prepare_latent_ids(latents).to(device)
        return self._pack_latents(latents), latent_ids

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        instruction=None,
        prompt=None,
        prompt_embeds=None,
        height=None,
        width=None,
        num_inference_steps=4,
        guidance_scale=3.0,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
        target_pixels=0,
        mu_mode="empirical",
    ):
        prompt = instruction if instruction is not None else prompt
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if mu_mode not in ("empirical", "training"):
            raise ValueError(f"Unknown mu_mode: {mu_mode}")

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        if prompt_embeds is None:
            prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
        else:
            prompt_embeds = prompt_embeds.to(device)
            text_ids = self._prepare_text_ids(prompt_embeds).to(device)

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=[""] * batch_size,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        condition_images = None
        if image is not None:
            images = image if isinstance(image, (list, tuple)) else [image]
            condition_images = []
            for input_image in images:
                input_image = input_image.convert("RGB")
                original_width, original_height = input_image.size
                processed_width, processed_height = original_width, original_height
                if target_pixels > 0:
                    scale = (target_pixels / (processed_width * processed_height)) ** 0.5
                    processed_width = max(1, int(processed_width * scale))
                    processed_height = max(1, int(processed_height * scale))
                target_ratio = original_width / original_height
                width_min = max(64, (processed_width - 64) // 64 * 64)
                width_max = (processed_width + 64) // 64 * 64
                height_min = max(64, (processed_height - 64) // 64 * 64)
                height_max = (processed_height + 64) // 64 * 64
                best_size = (processed_width, processed_height)
                best_error = float("inf")
                for candidate_width in range(width_min, width_max + 1, 64):
                    for candidate_height in range(height_min, height_max + 1, 64):
                        error = abs(candidate_width / candidate_height - target_ratio)
                        if error < best_error:
                            best_error = error
                            best_size = (candidate_width, candidate_height)
                input_image = input_image.resize(best_size, Image.Resampling.LANCZOS)
                processed_width, processed_height = best_size
                condition_images.append(self.image_processor.preprocess(input_image, height=processed_height, width=processed_width))
                height = height or processed_height
                width = width or processed_width

        height = height or 1024
        width = width or 1024
        num_latents_channels = self.transformer.config.in_channels
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_latents_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        if mu_mode == "training":
            if not self.scheduler.config.use_dynamic_shifting or self.scheduler.config.time_shift_type != "exponential": # pylint: disable=no-member
                raise ValueError("Training mu requires an exponential dynamic scheduler")
            mu = math.log(1 + image_seq_len) / (math.log(2.0) + image_seq_len)
        else:
            mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        dynamic_shifting = self.scheduler.config.use_dynamic_shifting # pylint: disable=no-member
        self.last_mu = mu if dynamic_shifting else None
        self.last_mu_mode = mu_mode if dynamic_shifting else "fixed"
        self.last_image_seq_len = int(image_seq_len)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        guidance = torch.full((1,), guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for index, timestep_value in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = timestep_value
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :latents.size(1)]
                if do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        negative_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    negative_noise_pred = negative_noise_pred[:, :latents.size(1)]
                    noise_pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)
                latents = self.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                if callback_on_step_end is not None:
                    callback_inputs = {"latents": latents, "prompt_embeds": prompt_embeds}
                    callback_kwargs = {name: callback_inputs[name] for name in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, index, timestep_value, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                if index == len(timesteps) - 1 or ((index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)
            latents = (latents / self.vae.config.scaling_factor).to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)
