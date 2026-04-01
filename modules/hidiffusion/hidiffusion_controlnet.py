from typing import Dict, Any, Tuple, Callable, Optional, Union, List
import torch
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models import ControlNetModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


controlnet_apply_steps_rate = 0.6


def make_diffusers_unet_2d_condition(block_class):

    class unet_2d_condition(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:
            default_overall_up_factor = 2**self.num_upsamplers
            forward_upsample_size = False
            upsample_size = None
            for dim in sample.shape[-2:]:
                if dim % default_overall_up_factor != 0:
                    forward_upsample_size = True
                    break
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)
            if encoder_attention_mask is not None:
                encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
            timesteps = timesteps.expand(sample.shape[0])
            t_emb = self.time_proj(timesteps)
            t_emb = t_emb.to(dtype=sample.dtype)
            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None
            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")
                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)
                    class_labels = class_labels.to(dtype=sample.dtype)
                class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_image":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`")
                image_embs = added_cond_kwargs.get("image_embeds")
                text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
                aug_emb = self.add_embedding(text_embs, image_embs)
            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`")
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`")
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)
            elif self.config.addition_embed_type == "image":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`")
                image_embs = added_cond_kwargs.get("image_embeds")
                aug_emb = self.add_embedding(image_embs)
            elif self.config.addition_embed_type == "image_hint":
                if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`")
                image_embs = added_cond_kwargs.get("image_embeds")
                hint = added_cond_kwargs.get("hint")
                aug_emb, hint = self.add_embedding(image_embs, hint)
                sample = torch.cat([sample, hint], dim=1)
            emb = emb + aug_emb if aug_emb is not None else emb
            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)
            if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`")
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`")
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(image_embeds)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`")
                image_embeds = added_cond_kwargs.get("image_embeds")
                image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)
            sample = self.conv_in(sample)
            if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)

            is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
            # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
            is_adapter = down_intrablock_additional_residuals is not None

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    # For t2i-adapter CrossAttnDownBlock2D
                    additional_residuals = {}
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
                else:
                    # sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        sample += down_intrablock_additional_residuals.pop(0)

                down_block_res_samples += res_samples

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    _, _, ori_H, ori_W = down_block_res_sample.shape
                    down_block_additional_residual = F.interpolate(down_block_additional_residual, (ori_H, ori_W), mode='bicubic')
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                _, _, ori_H, ori_W = sample.shape
                mid_block_additional_residual = F.interpolate(mid_block_additional_residual, (ori_H, ori_W), mode='bicubic')
                sample = sample + mid_block_additional_residual

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        # scale=lora_scale,
                    )
                    # sample = upsample_block(
                    #     hidden_states=sample,
                    #     temb=emb,
                    #     res_hidden_states_tuple=res_samples,
                    #     upsample_size=upsample_size,
                    #     scale=lora_scale,
                    # )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (sample,)

            return UNet2DConditionOutput(sample=sample)
    return unet_2d_condition


def make_diffusers_sdxl_contrtolnet_ppl(block_class):
    class sdxl_contrtolnet_ppl(block_class):
        # Save for unpatching later
        _parent = block_class

        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            control_image: PipelineImageInput = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            strength: float = 0.8,
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
            guess_mode: bool = False,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):
            # convert image to control_image to fit sdxl_controlnet ppl.
            if control_image is None:
                control_image = image
                image = None
                self.info['text_to_img_controlnet'] = True
            else:
                self.info['text_to_img_controlnet'] = False

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)
            controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
                control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
                control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
                mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
                control_guidance_start, control_guidance_end = (
                    mult * [control_guidance_start],
                    mult * [control_guidance_end],
                )

            # 1. Check inputs. Raise error if not correct
            if image is not None:
                # image-to-image controlnet
                self.check_inputs(
                    prompt,
                    prompt_2,
                    control_image,
                    strength,
                    num_inference_steps,
                    callback_steps,
                    negative_prompt,
                    negative_prompt_2,
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    None,
                    None,
                    controlnet_conditioning_scale,
                    control_guidance_start,
                    control_guidance_end,
                    callback_on_step_end_tensor_inputs,
                )
            else:
                # text-to-image controlnet
                self.check_inputs(
                    prompt,
                    prompt_2,
                    control_image,
                    callback_steps,
                    negative_prompt,
                    negative_prompt_2,
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    None,
                    None,
                    negative_pooled_prompt_embeds,
                    controlnet_conditioning_scale,
                    control_guidance_start,
                    control_guidance_end,
                    callback_on_step_end_tensor_inputs,
                )

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt,
                prompt_2,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # 4. Prepare image and controlnet_conditioning_image
            if image is not None:
                image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
                if isinstance(controlnet, ControlNetModel):
                    control_image = self.prepare_control_image(
                        image=control_image,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    height, width = control_image.shape[-2:]
                elif isinstance(controlnet, MultiControlNetModel):
                    control_images = []

                    for control_image_ in control_image:
                        control_image_ = self.prepare_control_image(
                            image=control_image_,
                            width=width,
                            height=height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=controlnet.dtype,
                            do_classifier_free_guidance=self.do_classifier_free_guidance,
                            guess_mode=guess_mode,
                        )

                        control_images.append(control_image_)

                    control_image = control_images
                    height, width = control_image[0].shape[-2:]
                else:
                    raise AssertionError
            else:
                if isinstance(controlnet, ControlNetModel):
                    control_image = self.prepare_image(
                        image=control_image,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    height, width = control_image.shape[-2:]
                elif isinstance(controlnet, MultiControlNetModel):
                    images = []

                    for image_ in control_image:
                        image_ = self.prepare_image(
                            image=image_,
                            width=width,
                            height=height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=controlnet.dtype,
                            do_classifier_free_guidance=self.do_classifier_free_guidance,
                            guess_mode=guess_mode,
                        )

                        images.append(image_)

                    control_image = images
                    height, width = image[0].shape[-2:]
                else:
                    raise AssertionError
            # 5. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            if image is not None:
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
                latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            else:
                timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            # 6. Prepare latent variables
            if image is not None:
                # image-to-image controlnet
                latents = self.prepare_latents(
                    image,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    True,
                )
            else:
                # text-to-image controlnet
                num_channels_latents = self.unet.config.in_channels
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )
                # num_channels_latents = self.unet.config.in_channels
                # shape = (batch_size * num_images_per_prompt, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
                # if isinstance(generator, list) and len(generator) != batch_size:
                #     raise ValueError(
                #         f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                #         f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                #     )

                # if latents is None:
                #     latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                # else:
                #     latents = latents.to(device)

                # # scale the initial noise by the standard deviation required by the scheduler
                # latents = latents * self.scheduler.init_noise_sigma

            # 7. Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7.1 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

            # 7.2 Prepare added time ids & embeddings
            if image is not None:
                if isinstance(control_image, list):
                    original_size = original_size or control_image[0].shape[-2:]
                else:
                    original_size = original_size or control_image.shape[-2:]
                target_size = target_size or (height, width)

                if negative_original_size is None:
                    negative_original_size = original_size
                if negative_target_size is None:
                    negative_target_size = target_size
                add_text_embeds = pooled_prompt_embeds

                if self.text_encoder_2 is None:
                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                else:
                    text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

                add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    aesthetic_score,
                    negative_aesthetic_score,
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
                add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                add_text_embeds = add_text_embeds.to(device)
                add_time_ids = add_time_ids.to(device)
            else:
                if isinstance(control_image, list):
                    original_size = original_size or control_image[0].shape[-2:]
                else:
                    original_size = original_size or control_image.shape[-2:]
                target_size = target_size or (height, width)

                add_text_embeds = pooled_prompt_embeds
                if self.text_encoder_2 is None:
                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                else:
                    text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

                add_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )

                if negative_original_size is not None and negative_target_size is not None:
                    negative_add_time_ids = self._get_add_time_ids(
                        negative_original_size,
                        negative_crops_coords_top_left,
                        negative_target_size,
                        dtype=prompt_embeds.dtype,
                        text_encoder_projection_dim=text_encoder_projection_dim,
                    )
                else:
                    negative_add_time_ids = add_time_ids

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                add_text_embeds = add_text_embeds.to(device)
                add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    # controlnet(s) inference
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        controlnet_added_cond_kwargs = {
                            "text_embeds": add_text_embeds.chunk(2)[1],
                            "time_ids": add_time_ids.chunk(2)[1],
                        }
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds
                        controlnet_added_cond_kwargs = added_cond_kwargs

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]


                    if i < controlnet_apply_steps_rate * num_inference_steps:

                        original_h, original_w = (128,128)
                        _, _, model_input_h, model_input_w = control_model_input.shape
                        downsample_factor = max(model_input_h/original_h, model_input_w/original_w)
                        downsample_size = (int(model_input_h//downsample_factor)//8*8, int(model_input_w//downsample_factor)//8*8)

                        # original_pixel_h, original_pixel_w = (1024,1024)
                        # _, _, pixel_h, pixel_w = control_image.shape
                        # downsample_pixel_factor = max(pixel_h/original_pixel_h, pixel_w/original_pixel_w)
                        # downsample_pixel_size = (int(pixel_h//downsample_pixel_factor)//8*8, int(pixel_w//downsample_pixel_factor)//8*8)
                        downsample_pixel_size = [downsample_size[0]*8, downsample_size[1]*8]

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            F.interpolate(control_model_input, downsample_size),
                            # control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=F.interpolate(control_image, downsample_pixel_size),
                            # controlnet_cond=control_image,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            added_cond_kwargs=controlnet_added_cond_kwargs,
                            return_dict=False,
                        )

                    if guess_mode and self.do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    if i < controlnet_apply_steps_rate * num_inference_steps:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=None,
                            mid_block_additional_residual=None,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            # If we do sequential model offloading, let's offload unet and controlnet
            # manually for max memory savings
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.unet.to("cpu")
                self.controlnet.to("cpu")
                torch.cuda.empty_cache()

            if output_type != "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                if needs_upcasting:
                    self.upcast_vae()
                    latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
            else:
                image = latents
                return StableDiffusionXLPipelineOutput(images=image)

            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)
    return sdxl_contrtolnet_ppl
