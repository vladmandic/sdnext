import os
import torch
import diffusers
from modules import shared, devices, sd_models


def get_timestep_ratio_conditioning(t, alphas_cumprod):
    s = torch.tensor([0.008])
    clamp_range = [0, 1]
    min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
    var = alphas_cumprod[t]
    var = var.clamp(*clamp_range)
    s, min_var = s.to(var.device), min_var.to(var.device)
    ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
    return ratio


def load_text_encoder(path):
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection
    from accelerate.utils.modeling import set_module_tensor_to_device
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    try:
        config = CLIPTextConfig(
            architectures=["CLIPTextModelWithProjection"],
            attention_dropout=0.0,
            bos_token_id=49406,
            dropout=0.0,
            eos_token_id=49407,
            hidden_act="gelu",
            hidden_size=1280,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=5120,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            model_type="clip_text_model",
            num_attention_heads=20,
            num_hidden_layers=32,
            pad_token_id=1,
            projection_dim=1280,
            vocab_size=49408
        )

        shared.log.info(f'Load Text Encoder: name="{os.path.basename(os.path.splitext(path)[0])}" file="{path}"')

        with init_empty_weights():
            text_encoder = CLIPTextModelWithProjection(config)

        state_dict = load_file(path)

        for key in list(state_dict.keys()):
            set_module_tensor_to_device(text_encoder, key, devices.device, value=state_dict.pop(key), dtype=devices.dtype)

        return text_encoder

    except Exception as e:
        text_encoder = None
        shared.log.error(f'Failed to load Text Encoder model: {e}')
        return None


def load_prior(path, config_file="default"):
    from diffusers.models.unets import StableCascadeUNet
    prior_text_encoder = None

    if config_file == "default":
        config_file = os.path.splitext(path)[0] + '.json'
    if not os.path.exists(config_file):
        if round(os.path.getsize(path) / 1024 / 1024 / 1024) < 5: # diffusers fails to find the configs from huggingface
            config_file = "configs/stable-cascade/prior_lite/config.json"
        else:
            config_file = "configs/stable-cascade/prior/config.json"

    shared.log.info(f'Load UNet: name="{os.path.basename(os.path.splitext(path)[0])}" file="{path}" config="{config_file}"')
    prior_unet = StableCascadeUNet.from_single_file(path, config=config_file, torch_dtype=devices.dtype_unet, cache_dir=shared.opts.diffusers_dir)

    if os.path.isfile(os.path.splitext(path)[0] + "_text_encoder.safetensors"): # OneTrainer
        prior_text_encoder = load_text_encoder(os.path.splitext(path)[0] + "_text_encoder.safetensors")
    elif os.path.isfile(os.path.splitext(path)[0] + "_text_model.safetensors"): # KohyaSS
        prior_text_encoder = load_text_encoder(os.path.splitext(path)[0] + "_text_model.safetensors")

    return prior_unet, prior_text_encoder


def load_cascade_combined(checkpoint_info, diffusers_load_config=None):
    from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline, StableCascadeCombinedPipeline
    from diffusers.models.unets import StableCascadeUNet
    from modules.sd_unet import unet_dict

    if diffusers_load_config is None:
        diffusers_load_config = {}

    diffusers_load_config.pop("vae", None)
    if 'cascade' in checkpoint_info.name.lower():
        diffusers_load_config["variant"] = 'bf16'

    if shared.opts.sd_unet != "Default" or 'stabilityai' in checkpoint_info.name.lower():
        if 'cascade' in checkpoint_info.name and ('lite' in checkpoint_info.name or (checkpoint_info.hash is not None and 'abc818bb0d' in checkpoint_info.hash)):
            decoder_folder = 'decoder_lite'
            prior_folder = 'prior_lite'
        else:
            decoder_folder = 'decoder'
            prior_folder = 'prior'
        if 'cascade' in checkpoint_info.name.lower():
            decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder=decoder_folder, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
            decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", cache_dir=shared.opts.diffusers_dir, decoder=decoder_unet, text_encoder=None, **diffusers_load_config)
        else:
            decoder = StableCascadeDecoderPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, text_encoder=None, **diffusers_load_config)
        # shared.log.debug(f'StableCascade {decoder_folder}: scale={decoder.latent_dim_scale}')
        prior_text_encoder = None
        if shared.opts.sd_unet != "Default":
            prior_unet, prior_text_encoder = load_prior(unet_dict[shared.opts.sd_unet])
        else:
            prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder=prior_folder, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
        if prior_text_encoder is not None:
            prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", cache_dir=shared.opts.diffusers_dir, prior=prior_unet, text_encoder=prior_text_encoder, image_encoder=None, feature_extractor=None, **diffusers_load_config)
        else:
            prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", cache_dir=shared.opts.diffusers_dir, prior=prior_unet, image_encoder=None, feature_extractor=None, **diffusers_load_config)
        # shared.log.debug(f'StableCascade {prior_folder}: scale={prior.resolution_multiple}')
        sd_model = StableCascadeCombinedPipeline(
            tokenizer=decoder.tokenizer,
            text_encoder=None,
            decoder=decoder.decoder,
            scheduler=decoder.scheduler,
            vqgan=decoder.vqgan,
            prior_prior=prior.prior,
            prior_text_encoder=prior.text_encoder,
            prior_tokenizer=prior.tokenizer,
            prior_scheduler=prior.scheduler,
            prior_feature_extractor=None,
            prior_image_encoder=None)
    else:
        sd_model = StableCascadeCombinedPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)

    sd_model.prior_pipe.scheduler.config.clip_sample = False
    sd_model.decoder_pipe.text_encoder = sd_model.text_encoder = None  # Nothing uses the decoder's text encoder
    sd_model.prior_pipe.image_encoder = sd_model.prior_image_encoder = None # No img2img is implemented yet
    sd_model.prior_pipe.feature_extractor = sd_model.prior_feature_extractor = None # No img2img is implemented yet

    #de-dupe
    del sd_model.decoder_pipe.text_encoder
    del sd_model.prior_prior
    del sd_model.prior_text_encoder
    del sd_model.prior_tokenizer
    del sd_model.prior_scheduler
    del sd_model.prior_feature_extractor
    del sd_model.prior_image_encoder

    # Custom sampler support
    sd_model.decoder_pipe = StableCascadeDecoderPipelineFixed(
        decoder=sd_model.decoder_pipe.decoder,
        tokenizer=sd_model.decoder_pipe.tokenizer,
        scheduler=sd_model.decoder_pipe.scheduler,
        vqgan=sd_model.decoder_pipe.vqgan,
        text_encoder=None,
        latent_dim_scale=sd_model.decoder_pipe.config.latent_dim_scale,
    )

    devices.torch_gc(force=True, reason='load')
    shared.log.debug(f'StableCascade combined: {sd_model.__class__.__name__}')
    return sd_model


# Balanced offload hooks:
class StableCascadeDecoderPipelineFixed(diffusers.StableCascadeDecoderPipeline):
    def guidance_scale(self): # pylint: disable=invalid-overridden-method
        return self._guidance_scale

    def do_classifier_free_guidance(self): # pylint: disable=invalid-overridden-method
        return self._guidance_scale > 1

    @torch.no_grad()
    def __call__(
        self,
        image_embeddings,
        prompt=None,
        num_inference_steps=10,
        guidance_scale=0.0,
        negative_prompt=None,
        prompt_embeds=None,
        prompt_embeds_pooled=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_pooled=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
    ):
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        # 0. Define commonly used variables
        guidance_scale = guidance_scale or 0.0
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1
        device = self._execution_device
        dtype = self.decoder.dtype

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if isinstance(image_embeddings, list):
            image_embeddings = torch.cat(image_embeddings, dim=0)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Compute the effective number of images per prompt
        # We must account for the fact that the image embeddings from the prior can be generated with num_images_per_prompt > 1
        # This results in a case where a single prompt is associated with multiple image embeddings
        # Divide the number of image embeddings by the batch size to determine if this is the case.
        num_images_per_prompt = num_images_per_prompt * (image_embeddings.shape[0] // batch_size)

        # 2. Encode caption
        if prompt_embeds is None and negative_prompt_embeds is None:
            _, prompt_embeds_pooled, _, negative_prompt_embeds_pooled = self.encode_prompt(
                prompt=prompt,
                device=device,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_pooled=prompt_embeds_pooled,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
            )

        # The pooled embeds from the prior are pooled again before being passed to the decoder
        prompt_embeds_pooled = (
            torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
            if self.do_classifier_free_guidance
            else prompt_embeds_pooled
        )
        effnet = (
            torch.cat([image_embeddings, torch.zeros_like(image_embeddings)])
            if self.do_classifier_free_guidance
            else image_embeddings
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latents
        latents = self.prepare_latents(
            batch_size, image_embeddings, num_images_per_prompt, dtype, device, generator, latents, self.scheduler
        )

        if isinstance(self.scheduler, diffusers.DDPMWuerstchenScheduler):
            timesteps = timesteps[:-1]
        else:
            if hasattr(self.scheduler.config, "clip_sample") and self.scheduler.config.clip_sample: # pylint: disable=no-member
                self.scheduler.config.clip_sample = False  # disample sample clipping

        # 6. Run denoising loop
        if hasattr(self.scheduler, "betas"):
            alphas = 1.0 - self.scheduler.betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = []

        self._num_timesteps = len(timesteps) # pylint: disable=attribute-defined-outside-init
        for i, t in enumerate(self.progress_bar(timesteps)):
            if not isinstance(self.scheduler, diffusers.DDPMWuerstchenScheduler):
                if len(alphas_cumprod) > 0:
                    timestep_ratio = get_timestep_ratio_conditioning(t.long().cpu(), alphas_cumprod)
                    timestep_ratio = timestep_ratio.expand(latents.size(0)).to(dtype).to(device)
                else:
                    timestep_ratio = t.float().div(self.scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
            else:
                timestep_ratio = t.expand(latents.size(0)).to(dtype)

            # 7. Denoise latents
            predicted_latents = self.decoder(
                sample=torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents,
                timestep_ratio=torch.cat([timestep_ratio] * 2) if self.do_classifier_free_guidance else timestep_ratio,
                clip_text_pooled=prompt_embeds_pooled,
                effnet=effnet,
                return_dict=False,
            )[0]

            # 8. Check for classifier free guidance and apply it
            if self.do_classifier_free_guidance:
                predicted_latents_text, predicted_latents_uncond = predicted_latents.chunk(2)
                predicted_latents = torch.lerp(predicted_latents_uncond, predicted_latents_text, self.guidance_scale)

            # 9. Renoise latents to next timestep
            if not isinstance(self.scheduler, diffusers.DDPMWuerstchenScheduler):
                timestep_ratio = t
            latents = self.scheduler.step(
                model_output=predicted_latents,
                timestep=timestep_ratio,
                sample=latents,
                generator=generator,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        if output_type not in ["pt", "np", "pil", "latent"]:
            raise ValueError(
                f"Only the output types `pt`, `np`, `pil` and `latent` are supported not output_type={output_type}"
            )

        if output_type != "latent":
            if shared.opts.diffusers_offload_mode == "balanced":
                shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            else:
                self.maybe_free_model_hooks()
            # 10. Scale and decode the image latents with vq-vae
            latents = self.vqgan.config.scale_factor * latents
            images = self.vqgan.decode(latents).sample.clamp(0, 1)
            if output_type == "np":
                images = images.permute(0, 2, 3, 1).cpu().float().numpy()  # float() as bfloat16-> numpy doesnt work
            elif output_type == "pil":
                from modules import images_sharpfin
                images = [images_sharpfin.to_pil(images[i]) for i in range(images.shape[0])]
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        else:
            images = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return images
        return diffusers.ImagePipelineOutput(images)
