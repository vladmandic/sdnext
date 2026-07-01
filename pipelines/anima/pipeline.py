# pylint: disable

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLWan, CosmosTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.cosmos.pipeline_output import CosmosImagePipelineOutput

logger = logging.get_logger(__name__)


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class AnimaTextToImagePipeline(DiffusionPipeline):
    """Pipeline for text-to-image generation using the Anima model.

    Anima uses a Cosmos Predict2 backbone with a Qwen3 text encoder and an LLM adapter
    that cross-attends T5 token embeddings to Qwen3 hidden states.
    """

    model_cpu_offload_seq = "text_encoder->llm_adapter->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        t5_tokenizer: PreTrainedTokenizerFast,
        llm_adapter,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            llm_adapter=llm_adapter,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int = 512,
    ):
        """Encode prompt through Qwen3 and run LLM adapter with T5 token IDs."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Check for empty prompts - return zero embeddings directly
        all_empty = all(p.strip() == "" for p in prompt)
        if all_empty:
            return torch.zeros(batch_size, 512, self.llm_adapter.config.target_dim, device=device, dtype=dtype)

        # Tokenize with Qwen3 tokenizer
        qwen_inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        qwen_input_ids = qwen_inputs.input_ids.to(device)
        qwen_attention_mask = qwen_inputs.attention_mask.to(device)

        # Get Qwen3 hidden states
        qwen_outputs = self.text_encoder(
            input_ids=qwen_input_ids,
            attention_mask=qwen_attention_mask,
        )
        qwen_hidden_states = qwen_outputs.last_hidden_state.to(dtype=dtype)

        # Tokenize with T5 tokenizer (we only need the IDs for the adapter embedding)
        t5_inputs = self.t5_tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs.input_ids.to(device)

        # Run LLM adapter: T5 token embeddings attend to Qwen3 hidden states
        adapted_embeds = self.llm_adapter(
            source_hidden_states=qwen_hidden_states,
            target_input_ids=t5_input_ids,
        )

        # Pad to 512 sequence length if shorter
        if adapted_embeds.shape[1] < 512:
            adapted_embeds = torch.nn.functional.pad(
                adapted_embeds, (0, 0, 0, 512 - adapted_embeds.shape[1])
            )

        return adapted_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._encode_prompt(prompt, device, dtype, max_sequence_length)
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._encode_prompt(negative_prompt, device, dtype, max_sequence_length)
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int = 1,
        dtype: torch.dtype = None,
        device: torch.device = None,
        generator=None,
        latents: torch.Tensor = None,
    ):
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def check_inputs(self, prompt, height, width, prompt_embeds=None):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: Optional[str | list[str]] = None,
        height: int = 768,
        width: int = 1360,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator | list[torch.Generator]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Callable[[int, int, Dict], None] | PipelineCallback | MultiPipelineCallbacks
        ] = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_frames = 1

        self.check_inputs(prompt, height, width, prompt_embeds)
        self._guidance_scale = guidance_scale # pylint: disable=attribute-defined-outside-init
        self._current_timestep = None # pylint: disable=attribute-defined-outside-init
        self._interrupt = False # pylint: disable=attribute-defined-outside-init

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # Prepare timesteps - use default descending schedule (1→0)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps=num_inference_steps, device=device
        )

        # Prepare latents
        transformer_dtype = self.transformer.dtype
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        # Denoising loop using CONST preconditioning (flow matching velocity model):
        #   - c_in = 1.0 (no input scaling)
        #   - timestep = sigma (passed directly)
        #   - model output is the velocity: denoised = x - velocity * sigma
        #   - CFG applied to velocity (equivalent to applying to denoised for linear preconditioning)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps) # pylint: disable=attribute-defined-outside-init

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t # pylint: disable=attribute-defined-outside-init
                sigma = self.scheduler.sigmas[i]

                # Pass sigma directly as timestep (CONST preconditioning)
                timestep = sigma.expand(latents.shape[0]).to(transformer_dtype)
                latent_model_input = latents.to(transformer_dtype)

                # Model predicts velocity (raw output IS the velocity for CONST)
                velocity = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0].float()

                if self.do_classifier_free_guidance:
                    velocity_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    velocity = velocity_uncond + self.guidance_scale * (velocity - velocity_uncond)

                # Euler step: scheduler computes x_next = x + (sigma_next - sigma) * velocity
                latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None # pylint: disable=attribute-defined-outside-init

        if output_type != "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            image = [batch[0] for batch in video]
            if isinstance(video, torch.Tensor):
                image = torch.stack(image)
            elif isinstance(video, np.ndarray):
                image = np.stack(image)
        else:
            image = latents[:, :, 0]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return CosmosImagePipelineOutput(images=image)
