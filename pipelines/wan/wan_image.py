from typing import Any, Callable, Dict, List, Optional, Union
import torch
import diffusers
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput

from modules import devices


class WanImagePipeline(diffusers.WanPipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        strength: float = 0.3, # new
        image: PipelineImageInput = None, # new
    ):
        # get img2img timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=devices.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        # monkey patch original pipeline
        self.scheduler.timesteps = timesteps
        # self.scheduler._step_index = 0
        self.scheduler.orig_set_timesteps = self.scheduler.set_timesteps
        self.scheduler.set_timesteps = lambda *args, **kwargs: None

        # prepare latents
        latents = self.img2img_prepare_latents(
            image=image,
            timesteps=timesteps,
            dtype=devices.dtype,
            device=devices.device,
            generator=generator,
        )

        # call original pipeline
        result = super().__call__( # pylint: disable=no-member
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        # un-monkey patch original pipeline
        self.scheduler.set_timesteps = self.scheduler.orig_set_timesteps
        return result

    def get_timesteps(self, num_inference_steps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            # self.scheduler.set_begin_index(t_start * self.scheduler.order)
            self.scheduler.set_begin_index(0)
        return timesteps, num_inference_steps - t_start

    def img2img_prepare_latents(
        self,
        image: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor
        from diffusers.video_processor import VideoProcessor

        if isinstance(image, list):
            image = image[0] # ignore batch for now

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        image_tensor = video_processor.preprocess(image, None, None) # convert PIL to [B, C, H, W] # channels may need rearrange
        image_tensor = image_tensor.squeeze(0).to(device=device, dtype=dtype)
        image_tensor = image_tensor[None, :, None, :, :] # expand before encode to [B, C, N, H, W]
        encoder_output = self.vae.encode(image_tensor)
        # init_latents = encoder_output.latent_dist.mode() # argmax or sample?
        init_latents = encoder_output.latent_dist.sample(generator)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(1, self.vae.config.z_dim, 1, 1, 1)
        init_latents = ((init_latents.float() - latents_mean) * latents_std).to(dtype) # normalized to standard distribution range

        init_noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)
        init_timestep = timesteps[:1]
        noised_latents = self.scheduler.add_noise(init_latents, init_noise, init_timestep)

        return noised_latents
