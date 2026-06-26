"""Krea 2 (K2) text-to-image pipeline.

A single-stream flow-matching pipeline that conditions a custom DiT on stacked Qwen3-VL
hidden states and decodes with the Qwen-Image VAE. The encode, packing, denoise and decode
steps mirror the reference K2 inference code. This module imports only diffusers/transformers
so the repos can ship it for standalone use; SD.Next-specific wiring lives in the loader.
"""

import torch
from einops import rearrange, repeat

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class Krea2Pipeline(DiffusionPipeline, FromSingleFileMixin):
    r"""Text-to-image generation with Krea 2.

    Args:
        transformer (`Krea2Transformer2DModel`): single-stream flow-matching DiT.
        text_encoder (`Qwen3VLModel`): multimodal backbone tapped for stacked hidden states.
        tokenizer (`Qwen2Tokenizer`): tokenizer paired with `text_encoder`.
        vae (`AutoencoderKLQwenImage`): f8/16-channel latent autoencoder.
        scheduler (`FlowMatchEulerDiscreteScheduler`): exponential-shift flow-matching scheduler.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    # Conditioning template and layer taps (reference encoder.py / inference.py).
    PROMPT_TEMPLATE_PREFIX = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n"
    )
    PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PREFIX_TOKENS = 34
    SUFFIX_START = 5
    MAX_LENGTH = 512
    SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    MIN_RES = 256
    MAX_RES = 1280

    def __init__(self, transformer, text_encoder, tokenizer, vae, scheduler):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
        self.patch = int(getattr(transformer.config, "patch", 2))
        self.latent_channels = int(getattr(transformer.config, "channels", 16))
        self.vae_compression = 8  # AutoencoderKLQwenImage is f8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_compression)
        self._interrupt = False
        self._guidance_scale = None
        self._num_timesteps = 0

    # --- text conditioning: port of encoder.py Qwen3VLConditioner.forward ---
    def encode_prompt(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(hidden, mask)` where hidden is `(B, L, len(SELECT_LAYERS), txtdim)`.

        The user prompt is wrapped in the image-description chat template, padded to a fixed
        length, and the assistant suffix appended; the system prefix tokens are then dropped.
        """
        texts = [self.PROMPT_TEMPLATE_PREFIX + p for p in prompts]
        suffix = self.tokenizer([self.PROMPT_TEMPLATE_SUFFIX] * len(texts), return_tensors="pt").to(device)
        main = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.MAX_LENGTH + self.PREFIX_TOKENS - self.SUFFIX_START,
            return_tensors="pt",
        ).to(device)

        input_ids = torch.cat([main.input_ids, suffix.input_ids], dim=1)
        mask = torch.cat([main.attention_mask.bool(), suffix.attention_mask.bool()], dim=1)

        out = self.text_encoder(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        hidden = torch.stack([out.hidden_states[i] for i in self.SELECT_LAYERS], dim=2)
        return hidden[:, self.PREFIX_TOKENS:], mask[:, self.PREFIX_TOKENS:]

    # --- packing: port of sampling.prepare ---
    def pack_sequence(self, latent: torch.Tensor, text_mask: torch.Tensor):
        """Patchify the latent and build the joint text+image `(tokens, position_ids, mask)`."""
        batch, _, height, width = latent.shape
        patch = self.patch
        grid_h, grid_w = height // patch, width // patch
        device = latent.device

        img_ids = torch.zeros(grid_h, grid_w, 3, device=device)
        img_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        img_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        img_pos = repeat(img_ids, "h w c -> b (h w) c", b=batch)
        img_mask = torch.ones(batch, grid_h * grid_w, dtype=torch.bool, device=device)
        img_tokens = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        txt_pos = torch.zeros(batch, text_mask.shape[1], 3, device=device)
        pos = torch.cat([txt_pos, img_pos], dim=1)
        mask = torch.cat([text_mask, img_mask], dim=1)
        return img_tokens, pos, mask

    def prepare_latents(self, batch_size, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, self.latent_channels, height // self.vae_compression, width // self.vae_compression)
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        return image_seq_len * slope + (base_shift - slope * base_seq_len)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalize and decode latents with the Qwen-Image VAE (port of autoencoder.decode)."""
        cfg = self.vae.config
        mean = torch.tensor(cfg.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(cfg.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents = latents.unsqueeze(2) * std + mean  # (B, C, 1, H, W); the VAE treats images as 1-frame video
        image = self.vae.decode(latents).sample
        return image.squeeze(2)

    def encode_image(self, image, height, width, dtype, device):
        """Encode a pixel image to a normalized latent (inverse of decode_latents)."""
        pixels = self.image_processor.preprocess(image, height=height, width=width)
        pixels = pixels.to(device=device, dtype=self.vae.dtype).unsqueeze(2)  # (B, C, 1, H, W)
        cfg = self.vae.config
        mean = torch.tensor(cfg.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(cfg.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        raw = self.vae.encode(pixels).latent_dist.mode()
        return ((raw.to(dtype) - mean) / std).squeeze(2)

    @property
    def guidance_scale(self):
        return self._guidance_scale

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
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        image=None,
        strength: float = 0.6,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ):
        align = self.vae_compression * self.patch
        height = (height // align) * align
        width = (width // align) * align

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        device = self._execution_device
        dtype = self.transformer.dtype
        self._guidance_scale = guidance_scale
        self._interrupt = False

        is_distilled = bool(getattr(self.transformer.config, "is_distilled", False))
        do_cfg = guidance_scale is not None and guidance_scale > 0 and not is_distilled

        text, text_mask = self.encode_prompt(prompts, device)
        text = text.to(dtype)
        if do_cfg:
            negatives = [negative_prompt or ""] * len(prompts) if not isinstance(negative_prompt, list) else negative_prompt
            uncond, uncond_mask = self.encode_prompt(negatives, device)
            uncond = uncond.to(dtype)

        batch = len(prompts) * num_images_per_prompt
        text = text.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)
        if do_cfg:
            uncond = uncond.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_mask = uncond_mask.repeat_interleave(num_images_per_prompt, dim=0)

        cfg = self.scheduler.config
        grid_h = height // (self.vae_compression * self.patch)
        grid_w = width // (self.vae_compression * self.patch)
        mu = self.calculate_shift(
            grid_h * grid_w,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 6400),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu) # pylint: disable=unexpected-keyword-arg
        timesteps = self.scheduler.timesteps

        if image is not None:
            clean = self.encode_image(image, height, width, dtype, device)
            clean = clean.repeat_interleave(num_images_per_prompt, dim=0)
            init_steps = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_steps, 0)
            timesteps = timesteps[t_start:]
            noise = self.prepare_latents(batch, height, width, dtype, device, generator)
            latents = self.scheduler.scale_noise(clean, timesteps[:1], noise)
        else:
            latents = self.prepare_latents(batch, height, width, dtype, device, generator, latents)

        img, pos, mask = self.pack_sequence(latents, text_mask)
        if do_cfg:
            _, uncond_pos, uncond_full_mask = self.pack_sequence(latents, uncond_mask)
        self._num_timesteps = len(timesteps)
        num_train = cfg.get("num_train_timesteps", 1000)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                model_t = (t.float() / num_train).reshape(1).expand(batch).to(device=device, dtype=img.dtype)
                cond = self.transformer(
                    hidden_states=img, encoder_hidden_states=text, timestep=model_t,
                    position_ids=pos, attention_mask=mask, return_dict=False,
                )[0]
                if do_cfg:
                    neg = self.transformer(
                        hidden_states=img, encoder_hidden_states=uncond, timestep=model_t,
                        position_ids=uncond_pos, attention_mask=uncond_full_mask, return_dict=False,
                    )[0]
                    velocity = cond + guidance_scale * (cond - neg)
                else:
                    velocity = cond
                img = self.scheduler.step(velocity, t, img, return_dict=False)[0]

                if callback_on_step_end is not None:
                    # Unpack the packed tokens to a standard [B, C, h, w] latent for the callback's preview.
                    cb_kwargs = {}
                    if "latents" in (callback_on_step_end_tensor_inputs or ["latents"]):
                        cb_kwargs["latents"] = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=grid_h, w=grid_w, ph=self.patch, pw=self.patch)
                    callback_on_step_end(self, i, t, cb_kwargs)
                progress_bar.update()

        latent = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=grid_h, w=grid_w, ph=self.patch, pw=self.patch)

        if output_type == "latent":
            image = latent
        else:
            image = self.decode_latents(latent.to(dtype))
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


class Krea2Img2ImgPipeline(Krea2Pipeline):
    """Image-to-image task variant.

    The denoise path is identical to the base pipeline (which already accepts `image` + `strength`);
    this distinct class exists so the diffusers AUTO maps and `get_diffusers_task` can tell the two
    tasks apart, following the per-task-class convention (e.g. ChromaPipeline / ChromaImg2ImgPipeline).
    """
