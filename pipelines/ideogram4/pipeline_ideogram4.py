"""Ideogram 4 text-to-image pipeline (diffusers-native, SD.Next integration).

Owns its sampling loop: a packed text+image sequence is denoised with asymmetric
dual-branch classifier-free guidance over two transformers (conditional and a
separately-trained unconditional tower), then the image latents are denormalized,
unpatchified, and decoded by the VAE.

The ``__call__`` signature names every argument SD.Next forwards (it prunes kwargs
against the signature) and drives the diffusers ``callback_on_step_end`` contract
for progress, interrupt, and preview.
"""

from __future__ import annotations

import torch
from PIL import Image

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from pipelines.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from pipelines.ideogram4.latent_norm import get_latent_norm
from pipelines.ideogram4.scheduler_ideogram4 import (
    get_schedule_for_resolution,
    make_step_intervals,
)
from pipelines.ideogram4.text_encoder_ideogram4 import encode_text, tokenize


class Ideogram4Pipeline(DiffusionPipeline):
    """Ideogram 4 flow-matching text-to-image pipeline."""

    def __init__(self, transformer, unconditional_transformer, text_encoder, tokenizer, vae, scheduler) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            unconditional_transformer=unconditional_transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
        self.patch_size = 2
        self.ae_scale_factor = 8
        self.max_text_tokens = 2048
        self.latent_shift = None
        self.latent_scale = None
        self._num_timesteps = 0

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    def latent_norm(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent_shift is None:
            self.latent_shift, self.latent_scale = get_latent_norm()
        return self.latent_shift.to(device=device, dtype=dtype), self.latent_scale.to(device=device, dtype=dtype)

    def build_inputs(self, prompts: list[str], height: int, width: int, device) -> dict:
        """Build the packed (text tokens + image latent tokens) sequence for one batch."""
        tokenized = [tokenize(self.tokenizer, p, self.max_text_tokens) for p in prompts]
        batch_size = len(prompts)

        patch = self.patch_size * self.ae_scale_factor
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"height/width must be divisible by patch_size*ae_scale_factor={patch}")
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w

        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        # Image position ids (t=0, h, w), offset to stay disjoint from text positions.
        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, (toks, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len  # layout: [pad] [text] [image]

            token_ids[b, offset : offset + num_text] = toks

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
            segment_ids[b, offset : offset + total_unpadded] = 1

        return {
            "token_ids": token_ids.to(device),
            "text_position_ids": text_position_ids.to(device),
            "position_ids": position_ids.to(device),
            "segment_ids": segment_ids.to(device),
            "indicator": indicator.to(device),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def init_noise(self, batch_size: int, num_image_tokens: int, latent_dim: int, generator, device) -> torch.Tensor:
        if isinstance(generator, list):
            samples = [
                torch.randn((1, num_image_tokens, latent_dim), generator=generator[b % len(generator)], device=device, dtype=torch.float32)
                for b in range(batch_size)
            ]
            return torch.cat(samples, dim=0)
        return torch.randn((batch_size, num_image_tokens, latent_dim), generator=generator, device=device, dtype=torch.float32)

    def resolve_sampling(self, num_inference_steps: int, guidance_scale: float, device):
        """Map SD.Next's steps + CFG scale to a flat per-step guidance schedule."""
        num_steps = int(num_inference_steps)
        guidance_schedule = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=device)
        return num_steps, guidance_schedule

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        generator=None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        **kwargs,
    ):
        device = self._execution_device

        if prompt is None:
            prompts = [""]
        elif isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)
        batch_size = len(prompts)

        num_steps, gw_per_step = self.resolve_sampling(num_inference_steps, guidance_scale, device)
        schedule = get_schedule_for_resolution((height, width), known_mean=self.scheduler.config.mu, std=self.scheduler.config.std)
        step_intervals = make_step_intervals(num_steps).to(device)

        inputs = self.build_inputs(prompts, height=height, width=width, device=device)
        num_image_tokens = inputs["num_image_tokens"]
        grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
        max_text_tokens = inputs["max_text_tokens"]
        latent_dim = self.transformer.config.in_channels

        # The tapped forward bypasses the offload hook, so move the encoder on-device, then free it after.
        from modules import devices, shared
        self.text_encoder.to(device)
        llm_features = encode_text(self.text_encoder, inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
        self.text_encoder.to(devices.cpu)

        # At guidance 1.0 the unconditional velocity has zero weight, so skip the second
        # tower; only build the negative branch when some step needs it.
        gw_values = gw_per_step.tolist()
        use_cfg = any(gw != 1.0 for gw in gw_values)

        neg_position_ids = neg_segment_ids = neg_indicator = neg_llm_features = None
        if use_cfg:
            # Negative branch is image-only (asymmetric CFG) with zeroed conditioning.
            neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
            neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
            neg_indicator = inputs["indicator"][:, max_text_tokens:]
            neg_llm_features = torch.zeros(batch_size, num_image_tokens, llm_features.shape[-1], dtype=llm_features.dtype, device=device)

        z = self.init_noise(batch_size, num_image_tokens, latent_dim, generator, device)
        text_z_padding = torch.zeros(batch_size, max_text_tokens, latent_dim, dtype=torch.float32, device=device)

        self._num_timesteps = num_steps
        with self.progress_bar(total=num_steps) as progress_bar:
            for i in range(num_steps - 1, -1, -1):
                t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
                s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
                t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

                pos_z = torch.cat([text_z_padding, z], dim=1)
                pos_out = self.transformer(
                    llm_features=llm_features,
                    x=pos_z,
                    t=t,
                    position_ids=inputs["position_ids"],
                    segment_ids=inputs["segment_ids"],
                    indicator=inputs["indicator"],
                )
                pos_v = pos_out[:, max_text_tokens:]

                gw_i = gw_values[i]
                if gw_i == 1.0:
                    v = pos_v  # unconditional weight is zero; skip the second tower
                else:
                    neg_v = self.unconditional_transformer(
                        llm_features=neg_llm_features,
                        x=z,
                        t=t,
                        position_ids=neg_position_ids,
                        segment_ids=neg_segment_ids,
                        indicator=neg_indicator,
                    )
                    v = gw_i * pos_v + (1.0 - gw_i) * neg_v
                z = z + v * (s_val - t_val)

                if callback_on_step_end is not None:
                    cb = callback_on_step_end(self, num_steps - 1 - i, t_val, {"latents": z})
                    if isinstance(cb, dict):
                        z = cb.get("latents", z)
                # Live preview: denorm+unpack into Flux.2 latent space for TAE FLUX.2.
                shared.state.current_latent = self.denorm_unpack(z, grid_h, grid_w)
                progress_bar.update()

        if output_type == "latent":
            images = z
        else:
            images = self.decode_latents(z, grid_h=grid_h, grid_w=grid_w, output_type=output_type)

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)

    def denorm_unpack(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Denormalize the packed latent and unpatchify to (B, ae_channels, H, W) in VAE space."""
        batch_size = z.shape[0]
        patch = self.patch_size
        shift, scale = self.latent_norm(z.device, torch.float32)
        z = z.float() * scale + shift
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        return z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

    def decode_latents(self, z: torch.Tensor, grid_h: int, grid_w: int, output_type: str = "pil"):
        """Denormalize, unpatchify, and VAE-decode the image latents."""
        z = self.denorm_unpack(z, grid_h, grid_w)
        vae_dtype = next((p.dtype for p in self.vae.parameters() if torch.is_floating_point(p)), torch.float32)
        decoded = self.vae.decode(z.to(vae_dtype), return_dict=False)[0]

        decoded = decoded.float().clamp(-1.0, 1.0)
        decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
        decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return decoded
        return [Image.fromarray(arr) for arr in decoded]
