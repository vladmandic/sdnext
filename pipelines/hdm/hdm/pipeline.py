from typing import Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers import AutoencoderKL
from transformers import Qwen3Model, Qwen2Tokenizer

from .modules.xut import XUDiTConditionModel
from ..xut.modules.axial_rope import make_axial_pos_no_cache


class HDMXUTPipeline(DiffusionPipeline):
    transformer: XUDiTConditionModel
    tokenizer = Qwen2Tokenizer
    text_encoder: Qwen3Model
    vae: AutoencoderKL

    def __init__(
        self,
        transformer: XUDiTConditionModel,
        text_encoder: Qwen3Model,
        tokenizer: Qwen2Tokenizer,
        vae: AutoencoderKL,
        scheduler,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_mean = torch.tensor(self.vae.config.latents_mean)[None, :, None, None]
        self.vae_std = torch.tensor(self.vae.config.latents_std)[None, :, None, None]

    def apply_compile(self, *args, **kwargs):
        self.transformer.model.prev_tread_trns = torch.compile(
            self.transformer.model.prev_tread_trns, *args, **kwargs
        )
        self.transformer.model.backbone = torch.compile(
            self.transformer.model.backbone, *args, **kwargs
        )
        self.transformer.model.post_tread_trns = torch.compile(
            self.transformer.model.post_tread_trns, *args, **kwargs
        )
        self.vae.encoder = torch.compile(self.vae.encoder, *args, **kwargs)
        self.vae.decoder = torch.compile(self.vae.decoder, *args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "a photo of a dog",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        cfg_scale: float = 3.0,
        num_inference_steps: int = 16,
        camera_param: dict[str, float] = {
            "zoom": 1.0,
            "x_shift": 0.0,
            "y_shift": 0.0,
        },
        tread_gamma1: float = 0.0,
        tread_gamma2: float = 0.25,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * len(prompt)

        prompt_tokens = self.tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
        )
        negative_prompt_tokens = self.tokenizer(
            negative_prompt,
            padding="longest",
            return_tensors="pt",
        )

        prompt_emb = self.text_encoder(
            input_ids=prompt_tokens.input_ids.to(self.device),
            attention_mask=prompt_tokens.attention_mask.to(self.device),
        ).last_hidden_state
        negative_prompt_emb = self.text_encoder(
            input_ids=negative_prompt_tokens.input_ids.to(self.device),
            attention_mask=negative_prompt_tokens.attention_mask.to(self.device),
        ).last_hidden_state

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (
                len(prompt),
                self.transformer.config.input_dim,
                height // 16 * 2,
                width // 16 * 2,
            ),
            generator=generator[0],
        )
        image = image.to(self.device).to(self.dtype)
        aspect_ratio = (
            torch.tensor([width / height], device=self.device)
            .log()
            .repeat(image.size(0))
        ).to(self.dtype)

        latent_h, latent_w = image.shape[-2:]
        pos_map = make_axial_pos_no_cache(latent_h, latent_w, device=self.device)
        pos_map[..., 0] = pos_map[..., 0] + camera_param.get("y_shift", 0.0)
        pos_map[..., 1] = pos_map[..., 1] + camera_param.get("x_shift", 0.0)
        pos_map = pos_map / camera_param.get("zoom", 1.0)
        pos_map = pos_map[None].expand(image.size(0), -1, -1).to(self.dtype)

        t = torch.tensor([1] * image.size(0), device=self.device).to(self.dtype)
        current_t = 1.0
        dt = 1.0 / num_inference_steps

        for _ in (pbar := self.progress_bar(range(num_inference_steps))):
            cond = self.transformer(
                image.to(self.dtype),
                t,
                prompt_emb,
                added_cond_kwargs={
                    "addon_info": aspect_ratio,
                    "tread_rate": tread_gamma1,
                },
                pos_map=pos_map,
            ).sample.float()
            uncond = self.transformer(
                image.to(self.dtype),
                t,
                negative_prompt_emb,
                added_cond_kwargs={
                    "addon_info": aspect_ratio,
                    "tread_rate": tread_gamma2,
                },
                pos_map=pos_map,
            ).sample.float()
            cfg_flow = uncond + cfg_scale * (cond - uncond)
            image = image - dt * cfg_flow
            t = t - dt
            current_t -= dt

        torch.cuda.empty_cache()
        image = image * self.vae_std.to(self.device) + self.vae_mean.to(self.device)
        image = torch.concat([self.vae.decode(i[None].to(self.dtype)).sample for i in image])
        image = (image.float() / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
