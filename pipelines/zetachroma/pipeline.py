import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, Qwen3ForCausalLM

from .model import DEFAULT_TEXT_ENCODER_REPO, NextDiTPixelSpace, load_zetachroma_transformer


logger = logging.getLogger(__name__)


@dataclass
class ZetaChromaPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> torch.Tensor:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = base_shift + (image_seq_len / 4096) * (max_shift - base_shift)
        timesteps = math.exp(mu) / (math.exp(mu) + (1.0 / timesteps.clamp(min=1e-9) - 1.0))
        timesteps[0] = 1.0
        timesteps[-1] = 0.0
    return timesteps


class ZetaChromaPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->dit_model"

    dit_model: NextDiTPixelSpace
    text_encoder: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    _progress_bar_config: Dict[str, Any]

    def __init__(self, dit_model: NextDiTPixelSpace, text_encoder: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        DiffusionPipeline.register_modules(self, dit_model=dit_model, text_encoder=text_encoder, tokenizer=tokenizer)
        if hasattr(self.text_encoder, "requires_grad_"):
            self.text_encoder.requires_grad_(False)
        if hasattr(self.dit_model, "requires_grad_"):
            self.dit_model.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dit_model: Optional[NextDiTPixelSpace] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        text_encoder_repo: str = DEFAULT_TEXT_ENCODER_REPO,
        checkpoint_filename: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: Optional[bool] = None,
        model_config: Optional[dict] = None,
        axes_dims: Optional[list[int]] = None,
        axes_lens: Optional[list[int]] = None,
        rope_theta: float = 256.0,
        time_scale: float = 1000.0,
        pad_tokens_multiple: int | None = 128,
        **kwargs,
    ):
        _ = kwargs
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                text_encoder_repo,
                subfolder="tokenizer",
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        if text_encoder is None:
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                text_encoder_repo,
                subfolder="text_encoder",
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
            )
        if dit_model is None:
            dit_model = load_zetachroma_transformer(
                pretrained_model_name_or_path,
                checkpoint_filename=checkpoint_filename,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                model_config=model_config,
                axes_dims=axes_dims,
                axes_lens=axes_lens,
                rope_theta=rope_theta,
                time_scale=time_scale,
                pad_tokens_multiple=pad_tokens_multiple,
            )
        pipe = cls(dit_model=dit_model, text_encoder=text_encoder, tokenizer=tokenizer)
        if torch_dtype is not None:
            pipe.to(torch_dtype=torch_dtype)
        return pipe

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def progress_bar(self, iterable=None, **kwargs): # pylint: disable=arguments-differ
        self._progress_bar_config = getattr(self, "_progress_bar_config", None) or {}
        config = {**self._progress_bar_config, **kwargs}
        return tqdm(iterable, **config)

    def to(self, torch_device=None, torch_dtype=None, silence_dtype_warnings=False): # pylint: disable=arguments-differ
        _ = silence_dtype_warnings
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(device=torch_device, dtype=torch_dtype)
        if hasattr(self, "dit_model"):
            self.dit_model.to(device=torch_device, dtype=torch_dtype)
        return self

    def _encode_single_prompt(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
        ).to(device)

        outputs = self.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-2].to(dtype)
        hidden = hidden * inputs.attention_mask.unsqueeze(-1).to(dtype)
        actual_len = int(inputs.attention_mask.sum(dim=1).max().item())
        return hidden[:, :actual_len, :]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        elif len(negative_prompt) != len(prompt):
            raise ValueError("negative_prompt must have the same batch length as prompt")

        device = device or self._execution_device
        dtype = dtype or next(self.text_encoder.parameters()).dtype

        positive = [self._encode_single_prompt(text, device, dtype, max_sequence_length) for text in prompt]
        negative = [self._encode_single_prompt(text, device, dtype, max_sequence_length) for text in negative_prompt]
        max_len = max(max(tensor.shape[1] for tensor in positive), max(tensor.shape[1] for tensor in negative))

        def pad_sequence(sequence: torch.Tensor) -> torch.Tensor:
            if sequence.shape[1] == max_len:
                return sequence
            return torch.cat(
                [
                    sequence,
                    torch.zeros(
                        sequence.shape[0],
                        max_len - sequence.shape[1],
                        sequence.shape[2],
                        dtype=sequence.dtype,
                        device=sequence.device,
                    ),
                ],
                dim=1,
            )

        prompt_embeds = torch.cat([pad_sequence(tensor) for tensor in positive], dim=0)
        negative_embeds = torch.cat([pad_sequence(tensor) for tensor in negative], dim=0)
        return prompt_embeds, negative_embeds

    def _postprocess_images(self, images: torch.Tensor, output_type: str):
        images = images.float().clamp(-1, 1) * 0.5 + 0.5
        if output_type == "pt":
            return images

        images = images.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return images
        if output_type == "pil":
            return self.numpy_to_pil((images * 255).round().clip(0, 255).astype(np.uint8))
        raise ValueError(f"Unsupported output_type: {output_type}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 512,
        shift_schedule: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end=None,
        **kwargs,
    ):
        _ = kwargs
        height = height or 1024
        width = width or 1024
        dtype = dtype or next(self.dit_model.parameters()).dtype
        device = self._execution_device

        prompt_embeds, negative_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_embeds = negative_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        batch_size = prompt_embeds.shape[0]

        latents = randn_tensor((batch_size, 3, height, width), generator=generator, device=device, dtype=dtype)
        image_seq_len = (height // self.dit_model.patch_size) * (width // self.dit_model.patch_size)
        sigmas = get_schedule(
            num_inference_steps,
            image_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
            shift=shift_schedule,
        ).to(device=device, dtype=dtype)

        self.dit_model.eval()
        do_classifier_free_guidance = guidance_scale > 1.0
        for step, timestep in enumerate(self.progress_bar(range(num_inference_steps), desc="Sampling")):
            _ = timestep
            t_curr = sigmas[step]
            t_next = sigmas[step + 1]
            t_batch = torch.full((batch_size,), t_curr, device=device, dtype=dtype)

            if do_classifier_free_guidance:
                latent_input = torch.cat([latents, latents], dim=0)
                timestep_input = torch.cat([t_batch, t_batch], dim=0)
                context_input = torch.cat([prompt_embeds, negative_embeds], dim=0)
                velocity = self.dit_model(latent_input, timestep_input, context_input, context_input.shape[1])
                v_cond, v_uncond = velocity.chunk(2, dim=0)
                velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                velocity = self.dit_model(latents, t_batch, prompt_embeds, prompt_embeds.shape[1])

            latents = latents + (t_next - t_curr) * velocity

            if callback_on_step_end is not None:
                callback_kwargs = {
                    "latents": latents,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_embeds,
                }
                callback_result = callback_on_step_end(self, step, int(t_curr.item() * 1000), callback_kwargs)
                if isinstance(callback_result, dict) and "latents" in callback_result:
                    latents = callback_result["latents"]

        images = self._postprocess_images(latents, output_type=output_type)
        if not return_dict:
            return (images,)
        return ZetaChromaPipelineOutput(images=images)
