"""Lens text-to-image pipeline.

The pipeline follows the standard ``diffusers`` component and call conventions:
components are registered via ``register_modules`` and the call signature
supports ``height``/``width``, ``generator``, ``prompt_embeds``, ``output_type``,
``return_dict``, and ``callback_on_step_end``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKLFlux2,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image

from .reasoner import PromptReasoner
from .resolution import resolve_resolution

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from .text_encoder import LensGptOssEncoder
    from .transformer import LensTransformer2DModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical ``mu`` for ``FlowMatchEulerDiscreteScheduler`` dynamic shift.

    Constants are calibrated for the Lens inference schedule.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# Chat template constants used by the Lens text encoder.
_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97


# Default Lens transformer architecture.
DEFAULT_TRANSFORMER_CONFIG = dict(
    patch_size=2,
    in_channels=128,
    out_channels=32,
    num_layers=48,
    attention_head_dim=64,
    num_attention_heads=24,
    inner_dim=1536,
    enc_hidden_dim=2880,
    axes_dims_rope=(8, 28, 28),
    gate_mlp=True,
    rms_norm=True,
    multi_layer_encoder_feature=True,
    selected_layer_index=(5, 11, 17, 23),
)


@dataclass
class LensPipelineOutput(BaseOutput):
    """Output of :class:`LensPipeline`.

    Args:
        images: list of decoded PIL images, or a numpy array of shape
            ``[B, H, W, C]`` when ``output_type='np'``, or the raw latent
            tensor when ``output_type='latent'``.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class LensPipeline(DiffusionPipeline):
    r"""Lens text-to-image pipeline (GPT-OSS multi-layer features + Flux2 VAE).

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler used together with ``transformer`` to denoise the
            encoded image latents.
        vae ([`AutoencoderKLFlux2`]):
            Flux2 VAE used to decode latents into pixel images.
        text_encoder ([`LensGptOssEncoder`]):
            ``GptOssForCausalLM`` subclass that exposes hidden states at the
            configured ``selected_layer_index`` via ``encode_layers(...)``.
        tokenizer ([`PreTrainedTokenizerBase`]):
            GPT-OSS tokenizer.
        transformer ([`LensTransformer2DModel`]):
            The Lens denoising DiT.
        reasoner ([`PromptReasoner`], *optional*):
            Optional prompt rewriter (local OSS ``generate`` or
            OpenAI-compatible API).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = [
        "latents", "prompt_embeds", "negative_prompt_embeds", "noise_pred",
    ]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: LensGptOssEncoder,
        tokenizer: PreTrainedTokenizerBase,
        transformer: LensTransformer2DModel,
    ) -> None:
        super().__init__()
        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # Flux2 latent tile factor (4x4 patchify) and Lens DiT in_channels=128.
        self.vae_scale_factor = 16
        self.latent_channels = self.transformer.config.in_channels
        self.txt_offset = DEFAULT_TXT_OFFSET
        self.default_sample_size = 1024

        if not hasattr(self.text_encoder, "_lens_selected_layers"):
            self.text_encoder.set_selected_layers(
                self.transformer.config.selected_layer_index
            )

        self.reasoner = PromptReasoner(
            text_encoder=self.text_encoder, tokenizer=self.tokenizer
        )

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def _build_chat_inputs(
        self, prompts: Sequence[str], max_sequence_length: int, device: torch.device
    ):
        rendered: List[str] = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
            ]
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            text = text.split("<|return|>")[0]
            rendered.append(text)

        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    @torch.no_grad()
    def _get_text_embeddings(
        self, prompts: List[str], max_sequence_length: int, device: torch.device
    ):
        input_ids, attn_mask = self._build_chat_inputs(prompts, max_sequence_length, device)
        layer_outputs = self.text_encoder.encode_layers(input_ids, attn_mask)

        offset = self.txt_offset
        if input_ids.shape[1] > offset:
            features = [feat[:, offset:, :].contiguous() for feat in layer_outputs]
            mask = attn_mask[:, offset:].bool()
        else:
            zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
            features = [layer_outputs[0].new_zeros(zero_shape) for _ in layer_outputs]
            mask = torch.zeros(
                (input_ids.shape[0], 0), dtype=torch.bool, device=device
            )
        return features, mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[List[torch.Tensor]] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ):
        """Encode positives and negatives. Returns
        ``(prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask)``
        where each ``*_embeds`` is a list of per-layer tensors and each
        ``*_mask`` is a bool ``[B*N, S]`` tensor.

        Each unique prompt is encoded **once**; the resulting features and mask
        are then ``repeat_interleave``-d ``num_images_per_prompt`` times along
        the batch axis. This preserves the ``[p0,p0,...,p1,p1,...]`` ordering
        downstream consumers expect.
        """
        device = device or self._execution_device

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        n = int(num_images_per_prompt)

        # Negatives broadcast.
        if isinstance(negative_prompt, str):
            negatives = [negative_prompt] * len(prompts)
        else:
            negatives = list(negative_prompt)
            if len(negatives) == 1:
                negatives = negatives * len(prompts)
            if len(negatives) != len(prompts):
                raise ValueError(
                    "negative_prompt must be a string or a list of the same "
                    "length as prompt"
                )

        if prompt_embeds is None:
            prompt_embeds, prompt_mask = self._get_text_embeddings(
                prompts, max_sequence_length, device
            )
            prompt_embeds, prompt_mask = self._repeat_for_n(prompt_embeds, prompt_mask, n)
        elif prompt_mask is None:
            raise ValueError("`prompt_mask` must be provided when passing `prompt_embeds`.")
        if negative_prompt_embeds is None:
            if all(isinstance(neg, str) and not neg.strip() for neg in negatives):
                # Empty negatives use an unconditional branch with no text tokens.
                negative_prompt_embeds = [
                    feat.new_zeros(feat.shape) for feat in prompt_embeds
                ]
                negative_prompt_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
            else:
                negative_prompt_embeds, negative_prompt_mask = self._get_text_embeddings(
                    negatives, max_sequence_length, device
                )
                negative_prompt_embeds, negative_prompt_mask = self._repeat_for_n(
                    negative_prompt_embeds, negative_prompt_mask, n
                )
        elif negative_prompt_mask is None:
            raise ValueError(
                "`negative_prompt_mask` must be provided when passing "
                "`negative_prompt_embeds`."
            )
        return prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask

    @staticmethod
    def _repeat_for_n(features: List[torch.Tensor], mask: torch.Tensor, n: int):
        """Repeat each sample ``n`` times along the batch axis (interleaved)."""
        if n == 1:
            return features, mask
        features = [f.repeat_interleave(n, dim=0) for f in features]
        mask = mask.repeat_interleave(n, dim=0)
        return features, mask

    # ------------------------------------------------------------------
    # Reasoner shim
    # ------------------------------------------------------------------

    def refine_prompt(
        self, prompts: Sequence[str], enable_reasoner: bool = False
    ) -> List[str]:
        if enable_reasoner and self.reasoner is None:
            return list(prompts)
        return self.reasoner.refine(prompts, enable=enable_reasoner)

    # ------------------------------------------------------------------
    # Latent prep
    # ------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        shape = (batch_size, latent_h * latent_w, num_channels_latents)
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds,
        callback_on_step_end_tensor_inputs,
    ) -> None:
        if height is None or width is None:
            raise ValueError(
                "height and width must be provided (or use base_resolution + aspect_ratio)."
            )
        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError(
                f"height and width must be divisible by {self.vae_scale_factor}; "
                f"got ({height}, {width})."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        if callback_on_step_end_tensor_inputs is not None:
            for k in callback_on_step_end_tensor_inputs:
                if k not in self._callback_tensor_inputs:
                    raise ValueError(
                        f"callback_on_step_end_tensor_inputs entry {k!r} is not "
                        f"in {self._callback_tensor_inputs}."
                    )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    def _pack_latents(
        self,
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor
        latents = latents.view(batch_size, num_channels_latents, height, 2, width, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, height * width, num_channels_latents * 4)

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int, # pylint: disable=unused-argument
    ) -> torch.Tensor:
        batch_size, _seq_len, patch_ch = latents.shape
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        return (
            latents
            .view(batch_size, latent_h, latent_w, patch_ch // 4, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch_size, patch_ch // 4, latent_h * 2, latent_w * 2)
        )

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor, latent_h: int, latent_w: int):
        latents = rearrange(
            latents,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            p1=2, p2=2, h=latent_h, w=latent_w,
        )
        latents = latents.to(self.vae.dtype)
        # Reverse the VAE latent normalization used by Lens. We compute the
        # shift/scale at runtime from the live ``vae.bn`` so this stays correct
        # under cpu-offload (where the VAE may be moved between devices).
        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1)
        var = bn.running_var.view(1, -1, 1, 1)
        std = torch.sqrt(var + self.vae.config.batch_norm_eps)
        shift = (-mean).to(device=latents.device, dtype=latents.dtype)
        scale = (1.0 / std).to(device=latents.device, dtype=latents.dtype)
        x = self._patchify_latents(latents)
        x = x / scale - shift
        x = self._unpatchify_latents(x)
        return self.vae.decode(x).sample

    @staticmethod
    def _to_pil(image: torch.Tensor) -> List[Image.Image]:
        # image: [B, C, H, W] in [-1, 1].
        image = image.clamp(-1.0, 1.0)
        image = (image + 1.0) * (255.0 / 2.0)
        image = image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
        return [Image.fromarray(im) for im in image]

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None, # noqa: RUF013
        negative_prompt: Union[str, List[str]] = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        base_resolution: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.Tensor]] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_reasoner: bool = False,
    ):
        # 0. Resolution defaulting.
        if base_resolution is not None and aspect_ratio is not None:
            height, width = resolve_resolution(base_resolution, aspect_ratio)
        elif height is None or width is None:
            height = width = self.default_sample_size

        # 1. Input validation.
        self.check_inputs(
            prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs
        )

        device = self._execution_device
        dtype = self.transformer.dtype

        # 2. Reasoner refinement (no-op when disabled and no API).
        if prompt is not None:
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            prompts = self.refine_prompt(prompts, enable_reasoner=enable_reasoner)
            self._last_refined_prompts = prompts # pylint: disable=attribute-defined-outside-init
        else:
            prompts = None

        # 3. Encode positives and negatives.
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 4. Pad pos/neg to a shared S_txt for joint CFG batching.
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._align_text_features(
            prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask
        )

        encoder_features = [
            torch.cat([pf, nf], dim=0).to(dtype=dtype)
            for pf, nf in zip(prompt_embeds, negative_prompt_embeds)
        ]
        encoder_mask = torch.cat([prompt_mask, negative_prompt_mask], dim=0)

        # 5. Prepare latents.
        batch_size = prompt_embeds[0].shape[0]
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        seq_len = latent_h * latent_w
        latents = self.prepare_latents(
            batch_size, self.latent_channels, height, width,
            dtype=dtype, device=device, generator=generator, latents=latents,
        )

        # 6. Scheduler.
        mu = compute_empirical_mu(seq_len, num_inference_steps)
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

        # 7. Denoising loop.
        img_shapes = [(1, latent_h, latent_w)]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                timestep = t.expand(batch_size * 2).to(latents.dtype)
                hidden_states = latents.repeat(2, 1, 1)

                noise = self.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_features,
                    encoder_hidden_states_mask=encoder_mask,
                    timestep=timestep / 1000,
                    img_shapes=img_shapes,
                )

                cond, uncond = noise.chunk(2)
                comb = uncond + guidance_scale * (cond - uncond)
                cond_norm = torch.norm(cond, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                scale = torch.where(
                    comb_norm > 0,
                    cond_norm / comb_norm.clamp_min(1e-12),
                    torch.ones_like(comb_norm),
                )
                noise_pred = comb * scale

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    cb_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "latents":
                            cb_kwargs[k] = latents
                        elif k == "prompt_embeds":
                            cb_kwargs[k] = prompt_embeds
                        elif k == "negative_prompt_embeds":
                            cb_kwargs[k] = negative_prompt_embeds
                        elif k == "noise_pred":
                            cb_kwargs[k] = noise_pred
                        else:
                            raise ValueError(
                                f"callback_on_step_end_tensor_inputs entry {k!r} is not "
                                f"in {self._callback_tensor_inputs}."
                            )
                    cb_out = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_out.pop("latents", latents)
                    prompt_embeds = cb_out.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = cb_out.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                progress_bar.update()

        # 8. Decode.
        if output_type == "latent":
            images: Any = latents
        else:
            decoded = self._decode(latents, latent_h, latent_w)
            if output_type == "pil":
                images = self._to_pil(decoded)
            elif output_type == "np":
                decoded = decoded.clamp(-1.0, 1.0)
                decoded = (decoded + 1.0) * 0.5
                images = decoded.permute(0, 2, 3, 1).to("cpu", torch.float32).numpy()
            else:
                raise ValueError(
                    f"output_type must be one of 'pil', 'np', 'latent'; got {output_type!r}."
                )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return LensPipelineOutput(images=images)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_text_features(
        pos_features: List[torch.Tensor],
        pos_mask: torch.Tensor,
        neg_features: List[torch.Tensor],
        neg_mask: torch.Tensor,
    ):
        """Pad pos/neg encodings and masks to a common ``S_txt``."""
        if not pos_features or not neg_features:
            raise ValueError("Positive and negative text feature lists must be non-empty.")
        if len(pos_features) != len(neg_features):
            raise ValueError(
                "Positive and negative text feature lists must have the same "
                f"number of layers; got {len(pos_features)} and {len(neg_features)}."
            )
        seq_pos = pos_features[0].shape[1]
        seq_neg = neg_features[0].shape[1]
        if pos_mask.shape[1] != seq_pos:
            raise ValueError(
                f"prompt_mask length {pos_mask.shape[1]} does not match "
                f"prompt feature length {seq_pos}."
            )
        if pos_mask.shape[0] != pos_features[0].shape[0]:
            raise ValueError(
                f"prompt_mask batch size {pos_mask.shape[0]} does not match "
                f"prompt feature batch size {pos_features[0].shape[0]}."
            )
        if neg_mask.shape[1] != seq_neg:
            raise ValueError(
                f"negative_prompt_mask length {neg_mask.shape[1]} does not "
                f"match negative prompt feature length {seq_neg}."
            )
        if neg_mask.shape[0] != neg_features[0].shape[0]:
            raise ValueError(
                f"negative_prompt_mask batch size {neg_mask.shape[0]} does "
                f"not match negative prompt feature batch size {neg_features[0].shape[0]}."
            )
        if pos_features[0].shape[0] != neg_features[0].shape[0]:
            raise ValueError(
                "Positive and negative text features must have the same batch "
                f"size; got {pos_features[0].shape[0]} and {neg_features[0].shape[0]}."
            )
        for i, feat in enumerate(pos_features):
            if feat.shape[:2] != pos_features[0].shape[:2]:
                raise ValueError(
                    f"Positive feature layer {i} shape {feat.shape[:2]} does "
                    f"not match layer 0 shape {pos_features[0].shape[:2]}."
                )
        for i, feat in enumerate(neg_features):
            if feat.shape[:2] != neg_features[0].shape[:2]:
                raise ValueError(
                    f"Negative feature layer {i} shape {feat.shape[:2]} does "
                    f"not match layer 0 shape {neg_features[0].shape[:2]}."
                )

        target = max(seq_pos, seq_neg)

        def pad(features: List[torch.Tensor], cur: int) -> List[torch.Tensor]:
            if cur == target:
                return features
            pad_len = target - cur
            return [
                torch.cat(
                    [feat, feat.new_zeros((feat.shape[0], pad_len, feat.shape[-1]))],
                    dim=1,
                )
                for feat in features
            ]

        def pad_mask(mask: torch.Tensor, cur: int) -> torch.Tensor:
            if cur == target:
                return mask
            return torch.cat(
                [
                    mask,
                    torch.zeros(
                        (mask.shape[0], target - cur),
                        dtype=torch.bool, device=mask.device,
                    ),
                ],
                dim=1,
            )

        pos_features = pad(pos_features, seq_pos)
        neg_features = pad(neg_features, seq_neg)
        pos_mask = pad_mask(pos_mask.bool(), seq_pos)
        neg_mask = pad_mask(neg_mask.bool(), seq_neg)
        return pos_features, pos_mask, neg_features, neg_mask
