from typing import Any, Dict, Optional
import torch
import numpy as np
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers, logging
from diffusers.models.modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def teacache_mochi_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p = self.config.patch_size

    post_patch_height = height // p
    post_patch_width = width // p

    temb, encoder_hidden_states = self.time_embed(
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        hidden_dtype=hidden_states.dtype,
    )

    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
    hidden_states = self.patch_embed(hidden_states)
    hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

    image_rotary_emb = self.rope(
        self.pos_frequencies,
        num_frames,
        post_patch_height,
        post_patch_width,
        device=hidden_states.device,
        dtype=torch.float32,
    )

    if self.enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, temb_)
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [-3.51241319e+03,  8.11675948e+02, -6.09400215e+01,  2.42429681e+00, 3.05291719e-03]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            encoder_attention_mask,
                            image_rotary_emb,
                            **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            encoder_attention_mask=encoder_attention_mask,
                            image_rotary_emb=image_rotary_emb,
                    )
            hidden_states = self.norm_out(hidden_states, temb)
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            encoder_attention_mask,
                            image_rotary_emb,
                            **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            encoder_attention_mask=encoder_attention_mask,
                            image_rotary_emb=image_rotary_emb,
                    )
        hidden_states = self.norm_out(hidden_states, temb)

    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
    hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
    output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
