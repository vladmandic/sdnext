import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Union, List

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def teacache_lumina2_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, _, height, width = hidden_states.shape
    temb, encoder_hidden_states_processed = self.time_caption_embed(hidden_states, timestep, encoder_hidden_states)
    (image_patch_embeddings, context_rotary_emb, noise_rotary_emb, joint_rotary_emb,
     encoder_seq_lengths, seq_lengths) = self.rope_embedder(hidden_states, encoder_attention_mask)
    image_patch_embeddings = self.x_embedder(image_patch_embeddings)
    for layer in self.context_refiner:
        encoder_hidden_states_processed = layer(encoder_hidden_states_processed, encoder_attention_mask, context_rotary_emb)
    for layer in self.noise_refiner:
        image_patch_embeddings = layer(image_patch_embeddings, None, noise_rotary_emb, temb)

    max_seq_len = max(seq_lengths)
    input_to_main_loop = image_patch_embeddings.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
    for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        input_to_main_loop[i, :enc_len] = encoder_hidden_states_processed[i, :enc_len]
        input_to_main_loop[i, enc_len:seq_len_val] = image_patch_embeddings[i]

    use_mask = len(set(seq_lengths)) > 1
    attention_mask_for_main_loop_arg = None
    if use_mask:
        mask = input_to_main_loop.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            mask[i, :seq_len_val] = True
        attention_mask_for_main_loop_arg = mask

    should_calc = True
    if self.enable_teacache:
        cache_key = max_seq_len
        if cache_key not in self.cache:
            self.cache[cache_key] = {
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_residual": None,
            }

        current_cache = self.cache[cache_key]
        modulated_inp, _, _, _ = self.layers[0].norm1(input_to_main_loop, temb)

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache["previous_modulated_input"] is not None:
                # teacache v1 coefficients:
                coefficients = [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344]
                # teacache v2 coefficients:
                #coefficients = [225.7042019806413, -608.8453716535591, 304.1869942338369, 124.21267720116742, -1.4089066892956552]
                rescale_func = np.poly1d(coefficients)
                prev_mod_input = current_cache["previous_modulated_input"]
                prev_mean = prev_mod_input.abs().mean()

                if prev_mean.item() > 1e-9:
                    rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item()
                else:
                    rel_l1_change = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float('inf')

                current_cache["accumulated_rel_l1_distance"] += rescale_func(rel_l1_change)

                if current_cache["accumulated_rel_l1_distance"] < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    current_cache["accumulated_rel_l1_distance"] = 0.0
            else:
                should_calc = True
                current_cache["accumulated_rel_l1_distance"] = 0.0

        current_cache["previous_modulated_input"] = modulated_inp.clone()

        if self.uncond_seq_len is None:
            self.uncond_seq_len = cache_key
        if cache_key != self.uncond_seq_len:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0

    if self.enable_teacache and not should_calc:
        if max_seq_len in self.cache and "previous_residual" in self.cache[max_seq_len] and self.cache[max_seq_len]["previous_residual"] is not None:
             processed_hidden_states = input_to_main_loop + self.cache[max_seq_len]["previous_residual"]
        else:
             should_calc = True
             current_processing_states = input_to_main_loop
             for layer in self.layers:
                current_processing_states = layer(current_processing_states, attention_mask_for_main_loop_arg, joint_rotary_emb, temb)
             processed_hidden_states = current_processing_states


    if not (self.enable_teacache and not should_calc) :
        current_processing_states = input_to_main_loop
        for layer in self.layers:
            current_processing_states = layer(current_processing_states, attention_mask_for_main_loop_arg, joint_rotary_emb, temb)

        if self.enable_teacache:
            if max_seq_len in self.cache:
                 self.cache[max_seq_len]["previous_residual"] = current_processing_states - input_to_main_loop
            else:
                 logger.warning(f"TeaCache: Cache key {max_seq_len} not found when trying to save residual.")

        processed_hidden_states = current_processing_states

    output_after_norm = self.norm_out(processed_hidden_states, temb)
    p = self.config.patch_size
    final_output_list = []
    for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        image_part = output_after_norm[i][enc_len:seq_len_val]
        h_p, w_p = height // p, width // p
        reconstructed_image = image_part.view(h_p, w_p, p, p, self.out_channels) \
                                        .permute(4, 0, 2, 1, 3) \
                                        .flatten(3, 4) \
                                        .flatten(1, 2)
        final_output_list.append(reconstructed_image)

    final_output_tensor = torch.stack(final_output_list, dim=0)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (final_output_tensor,)

    return Transformer2DModelOutput(sample=final_output_tensor)
