"""GPT-OSS text encoder for Lens.

We subclass ``transformers.GptOssForCausalLM`` so we can:

1. Return hidden states *only* at a configured layer subset (default
   ``[5, 11, 17, 23]``), avoiding the memory cost of HF's stock
   ``output_hidden_states=True`` which materializes every layer.
2. Early-exit after the last selected layer, since we don't need the
   downstream LM head at all when extracting features.

Standard ``generate(...)`` is inherited unchanged and is used by the optional
prompt reasoner.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


class LensGptOssEncoder(GptOssForCausalLM):
    """``GptOssForCausalLM`` subclass that exposes selected hidden states."""

    def set_selected_layers(self, layer_indices: Sequence[int]) -> None:
        layers = [int(i) for i in layer_indices]
        if not layers:
            raise ValueError("layer_indices must be non-empty")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layer_indices must be unique; got {layers}")
        if min(layers) < 0 or max(layers) >= len(self.model.layers):
            raise ValueError(
                f"layer_indices out of range; got {layers}, "
                f"model has {len(self.model.layers)} layers"
            )
        self._lens_selected_layers = layers # pylint: disable=attribute-defined-outside-init
        self._lens_max_layer = max(layers) # pylint: disable=attribute-defined-outside-init

    @torch.no_grad()
    def forward(  # type: ignore[override] # pylint: keyword-arg-before-vararg
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """Lens-specific forward.

        When ``input_ids`` and ``attention_mask`` are provided AND
        ``set_selected_layers(...)`` has been called, this returns the list of
        hidden states at the configured selected layers (the Lens feature
        extraction path).

        Otherwise, falls back to ``GptOssForCausalLM.forward`` so that
        ``generate(...)`` (used by the prompt reasoner) still works unchanged.
        """
        is_lens_feature_call = (
            input_ids is not None
            and attention_mask is not None
            and hasattr(self, "_lens_selected_layers")
            and not args
            and not kwargs
        )

        target_device = self.model.embed_tokens.weight.device
        if input_ids is not None and input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)

        if not is_lens_feature_call:
            return super().forward(input_ids, attention_mask, *args, **kwargs)

        model = self.model
        inputs_embeds = model.embed_tokens(input_ids)
        target_dtype = inputs_embeds.dtype
        if len(model.layers) > 0:
            target_dtype = model.layers[0].self_attn.k_proj.weight.dtype
        if inputs_embeds.dtype != target_dtype:
            inputs_embeds = inputs_embeds.to(dtype=target_dtype)

        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0).expand_as(input_ids)

        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        captured: List[torch.Tensor] = [None] * len(self._lens_selected_layers)
        index_lookup = {idx: pos for pos, idx in enumerate(self._lens_selected_layers)}

        for i, decoder_layer in enumerate(model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[model.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if i in index_lookup:
                captured[index_lookup[i]] = hidden_states
            if i == self._lens_max_layer:
                break

        for pos, layer_idx in enumerate(self._lens_selected_layers):
            if captured[pos] is None:
                raise RuntimeError(
                    f"Failed to capture hidden state for layer {layer_idx}"
                )
        return captured

    def encode_layers(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Backwards-compatible alias for the Lens feature path.

        Kept so existing call sites (``LensPipeline._get_text_embeddings``,
        external users) keep working. New code should call the encoder
        directly: ``encoder(input_ids, attention_mask)``.
        """
        if not hasattr(self, "_lens_selected_layers"):
            raise RuntimeError("Call set_selected_layers(...) before encode_layers().")
        return self(input_ids=input_ids, attention_mask=attention_mask)
