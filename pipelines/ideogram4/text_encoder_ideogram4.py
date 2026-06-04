"""Qwen3-VL text conditioning for Ideogram 4.

The prompt (plain text or a structured JSON caption) is wrapped in the Qwen3
chat template and run through the Qwen3-VL language model. Hidden states are
captured from 13 intermediate layers (pre final-norm) and concatenated along the
feature dim, giving the DiT multi-scale semantic features.

v1 feeds the prompt verbatim (no magic-prompt expansion, no caption verifier).
These weights require a structured JSON caption: a plain-text prompt lands
out-of-distribution and the model renders a baked-in "Image blocked by safety
filter" placeholder. The caption schema is a ``compositional_deconstruction``
object (with ``background`` + ``elements``); see the upstream prompting guide.
"""

from __future__ import annotations

import torch
from transformers.masking_utils import create_causal_mask

from pipelines.ideogram4.constants import LLM_TOKEN_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS


def tokenize(tokenizer, prompt: str, max_text_tokens: int) -> tuple[torch.Tensor, int]:
    """Chat-template tokenize a single prompt (passed verbatim)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > max_text_tokens:
        raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}")
    return token_ids, num_text_tokens


def qwen3_vl_layer_features(text_encoder, token_ids: torch.Tensor, attention_mask: torch.Tensor, pos_2d: torch.Tensor) -> list[torch.Tensor]:
    """Run the Qwen3-VL language model and return hidden states at the tap layers.

    The layer loop is driven manually so the tapped states are captured before the
    model's final norm, matching the reference.
    """
    language_model = text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]


def encode_text(text_encoder, token_ids: torch.Tensor, text_position_ids: torch.Tensor, indicator: torch.Tensor) -> torch.Tensor:
    """Stack the tap-layer hidden states into (B, L, hidden_size * num_taps) float32.

    Non-LLM positions (left padding / image slots) are zeroed so the DiT only sees
    real text features at LLM_TOKEN_INDICATOR positions.
    """
    batch_size, seq_len = token_ids.shape

    attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
    pos_2d = text_position_ids[..., 0].contiguous()

    with torch.no_grad():
        selected = qwen3_vl_layer_features(text_encoder, token_ids, attention_mask, pos_2d)

    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))  # (B, L, H, num_taps)
    stacked = stacked.reshape(batch_size, seq_len, -1)  # (B, L, H * num_taps)

    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    return stacked.to(torch.float32)
