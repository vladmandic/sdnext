"""Workaround for huggingface/diffusers#13564 connectors padding regression.

PR #13564 (merged 2026-05-08) refactored LTX2ConnectorTransformer1d's padding
logic from a loop-based gather-and-pad into a vectorized mask-then-flip. The
new code applies torch.flip(hidden_states, dims=[1]) after replacing padding
positions with learned registers, which reverses the order of valid prompt
tokens. Audio cross-attention is position-sensitive, so reversed token order
produces jumbled dialogue (right vocabulary, wrong word order). Visual quality
is mostly unaffected because spatial cross-attention is less position-sensitive.

This module restores the pre-#13564 forward at import time when the broken
pattern is detected. Safe to leave in place after upstream fixes the bug:
detection will skip the monkey-patch when the source no longer matches.
"""

import inspect

import torch
import torch.nn.functional as F

from modules.logger import log


_PATCH_APPLIED = False
_BROKEN_MARKER = 'torch.flip(hidden_states, dims=[1])'


def _patched_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    attn_mask_binarize_threshold: float = -9000.0,
):
    batch_size, seq_len, _ = hidden_states.shape

    if self.learnable_registers is not None:
        if seq_len % self.num_learnable_registers != 0:
            raise ValueError(
                f"The `hidden_states` sequence length {hidden_states.shape[1]} should be divisible by the number"
                f" of learnable registers {self.num_learnable_registers}"
            )

        num_register_repeats = seq_len // self.num_learnable_registers
        registers = (
            self.learnable_registers.unsqueeze(0).expand(num_register_repeats, -1, -1).reshape(seq_len, -1)
        )

        binary_attn_mask = (attention_mask >= attn_mask_binarize_threshold).int()
        if binary_attn_mask.ndim == 4:
            binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(1)

        hidden_states_non_padded = [hidden_states[i, binary_attn_mask[i].bool(), :] for i in range(batch_size)]
        valid_seq_lens = [x.shape[0] for x in hidden_states_non_padded]
        pad_lengths = [seq_len - vsl for vsl in valid_seq_lens]
        padded_hidden_states = [
            F.pad(x, pad=(0, 0, 0, p), value=0) for x, p in zip(hidden_states_non_padded, pad_lengths)
        ]
        padded_hidden_states = torch.cat([x.unsqueeze(0) for x in padded_hidden_states], dim=0)

        flipped_mask = torch.flip(binary_attn_mask, dims=[1]).unsqueeze(-1)
        hidden_states = flipped_mask * padded_hidden_states + (1 - flipped_mask) * registers

        attention_mask = torch.zeros_like(attention_mask)

    rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(block, hidden_states, attention_mask, rotary_emb) # pylint: disable=protected-access
        else:
            hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)

    hidden_states = self.norm_out(hidden_states)
    return hidden_states, attention_mask


def apply_patch():
    global _PATCH_APPLIED # pylint: disable=global-statement
    if _PATCH_APPLIED:
        return
    try:
        from diffusers.pipelines.ltx2.connectors import LTX2ConnectorTransformer1d
    except ImportError:
        _PATCH_APPLIED = True
        return
    try:
        source = inspect.getsource(LTX2ConnectorTransformer1d.forward)
    except (OSError, TypeError):
        source = ''
    if _BROKEN_MARKER in source:
        LTX2ConnectorTransformer1d.forward = _patched_forward # TODO ltx: patched diffusers connectors padding to fix audio token order (upstream #13564 regression)
    _PATCH_APPLIED = True
