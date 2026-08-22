"""Local fixes for LTX-2.x gaps in the pinned diffusers.

Both patches are installed at import time by ltx_process and are safe to leave in
place once upstream fixes them: the first skips when the source no longer matches,
the second is a no-op as soon as no misrouted keys appear.

Connector padding (huggingface/diffusers#13564): PR #13564 (merged 2026-05-08)
refactored LTX2ConnectorTransformer1d's padding logic from a loop-based
gather-and-pad into a vectorized mask-then-flip. The new code applies
torch.flip(hidden_states, dims=[1]) after replacing padding positions with learned
registers, which reverses the order of valid prompt tokens. Audio cross-attention is
position-sensitive, so reversed token order produces jumbled dialogue (right
vocabulary, wrong word order). Visual quality is mostly unaffected because spatial
cross-attention is less position-sensitive.

Stage-2 LoRA connectors: LTX2LoraLoaderMixin.lora_state_dict recognizes connector
weights only under the 2.3-era text_embedding_projection prefix, so a
diffusion_model.* checkpoint is routed wholesale into the transformer namespace. The
2.5 stage-2 distilled LoRA carries its connector deltas as
diffusion_model.{video,audio}_embeddings_connector.*, so 224 of its 3544 keys reach a
module that cannot host them and peft drops them. Re-routing uses the rename table
from the convert_ltx2_to_diffusers script.
"""

import functools
import inspect

import torch
import torch.nn.functional as F

from modules.logger import log


PATCH_APPLIED = False
BROKEN_MARKER = 'torch.flip(hidden_states, dims=[1])'
CONNECTOR_LORA_PREFIXES = ('video_embeddings_connector.', 'audio_embeddings_connector.')
CONNECTOR_LORA_RENAME = {
    'video_embeddings_connector': 'video_connector',
    'audio_embeddings_connector': 'audio_connector',
    'transformer_1d_blocks': 'transformer_blocks',
}


def patched_connector_forward(
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


def reroute_connector_keys(state_dict):
    converted = {}
    moved = 0
    for key, value in state_dict.items():
        name = key.removeprefix('transformer.')
        if name.startswith(CONNECTOR_LORA_PREFIXES):
            for src, dst in CONNECTOR_LORA_RENAME.items():
                name = name.replace(src, dst)
            converted[f'connectors.{name}'] = value
            moved += 1
        else:
            converted[key] = value
    if moved == 0:
        return state_dict
    log.debug(f'LTX: lora=connectors rerouted={moved} total={len(state_dict)}')
    return converted


def apply_connectors_forward_patch():
    try:
        from diffusers.pipelines.ltx2.connectors import LTX2ConnectorTransformer1d
    except ImportError:
        return
    try:
        source = inspect.getsource(LTX2ConnectorTransformer1d.forward)
    except (OSError, TypeError):
        source = ''
    if BROKEN_MARKER in source:
        LTX2ConnectorTransformer1d.forward = patched_connector_forward # TODO ltx: patched diffusers connectors padding to fix audio token order (upstream #13564 regression)


def apply_lora_patch():
    try:
        from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin
    except ImportError:
        return
    original = LTX2LoraLoaderMixin.lora_state_dict.__func__

    @functools.wraps(original)
    def lora_state_dict(cls, *args, **kwargs): # TODO ltx: diffusers routes 2.5 stage-2 lora connector keys into the transformer namespace
        loaded = original(cls, *args, **kwargs)
        if isinstance(loaded, tuple):
            return (reroute_connector_keys(loaded[0]), *loaded[1:])
        return reroute_connector_keys(loaded)

    LTX2LoraLoaderMixin.lora_state_dict = classmethod(lora_state_dict)


def apply_patch():
    global PATCH_APPLIED # pylint: disable=global-statement
    if PATCH_APPLIED:
        return
    apply_connectors_forward_patch()
    apply_lora_patch()
    PATCH_APPLIED = True
