# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import torch
from transformers import AutoTokenizer

from diffusers import AutoModel
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


SYSTEM_PROMPT = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
USER_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"


def build_prompt_text(prompt: str, eos_token: str) -> str:
    """The single-turn chat string the Ming language model is prompted with."""
    return SYSTEM_PROMPT + eos_token + USER_PREFIX + prompt + eos_token + ASSISTANT_PREFIX


def insert_query_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, text_encoder_config):
    """Insert a `[image_start, image_patch x num_query_tokens, image_end]` block after each item's text, before its
    right padding. Returns the new ids and attention mask, and a mask of the query-token (patch) positions."""
    num_query = text_encoder_config.num_query_tokens
    block = [text_encoder_config.image_start_token] + [text_encoder_config.image_patch_token] * num_query
    block += [text_encoder_config.image_end_token]

    rows, masks, query_masks = [], [], []
    for ids, mask in zip(input_ids.tolist(), attention_mask.tolist()):
        text_len = sum(mask)
        rows.append(ids[:text_len] + block + ids[text_len:])
        masks.append(mask[:text_len] + [1] * len(block) + mask[text_len:])
        query_masks.append([0] * (text_len + 1) + [1] * num_query + [0] * (1 + len(ids) - text_len))

    to_tensor = lambda x: torch.tensor(x, dtype=input_ids.dtype, device=input_ids.device)
    return to_tensor(rows), to_tensor(masks), to_tensor(query_masks).bool()


class MingImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that runs the Ming language model on each prompt followed by the image-generation query "
            "tokens, and turns its hidden states into the two conditioning streams of the transformer: "
            "`prompt_embeds` (query tokens, through the connector) and `prompt_embeds_2` (text tokens of the selected "
            "layers). The unconditional embeddings are zeros, as in the original pipeline."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", AutoModel),
            ComponentSpec("tokenizer", AutoTokenizer),
            ComponentSpec("connector", AutoModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("prompt")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Query-token conditioning, one `(num_query_tokens, cond_dim)` tensor per prompt.",
            ),
            OutputParam(
                "prompt_embeds_2",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Text-token conditioning, one `(text_seq_len, inner_dim)` tensor per prompt.",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Unconditional query-token conditioning (zeros).",
            ),
            OutputParam(
                "negative_prompt_embeds_2",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Unconditional text-token conditioning (zeros).",
            ),
        ]

    @staticmethod
    def check_inputs(prompt):
        if not isinstance(prompt, str) and not (isinstance(prompt, list) and all(isinstance(p, str) for p in prompt)):
            raise ValueError(f"`prompt` has to be of type `str` or `list[str]` but is {type(prompt)}")

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt)
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        device = components._execution_device
        text_encoder_config = components.text_encoder.config

        tokenizer = components.tokenizer
        text_inputs = tokenizer(
            [build_prompt_text(p, tokenizer.eos_token) for p in prompts],
            padding=True,
            return_tensors="pt",
        )
        input_ids, attention_mask, query_mask = insert_query_tokens(
            text_inputs.input_ids, text_inputs.attention_mask, text_encoder_config
        )
        hidden_states = components.text_encoder(input_ids.to(device), attention_mask.to(device))

        # Query tokens: last (normalized) layer at the query positions
        query_mask = query_mask.to(device)
        query_hidden_states = hidden_states[-1][query_mask].view(
            len(prompts), text_encoder_config.num_query_tokens, -1
        )

        # Text tokens: selected layers concatenated, zero-padded to the longest prompt
        text_lens = text_inputs.attention_mask.sum(dim=1).tolist()
        max_text_len = max(text_lens)
        text_mask = text_inputs.attention_mask[:, :max_text_len].to(device)
        text_hidden_states = torch.cat(
            [hidden_states[i][:, :max_text_len] for i in components.connector.config.hidden_state_layers], dim=-1
        )
        text_hidden_states = text_hidden_states * text_mask.unsqueeze(-1).to(text_hidden_states.dtype)

        connector_dtype = components.connector.dtype
        prompt_embeds, prompt_embeds_2 = components.connector(
            query_hidden_states.to(connector_dtype), text_hidden_states.to(connector_dtype)
        )

        block_state.prompt_embeds = list(prompt_embeds.unbind(0))
        block_state.prompt_embeds_2 = [embeds[:text_len] for embeds, text_len in zip(prompt_embeds_2, text_lens)]
        block_state.negative_prompt_embeds = [torch.zeros_like(e) for e in block_state.prompt_embeds]
        block_state.negative_prompt_embeds_2 = [torch.zeros_like(e) for e in block_state.prompt_embeds_2]

        self.set_block_state(state, block_state)
        return components, state
