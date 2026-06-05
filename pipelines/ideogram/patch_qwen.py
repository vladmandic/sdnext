from transformers.utils.generic import can_return_tuple
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLCausalLMOutputWithPast


_original_qwen3vl_forward = None


@can_return_tuple
def _qwen3vl_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    logits_to_keep=0,
    **kwargs,
):
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        mm_token_type_ids=mm_token_type_ids,
        **kwargs,
    )
    hidden_states = outputs[0]
    if hidden_states is not None and hasattr(self, "lm_head"):
        target_device = hidden_states.device
        if self.lm_head.weight.device != target_device:
            self.lm_head = self.lm_head.to(target_device)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )


def hijack_qwen3vl():
    # qwen forward makes a mess of .to() calls when model is manually assembled with lm_head attached
    global _original_qwen3vl_forward # pylint: disable=global-statement
    if _original_qwen3vl_forward is None:
        _original_qwen3vl_forward = Qwen3VLForConditionalGeneration.forward
        Qwen3VLForConditionalGeneration.forward = _qwen3vl_forward
