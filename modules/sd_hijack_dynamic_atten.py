
from functools import cache, wraps
import torch
from modules import shared, devices


# Find something divisible with the input_tokens
@cache
def find_split_size(original_size: int, slice_block_size: int, slice_rate: int = 2) -> int:
    split_size = original_size
    while True:
        if (split_size * slice_block_size) <= slice_rate and original_size % split_size == 0:
            return split_size
        split_size = split_size - 1
        if split_size <= 1:
            return 1
    return split_size


# Find slice sizes for SDPA
@cache
def find_sdpa_slice_sizes(query_shape: tuple[int], key_shape: tuple[int], query_element_size: int, slice_rate: int = 2, trigger_rate: int = 3) -> tuple[bool, int]:
    batch_size, attn_heads, query_len, _ = query_shape
    _, _, key_len, _ = key_shape

    slice_batch_size = attn_heads * (query_len * key_len) * query_element_size / 1024 / 1024 / 1024

    split_batch_size = batch_size
    split_head_size = attn_heads
    split_query_size = query_len

    do_batch_split = False
    do_head_split = False
    do_query_split = False

    if batch_size * slice_batch_size >= trigger_rate:
        do_batch_split = True
        split_batch_size = find_split_size(batch_size, slice_batch_size, slice_rate=slice_rate)

        if split_batch_size * slice_batch_size > slice_rate:
            slice_head_size = split_batch_size * (query_len * key_len) * query_element_size / 1024 / 1024 / 1024
            do_head_split = True
            split_head_size = find_split_size(attn_heads, slice_head_size, slice_rate=slice_rate)

            if split_head_size * slice_head_size > slice_rate:
                slice_query_size = split_batch_size * split_head_size * (key_len) * query_element_size / 1024 / 1024 / 1024
                do_query_split = True
                split_query_size = find_split_size(query_len, slice_query_size, slice_rate=slice_rate)

    return do_batch_split, do_head_split, do_query_split, split_batch_size, split_head_size, split_query_size


if devices.sdpa_pre_dyanmic_atten is None:
    devices.sdpa_pre_dyanmic_atten = torch.nn.functional.scaled_dot_product_attention
@wraps(devices.sdpa_pre_dyanmic_atten)
def dynamic_scaled_dot_product_attention(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.FloatTensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
    is_unsqueezed = False
    if query.dim() == 3:
        query = query.unsqueeze(0)
        is_unsqueezed = True
        if key.dim() == 3:
            key = key.unsqueeze(0)
        if value.dim() == 3:
            value = value.unsqueeze(0)
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    do_batch_split, do_head_split, do_query_split, split_batch_size, split_head_size, split_query_size = find_sdpa_slice_sizes(query.shape, key.shape, query.element_size(), slice_rate=shared.opts.dynamic_attention_slice_rate, trigger_rate=shared.opts.dynamic_attention_trigger_rate)

    # Slice SDPA
    if do_batch_split:
        batch_size, attn_heads, query_len, _ = query.shape
        _, _, _, head_dim = value.shape
        hidden_states = torch.zeros((batch_size, attn_heads, query_len, head_dim), device=query.device, dtype=query.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.expand((query.shape[0], query.shape[1], query.shape[2], key.shape[-2]))
        for ib in range(batch_size // split_batch_size):
            start_idx = ib * split_batch_size
            end_idx = (ib + 1) * split_batch_size
            if do_head_split:
                for ih in range(attn_heads // split_head_size): # pylint: disable=invalid-name
                    start_idx_h = ih * split_head_size
                    end_idx_h = (ih + 1) * split_head_size
                    if do_query_split:
                        for iq in range(query_len // split_query_size): # pylint: disable=invalid-name
                            start_idx_q = iq * split_query_size
                            end_idx_q = (iq + 1) * split_query_size
                            hidden_states[start_idx:end_idx, start_idx_h:end_idx_h, start_idx_q:end_idx_q, :] = devices.sdpa_pre_dyanmic_atten(
                                query[start_idx:end_idx, start_idx_h:end_idx_h, start_idx_q:end_idx_q, :],
                                key[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                                value[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                                attn_mask=attn_mask[start_idx:end_idx, start_idx_h:end_idx_h, start_idx_q:end_idx_q, :] if attn_mask is not None else attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_h:end_idx_h, :, :] = devices.sdpa_pre_dyanmic_atten(
                            query[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                            key[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                            value[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                            attn_mask=attn_mask[start_idx:end_idx, start_idx_h:end_idx_h, :, :] if attn_mask is not None else attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
                        )
            else:
                hidden_states[start_idx:end_idx, :, :, :] = devices.sdpa_pre_dyanmic_atten(
                    query[start_idx:end_idx, :, :, :],
                    key[start_idx:end_idx, :, :, :],
                    value[start_idx:end_idx, :, :, :],
                    attn_mask=attn_mask[start_idx:end_idx, :, :, :] if attn_mask is not None else attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
                )
        if devices.backend != "directml":
            getattr(torch, query.device.type).synchronize()
    else:
        hidden_states = devices.sdpa_pre_dyanmic_atten(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
    if is_unsqueezed:
        hidden_states = hidden_states.squeeze(0)
    return hidden_states


@cache
def find_bmm_slice_sizes(query_shape, query_element_size, slice_rate=2, trigger_rate=4):
    if len(query_shape) == 3:
        batch_size_attention, query_tokens, shape_three = query_shape
        shape_four = 1
    else:
        batch_size_attention, query_tokens, shape_three, shape_four = query_shape

    slice_block_size = query_tokens * shape_three * shape_four / 1024 / 1024 * query_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = query_tokens
    split_3_slice_size = shape_three

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if block_size > trigger_rate:
        do_split = True
        split_slice_size = find_split_size(split_slice_size, slice_block_size, slice_rate=slice_rate)
        if split_slice_size * slice_block_size > slice_rate:
            slice_2_block_size = split_slice_size * shape_three * shape_four / 1024 / 1024 * query_element_size
            do_split_2 = True
            split_2_slice_size = find_split_size(split_2_slice_size, slice_2_block_size, slice_rate=slice_rate)
            if split_2_slice_size * slice_2_block_size > slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size * shape_four / 1024 / 1024 * query_element_size
                do_split_3 = True
                split_3_slice_size = find_split_size(split_3_slice_size, slice_3_block_size, slice_rate=slice_rate)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size


class DynamicAttnProcessorBMM:
    r"""
    dynamically slices attention queries in order to keep them under the slice rate
    slicing will not get triggered if the query size is smaller than the slice rate to gain performance

    slice rate is in GB
    based on AttnProcessor V1
    """

    def __call__(self, attn, hidden_states: torch.Tensor, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs) -> torch.Tensor: # pylint: disable=too-many-statements, too-many-locals, too-many-branches, keyword-arg-before-vararg

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        ####################################################################
        # Slicing parts:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
        do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_bmm_slice_sizes(query.shape, query.element_size(), slice_rate=shared.opts.dynamic_attention_slice_rate*4, trigger_rate=shared.opts.dynamic_attention_trigger_rate*4)

        if do_split:
            for i in range(batch_size_attention // split_slice_size):
                start_idx = i * split_slice_size
                end_idx = (i + 1) * split_slice_size
                if do_split_2:
                    for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
                        start_idx_2 = i2 * split_2_slice_size
                        end_idx_2 = (i2 + 1) * split_2_slice_size
                        if do_split_3:
                            for i3 in range(shape_three // split_3_slice_size): # pylint: disable=invalid-name
                                start_idx_3 = i3 * split_3_slice_size
                                end_idx_3 = (i3 + 1) * split_3_slice_size

                                query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                                key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                                attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] if attention_mask is not None else None

                                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                                del query_slice
                                del key_slice
                                del attn_mask_slice
                                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3])

                                hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = attn_slice
                                del attn_slice
                        else:
                            query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2]
                            key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2]
                            attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attention_mask is not None else None

                            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                            del query_slice
                            del key_slice
                            del attn_mask_slice
                            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2])

                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = attn_slice
                            del attn_slice
                else:
                    query_slice = query[start_idx:end_idx]
                    key_slice = key[start_idx:end_idx]
                    attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

                    attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                    del query_slice
                    del key_slice
                    del attn_mask_slice
                    attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                    hidden_states[start_idx:end_idx] = attn_slice
                    del attn_slice
            if devices.backend != "directml":
                getattr(torch, query.device.type).synchronize()
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        ####################################################################
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
