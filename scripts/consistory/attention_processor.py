# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling ConsiStory or otherwise documented as NVIDIA-proprietary
# are not a contribution and subject to the license under the LICENSE file located at the root directory.


from typing import Callable, Optional
import torch
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.attention_processor import Attention
from .consistory_utils import AnchorCache, FeatureInjector, QueryStore


class ConsistoryAttnStoreProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, record_attention=True, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        # if attention_probs.requires_grad:
        if record_attention:
            self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class ConsistoryExtendedAttnXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, place_in_unet, attnstore, extended_attn_kwargs, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        self.t_range = extended_attn_kwargs.get('t_range', [])
        self.extend_kv_unet_parts = extended_attn_kwargs.get('extend_kv_unet_parts', ['down', 'mid', 'up'])

        self.place_in_unet = place_in_unet
        self.curr_unet_part = self.place_in_unet.split('_')[0]
        self.attnstore = attnstore

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        perform_extend_attn: bool = False,
        query_store: Optional[QueryStore] = None,
        feature_injector: Optional[FeatureInjector] = None,
        anchors_cache: Optional[AnchorCache] = None,
        **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = hidden_states.shape
            height = width = int(wh ** 0.5)

        is_cross = encoder_hidden_states is not None
        perform_extend_attn = perform_extend_attn and (not is_cross) and \
                              any([self.attnstore.curr_iter >= x[0] and self.attnstore.curr_iter <= x[1] for x in self.t_range]) and \
                              self.curr_unet_part in self.extend_kv_unet_parts

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if (self.curr_unet_part in self.extend_kv_unet_parts) and query_store and query_store.mode == 'cache':
            query_store.cache_query(query, self.place_in_unet)
        elif perform_extend_attn and query_store and query_store.mode == 'inject':
            query = query_store.inject_query(query, self.place_in_unet, self.attnstore.curr_iter)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query).contiguous()

        if perform_extend_attn:
            # Anchor Caching
            if anchors_cache and anchors_cache.is_cache_mode():
                if self.place_in_unet not in anchors_cache.input_h_cache:
                    anchors_cache.input_h_cache[self.place_in_unet] = {}

                # Hidden states inside the mask, for uncond (index 0) and cond (index 1) prompts
                subjects_hidden_states = torch.stack([x[self.attnstore.last_mask_dropout[width]] for x in hidden_states.chunk(2)])
                anchors_cache.input_h_cache[self.place_in_unet][self.attnstore.curr_iter] = subjects_hidden_states

            if anchors_cache and anchors_cache.is_inject_mode():
                # We make extended key and value by concatenating the original key and value with the query.
                anchors_hidden_states = anchors_cache.input_h_cache[self.place_in_unet][self.attnstore.curr_iter]

                anchors_keys = attn.to_k(anchors_hidden_states, *args)
                anchors_values = attn.to_v(anchors_hidden_states, *args)

                extended_key = torch.cat([torch.cat([key.chunk(2, dim=0)[x], anchors_keys[x].unsqueeze(0)], dim=1) for x in range(2)])
                extended_value = torch.cat([torch.cat([value.chunk(2, dim=0)[x], anchors_values[x].unsqueeze(0)], dim=1) for x in range(2)])

                extended_key = attn.head_to_batch_dim(extended_key).contiguous()
                extended_value = attn.head_to_batch_dim(extended_value).contiguous()

                # attn_masks needs to be of shape [batch_size, query_tokens, key_tokens]
                # hidden_states = xformers.ops.memory_efficient_attention(query, extended_key, extended_value,  op=self.attention_op, scale=attn.scale)
                hidden_states = F.scaled_dot_product_attention(query, extended_key, extended_value, scale=attn.scale)
            else:
                # # We make extended key and value by concatenating the original key and value with the query.
                # attention_mask_bias = self.attnstore.get_attn_mask_bias(tgt_size = width, bsz = batch_size)

                # if attention_mask_bias is not None:
                #     attention_mask_bias = torch.cat([x.unsqueeze(0).expand(attn.heads, -1, -1) for x in attention_mask_bias])

                # Pre-allocate the output tensor
                ex_out = torch.empty_like(query)

                for i in range(batch_size):
                    start_idx = i * attn.heads
                    end_idx = start_idx + attn.heads

                    attention_mask = self.attnstore.get_extended_attn_mask_instance(width, i%(batch_size//2))

                    curr_q = query[start_idx:end_idx]

                    if i < batch_size//2:
                        curr_k = key[:batch_size//2]
                        curr_v = value[:batch_size//2]
                    else:
                        curr_k = key[batch_size//2:]
                        curr_v = value[batch_size//2:]

                    curr_k = curr_k.flatten(0,1)[attention_mask].unsqueeze(0)
                    curr_v = curr_v.flatten(0,1)[attention_mask].unsqueeze(0)

                    curr_k = attn.head_to_batch_dim(curr_k).contiguous()
                    curr_v = attn.head_to_batch_dim(curr_v).contiguous()

                    # hidden_states = xformers.ops.memory_efficient_attention(curr_q, curr_k, curr_v, op=self.attention_op, scale=attn.scale)
                    hidden_states = F.scaled_dot_product_attention(curr_q, curr_k, curr_v, scale=attn.scale)

                    ex_out[start_idx:end_idx] = hidden_states

                hidden_states = ex_out
        else:
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            # attn_masks needs to be of shape [batch_size, query_tokens, key_tokens]
            # hidden_states = xformers.ops.memory_efficient_attention(query, key, value, op=self.attention_op, scale=attn.scale)
            hidden_states = F.scaled_dot_product_attention(query, key, value, scale=attn.scale)

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if feature_injector is not None:
            output_res = int(hidden_states.shape[1] ** 0.5)

            if anchors_cache and anchors_cache.is_inject_mode():
                hidden_states[batch_size//2:] = feature_injector.inject_anchors(hidden_states[batch_size//2:], self.attnstore.curr_iter, output_res, self.attnstore.extended_mapping, self.place_in_unet, anchors_cache)
            else:
                hidden_states[batch_size//2:] = feature_injector.inject_outputs(hidden_states[batch_size//2:], self.attnstore.curr_iter, output_res, self.attnstore.extended_mapping, self.place_in_unet, anchors_cache)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_extended_self_attn(unet, attnstore, extended_attn_kwargs):
    DICT_PLACE_TO_RES = {'down_0': 64, 'down_1': 64, 'down_2': 64, 'down_3': 64, 'down_4': 64, 'down_5': 64, 'down_6': 64, 'down_7': 64,
                         'down_8': 32, 'down_9': 32, 'down_10': 32, 'down_11': 32, 'down_12': 32, 'down_13': 32, 'down_14': 32, 'down_15': 32,
                         'down_16': 32, 'down_17': 32, 'down_18': 32, 'down_19': 32, 'down_20': 32, 'down_21': 32, 'down_22': 32, 'down_23': 32,
                         'down_24': 32, 'down_25': 32, 'down_26': 32, 'down_27': 32, 'down_28': 32, 'down_29': 32, 'down_30': 32, 'down_31': 32,
                         'down_32': 32, 'down_33': 32, 'down_34': 32, 'down_35': 32, 'down_36': 32, 'down_37': 32, 'down_38': 32, 'down_39': 32,
                         'down_40': 32, 'down_41': 32, 'down_42': 32, 'down_43': 32, 'down_44': 32, 'down_45': 32, 'down_46': 32, 'down_47': 32,
                         'mid_120': 32, 'mid_121': 32, 'mid_122': 32, 'mid_123': 32, 'mid_124': 32, 'mid_125': 32, 'mid_126': 32, 'mid_127': 32,
                         'mid_128': 32, 'mid_129': 32, 'mid_130': 32, 'mid_131': 32, 'mid_132': 32, 'mid_133': 32, 'mid_134': 32, 'mid_135': 32,
                         'mid_136': 32, 'mid_137': 32, 'mid_138': 32, 'mid_139': 32, 'up_49': 32, 'up_51': 32, 'up_53': 32, 'up_55': 32, 'up_57': 32,
                         'up_59': 32, 'up_61': 32, 'up_63': 32, 'up_65': 32, 'up_67': 32, 'up_69': 32, 'up_71': 32, 'up_73': 32, 'up_75': 32,
                         'up_77': 32, 'up_79': 32, 'up_81': 32, 'up_83': 32, 'up_85': 32, 'up_87': 32, 'up_89': 32, 'up_91': 32, 'up_93': 32,
                         'up_95': 32, 'up_97': 32, 'up_99': 32, 'up_101': 32, 'up_103': 32, 'up_105': 32, 'up_107': 32, 'up_109': 64, 'up_111': 64,
                         'up_113': 64, 'up_115': 64, 'up_117': 64, 'up_119': 64}
    attn_procs = {}
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attn = i % 2 == 0
        if name.startswith("mid_block"):
            place_in_unet = f"mid_{i}"
        elif name.startswith("up_blocks"):
            place_in_unet = f"up_{i}"
        elif name.startswith("down_blocks"):
            place_in_unet = f"down_{i}"
        else:
            continue

        if is_self_attn:
            attn_procs[name] = ConsistoryExtendedAttnXFormersAttnProcessor(place_in_unet, attnstore, extended_attn_kwargs)
        else:
            attn_procs[name] = ConsistoryAttnStoreProcessor(attnstore, place_in_unet)

    unet.set_attn_processor(attn_procs)
