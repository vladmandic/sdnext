# copied from https://github.com/Birch-san/sdxl-play/blob/main/src/attn/natten_attn_processor.py

import os
from diffusers.models.attention import Attention
import torch
from torch.nn import Linear
from einops import rearrange
from installer import install
from modules.logger import log


def init():
    try:
        os.environ['NATTEN_CUDA_ARCH'] = '8.0;8.6'
        install('natten')
        import natten
        return natten
    except Exception as e:
        log.error(f'Init natten: {e}')
        return None


def fuse_qkv(attn: Attention) -> None:
    has_bias = attn.to_q.bias is not None
    qkv = Linear(in_features=attn.to_q.in_features, out_features=attn.to_q.out_features*3, bias=has_bias, dtype=attn.to_q.weight.dtype, device=attn.to_q.weight.device)
    qkv.weight.data.copy_(torch.cat([attn.to_q.weight.data * attn.scale, attn.to_k.weight.data, attn.to_v.weight.data]))
    if has_bias:
        qkv.bias.data.copy_(torch.cat([attn.to_q.bias.data * attn.scale, attn.to_k.bias.data, attn.to_v.bias.data]))
    setattr(attn, 'qkv', qkv) # noqa: B010
    del attn.to_q, attn.to_k, attn.to_v


def fuse_vae_qkv(vae) -> None:
    for attn in [*vae.encoder.mid_block.attentions, *vae.decoder.mid_block.attentions]:
        fuse_qkv(attn)


class NattenAttnProcessor:
    kernel_size: int

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        temb: torch.FloatTensor | None = None,
    ):
        import natten
        assert hasattr(attn, 'qkv'), "Did not find property qkv on attn. Expected you to fuse its q_proj, k_proj, v_proj weights and biases beforehand, and multiply attn.scale into the q weights and bias."
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        # assumes MHA (as opposed to GQA)
        inner_dim: int = attn.qkv.out_features // 3
        if attention_mask is not None:
            raise ValueError("No mask customization for neighbourhood attention; the mask is already complicated enough as it is")
        if encoder_hidden_states is not None:
            raise ValueError("NATTEN cannot be used for cross-attention. I think.")
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)
            hidden_states = rearrange(hidden_states, '... c h w -> ... h w c')
        qkv = attn.qkv(hidden_states)
        # assumes MHA (as opposed to GQA)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=inner_dim)
        qk = natten.functional.na2d_qk(q, k, self.kernel_size, 1) # natten2dqk
        a = torch.softmax(qk, dim=-1)
        hidden_states = natten.functional.na2d_av(a, v, self.kernel_size, 1) # natten2dav
        hidden_states = rearrange(hidden_states, "n nh h w e -> n h w (nh e)")
        linear_proj, dropout = attn.to_out
        hidden_states = linear_proj(hidden_states)
        hidden_states = dropout(hidden_states)
        hidden_states = rearrange(hidden_states, '... h w c -> ... c h w')
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


def enable_natten(pipe):
    if not hasattr(pipe, 'vae'):
        return
    natten = init()
    kernel_size = 17
    if natten is not None:
        log.info(f'VAE natten: version={natten.__version__} kernel={kernel_size}')
        fuse_vae_qkv(pipe.vae)
        pipe.vae.set_attn_processor(NattenAttnProcessor(kernel_size=kernel_size))
