# based on <https://huggingface.co/spaces/fancyfeast/joytag>

import os
import math
import json
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import QuickGELUActivation
import torchvision
import torchvision.transforms.functional as TVF
import einops
from einops.layers.torch import Rearrange
import huggingface_hub
from modules import shared, devices, sd_models


model = None
tags = None
MODEL_REPO = "fancyfeast/joytag"
THRESHOLD = 0.4
MODEL_CONFIGS = {
    # Custom models trained from scratch
    # "Standard" definitions:
    # name | layers | width | heads
    #  B   |   12   |  768  |   12
    #  L   |   24   | 1024  |   16
    #  H   |   32   | 1280  |   16
    #  G   |   48   | 1664  |   16
    #  e   |   56   | 1792  |   16
    #  22  |   48   | 6144  |   48

    # B/16, 224, PaLM, GELU
    'CustomTest6': {
        'class': 'CLIPLikeModel',
        'embedding_dim': 768,
        'num_attention_heads': 12,
        'activation_cls': nn.GELU,
        'num_channels': 3,
        'patch_size': 16,
        'use_palm_alt': True,
        'num_layers': 12,
        'use_mha_alt': False,
        'good_dropout': False,
    },

    # GAP head + Sinusoidal positional embeddings + 448 image size
    'CustomTest18': {
        'class': 'CLIPLikeModel',
        'embedding_dim': 768,
        'num_attention_heads': 12,
        'activation_cls': nn.GELU,
        'num_channels': 3,
        'patch_size': 16,
        'use_palm_alt': True,
        'num_layers': 12,
        'use_mha_alt': False,
        'good_dropout': False,
        'use_gap_head': True,
        'sine_positional_embeddings': True,
    },

    # SW Model + B/16 + ASL + 448 image size
    # cutout_max_pct = 0
    # mixup_alpha = 0.8
    # noise_level = 2
    # random_resize_method = true
    # total_labels = 6549
    'SWModel1': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': False},
    # Sinusoidal positional embeddings
    'SWModel2': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    # Sinusoidal positional embeddings + 224 image size + L/14
    'SWModel3': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 14, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.05, 'layerscale_init': 1e-1, 'use_sine': True},
    # Sinusoidal positional embeddings + 224 image size + G/14
    'SWModel4': {'class': 'ViT', 'num_blocks': 48, 'patch_size': 14, 'd_model': 1664, 'mlp_dim': 1664*4, 'num_heads': 16, 'stochdepth_rate': 0.05, 'layerscale_init': 1e-1, 'use_sine': True},
    # Sinusoidal positional embeddings + focal loss
    'SWModel5': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    'SWModel6': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    'SWModel7': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    'SWModel8': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    'SWModel9': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    'SWModel10': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    'SWModel11': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0, 'use_sine': True},
    # Trying head_mean_after
    'SWModel12': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'head_mean_after': True},
    # Fat boy
    'SWModel13': {'class': 'ViT', 'num_blocks': 6, 'patch_size': 16, 'd_model': 1536, 'mlp_dim': 1536*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True},
    # L/14
    'SWModel14': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 14, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.05, 'layerscale_init': 1e-1, 'use_sine': True},
    'SWModel15': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 14, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.05, 'layerscale_init': 1e-5, 'use_sine': True},
    'SWModel16': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 14, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.10, 'layerscale_init': 1e-1, 'use_sine': True},
    'SWModel16f': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 14, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.10, 'layerscale_init': 1e-1, 'use_sine': True},
    'SWModel22': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 14, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.20, 'layerscale_init': 1e-1, 'use_sine': True},
    'SWModel25': {'class': 'ViT', 'num_blocks': 24, 'patch_size': 16, 'd_model': 1024, 'mlp_dim': 1024*4, 'num_heads': 16, 'stochdepth_rate': 0.15, 'layerscale_init': 1e-1, 'use_sine': True, 'cnn_stem': 'conv:c=128;ln;relu;conv:c=256;ln;relu;conv:c=512;ln;relu;conv:c=1024;ln;relu;conv:c=1024,s=1,k=1,p=0'},
    # CNN stem
    'SWModel18': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'cnn_stem': 'conv:c=64;bn;relu;conv:c=128;bn;relu;conv:c=256;bn;relu;conv:c=512;bn;relu;conv:c=768,s=1,k=1'},
    'SWModel19': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'cnn_stem': 'conv:c=64;bn;relu;conv:c=128;bn;relu;conv:c=128,s=1;bn;relu;conv:c=256;bn;relu;conv:c=256,s=1;bn;relu;conv:c=512;bn;relu;conv:c=768,s=1,k=1,p=0'},
    'SWModel20': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'cnn_stem': 'conv:c=64;ln;relu;conv:c=128;ln;relu;conv:c=256;ln;relu;conv:c=512;ln;relu;conv:c=768,s=1,k=1,p=0'},
    'SWModel21': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'cnn_stem': 'conv:c=64;ln;gelu;conv:c=128;ln;gelu;conv:c=256;ln;gelu;conv:c=512;ln;gelu;conv:c=768,s=1,k=1,p=0'},
    'SWModel23': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'cnn_stem': 'conv:c=64;ln;relu;conv:c=128;ln;relu;conv:c=256;ln;relu;conv:c=512;ln;relu;conv:c=768,s=1,k=1,p=0'},
    'SWModel24': {'class': 'ViT', 'num_blocks': 12, 'patch_size': 16, 'd_model': 768, 'mlp_dim': 768*4, 'num_heads': 12, 'stochdepth_rate': 0.05, 'use_sine': True, 'cnn_stem': 'conv:c=64;ln;relu;conv:c=128;ln;relu;conv:c=256;ln;relu;conv:c=512;ln;relu;conv:c=768,s=1,k=1,p=0'},
    # H/14
    'SWModel17': {'class': 'ViT', 'num_blocks': 32, 'patch_size': 14, 'd_model': 1280, 'mlp_dim': 1280*4, 'num_heads': 16, 'stochdepth_rate': 0.05, 'layerscale_init': 1e-1, 'use_sine': True},
    'SWModel26': {'class': 'ViT', 'num_blocks': 32, 'patch_size': 14, 'd_model': 1280, 'mlp_dim': 1280*4, 'num_heads': 16, 'stochdepth_rate': 0.15, 'layerscale_init': 1e-1, 'use_sine': True},
}


class VisionModel(nn.Module):
    image_size: int
    n_tags: int

    def __init__(self, image_size: int, n_tags: int):
        super().__init__()
        self.image_size = image_size
        self.n_tags = n_tags

    @staticmethod
    def load_model(path: str) -> 'VisionModel':
        with open(Path(path) / 'config.json', 'r', encoding='utf8') as f:
            config = json.load(f)
        from safetensors.torch import load_file
        resume = load_file(Path(path) / 'model.safetensors', device='cpu')
        model_classes = VisionModel.__subclasses__()
        model_cls = next(cls for cls in model_classes if cls.__name__ == config['class'])
        instance = model_cls(**{k: v for k, v in config.items() if k != 'class'})
        instance.load(resume)
        return instance

    @staticmethod
    def from_config(config: dict) -> 'VisionModel':
        model_classes = VisionModel.__subclasses__()
        model_cls = next(cls for cls in model_classes if cls.__name__ == config['class'])
        return model_cls(**{k: v for k, v in config.items() if k != 'class'})

    def get_optimized_parameters(self, lr: float):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self, state_dict):
        raise NotImplementedError


def basic_calculate_loss(preds: dict[str, torch.Tensor], batch: dict, pos_weight: torch.Tensor, loss_type: str):
    def asl_helper(preds, target):
        p = F.softmax(preds, dim=1)
        xs_pos = p.clamp(min=1e-6)
        xs_neg = (1 - p).clamp(min=1e-6)
        los_pos = torch.log(torch.gather(xs_pos, 1, target.unsqueeze(1))).sum()
        los_neg = torch.log(xs_neg)
        los_neg = los_neg.sum() - torch.gather(los_neg, 1, target.unsqueeze(1)).sum()
        loss = los_pos + los_neg
        return -loss

    if loss_type == "ce":
        loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'])
    elif loss_type == "weighted":
        loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'], pos_weight=pos_weight)
    elif loss_type == "focal":
        gamma = 2
        p = torch.sigmoid(preds['tags'])
        ce_loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'], reduction='none')
        p_t = p * batch['tags'] + (1 - p) * (1 - batch['tags'])
        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = loss.mean()
    elif loss_type == "focal2":
        gamma = 2
        p = torch.sigmoid(preds['tags'])
        ce_loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'], reduction='none')
        p_t = p * batch['tags'] + (1 - p) * (1 - batch['tags'])
        loss = ce_loss * ((1 - p_t) ** gamma) * 256
        loss = loss.mean()
    elif loss_type == "asl":
        p = torch.sigmoid(preds['tags'])
        xs_pos = p
        xs_neg = 1 - p
        los_pos = batch['tags'] * torch.log(xs_pos.clamp(min=1e-6))
        los_neg = (1 - batch['tags']) * torch.log(xs_neg.clamp(min=1e-6))
        loss = los_pos + los_neg
        loss = -loss.sum()
        # Rating
        loss = loss + asl_helper(preds['rating'], batch['rating'])
        # Score
        loss = loss + asl_helper(preds['score'], batch['score'])
    elif loss_type == "asl2":
        p = torch.sigmoid(preds['tags'])
        xs_pos = p
        xs_neg = 1 - p
        los_pos = batch['tags'] * torch.log(xs_pos.clamp(min=1e-6))
        los_neg = (1 - batch['tags']) * torch.log(xs_neg.clamp(min=1e-6))
        loss = -los_pos - los_neg
        loss = loss.sum()
    elif loss_type == "asl3":
        p = torch.sigmoid(preds['tags'])
        xs_pos = p
        xs_neg = 1 - p
        los_pos = batch['tags'] * torch.log(xs_pos.clamp(min=1e-6))
        los_neg = (1 - batch['tags']) * torch.log(xs_neg.clamp(min=1e-6))
        loss = -los_pos - los_neg
        loss = loss.mean()
    elif loss_type == "asl4":
        p = torch.sigmoid(preds['tags'])
        xs_pos = p
        xs_neg = 1 - p
        los_pos = batch['tags'] * torch.log(xs_pos.clamp(min=1e-6))
        los_neg = (1 - batch['tags']) * torch.log(xs_neg.clamp(min=1e-6))
        loss = -los_pos - los_neg
        loss = loss.mean() * 128
    elif loss_type == "asl5":
        loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'], pos_weight=pos_weight) * 128
    elif loss_type == "asl6":
        loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'], pos_weight=pos_weight) * 256
    elif loss_type == "asl7":
        loss = F.binary_cross_entropy_with_logits(preds['tags'], batch['tags'], pos_weight=pos_weight) * 2
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


class CLIPMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation_cls):
        super().__init__()
        self.activation_fn = activation_cls()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FastCLIPAttention2(nn.Module):
    """Fast Attention module for CLIP-like. This is NOT a drop-in replacement for CLIPAttention, since it adds additional flexibility.  Mainly uses xformers."""
    def __init__(self, hidden_size: int, out_dim: int, num_attention_heads: int, out_seq_len: Optional[int] = None, norm_qk: bool = False):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.embed_dim = hidden_size
        self.out_dim = out_dim
        self.norm_qk = norm_qk
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == self.embed_dim, "embed_dim must be divisible by num_attention_heads"
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.out_proj = nn.Linear(self.embed_dim, self.out_dim)
        if self.norm_qk:
            self.query_norm = nn.LayerNorm(self.embed_dim)
            self.key_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query_states: torch.Tensor, kv_states: torch.Tensor) -> torch.Tensor:
        bsz, src_len, embed_dim = kv_states.size()
        if self.out_seq_len is not None:
            tgt_len = self.out_seq_len
        else:
            tgt_len = src_len
        kv_states = self.kv_proj(kv_states)  # (bsz, src_len, embed_dim * 2)
        q_states = self.q_proj(query_states[:, :tgt_len])   # (bsz, tgt_len, embed_dim)
        # NOTE: It is not clear if LayerNorm should be applied to the embed_dim, or to the head_dim
        if self.norm_qk:
            q_states = self.query_norm(q_states).type(q_states.dtype)
            k_states = self.key_norm(kv_states[:, :, :embed_dim]).type(kv_states.dtype)
            v_states = kv_states[:, :, embed_dim:]
        else:
            k_states = kv_states[:, :, :embed_dim]
            v_states = kv_states[:, :, embed_dim:]
        q_states = q_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        k_states = k_states.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        v_states = v_states.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        # Performs scale of query_states, attention, and softmax
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x = F.scaled_dot_product_attention(q_states, k_states, v_states)   # (bsz, num_heads, tgt_len, head_dim)
            x = x.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)   # (bsz, tgt_len, embed_dim)
        # Projection
        x = self.out_proj(x)  # (bsz, tgt_len, out_dim)
        return x


class SkipInit(nn.Module):
    def __init__(self, hidden_size: int, channel_wise: bool, init_scale: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.channel_wise = channel_wise
        self.init_scale = init_scale
        if self.channel_wise:
            self.scale = nn.Parameter(torch.ones(hidden_size) * init_scale)
        else:
            self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FastCLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        out_seq_len: Optional[int],
        activation_cls = QuickGELUActivation,
        use_palm_alt: bool = False,
        norm_qk: bool = False,
        skip_init: Optional[float] = None,
        stochastic_depth: Optional[float] = None,
    ):
        super().__init__()
        self.use_palm_alt = use_palm_alt
        self.stochastic_depth = stochastic_depth
        self.self_attn = FastCLIPAttention2(
            hidden_size=hidden_size,
            out_dim=hidden_size,
            num_attention_heads=num_attention_heads,
            out_seq_len=out_seq_len,
            norm_qk=norm_qk,
        )
        self.mlp = CLIPMlp(hidden_size, 4 * hidden_size, activation_cls)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        if not use_palm_alt:
            self.layer_norm2 = nn.LayerNorm(hidden_size)
        if skip_init is not None:
            self.attn_skip_init = SkipInit(hidden_size, channel_wise=True, init_scale=skip_init)
            self.mlp_skip_init = SkipInit(hidden_size, channel_wise=True, init_scale=skip_init)
        else:
            self.attn_skip_init = nn.Identity()
            self.mlp_skip_init = nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if not self.use_palm_alt:
            hidden_states = self.self_attn(query_states=hidden_states, kv_states=hidden_states)
            hidden_states = self.attn_skip_init(hidden_states)
            hidden_states = hidden_states + residual[:, :hidden_states.size(1)]
            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.mlp_skip_init(hidden_states)
            hidden_states = hidden_states + residual
        else:
            # An alternative implementation inspired by the PALM paper
            # By performing the attention and MLP in parallel it's possible to fuse the linear projections of the attention and MLP layers
            # We don't do that here yet, but that supposedly improves efficiency without hurting performance
            attn = self.self_attn(query_states=hidden_states, kv_states=hidden_states)
            attn = self.attn_skip_init(attn)
            mlp = self.mlp(hidden_states[:, :attn.size(1)])
            mlp = self.mlp_skip_init(mlp)
            if self.stochastic_depth is not None:
                attn = torchvision.ops.stochastic_depth(attn, self.stochastic_depth, mode='row', training=self.training)
                mlp = torchvision.ops.stochastic_depth(mlp, self.stochastic_depth, mode='row', training=self.training)
            hidden_states = residual[:, :attn.size(1)] + attn + mlp
        return hidden_states


def sinusoidal_position_embedding(width: int, height: int, depth: int, dtype, device, temperature = 10000):
    """
    Sinusoidal position embedding. Returns a flat tensor of shape (h * w, d).
    """
    assert depth % 4 == 0, "Embedding dimension must be divisible by 4."
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    omega = torch.arange(depth // 4, device=device) / (depth // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    embedding = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return embedding.type(dtype)


class CLIPEmbeddingLayer(nn.Module):
    def __init__(self, hidden_size: int, num_channels: int, image_size: int, patch_size: int, patch_dropout: float = 0.0, good_dropout: bool = False, dpn: bool = False, sine_positional_embeddings: bool = False):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        seq_len = (image_size // patch_size) ** 2
        self.patch_dropout = patch_dropout
        self.hidden_size = hidden_size
        self.good_dropout = good_dropout
        self.dpn = dpn
        self.sine_positional_embeddings = sine_positional_embeddings
        self.patch_size = patch_size
        self.patch_embeddings = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        if not self.sine_positional_embeddings:
            self.positional_embeddings = nn.Embedding(seq_len, hidden_size)
        self.register_buffer("position_ids", torch.arange(seq_len))
        if self.dpn:
            self.to_patch_embeddings = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.LayerNorm(3 * patch_size * patch_size),
                nn.Linear(3 * patch_size * patch_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.to_patch_embeddings = nn.Conv2d(
                in_channels=num_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        B, _C, H, W = pixel_values.shape
        assert H % self.patch_size == 0, f"Input image height ({H}) needs to be divisible by the patch size ({self.patch_size})."
        assert W % self.patch_size == 0, f"Input image width ({W}) needs to be divisible by the patch size ({self.patch_size})."
        if self.dpn:
            patches = self.to_patch_embeddings(pixel_values)
        else:
            patches = self.to_patch_embeddings(pixel_values)
            patches = patches.flatten(2).transpose(1, 2)
        seq_len = patches.shape[1]
        patch_dropout = int(math.ceil((1.0 - self.patch_dropout) * seq_len))
        if self.sine_positional_embeddings:
            position_embeddings = sinusoidal_position_embedding(W // self.patch_size, H // self.patch_size, self.hidden_size, pixel_values.dtype, pixel_values.device)
        else:
            position_embeddings = self.positional_embeddings(self.position_ids)
        if patch_dropout == seq_len or not self.training:
            embeddings = patches + position_embeddings
        elif self.good_dropout:
            # Pick random patches to drop out
            # The "good_dropout" variant uses random permutations for each batch item, but is slightly slower and involves more code
            # The below method is a nice trick to generate a batch of random permutations.
            # Torch (as of 1.13) doesn't have a built-in function to do this, and a for loop of torch.randperm is slow.
            # Based on some benchmarks I measured the generation of the mask and the fetching to be only 50% slower than the non-"good_dropout" variant.
            # And the time taken here is only a fraction of the time spent performing the embedding convolution.
            # Generate a matrix of random numbers between 0 and 1 of shape (B, seq_len)
            patch_mask = torch.rand(B, seq_len, device=patches.device)
            # For each batch tensor, use argsort to convert the random numbers into a permutation of the patch indices
            patch_mask = torch.argsort(patch_mask, dim=1)
            # Truncate
            patch_mask = patch_mask[:, :patch_dropout]

            embeddings = patches.gather(1, patch_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)) + position_embeddings[patch_mask]
        else:
            # The non-"good_dropout" variant uses a single random permutation for all batch items, but is faster and uses less code
            indices = torch.randperm(seq_len, device=pixel_values.device)[:patch_dropout]
            embeddings = patches[:, indices, :] + position_embeddings[indices.expand(1, -1)]
        return embeddings


class MHAPoolingHead(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, activation_cls, out_dim: int, alt_style: bool, norm_qk: bool):
        super().__init__()
        self.out_dim = out_dim if not alt_style else hidden_size
        self.probe = nn.Parameter(torch.randn(hidden_size))
        self.mlp = CLIPMlp(hidden_size, 4 * hidden_size, activation_cls)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pooling_head = nn.Linear(hidden_size, 1)
        self.self_attn = FastCLIPAttention2(
            hidden_size=hidden_size,
            out_dim=self.out_dim,
            num_attention_heads=num_attention_heads,
            out_seq_len=1,
            norm_qk=norm_qk,
        )
        self.mlp = CLIPMlp(self.out_dim, 4 * self.out_dim, activation_cls)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.out_dim)
        if alt_style:
            self.final_proj = nn.Linear(hidden_size, out_dim)
        else:
            self.final_proj = nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.layer_norm1(hidden_states)
        query_states = self.probe.unsqueeze(0).unsqueeze(0).expand(hidden_states.size(0), 1, -1)
        hidden_states = self.self_attn(query_states=query_states, kv_states=hidden_states)
        # We don't use a residual connection here because the out_dim is different from the hidden_size
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.final_proj(hidden_states)
        return hidden_states.squeeze(1)


class GAPHead(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.proj(x)
        return x


class CLIPLikeModel(VisionModel):
    def __init__(
        self,
        n_tags: int,
        embedding_dim: int,
        num_attention_heads: int,
        activation_cls,
        num_channels: int,
        image_size: int,
        patch_size: int,
        patch_dropout: float,
        use_palm_alt: bool,
        num_layers: int,
        use_mha_alt: bool,
        loss_type: str,
        good_dropout: bool=False,
        dpn: bool=False,
        sine_positional_embeddings: bool=False,
        norm_qk: bool = False,
        no_wd_bias: bool = False,
        use_gap_head: bool = False,
        skip_init: Optional[float] = None,
        stochastic_depth: Optional[float] = None,
    ):
        super().__init__(image_size, n_tags)
        out_dim = n_tags
        self.n_tags = n_tags
        self.loss_type = loss_type
        self.no_wd_bias = no_wd_bias
        stochastic_depth_space = torch.linspace(0, stochastic_depth, num_layers) if stochastic_depth is not None else None
        self.embedding_layer = CLIPEmbeddingLayer(embedding_dim, num_channels, image_size, patch_size, patch_dropout, good_dropout, dpn, sine_positional_embeddings)
        self.pre_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_layers = nn.ModuleList([FastCLIPEncoderLayer(
            hidden_size=embedding_dim,
            num_attention_heads=num_attention_heads,
            out_seq_len=None,
            activation_cls=activation_cls,
            use_palm_alt=use_palm_alt,
            norm_qk=norm_qk,
            skip_init=skip_init,
            stochastic_depth=stochastic_depth_space[i].item() if stochastic_depth_space is not None else None,
        ) for i in range(num_layers)])
        if use_gap_head:
            self.pooling_head = GAPHead(embedding_dim, out_dim)
        else:
            self.pooling_head = MHAPoolingHead(embedding_dim, num_attention_heads, activation_cls, out_dim, use_mha_alt, norm_qk=norm_qk)

    def forward(self, batch):
        hidden_states = self.embedding_layer(batch['image'])
        hidden_states = self.pre_layer_norm(hidden_states)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        preds = self.pooling_head(hidden_states)
        result = { 'tags': preds }
        return result

    def calculate_loss(self, preds, batch, pos_weight):
        return basic_calculate_loss(preds, batch, pos_weight, self.loss_type)

    def get_optimized_parameters(self, lr: float):
        if self.no_wd_bias:
            return self.get_optimized_parameters_no_wd_bias()
        else:
            return self.parameters()

    def get_optimized_parameters_no_wd_bias(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
                print(f'No decay: {name}')
            else:
                decay.append(param)

        return [
            {'params': decay},
            {'params': no_decay, 'weight_decay': 0.},
        ]

    def save(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)


class MaskedAutoEncoderViT(nn.Module):
    def __init__(
        self,
        n_tags: int,
        embedding_dim: int,
        num_attention_heads: int,
        activation_cls,
        num_channels: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        loss_type: str,
        sine_positional_embeddings: bool=False,
        decoder_embedding_dim: int = 512,
        decoder_num_attention_heads: int = 8,
        decoder_num_layers: int = 6,
        decoder_force_projection: bool = False,
        masking_ratio: float = 0.75,
        mae_loss_weight: float = 1.0,
        mae_normalize_targets: bool = False,
        mae_post_norm: bool = False,
    ):
        super().__init__()
        self.n_tags = n_tags
        self.seq_len = (image_size // patch_size) ** 2
        self.embedding_dim = embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.sine_positional_embeddings = sine_positional_embeddings
        self.image_size = image_size
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.loss_type = loss_type
        self.mae_loss_weight = mae_loss_weight
        self.mae_normalize_targets = mae_normalize_targets
        if not self.sine_positional_embeddings:
            self.positional_embeddings = nn.Embedding(self.seq_len, embedding_dim)
            self.decoder_positional_embeddings = nn.Embedding(self.seq_len, decoder_embedding_dim)
        self.register_buffer("position_ids", torch.arange(self.seq_len))
        self.to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.patch_embedder = nn.Linear(num_channels * patch_size * patch_size, embedding_dim)

        # Encoder
        self.pre_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_layers = nn.ModuleList([FastCLIPEncoderLayer(
            hidden_size=embedding_dim,
            num_attention_heads=num_attention_heads,
            out_seq_len=None,
            activation_cls=activation_cls,
            use_palm_alt=True,
            norm_qk=False,
            skip_init=None,
        ) for _ in range(num_layers)])
        # Head for classification
        self.pooling_head = GAPHead(embedding_dim, n_tags)
        # Decoder
        if embedding_dim != decoder_embedding_dim or decoder_force_projection:
            self.encoder_to_decoder_proj = nn.Linear(embedding_dim, decoder_embedding_dim)
        else:
            self.encoder_to_decoder_proj = nn.Identity()
        self.decoder_pre_layer_norm = nn.LayerNorm(decoder_embedding_dim)
        self.decoder_layers = nn.ModuleList([FastCLIPEncoderLayer(
            hidden_size=decoder_embedding_dim,
            num_attention_heads=decoder_num_attention_heads,
            out_seq_len=None,
            activation_cls=activation_cls,
            use_palm_alt=True,
            norm_qk=False,
            skip_init=None,
        ) for _ in range(decoder_num_layers)])
        if mae_post_norm:
            self.decoder_to_pixel_values = nn.Sequential(
                nn.LayerNorm(decoder_embedding_dim),
                nn.Linear(decoder_embedding_dim, num_channels * patch_size * patch_size)
            )
        else:
            self.decoder_to_pixel_values = nn.Linear(decoder_embedding_dim, num_channels * patch_size * patch_size)
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, batch):
        pixel_values = batch['image']
        device = pixel_values.device
        B, _C, H, W = pixel_values.shape
        assert H % self.patch_size == 0, f"Input image height ({H}) needs to be divisible by the patch size ({self.patch_size})."
        assert W % self.patch_size == 0, f"Input image width ({W}) needs to be divisible by the patch size ({self.patch_size})."
        # Convert image to patches (B, seq_len, C * patch_size * patch_size)
        patches = self.to_patches(pixel_values)
        seq_len = patches.shape[1]
        num_masked = int(self.masking_ratio * seq_len)
        # For each batch tensor, use argsort to convert the random numbers into a permutation of the patch indices
        # From this we can get the masked and unmasked indices
        patch_mask = torch.rand(B, seq_len, device=device)
        patch_mask = torch.argsort(patch_mask, dim=1)
        masked_indices, unmasked_indices = patch_mask[:, :num_masked], patch_mask[:, num_masked:]
        batch_range = torch.arange(B, device=device)[:, None]
        # Masked and unmasked patches
        unmasked_patches = patches[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]
        # Embed unmasked patches for the encoder (B, seq_len, embedding_dim)
        tokens = self.patch_embedder(unmasked_patches)
        if self.sine_positional_embeddings:
            position_embeddings = sinusoidal_position_embedding(W // self.patch_size, H // self.patch_size, self.embedding_dim, pixel_values.dtype, device)
            decoder_position_embeddings = sinusoidal_position_embedding(W // self.patch_size, H // self.patch_size, self.decoder_embedding_dim, pixel_values.dtype, device)
        else:
            position_embeddings = self.positional_embeddings(self.position_ids)
            decoder_position_embeddings = self.decoder_positional_embeddings(self.position_ids)
        # Add position embeddings
        tokens = tokens + position_embeddings[unmasked_indices]
        # Run the encoder
        encoded_tokens = self.pre_layer_norm(tokens)
        for layer in self.encoder_layers:
            encoded_tokens = layer(encoded_tokens)
        # Label predictions
        if self.training:
            preds = self.pooling_head(encoded_tokens)
        else:
            # During inference, classify using the entire image
            # But we'll do the usual for the MAE part, just so we can see how MAE is performing during validation
            tokens = self.patch_embedder(patches)
            tokens = tokens + position_embeddings
            tokens = self.pre_layer_norm(tokens)
            for layer in self.encoder_layers:
                tokens = layer(tokens)
            preds = self.pooling_head(tokens)
        # Projection for the decoder and position embeddings
        decoder_tokens = self.encoder_to_decoder_proj(encoded_tokens)
        decoder_tokens = decoder_tokens + decoder_position_embeddings[unmasked_indices]
        # Fill in the masked patches
        mask_tokens = einops.repeat(self.mask_token, 'd -> b n d', b = B, n = num_masked)
        mask_tokens = mask_tokens + decoder_position_embeddings[masked_indices]
        decoder_tokens = torch.cat([decoder_tokens, mask_tokens], dim=1)
        # Run the decoder
        decoded_tokens = self.decoder_pre_layer_norm(decoder_tokens)
        for layer in self.decoder_layers:
            decoded_tokens = layer(decoded_tokens)
        # Only predict the masked patches
        # All the masked patches are at the end of the sequence
        decoded_tokens = decoded_tokens[:, -num_masked:]
        pred_pixel_values = self.decoder_to_pixel_values(decoded_tokens)
        # Calculate the mae loss
        if self.mae_normalize_targets:
            # Normalize each patch by its mean and variance. The ViCHA paper says this provides better results
            means = masked_patches.mean(dim=-1, keepdim=True)
            variant = masked_patches.var(dim=-1, keepdim=True)
            target = (masked_patches - means) / (variant + 1e-6)**0.5
            mae_loss = F.mse_loss(pred_pixel_values, target)
        else:
            mae_loss = F.mse_loss(pred_pixel_values, masked_patches)
        mae_loss = mae_loss * self.mae_loss_weight
        return {
            'tags': preds,
            'mae_loss': mae_loss,
        }

    def calculate_loss(self, preds, batch, pos_weight):
        return basic_calculate_loss(preds, batch, pos_weight, self.loss_type) + preds['mae_loss']

    def get_optimized_parameters(self, _lr: float):
        return self.parameters()

    def save(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)


class StochDepth(nn.Module):
    def __init__(self, drop_rate: float, scale_by_keep: bool = False):
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if not self.training:
            return x
        batch_size = x.shape[0]
        r = torch.rand((batch_size, 1, 1), device=x.device)
        keep_prob = 1 - self.drop_rate
        binary_tensor = torch.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class SkipInitChannelwise(nn.Module):
    def __init__(self, channels, init_val=1e-6):
        super().__init__()
        self.channels = channels
        self.init_val = init_val
        self.skip = nn.Parameter(torch.ones(channels) * init_val)

    def forward(self, x):
        return x * self.skip


class PosEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, use_sine: bool, patch_size: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_sine = use_sine
        self.patch_size = patch_size
        if not self.use_sine:
            self.embedding = nn.Embedding(max_len, d_model)
            nn.init.trunc_normal_(self.embedding.weight, std=0.02)
            self.register_buffer("position_ids", torch.arange(max_len))

    def forward(self, x, width: int, height: int):
        if self.use_sine:
            position_embeddings = sinusoidal_position_embedding(width // self.patch_size, height // self.patch_size, self.d_model, x.dtype, x.device)
        else:
            position_embeddings = self.embedding(self.position_ids)
        return x + position_embeddings


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, stochdepth_rate: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        if stochdepth_rate > 0:
            self.stochdepth = StochDepth(stochdepth_rate, scale_by_keep=True)
        else:
            self.stochdepth = None

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        if self.stochdepth is not None:
            x = self.stochdepth(x)
        x = self.linear2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int, layerscale_init: float, stochdepth_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # MHA
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.skip_init1 = SkipInitChannelwise(channels=d_model, init_val=layerscale_init)
        self.stochdepth1 = StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0 else None
        # MLP
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLPBlock(d_model, d_ff, stochdepth_rate)
        self.skip_init2 = SkipInitChannelwise(channels=d_model, init_val=layerscale_init)
        self.stochdepth2 = StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0 else None

    def forward(self, x):
        bsz, src_len, embed_dim = x.shape
        out = x
        out = self.norm1(out)
        # MHA
        qkv_states = self.qkv_proj(out).split(self.d_model, dim=-1)
        q_states = qkv_states[0].view(bsz, src_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)  # (bsz, num_heads, src_len, embed_dim // num_heads)
        k_states = qkv_states[1].view(bsz, src_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)  # (bsz, num_heads, src_len, embed_dim // num_heads)
        v_states = qkv_states[2].view(bsz, src_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)  # (bsz, num_heads, src_len, embed_dim // num_heads)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q_states, k_states, v_states)   # (bsz, num_heads, tgt_len, head_dim)
            out = out.transpose(1, 2).contiguous().view(bsz, src_len, embed_dim)   # (bsz, tgt_len, embed_dim)
        out = self.out_proj(out)
        out = self.skip_init1(out)
        if self.stochdepth1 is not None:
            out = self.stochdepth1(out)
        x = out + x
        out = self.norm2(x)
        out = self.mlp(out)
        out = self.skip_init2(out)
        if self.stochdepth2 is not None:
            out = self.stochdepth2(out)
        out = out + x
        return out


def CaiT_LayerScale_init(network_depth):
    if network_depth <= 18:
        return 1e-1
    elif network_depth <= 24:
        return 1e-5
    else:
        return 1e-6


class CNNLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        x = self.norm(x)
        x = x.transpose(1, 3)
        return x


class CNNStem(nn.Module):
    def __init__(self, config: str):
        super().__init__()
        self.config = config
        layers = []
        channels = 3
        for line in config.split(";"):
            ty, line = line.split(":") if ":" in line else (line, "")
            options = line.split(",")
            options = [o.split("=") for o in options] if line else []
            options = {k: v for k, v in options} # pylint: disable=unnecessary-comprehension # noqa: C416
            if ty == 'conv':
                layers.append(nn.Conv2d(
                    in_channels=channels,
                    out_channels=int(options['c']),
                    kernel_size=int(options['k'] if 'k' in options else 3),
                    stride=int(options['s'] if 's' in options else 2),
                    bias=True,
                    padding=int(options['p'] if 'p' in options else 1),
                ))
                channels = int(options['c'])
            elif ty == 'bn':
                layers.append(nn.BatchNorm2d(channels))
            elif ty == 'ln':
                layers.append(CNNLayerNorm(channels))
            elif ty == 'relu':
                layers.append(nn.ReLU())
            elif ty == 'gelu':
                layers.append(nn.GELU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ViT(VisionModel):
    def __init__(self,
        n_tags: int,
        image_size: int,
        num_blocks: int,
        patch_size: int,
        d_model: int,
        mlp_dim: int,
        num_heads: int,
        stochdepth_rate: float,
        use_sine: bool,
        loss_type: str,
        layerscale_init: Optional[float] = None,
        head_mean_after: bool = False,
        cnn_stem: str = None,
        patch_dropout: float = 0.0,
    ):
        super().__init__(image_size, n_tags)
        #assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        out_dim = n_tags
        self.n_tags = n_tags
        self.loss_type = loss_type
        self.patch_size = patch_size
        self.head_mean_after = head_mean_after
        self.patch_dropout = patch_dropout
        layerscale_init = CaiT_LayerScale_init(num_blocks) if layerscale_init is None else layerscale_init
        self.patch_embeddings = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        ) if cnn_stem is None else CNNStem(cnn_stem)
        self.pos_embedding = PosEmbedding(d_model, (image_size // patch_size) ** 2, use_sine=use_sine, patch_size=patch_size)
        self.blocks = nn.ModuleList([
            ViTBlock(num_heads, d_model, mlp_dim, layerscale_init, stochdepth_rate)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, batch, return_embeddings=False, return_loss: bool = False, pos_weight = None):
        B, _C, H, W = batch['image'].shape
        assert H % self.patch_size == 0, f"Input image height ({H}) needs to be divisible by the patch size ({self.patch_size})."
        assert W % self.patch_size == 0, f"Input image width ({W}) needs to be divisible by the patch size ({self.patch_size})."
        x = self.patch_embeddings(batch['image'])  # (bsz, d_model, patch_num, patch_num)
        x = x.flatten(2).transpose(1, 2)  # (bsz, patch_num ** 2, d_model)
        x = self.pos_embedding(x, W, H)   # (bsz, patch_num ** 2, d_model)
        # Patch dropout
        seq_len = x.shape[1]
        patch_dropout = int(math.ceil((1.0 - self.patch_dropout) * seq_len))
        if patch_dropout != seq_len:
            # Generate a matrix of random numbers between 0 and 1 of shape (B, seq_len)
            patch_mask = torch.rand(B, seq_len, device=x.device)
            # For each batch tensor, use argsort to convert the random numbers into a permutation of the patch indices
            patch_mask = torch.argsort(patch_mask, dim=1)
            # Truncate
            patch_mask = patch_mask[:, :patch_dropout]
            x = x.gather(1, patch_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            #indices = torch.randperm(seq_len, device=x.device)[:patch_dropout]
            #x = x[:, indices, :]
        # Transformer
        for block in self.blocks:
            x = block(x)
        # Head
        result = {}
        x = self.norm(x)
        if self.head_mean_after:
            x = self.head(x)
            x = x.mean(dim=1)
        else:
            x = x.mean(dim=1)
            if return_embeddings:
                result['embeddings'] = x
            x = self.head(x)
        result['tags'] = x
        if return_loss:
            result['loss'] = self.calculate_loss(result, batch, pos_weight)
        return result

    def calculate_loss(self, preds, batch, pos_weight):
        return basic_calculate_loss(preds, batch, pos_weight, self.loss_type)

    def get_optimized_parameters(self, lr: float):
        return self.parameters()

    def save(self):
        return self.state_dict()

    def load(self, state_dict):
        if 'head.weight' in state_dict and 'head.bias' in state_dict and state_dict['head.weight'].shape[0] == (self.n_tags + 9):
            # Support old models which included 3 rating and 6 score dimensions
            state_dict['head.weight'] = state_dict['head.weight'][:self.n_tags]
            state_dict['head.bias'] = state_dict['head.bias'][:self.n_tags]
        self.load_state_dict(state_dict)


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2
    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return image_tensor


def load():
    """Load JoyTag model."""
    global model, tags  # pylint: disable=global-statement
    if model is None:
        folder = huggingface_hub.snapshot_download(MODEL_REPO, cache_dir=shared.opts.hfcache_dir)
        model = VisionModel.load_model(folder)
        model.to(dtype=devices.dtype)
        model.eval()
        with open(os.path.join(folder, 'top_tags.txt'), 'r', encoding='utf8') as f:
            tags = [line.strip() for line in f.readlines() if line.strip()]
        shared.log.info(f'Caption: type=vlm model="JoyTag" repo="{MODEL_REPO}" tags={len(tags)}')
    sd_models.move_model(model, devices.device)


def unload():
    """Release JoyTag model from GPU/memory."""
    global model, tags  # pylint: disable=global-statement
    if model is not None:
        shared.log.debug('JoyTag unload')
        sd_models.move_model(model, devices.cpu, force=True)
        model = None
        tags = None
        devices.torch_gc(force=True)
    else:
        shared.log.debug('JoyTag unload: no model loaded')


def predict(image: Image.Image):
    load()
    image_tensor = prepare_image(image, model.image_size).unsqueeze(0).to(device=devices.device, dtype=devices.dtype)
    with devices.inference_context():
        preds = model({'image': image_tensor})
        tag_preds = preds['tags'].sigmoid().cpu()
    scores = {tags[i]: tag_preds[0][i] for i in range(len(tags))}
    if shared.opts.tagger_show_scores:
        predicted_tags = [f'{tag}:{score:.2f}' for tag, score in scores.items() if score > THRESHOLD]
    else:
        predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    tag_string = ', '.join(predicted_tags)
    return tag_string
