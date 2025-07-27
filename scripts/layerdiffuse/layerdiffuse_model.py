import torch.nn as nn
import torch
import cv2
import numpy as np

import einops
from tqdm import tqdm
from typing import Optional, Tuple, Union
from diffusers import AutoencoderKL
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.attention_processor import Attention, AttnProcessor
try:
    from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
except Exception:
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1)),
        )

    def __call__(self, x):
        return self.blocks(x)


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = zero_module(nn.Conv2d(4, block_out_channels[2], kernel_size=1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent

            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


class TransparentVAEDecoder(AutoencoderKL):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
    ):
        super().__init__(in_channels, out_channels, down_block_types, up_block_types, block_out_channels, layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size, scaling_factor, latents_mean, latents_std, force_upcast)

    def set_transparent_decoder(self, sd, mod_number=1):
        model = UNet1024(in_channels=3, out_channels=4)
        model.load_state_dict(sd, strict=True)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()

        self.transparent_decoder = model
        self.mod_number = mod_number

    def estimate_single_pass(self, pixel, latent):
        y = self.transparent_decoder(pixel, latent)
        return y

    def estimate_augmented(self, pixel, latent):
        args = [
            [False, 0], [False, 1], [False, 2], [False, 3], [True, 0], [True, 1], [True, 2], [True, 3],
        ]

        result = []

        for flip, rok in tqdm(args):
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))

            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))

            eps = self.estimate_single_pass(feed_pixel, feed_latent).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))

            if flip:
                eps = torch.flip(eps, dims=(3,))

            result += [eps]

        result = torch.stack(result, dim=0)
        median = torch.median(result, dim=0).values
        return median

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None) -> Union[DecoderOutput, torch.Tensor]:
        pixel = super().decode(z, return_dict=False, generator=generator)[0]
        pixel = pixel / 2 + 0.5


        result_pixel = []
        for i in range(int(z.shape[0])):
            if self.mod_number != 1 and i % self.mod_number != 0:
                img = torch.cat((pixel[i:i+1], torch.ones_like(pixel[i:i+1,:1,:,:])), dim=1)
                result_pixel.append(img)
                continue

            y = self.estimate_augmented(pixel[i:i+1], z[i:i+1])

            y = y.clip(0, 1).movedim(1, -1)
            alpha = y[..., :1]
            fg = y[..., 1:]

            B, H, W, C = fg.shape
            cb = checkerboard(shape=(H // 64, W // 64))
            cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_LANCZOS4)
            cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
            cb = torch.from_numpy(cb).to(fg)

            png = torch.cat([fg, alpha], dim=3)
            png = png.permute(0, 3, 1, 2)
            result_pixel.append(png)

        result_pixel = torch.cat(result_pixel, dim=0)
        result_pixel = (result_pixel - 0.5) * 2

        if not return_dict:
            return (result_pixel, )
        return DecoderOutput(sample=result_pixel)


class TransparentVAEEncoder:
    def __init__(self, sd, device="cpu", torch_dtype=torch.float32):
        self.load_device = device
        self.dtype = torch_dtype

        model = LatentTransparencyOffsetEncoder()
        model.load_state_dict(sd, strict=True)
        model.to(device=self.offload_device, dtype=self.dtype)
        model.eval()


class HookerLayers(torch.nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layers = torch.nn.ModuleList(layer_list)


class AdditionalAttentionCondsEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks_0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 64*64*256

        self.blocks_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 32*32*256

        self.blocks_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 16*16*256

        self.blocks_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 8*8*256

        self.blks = [self.blocks_0, self.blocks_1, self.blocks_2, self.blocks_3]

    def __call__(self, h):
        results = {}
        for b in self.blks:
            h = b(h)
            results[int(h.shape[2]) * int(h.shape[3])] = h
        return results


class LoraLoader(torch.nn.Module):
    def __init__(self, layer_list, use_control=False):
        super().__init__()
        self.hookers = HookerLayers(layer_list)

        if use_control:
            self.kwargs_encoder = AdditionalAttentionCondsEncoder()
        else:
            self.kwargs_encoder = None


class LoRALinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 256):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_features, bias=False)

    def forward(self, h, org):
        org_weight = org.weight.to(h)
        org_bias = org.bias.to(h) if org.bias is not None else None
        down_weight = self.down.weight
        up_weight = self.up.weight
        final_weight = org_weight + torch.mm(up_weight, down_weight)
        return torch.nn.functional.linear(h, final_weight, org_bias)


class AttentionSharingProcessor(nn.Module):
    def __init__(self, module, frames=2, use_control=True, rank=256):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features

        hidden_size = k_out_channels

        self.to_q_lora = [LoRALinearLayer(q_in_channels, q_out_channels, rank) for _ in range(self.frames)]
        self.to_k_lora = [LoRALinearLayer(k_in_channels, k_out_channels, rank) for _ in range(self.frames)]
        self.to_v_lora = [LoRALinearLayer(v_in_channels, v_out_channels, rank) for _ in range(self.frames)]
        self.to_out_lora = [LoRALinearLayer(o_in_channels, o_out_channels, rank) for _ in range(self.frames)]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.control_convs = None

        if use_control:
            self.control_convs = [torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(256, hidden_size, kernel_size=1),
            ) for _ in range(self.frames)]
            self.control_convs = torch.nn.ModuleList(self.control_convs)

        self.control_signals = None
        self.processor = AttnProcessor()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        modified_hidden_states = einops.rearrange(hidden_states, '(b f) d c -> f b d c', f=self.frames)

        if self.control_convs is not None:
            context_dim = int(modified_hidden_states.shape[2])
            control_outs = []
            for f in range(self.frames):
                control_signal = self.control_signals[context_dim].to(modified_hidden_states)
                control = self.control_convs[f](control_signal)
                control = einops.rearrange(control, 'b c h w -> b (h w) c')
                control_outs.append(control)
            control_outs = torch.stack(control_outs, dim=0)
            modified_hidden_states = modified_hidden_states + control_outs.to(modified_hidden_states)

        if encoder_hidden_states is None:
            framed_context = modified_hidden_states
        else:
            framed_context = einops.rearrange(encoder_hidden_states, '(b f) d c -> f b d c', f=self.frames)


        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]

            if encoder_hidden_states is not None:
                framed_cond_mark = einops.rearrange(torch.ones(batch_size*self.frames), '(b f) -> f b', f=self.frames).to(modified_hidden_states)
                cond_overwrite = []
                if len(cond_overwrite) > f:
                    cond_overwrite = cond_overwrite[f]
                else:
                    cond_overwrite = None
                if cond_overwrite is not None:
                    cond_mark = framed_cond_mark[f][:, None, None]
                    fcf = cond_overwrite.to(fcf) * (1.0 - cond_mark) + fcf * cond_mark

            query = self.to_q_lora[f](modified_hidden_states[f], attn.to_q)
            key = self.to_k_lora[f](fcf, attn.to_k)
            value = self.to_v_lora[f](fcf, attn.to_v)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            output = torch.bmm(attention_probs, value)
            output = attn.batch_to_head_dim(output)
            output = self.to_out_lora[f](output, attn.to_out[0])
            output = attn.to_out[1](output)
            attn_outs.append(output)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        query = self.temporal_q(x)
        key = self.temporal_k(x)
        value = self.temporal_v(x)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        x = torch.bmm(attention_probs, value)
        x = attn.batch_to_head_dim(x)

        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        return modified_hidden_states - hidden_states
