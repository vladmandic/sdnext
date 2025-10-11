# Copyright (c) 2023 HuggingFace Team
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache License, Version 2.0 (the "License")
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 1st June 2025
#
# Original file was released under Apache License, Version 2.0 (the "License"), with the full license text
# available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.

from contextlib import nullcontext
from typing import Optional, Tuple, Literal, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange
from ....common.half_precision_fixes import safe_pad_operation

from ....common.distributed.advanced import get_sequence_parallel_world_size
from ....common.logger import get_logger
from .causal_inflation_lib import (
    InflatedCausalConv3d,
    causal_norm_wrapper,
    init_causal_conv3d,
    remove_head,
)
from .context_parallel_lib import (
    causal_conv_gather_outputs,
    causal_conv_slice_inputs,
)
from .global_config import set_norm_limit
from .types import (
    CausalAutoencoderOutput,
    CausalDecoderOutput,
    CausalEncoderOutput,
    MemoryState,
    _inflation_mode_t,
    _memory_device_t,
    _receptive_field_t,
    _selective_checkpointing_t,
)

logger = get_logger(__name__)  # pylint: disable=invalid-name

# Fake func, no checkpointing is required for inference
def gradient_checkpointing(module: Union[Callable, nn.Module], *args, enabled: bool, **kwargs):
    return module(*args, **kwargs)

class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer.
            If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self, *, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.nonlinearity = nn.SiLU()

        self.norm1 = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden = input_tensor

        hidden = self.norm1(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden

        return output_tensor

class Upsample3D(nn.Module):
    """A 3D upsampling layer."""

    def __init__(
        self,
        channels: int,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.conv = init_causal_conv3d(
            self.channels, self.channels, kernel_size=3, padding=1, inflation_mode=inflation_mode
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.slicing = slicing

        upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
        self.upscale_conv = nn.Conv3d(
            self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
        )
        identity = (
            torch.eye(self.channels).repeat(upscale_ratio, 1).reshape_as(self.upscale_conv.weight)
        )

        self.upscale_conv.weight.data.copy_(identity)
        nn.init.zeros_(self.upscale_conv.bias)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState,
    ) -> torch.FloatTensor:
        return gradient_checkpointing(
            self.custom_forward,
            hidden_states,
            memory_state,
            enabled=self.training and self.gradient_checkpointing,
        )

    def custom_forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.slicing:
            split_size = hidden_states.size(2) // 2
            hidden_states = list(
                hidden_states.split([split_size, hidden_states.size(2) - split_size], dim=2)
            )
        else:
            hidden_states = [hidden_states]

        for i in range(len(hidden_states)):
            hidden_states[i] = self.upscale_conv(hidden_states[i])
            hidden_states[i] = rearrange(
                hidden_states[i],
                "b (x y z c) f h w -> b c (f z) (h x) (w y)",
                x=self.spatial_ratio,
                y=self.spatial_ratio,
                z=self.temporal_ratio,
            )

        # [Overridden] For causal temporal conv
        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states[0] = remove_head(hidden_states[0])

        if self.slicing:
            hidden_states = self.conv(hidden_states, memory_state=memory_state)
            return torch.cat(hidden_states, dim=2)
        else:
            return self.conv(hidden_states[0], memory_state=memory_state)


class Downsample3D(nn.Module):
    """A 3D downsampling layer."""

    def __init__(
        self,
        channels: int,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_down: bool = False,
        spatial_down: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        self.conv = init_causal_conv3d(
            self.channels,
            self.channels,
            kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
            stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            padding=((1 if self.temporal_down else 0), 0, 0),
            inflation_mode=inflation_mode,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState,
    ) -> torch.FloatTensor:
        return gradient_checkpointing(
            self.custom_forward,
            hidden_states,
            memory_state,
            enabled=self.training and self.gradient_checkpointing,
        )

    def custom_forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState,
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if self.spatial_down:
            hidden_states = safe_pad_operation(hidden_states, (0, 1, 0, 1), mode="constant", value=0)

        hidden_states = self.conv(hidden_states, memory_state=memory_state)
        return hidden_states


class ResnetBlock3D(ResnetBlock2D):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = init_causal_conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.conv2 = init_causal_conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        if self.use_in_shortcut:
            self.conv_shortcut = init_causal_conv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )
        self.gradient_checkpointing = False

    def forward(self, input_tensor: torch.Tensor, memory_state: MemoryState = MemoryState.UNSET):
        return gradient_checkpointing(
            self.custom_forward,
            input_tensor,
            memory_state,
            enabled=self.training and self.gradient_checkpointing,
        )

    def custom_forward(
        self, input_tensor: torch.Tensor, memory_state: MemoryState = MemoryState.UNSET
    ):
        assert memory_state != MemoryState.UNSET
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, memory_state=memory_state)

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_downsample: bool = True,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if add_downsample:
            # Todo: Refactor this line before V5 Image VAE Training.
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        channels=out_channels,
                        inflation_mode=inflation_mode,
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                    )
                ]
            )

    def forward(
        self, hidden_states: torch.FloatTensor, memory_state: MemoryState
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, memory_state=memory_state)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_upsample: bool = True,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        # Todo: Refactor this line before V5 Image VAE Training.
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        channels=out_channels,
                        inflation_mode=inflation_mode,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        slicing=slicing,
                    )
                ]
            )

    def forward(
        self, hidden_states: torch.FloatTensor, memory_state: MemoryState
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, memory_state=memory_state)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock3D(
                    in_channels=channels,
                    out_channels=channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                ),
                ResnetBlock3D(
                    in_channels=channels,
                    out_channels=channels,
                    dropout=dropout,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                ),
            ]
        )

    def forward(self, hidden_states: torch.Tensor, memory_state: MemoryState):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, memory_state)
        return hidden_states


class Encoder3D(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes
    its input into a latent representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        double_z: bool = True,
        temporal_down_num: int = 2,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        selective_checkpointing: Tuple[_selective_checkpointing_t] = ("none",),
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.temporal_down_num = temporal_down_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1
            # Note: take the last one

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3D(
            channels=block_out_channels[-1],
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=32, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = init_causal_conv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        assert len(selective_checkpointing) == len(self.down_blocks)
        self.set_gradient_checkpointing(selective_checkpointing)

    def set_gradient_checkpointing(self, checkpointing_types):
        gradient_checkpointing = []
        for down_block, sac_type in zip(self.down_blocks, checkpointing_types):
            if sac_type == "coarse":
                gradient_checkpointing.append(True)
            elif sac_type == "fine":
                for n, m in down_block.named_modules():
                    if hasattr(m, "gradient_checkpointing"):
                        m.gradient_checkpointing = True
                        logger.debug(f"set gradient_checkpointing: {n}")
                gradient_checkpointing.append(False)
            else:
                gradient_checkpointing.append(False)
        self.gradient_checkpointing = gradient_checkpointing
        logger.info(f"[Encoder3D] gradient_checkpointing: {checkpointing_types}")

    def forward(self, sample: torch.FloatTensor, memory_state: MemoryState) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample, memory_state=memory_state)
        # down
        for down_block, sac in zip(self.down_blocks, self.gradient_checkpointing):
            sample = gradient_checkpointing(
                down_block,
                sample,
                memory_state=memory_state,
                enabled=self.training and sac,
            )

        # middle
        sample = self.mid_block(sample, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that
    decodes its latent representation into an output sample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        slicing_up_num: int = 0,
        selective_checkpointing: Tuple[_selective_checkpointing_t] = ("none",),
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock3D(
            channels=block_out_channels[-1],
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            is_slicing_up_block = i >= len(block_out_channels) - slicing_up_num
            # Note: Keep symmetric

            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                temporal_up=is_temporal_up_block,
                slicing=is_slicing_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=32, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = init_causal_conv3d(
            block_out_channels[0], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        assert len(selective_checkpointing) == len(self.up_blocks)
        self.set_gradient_checkpointing(selective_checkpointing)

    def set_gradient_checkpointing(self, checkpointing_types):
        gradient_checkpointing = []
        for up_block, sac_type in zip(self.up_blocks, checkpointing_types):
            if sac_type == "coarse":
                gradient_checkpointing.append(True)
            elif sac_type == "fine":
                for n, m in up_block.named_modules():
                    if hasattr(m, "gradient_checkpointing"):
                        m.gradient_checkpointing = True
                        logger.debug(f"set gradient_checkpointing: {n}")
                gradient_checkpointing.append(False)
            else:
                gradient_checkpointing.append(False)
        self.gradient_checkpointing = gradient_checkpointing
        logger.info(f"[Decoder3D] gradient_checkpointing: {checkpointing_types}")

    def forward(self, sample: torch.FloatTensor, memory_state: MemoryState) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, memory_state=memory_state)

        # middle
        sample = self.mid_block(sample, memory_state=memory_state)

        # up
        for up_block, sac in zip(self.up_blocks, self.gradient_checkpointing):
            sample = gradient_checkpointing(
                up_block,
                sample,
                memory_state=memory_state,
                enabled=self.training and sac,
            )

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        latent_channels: int = 4,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        enc_selective_checkpointing: Tuple[_selective_checkpointing_t] = ("none",),
        dec_selective_checkpointing: Tuple[_selective_checkpointing_t] = ("none",),
        temporal_scale_num: int = 3,
        slicing_up_num: int = 0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        slicing_sample_min_size: int = None,
        spatial_downsample_factor: int = 16,
        temporal_downsample_factor: int = 8,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        if slicing_sample_min_size is None:
            slicing_sample_min_size = temporal_downsample_factor
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = slicing_sample_min_size // (2**temporal_scale_num)

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            double_z=True,
            temporal_down_num=temporal_scale_num,
            selective_checkpointing=enc_selective_checkpointing,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            selective_checkpointing=dec_selective_checkpointing,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.quant_conv = (
            init_causal_conv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            init_causal_conv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        h = self.slicing_encode(x)
        p = DiagonalGaussianDistribution(h)
        z = p.sample()
        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.FloatTensor) -> CausalDecoderOutput:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = self.slicing_decode(z)
        return CausalDecoderOutput(x)

    def _encode(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        x = causal_conv_slice_inputs(x, self.slicing_sample_min_size, memory_state=memory_state)
        h = self.encoder(x, memory_state=memory_state)
        h = self.quant_conv(h, memory_state=memory_state) if self.quant_conv is not None else h
        h = causal_conv_gather_outputs(h)
        return h

    def _decode(self, z: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        z = causal_conv_slice_inputs(z, self.slicing_latent_min_size, memory_state=memory_state)
        z = (
            self.post_quant_conv(z, memory_state=memory_state)
            if self.post_quant_conv is not None
            else z
        )
        x = self.decoder(z, memory_state=memory_state)
        x = causal_conv_gather_outputs(x)
        return x

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size:
            x_slices = x[:, :, 1:].split(split_size=self.slicing_sample_min_size * sp_size, dim=2)
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
            return torch.cat(encoded_slices, dim=2)
        else:
            return self._encode(x, memory_state=MemoryState.DISABLED)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size * sp_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
            return torch.cat(decoded_slices, dim=2)
        else:
            return self._decode(z, memory_state=MemoryState.DISABLED)

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % self.temporal_downsample_factor == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = split_size // self.temporal_downsample_factor
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    def __init__(
        self, *args, spatial_downsample_factor: int, temporal_downsample_factor: int, **kwargs
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        super().__init__(*args, **kwargs)

    def forward(self, x) -> CausalAutoencoderOutput:
        z, _, p = self.encode(x)
        x, _ = self.decode(z)
        return CausalAutoencoderOutput(x, z, None, p)

    def encode(self, x) -> CausalEncoderOutput:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        p = super().encode(x).latent_dist
        z = p.sample().squeeze(2)
        return CausalEncoderOutput(z, None, p)

    def decode(self, z) -> CausalDecoderOutput:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = super().decode(z).sample.squeeze(2)
        return CausalDecoderOutput(x, None)

    def preprocess(self, x):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1
        return x

    def postprocess(self, x):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: Optional[Literal["cpu", "same"]],
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
        else:
            self.disable_slicing()
        self.slicing_sample_min_size = split_size
        if split_size is not None:
            self.slicing_latent_min_size = split_size // self.temporal_downsample_factor
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)
