from typing import Type, Dict, Any, Tuple, Optional
import math
import torch
import torch.nn.functional as F
from diffusers.pipelines import auto_pipeline


current_steps = 50
def sd15_hidiffusion_key():
    modified_key = {}
    modified_key['down_module_key'] = ['down_blocks.0.downsamplers.0.conv']
    modified_key['down_module_key_extra'] = ['down_blocks.1']
    modified_key['up_module_key'] = ['up_blocks.2.upsamplers.0.conv']
    modified_key['up_module_key_extra'] = ['up_blocks.2']
    modified_key['windown_attn_module_key'] = [
                               'down_blocks.0.attentions.0.transformer_blocks.0',
                               'down_blocks.0.attentions.1.transformer_blocks.0',
                               'up_blocks.3.attentions.0.transformer_blocks.0',
                               'up_blocks.3.attentions.1.transformer_blocks.0',
                               'up_blocks.3.attentions.2.transformer_blocks.0']
    return modified_key

def sdxl_hidiffusion_key():
    modified_key = {}
    modified_key['down_module_key'] = ['down_blocks.1']
    modified_key['down_module_key_extra'] = ['down_blocks.1.downsamplers.0.conv']
    modified_key['up_module_key'] = ['up_blocks.1']
    modified_key['up_module_key_extra'] = ['up_blocks.0.upsamplers.0.conv']
    modified_key['windown_attn_module_key'] = [
                               'down_blocks.1.attentions.0.transformer_blocks.0',
                               'down_blocks.1.attentions.0.transformer_blocks.1',
                               'down_blocks.1.attentions.1.transformer_blocks.0',
                               'down_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.0.transformer_blocks.0',
                               'up_blocks.1.attentions.0.transformer_blocks.1',
                               'up_blocks.1.attentions.1.transformer_blocks.0',
                               'up_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.2.transformer_blocks.0',
                               'up_blocks.1.attentions.2.transformer_blocks.1']
    return modified_key


def sdxl_turbo_hidiffusion_key():
    modified_key = {}
    modified_key['down_module_key'] = ['down_blocks.1']
    modified_key['up_module_key'] = ['up_blocks.1']
    modified_key['windown_attn_module_key'] = [
                               'down_blocks.1.attentions.0.transformer_blocks.0',
                               'down_blocks.1.attentions.0.transformer_blocks.1',
                               'down_blocks.1.attentions.1.transformer_blocks.0',
                               'down_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.0.transformer_blocks.0',
                               'up_blocks.1.attentions.0.transformer_blocks.1',
                               'up_blocks.1.attentions.1.transformer_blocks.0',
                               'up_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.2.transformer_blocks.0',
                               'up_blocks.1.attentions.2.transformer_blocks.1']
    return modified_key


# T1_ratio: see T1 introduced in the main paper. T1 = number_inference_step * T1_ratio. A higher T1_ratio can better mitigate object duplication. We set T1_ratio=0.4 by default. You'd better adjust it to fit your prompt. Only active when apply_raunet=True.
# T2_ratio: see T2 introduced in the appendix, used in extreme resolution image generation. T2 = number_inference_step * T2_ratio. A higher T2_ratio can better mitigate object duplication. Only active when apply_raunet=True
switching_threshold_ratio_dict = {
    'sd15_1024': {'T1_ratio': 0.4, 'T2_ratio': 0.0},
    'sd15_2048': {'T1_ratio': 0.7, 'T2_ratio': 0.3},
    'sdxl_2048': {'T1_ratio': 0.4, 'T2_ratio': 0.0},
    'sdxl_4096': {'T1_ratio': 0.7, 'T2_ratio': 0.3},
    'sdxl_turbo_1024': {'T1_ratio': 0.5, 'T2_ratio': 0.0},
}


text_to_img_controlnet_switching_threshold_ratio_dict = {
    'sdxl_2048': {'T1_ratio': 0.5, 'T2_ratio': 0.0},
}
controlnet_apply_steps_rate = 0.6
is_aggressive_raunet = True
aggressive_step = 8
inpainting_is_aggressive_raunet = False
playground_is_aggressive_raunet = False


def _chunked_feed_forward(ff: torch.nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


def make_diffusers_transformer_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace global self-attention with MSW-MSA
    class transformer_block(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.FloatTensor:
            # reference: https://github.com/microsoft/Swin-Transformer
            def window_partition(x, window_size, shift_size, H, W):
                B, _N, C = x.shape
                x = x.view(B,H,W,C)
                if H % 2 != 0 or W % 2 != 0:
                    from modules.errors import log
                    log.warning('HiDiffusion: The feature size is not divisible by 2')
                    x = F.interpolate(x.permute(0,3,1,2).contiguous(), size=(window_size[0]*2, window_size[1]*2), mode='bicubic').permute(0,2,3,1).contiguous()
                if type(shift_size) == list or type(shift_size) == tuple:
                    if shift_size[0] > 0:
                        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
                else:
                    if shift_size > 0:
                        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
                x = x.view(B, 2, window_size[0], 2, window_size[1], C)
                windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
                windows = windows.view(-1, window_size[0] * window_size[1], C)
                return windows

            def window_reverse(windows, window_size, H, W, shift_size):
                B, _N, C = windows.shape
                windows = windows.view(-1, window_size[0], window_size[1], C)
                B = int(windows.shape[0] / 4) # 2x2
                x = windows.view(B, 2, 2, window_size[0], window_size[1], -1)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, window_size[0]*2, window_size[1]*2, -1)
                if type(shift_size) == list or type(shift_size) == tuple:
                    if shift_size[0] > 0:
                        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
                else:
                    if shift_size > 0:
                        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
                if H % 2 != 0 or W % 2 != 0:
                    x = F.interpolate(x.permute(0,3,1,2).contiguous(), size=(H, W), mode='bicubic').permute(0,2,3,1).contiguous()
                x = x.view(B, H*W, C)
                return x

            batch_size = hidden_states.shape[0]
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("HiDiffusion: Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # MSW-MSA
            rand_num = torch.rand(1)
            _B, N, _C = hidden_states.shape
            try:
                ori_H, ori_W = self.info['size']
            except Exception as e:
                raise RuntimeError(f'HiDiffusion: cls={self.__class__.__name__} info={hasattr(self, "info")} parent={hasattr(self, "_parent")} orphaned call') from e

            downsample_ratio = round(((ori_H*ori_W) / N)**0.5)
            H, W = (math.ceil(ori_H/downsample_ratio), math.ceil(ori_W/downsample_ratio))
            widow_size = (math.ceil(H/2), math.ceil(W/2))
            if rand_num <= 0.25:
                shift_size = (0,0)
            elif rand_num > 0.25 and rand_num <= 0.5:
                shift_size = (widow_size[0]//4, widow_size[1]//4)
            elif rand_num > 0.5 and rand_num <= 0.75:
                shift_size = (widow_size[0]//4*2, widow_size[1]//4*2)
            elif rand_num > 0.75 and rand_num <= 1:
                shift_size = (widow_size[0]//4*3, widow_size[1]//4*3)
            else:
                shift_size = (0,0)
            norm_hidden_states = window_partition(norm_hidden_states, widow_size, shift_size, H, W)

            # 1. Retrieve lora scale.
            # cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            # 2. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            attn_output = window_reverse(attn_output, widow_size, H, W, shift_size)

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.use_ada_layer_norm_single:
                    norm_hidden_states = hidden_states
                elif self.use_ada_layer_norm_continuous:
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("HiDiffusion: Incorrect norm")

                if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            if self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            if self._chunk_size is not None:
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output
            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
            return hidden_states

    return transformer_block


def make_diffusers_cross_attn_down_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional downsampler with resolution-aware downsampler
    class cross_attn_down_block(block_class):
        _parent = block_class # Save for unpatching later
        timestep = 0
        aggressive_raunet = False
        T1_ratio = 0
        T1_start = 0
        T1_end = 0
        T1 = 0 # to avoid confict with sdxl-turbo
        max_timestep = current_steps

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            additional_residuals: Optional[torch.FloatTensor] = None,
        ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
            if not hasattr(self.info['pipeline'], '_num_timesteps'):
                self.info['pipeline']._num_timesteps = self.max_timestep # pylint: disable=protected-access
            self.max_timestep = self.info['pipeline']._num_timesteps # pylint: disable=protected-access
            # self.max_timestep = len(self.info['scheduler'].timesteps)
            try:
                ori_H, ori_W = self.info['size']
            except Exception as e:
                raise RuntimeError(f'HiDiffusion: cls={self.__class__.__name__} info={hasattr(self, "info")} parent={hasattr(self, "_parent")} orphaned call') from e
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise RuntimeError('HiDiffusion: unsupported model type')

            if self.aggressive_raunet:
                self.T1_start = int(aggressive_step/50 * self.max_timestep)
                self.T1_end = int(self.max_timestep * self.T1_ratio)
                self.T1 = 0 # to avoid confict with sdxl-turbo
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)

            output_states = ()
            _scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            blocks = list(zip(self.resnets, self.attentions))

            for i, (resnet, attn) in enumerate(blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    # hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                # apply additional residuals to the output of the last pair of resnet and attention blocks
                if i == len(blocks) - 1 and additional_residuals is not None:
                    hidden_states = hidden_states + additional_residuals

                if i == 0:
                    if self.aggressive_raunet and self.timestep >= self.T1_start and self.timestep < self.T1_end:
                        self.info["upsample_size"] = (hidden_states.shape[2], hidden_states.shape[3])
                        hidden_states = F.avg_pool2d(hidden_states, kernel_size=(2,2),ceil_mode=True)
                    elif self.timestep < self.T1:
                        self.info["upsample_size"] = (hidden_states.shape[2], hidden_states.shape[3])
                        hidden_states = F.avg_pool2d(hidden_states, kernel_size=(2,2),ceil_mode=True)
                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                    # hidden_states = downsampler(hidden_states, scale=lora_scale)

                output_states = output_states + (hidden_states,)

            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0

            return hidden_states, output_states

    return cross_attn_down_block


def make_diffusers_cross_attn_up_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional downsampler with resolution-aware downsampler
    class cross_attn_up_block(block_class):
        # Save for unpatching later
        _parent = block_class
        timestep = 0
        aggressive_raunet = False
        T1_ratio = 0
        T1_start = 0
        T1_end = 0
        T1 = 0 # to avoid confict with sdxl-turbo
        max_timestep = 50

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:
            def fix_scale(first, second):
                if (first.shape[-1] != second.shape[-1] or first.shape[-2] != second.shape[-2]):
                    rescale = min(second.shape[-2] / first.shape[-2], second.shape[-1] / first.shape[-1])
                    # log.debug(f"HiDiffusion rescale: {hidden_states.shape} => {res_hidden_states_tuple[0].shape} scale={rescale}")
                    return F.interpolate(first, scale_factor=rescale, mode='bicubic')
                return first

            self.max_timestep = self.info['pipeline']._num_timesteps # pylint: disable=protected-access
            try:
                ori_H, ori_W = self.info['size']
            except Exception as e:
                raise RuntimeError(f'HiDiffusion: cls={self.__class__.__name__} info={hasattr(self, "info")} parent={hasattr(self, "_parent")} orphaned call') from e
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]

                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet

                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise RuntimeError('HiDiffusion: unsupported model type')

            if self.aggressive_raunet:
                self.T1_start = int(aggressive_step/50 * self.max_timestep)
                self.T1_end = int(self.max_timestep * self.T1_ratio)
                self.T1 = 0 # to avoid confict with sdxl-turbo
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)

            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = fix_scale(hidden_states, res_hidden_states)
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if i == 1:
                    if self.aggressive_raunet and self.timestep >= self.T1_start and self.timestep < self.T1_end:
                        hidden_states = F.interpolate(hidden_states, size=self.info["upsample_size"], mode='bicubic')
                    elif self.timestep < self.T1:
                        hidden_states = F.interpolate(hidden_states, size=self.info["upsample_size"], mode='bicubic')

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)
            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0
            return hidden_states

    return cross_attn_up_block


def make_diffusers_downsampler_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional downsampler with resolution-aware downsampler
    class downsampler_block(block_class):
        # Save for unpatching later
        _parent = block_class
        T1_ratio = 0
        T1 = 0
        timestep = 0
        aggressive_raunet = False
        max_timestep = 50

        def forward(self, hidden_states: torch.Tensor, scale = 1.0) -> torch.Tensor: # pylint: disable=unused-argument
            self.max_timestep = self.info['pipeline']._num_timesteps # pylint: disable=protected-access
            # self.max_timestep = len(self.info['scheduler'].timesteps)
            try:
                ori_H, ori_W = self.info['size']
            except Exception as e:
                raise RuntimeError(f'HiDiffusion: cls={self.__class__.__name__} info={hasattr(self, "info")} parent={hasattr(self, "_parent")} orphaned call') from e
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise RuntimeError('HiDiffusion: unsupported model type')

            if self.aggressive_raunet:
                self.T1 = int(aggressive_step/50 * self.max_timestep)
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)
            if self.timestep < self.T1:
                self.ori_stride = self.stride # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.ori_padding = self.padding # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.ori_dilation = self.dilation # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.stride = (4,4) # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.padding = (2,2) # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.dilation = (2,2) # pylint: disable=access-member-before-definition, attribute-defined-outside-init

            hidden_states = F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            if self.timestep < self.T1:
                self.stride = self.ori_stride # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.padding = self.ori_padding # pylint: disable=access-member-before-definition, attribute-defined-outside-init
                self.dilation = self.ori_dilation # pylint: disable=access-member-before-definition, attribute-defined-outside-init
            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0
            return hidden_states

    return downsampler_block


def make_diffusers_upsampler_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional upsampler with resolution-aware downsampler
    class upsampler_block(block_class):
        # Save for unpatching later
        _parent = block_class
        T1_ratio = 0
        T1 = 0
        timestep = 0
        aggressive_raunet = False
        max_timestep = 50

        def forward(self, hidden_states: torch.Tensor, scale = 1.0) -> torch.Tensor: # pylint: disable=unused-argument
            self.max_timestep = self.info['pipeline']._num_timesteps # pylint: disable=protected-access
            # self.max_timestep = len(self.info['scheduler'].timesteps)
            try:
                ori_H, ori_W = self.info['size']
            except Exception as e:
                raise RuntimeError(f'HiDiffusion: cls={self.__class__.__name__} info={hasattr(self, "info")} parent={hasattr(self, "_parent")} orphaned call') from e
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]

                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise RuntimeError('HiDiffusion: unsupported model type')

            if self.aggressive_raunet:
                self.T1 = int(aggressive_step/50 * self.max_timestep)
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)
            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0

            return F.conv2d(hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return upsampler_block


def hook_diffusion_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_hidiffusion. """
    def hook(module, args):
        module.info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model.info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_hidiffusion(
        model: torch.nn.Module,
        apply_raunet: bool = True,
        apply_window_attn: bool = True,
        model_type: str = 'None',
        steps: int = 50):
    """
    model: diffusers model. We support SD 1.5, 2.1, XL, XL Turbo.
    apply_raunet: whether to apply RAU-Net
    apply_window_attn: whether to apply MSW-MSA.
    """
    global current_steps # pylint: disable=global-statement
    current_steps = steps
    if hasattr(model, 'controlnet') and (model_type == 'sd' or model_type == 'sdxl'):
        from .hidiffusion_controlnet import make_diffusers_sdxl_contrtolnet_ppl, make_diffusers_unet_2d_condition
        make_ppl_fn = make_diffusers_sdxl_contrtolnet_ppl
        model.__class__ = make_ppl_fn(model.__class__)
        make_block_fn = make_diffusers_unet_2d_condition
        model.unet.__class__ = make_block_fn(model.unet.__class__)

    diffusion_model = model.unet if hasattr(model, "unet") else model
    diffusion_model.num_upsamplers += 12
    diffusion_model.info = {
        'size': None,
        'upsample_size': None,
        'hooks': [],
        'text_to_img_controlnet': hasattr(model, 'controlnet'),
        'is_inpainting_task': model.__class__ in auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.values(),
        'pipeline': model}

    if model_type == 'sd':
        modified_key = sd15_hidiffusion_key()
        for key, module in diffusion_model.named_modules():
            if hasattr(module, "_parent"):
                raise RuntimeError(f'HiDiffusion: key={key} module={module.__class__} already patched')
            if apply_raunet and key in modified_key['down_module_key']:
                module.__class__ = make_diffusers_downsampler_block(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'
            if apply_raunet and key in modified_key['down_module_key_extra']:
                module.__class__ = make_diffusers_cross_attn_down_block(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'
            if apply_raunet and key in modified_key['up_module_key']:
                module.__class__ = make_diffusers_upsampler_block(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'
            if apply_raunet and key in modified_key['up_module_key_extra']:
                module.__class__ = make_diffusers_cross_attn_up_block(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'
            if apply_window_attn and key in modified_key['windown_attn_module_key']:
                module.__class__ = make_diffusers_transformer_block(module.__class__)
            if hasattr(module, "_parent"):
                module.model = 'sd15'
                module.info = diffusion_model.info

    elif model_type == 'sdxl':
        modified_key = sdxl_hidiffusion_key()
        for key, module in diffusion_model.named_modules():
            if hasattr(module, "_parent"):
                raise RuntimeError(f'HiDiffusion: key={key} module={module.__class__} already patched')
            if apply_raunet and key in modified_key['down_module_key']:
                module.__class__ = make_diffusers_cross_attn_down_block(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'
            if apply_raunet and key in modified_key['down_module_key_extra']:
                module.__class__ = make_diffusers_downsampler_block(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'
            if apply_raunet and key in modified_key['up_module_key']:
                module.__class__ = make_diffusers_cross_attn_up_block(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'
            if apply_raunet and key in modified_key['up_module_key_extra']:
                module.__class__ = make_diffusers_upsampler_block(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'
            if apply_window_attn and key in modified_key['windown_attn_module_key']:
                module.__class__ = make_diffusers_transformer_block(module.__class__)
            if hasattr(module, "_parent"):
                module.model = 'sdxl'
                module.info = diffusion_model.info
    else:
        raise RuntimeError('HiDiffusion: unsupported model type')

    model.info = diffusion_model.info
    model.hidiffusion = True
    hook_diffusion_model(diffusion_model)


def remove_hidiffusion(model: torch.nn.Module):
    """ Removes hidiffusion from a Diffusion module if it was already patched. """
    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        while hasattr(module, "_parent"):
            model.hidiffusion = True
            module.__class__ = module._parent # pylint: disable=protected-access
        if hasattr(module, "info"):
            for hook in module.info.get("hooks", []):
                hook.remove()
            module.info["hooks"].clear()
            del module.info
