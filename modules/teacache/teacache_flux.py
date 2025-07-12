from typing import Any, Dict, Optional, Union
import torch
import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def teacache_flux_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # Track TeaCache attention operations
    from modules import pipeline_viz
    pipeline_viz.safe_track_operation('attention_slicing', {
        'operation_type': 'teacache_flux',
        'enable_teacache': getattr(self, 'enable_teacache', False),
        'hidden_states_shape': str(hidden_states.shape) if hasattr(hidden_states, 'shape') else None
    })

    if self.enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    # Track gradient checkpointing
                    pipeline_viz.safe_track_operation('gradient_checkpoint', {
                        'operation_type': 'transformer_block',
                        'block_index': index_block,
                        'total_blocks': len(self.transformer_blocks)
                    })

                    def create_custom_forward4(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward4(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                    
                    # Complete gradient checkpointing tracking
                    pipeline_viz.safe_track_operation_complete('gradient_checkpoint', {
                        'success': True,
                        'block_index': index_block
                    })

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    # Track gradient checkpointing for single blocks
                    pipeline_viz.safe_track_operation('gradient_checkpoint', {
                        'operation_type': 'single_transformer_block',
                        'block_index': index_block,
                        'total_blocks': len(self.single_transformer_blocks)
                    })

                    def create_custom_forward2(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward2(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                    
                    # Complete gradient checkpointing tracking
                    pipeline_viz.safe_track_operation_complete('gradient_checkpoint', {
                        'success': True,
                        'block_index': index_block,
                        'operation_type': 'single_transformer_block'
                    })

                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Track gradient checkpointing (non-TeaCache path)
                pipeline_viz.safe_track_operation('gradient_checkpoint', {
                    'operation_type': 'transformer_block_standard',
                    'block_index': index_block,
                    'total_blocks': len(self.transformer_blocks)
                })

                def create_custom_forward1(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward1(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
                
                # Complete gradient checkpointing tracking
                pipeline_viz.safe_track_operation_complete('gradient_checkpoint', {
                    'success': True,
                    'block_index': index_block,
                    'operation_type': 'transformer_block_standard'
                })

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Track gradient checkpointing for single blocks (non-TeaCache path)
                pipeline_viz.safe_track_operation('gradient_checkpoint', {
                    'operation_type': 'single_transformer_block_standard',
                    'block_index': index_block,
                    'total_blocks': len(self.single_transformer_blocks)
                })

                def create_custom_forward3(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward3(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
                
                # Complete gradient checkpointing tracking
                pipeline_viz.safe_track_operation_complete('gradient_checkpoint', {
                    'success': True,
                    'block_index': index_block,
                    'operation_type': 'single_transformer_block_standard'
                })

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    # Complete TeaCache attention operation tracking
    pipeline_viz.safe_track_operation_complete('attention_slicing', {
        'success': True,
        'operation_type': 'teacache_flux',
        'output_shape': str(output.shape) if hasattr(output, 'shape') else None
    })

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
