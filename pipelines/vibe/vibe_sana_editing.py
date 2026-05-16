"""Transformer Sana model."""

from typing import Any

import torch
from diffusers import ModelMixin, SanaTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.models.transformers.sana_transformer import (
    GLUMBConv,
    SanaAttnProcessor2_0,
    SanaCombinedTimestepGuidanceEmbeddings,
    SanaModulatedNorm,
)
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from torch import nn

from .edit_head import MetaConnector

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _normalize_qk_norm(qk_norm: str | bool | None) -> str | None:
    """Normalize legacy boolean qk_norm values to Diffusers-compatible strings."""
    if isinstance(qk_norm, bool):
        # Legacy checkpoints can store qk_norm as bool while current diffusers expects a string.
        return "rms_norm" if qk_norm else None
    return qk_norm


class SanaLinearAttnProcessor2_0:
    r"""Processor for implementing scaled dot-product linear attention."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute the forward pass of the SanaLinearAttnProcessor2_0.

        We need that redifinition because default implementation do not take into account the attention mask.

        Args:
            attn (Attention): The attention object.
            hidden_states (torch.Tensor): The hidden states.
            encoder_hidden_states (torch.Tensor | None): The encoder hidden states. In case of cross attention.
            attention_mask (torch.Tensor | None): The attention mask.

        Returns:
            torch.Tensor: The processed hidden states.
        """
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))

        query = nn.functional.relu(query)
        key = nn.functional.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        if attention_mask is not None:
            # attention_mask([B, S], dtype=torch.bool): True - keep, False - ignore.
            # transform to (1.0 - keep, 0.0 - ignore).
            mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1
            mask = mask.to(key.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(1)  # -> [B, 1, S]

            # Null masked tokens in key and value before torch.matmul operation
            key = key * mask.unsqueeze(-1)  # (B, _, S, _) * (B, 1, S, 1)
            value = value * mask.unsqueeze(2)  # (B, _, _, S) * (B, 1, 1, S)

        value = nn.functional.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class SanaTransformerBlock(nn.Module):
    """Transformer block introduced in [Sana](huggingface.co/papers/2410.10629)."""

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: int = 20,
        cross_attention_head_dim: int = 112,
        cross_attention_dim: int = 2240,
        norm_eps: float = 1e-6,
        mlp_ratio: float = 2.5,
        qk_norm: str | bool | None = None,
        *,
        attention_out_bias: bool = True,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
    ) -> None:
        """Initialize the SanaTransformerBlock.

        Args:
            dim (int): The dimension of the input/output hidden states.
            num_attention_heads (int): The number of heads for self-attention.
            attention_head_dim (int): The dimension of each self-attention head.
            dropout (float): The dropout probability for attention layers.
            num_cross_attention_heads (int): The number of heads for cross-attention.
            cross_attention_head_dim (int): The dimension of each cross-attention head.
            cross_attention_dim (int): The dimension of the cross-attention context.
            norm_eps (float): Epsilon value for layer normalization stability.
            mlp_ratio (float): Expansion ratio for the feed-forward network hidden dimension.
            qk_norm (str | bool | None): Normalization method for Query/Key vectors.
            attention_out_bias (bool): Whether to include bias in the attention output projection.
            attention_bias (bool): Whether to include bias in the attention Q/K/V projections.
            norm_elementwise_affine (bool): Whether to learn affine parameters for normalization.
        """
        super().__init__()
        qk_norm = _normalize_qk_norm(qk_norm)

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads if qk_norm is not None else None,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=SanaLinearAttnProcessor2_0(),  # type: ignore[arg-type]
        )

        # 2. Cross Attention
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                qk_norm=qk_norm,
                kv_heads=num_cross_attention_heads if qk_norm is not None else None,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                processor=SanaAttnProcessor2_0(),  # type: ignore[arg-type]
            )

        # 3. Feed-forward
        self.ff = GLUMBConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        timestep: torch.LongTensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Executes the forward pass of the transformer block.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape `(batch_size, seq_len, dim)`.
            attention_mask (torch.Tensor | None): Mask for self-attention.
            encoder_hidden_states (torch.Tensor | None): Conditional embeddings (e.g., text) for cross-attention.
            encoder_attention_mask (torch.Tensor | None): Mask for cross-attention.
            timestep (torch.LongTensor): Timestep embeddings used for adaptive modulation of normalization layers.
            height (int): Spatial height of the latent image (used for reshaping in FFN).
            width (int): Spatial width of the latent image (used for reshaping in FFN).

        Returns:
            torch.Tensor: Transformed hidden states of shape `(batch_size, seq_len, dim)`.
        """
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if width * height == norm_hidden_states.shape[1]:
            norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
            ff_output = self.ff(norm_hidden_states)
            ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        else:
            # "seq_cat" case
            # apply the same local processing to each of the parts independently of each other
            # global interaction has already occurred, work with spatial features separately
            norm_hidden_states_1, norm_hidden_states_2 = torch.chunk(norm_hidden_states, 2, dim=1)

            hidden_states_reshaped_1 = norm_hidden_states_1.unflatten(1, (height, width)).permute(0, 3, 1, 2)
            hidden_states_reshaped_2 = norm_hidden_states_2.unflatten(1, (height, width)).permute(0, 3, 1, 2)

            ff_output_1 = self.ff(hidden_states_reshaped_1).flatten(2, 3).permute(0, 2, 1)
            ff_output_2 = self.ff(hidden_states_reshaped_2).flatten(2, 3).permute(0, 2, 1)

            ff_output = torch.cat([ff_output_1, ff_output_2], dim=1)

        return hidden_states + gate_mlp * ff_output


class VIBESanaEditingModel(SanaTransformer2DModel):
    """VIBE Sana editing model specialized for instruction-based image editing.

    This model extends `SanaTransformer2DModel` to support advanced editing capabilities.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int | None = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: int = 20,
        cross_attention_head_dim: int = 112,
        cross_attention_dim: int = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_eps: float = 1e-6,
        interpolation_scale: int | None = None,
        qk_norm: str | bool | None = None,
        timestep_scale: float = 1.0,
        input_condition_type: str = "channel_cat",
        edit_head_input_dim: int = 2048,
        edit_head_stacks_num: int = 4,
        num_meta_queries: int = 224,
        meta_queries_dim: int = 2048,
        *,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = False,
        guidance_embeds: bool = False,
    ) -> None:
        """Initialize the VIBESanaEditingModel.

        Args:
            in_channels (int): Number of channels in the input latent representation.
            out_channels (int | None): Number of output channels. Defaults to `in_channels`.
            num_attention_heads (int): Number of attention heads in transformer blocks.
            attention_head_dim (int): Dimension of each attention head.
            num_layers (int): Number of transformer blocks in the model.
            num_cross_attention_heads (int): Number of heads for cross-attention layers.
            cross_attention_head_dim (int): Dimension of each cross-attention head.
            cross_attention_dim (int): Dimension of the cross-attention context embedding.
            caption_channels (int): Dimension of the input caption/text embeddings.
            mlp_ratio (float): Expansion ratio for the feed-forward networks.
            dropout (float): Dropout probability.
            sample_size (int): Spatial size of the input sample (height/width).
            patch_size (int): Size of the patches extracted from the input latent.
            norm_eps (float): Epsilon for layer normalization.
            interpolation_scale (int | None): Scale factor for positional embedding interpolation.
            qk_norm (str | bool | None): Normalization type for Query/Key (e.g., 'rms_norm').
            timestep_scale (float): Scale factor for the timestep.
            input_condition_type (str): Method for conditioning on the input image. Options: 'channel_cat', 'seq_cat'.
            edit_head_input_dim (int): Input dimension for the `MetaConnector` edit head.
            edit_head_stacks_num (int): Number of layers in the `MetaConnector`.
            num_meta_queries (int): Number of meta tokens to extract from the text embeddings for the edit head.
            meta_queries_dim (int): Dimension of the meta query embeddings.
            attention_bias (bool): Whether to use bias in attention projections.
            norm_elementwise_affine (bool): Whether to learn affine parameters in normalization.
            guidance_embeds (bool): Whether to use additional guidance embeddings.
        """
        super(SanaTransformer2DModel, self).__init__()
        qk_norm = _normalize_qk_norm(qk_norm)

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_type="sincos" if interpolation_scale is not None else None,
        )

        # 2. Additional condition embeddings
        if guidance_embeds:
            self.time_embed = SanaCombinedTimestepGuidanceEmbeddings(inner_dim)
        else:
            self.time_embed = AdaLayerNormSingle(inner_dim)  # type: ignore[assignment]

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        # 5. Meta queries embedding
        self.num_meta_queries = num_meta_queries
        self.meta_queries = nn.Parameter(torch.randn(num_meta_queries, meta_queries_dim) / meta_queries_dim**0.5)

        # 6. Edit head
        self.input_condition_type = input_condition_type
        self.edit_head = MetaConnector(
            input_dim=edit_head_input_dim,
            n_stacks=edit_head_stacks_num,
            output_dim=caption_channels,
        )
        self.gradient_checkpointing = False

    def get_hidden_states_for_meta_tokens(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extracts hidden states corresponding to meta tokens from the text encoder output.

        This method identifies the positions of meta tokens in the input sequence using
        the attention mask and extracts their embeddings. It assumes the meta tokens
        are located at the end of the valid sequence.

        Args:
            text_encoder_hidden_states (torch.Tensor): Output embeddings from the text encoder
                of shape `(batch_size, seq_len, hidden_dim)`.
            attention_mask (torch.Tensor): Boolean mask indicating valid tokens (True) and
                padding (False), used to locate the end of the prompt.

        Returns:
            torch.Tensor: Extracted meta token embeddings of shape `(batch_size, num_meta_queries, hidden_dim)`.
        """
        img_hidden_states: list[torch.Tensor] = []
        for i in range(text_encoder_hidden_states.shape[0]):
            hidden_states_masked = text_encoder_hidden_states[i][attention_mask[i]]
            img_hidden_states.append(hidden_states_masked[-self.num_meta_queries :, :].unsqueeze(0))
        return torch.cat(img_hidden_states, dim=0)

    def forward_edit_heads(
        self,
        batch_size: int,
        meta_tokens_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes meta token embeddings through the edit head for both conditional and unconditional generation.

        This method generates editing embeddings for:
        1.  **Conditional input** (`hid_edit`): Using the extracted `meta_tokens_hidden_states`.
        2.  **Unconditional/Null input** (`hid_null`): Using zeroed hidden states, facilitating
            Classifier-Free Guidance (CFG) during inference.

        Args:
            batch_size (int): The batch size of the generation.
            meta_tokens_hidden_states (torch.Tensor): Input hidden states for the meta tokens
                extracted from the text prompt.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - `hid_edit` (torch.Tensor): Editing embeddings for the conditional branch.
                - `hid_null` (torch.Tensor): Editing embeddings for the unconditional (null) branch.
        """
        # repeat meta tokens embeddings to fit batch_size
        meta_tokens_embeddings = self.meta_queries.unsqueeze(dim=0)
        meta_tokens_embeddings = meta_tokens_embeddings.repeat(batch_size, 1, 1)

        # create zero hidden states for null prompts
        zero_hidden_states = torch.zeros_like(meta_tokens_hidden_states)

        # get hidden states for real prompts
        hid_edit = self.edit_head(meta_tokens_hidden_states, meta_tokens_embeddings)

        # get hidden states for null prompts
        hid_null = self.edit_head(zero_hidden_states, meta_tokens_embeddings)

        return hid_edit, hid_null

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: tuple[torch.Tensor] | None = None,
        *,
        t2i_samples: list[bool] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor, ...] | Transformer2DModelOutput:
        """Execute the forward pass of the VIBE Sana Editing Model.

        Args:
            hidden_states (torch.Tensor): Noisy latent image input of shape
                `(batch_size, num_channels, height, width)`.
            encoder_hidden_states (torch.Tensor): Conditional text embeddings of shape
                `(batch_size, seq_len, hidden_dim)`.
            timestep (torch.Tensor): Denoising timesteps.
            guidance (torch.Tensor | None, optional): Guidance embeddings for the diffusion process.
            encoder_attention_mask (torch.Tensor | None, optional): Mask for `encoder_hidden_states`.
            attention_mask (torch.Tensor | None, optional): Mask for `hidden_states`.
            attention_kwargs (dict[str, Any] | None, optional): Extra arguments for attention processors.
            controlnet_block_samples (tuple[torch.Tensor] | None, optional): Additional residuals
                from a ControlNet model, if used.
            t2i_samples (list[bool] | None, optional): Marker if the sample is a t2i sample.
            return_dict (bool): If True, returns a `Transformer2DModelOutput`. Otherwise, returns a tuple.

        Returns:
            tuple[torch.Tensor, ...] | Transformer2DModelOutput: The denoised latent output.
            If `return_dict` is True, returns `Transformer2DModelOutput(sample=output)`.

        Raises:
            ValueError: If `input_condition_type` is not 'channel_cat' or 'seq_cat'.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            # ensure attention_mask is a bias
            # assume that mask is expressed as:
            #     #   (1 = keep,      0 = discard)
            #     # convert mask into a bias that can be added to attention scores:
            #     #       (keep = +0,     discard = -10000.0)
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)  # type: ignore[union-attr]

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape

        if self.input_condition_type == "channel_cat":  # type: ignore
            # For mixed batch of t2i and editing samples, we need to mask the input image embeddings for the t2i samples
            if t2i_samples is not None and any(t2i_samples):
                hidden_states, input_img_embeds = hidden_states.split(num_channels // 2, dim=1)  # type: ignore
                mask = torch.tensor(t2i_samples, dtype=torch.bool, device=input_img_embeds.device).view(-1, 1, 1, 1)
                input_img_embeds = torch.where(mask, torch.zeros_like(input_img_embeds), input_img_embeds)
                hidden_states = torch.cat([hidden_states, input_img_embeds], dim=1)  # type: ignore
            hidden_states = self.patch_embed(hidden_states)
            attention_mask = None  # There is no need for attention mask in channel_cat case

        elif self.input_condition_type == "seq_cat":
            num_channels = hidden_states.shape[1]
            hidden_states, input_img_embeds = hidden_states.split(num_channels // 2, dim=1)
            hidden_states = self.patch_embed(hidden_states)

            if t2i_samples is not None and all(t2i_samples):
                # All samples are t2i samples, so there is no need for attention mask
                # Also, we don't need to patch the input image embeddings
                attention_mask = None
            elif t2i_samples is None or (t2i_samples is not None and not any(t2i_samples)):
                # All samples are editing samples, still no need for attention mask
                # But we need to patch the input image embeddings
                input_img_embeds = self.patch_embed(input_img_embeds)
                hidden_states = torch.cat([hidden_states, input_img_embeds], dim=1)
                attention_mask = None
            else:
                # Mixed samples.
                input_img_embeds = self.patch_embed(input_img_embeds)
                # Create attnention masks for hidden_states and input_img_embeds
                hidden_stated_mask = torch.ones(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    dtype=torch.bool,
                    device=hidden_states.device,
                )  # Shape: [bs, seq_len]
                input_img_embeds_mask = torch.ones(
                    input_img_embeds.shape[0],
                    input_img_embeds.shape[1],
                    dtype=torch.bool,
                    device=input_img_embeds.device,
                )  # Shape: [bs, seq_len]

                # Set False for t2i samples in input_img_embeds_mask
                if t2i_samples is not None:
                    t2i_samples_tensor = torch.as_tensor(t2i_samples, dtype=torch.bool, device=hidden_states.device)
                    input_img_embeds_mask[t2i_samples_tensor] = False

                hidden_states = torch.cat([hidden_states, input_img_embeds], dim=1)  # type: ignore
                attention_mask = torch.cat([hidden_stated_mask, input_img_embeds_mask], dim=1)
        else:
            msg = f"Invalid input condition type: {self.input_condition_type}"
            raise ValueError(msg)  # type: ignore

        patch_size = self.config.patch_size  # type: ignore[attr-defined]
        post_patch_height, post_patch_width = height // patch_size, width // patch_size

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(timestep, guidance=guidance, hidden_dtype=hidden_states.dtype)
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(  # type: ignore[attr-defined]
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            patch_size,
            patch_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * patch_size, post_patch_width * patch_size)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
