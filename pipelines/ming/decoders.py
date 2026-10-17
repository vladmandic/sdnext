# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import torch

from diffusers import AutoencoderKLQwenImage
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class MingImageDecodeStep(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "Step that decodes the denoised latents into images. The Ming-Image VAE decodes four channels, so the "
            "images are RGBA."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The denoised latents of shape `(batch, 16, height // 8, width // 8)`.",
            ),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        # The checkpoint scales latents with a single `scaling_factor` / `shift_factor` pair (no per-channel
        # `latents_mean` / `latents_std`); both are extra keys of the VAE config.
        latents = block_state.latents.to(vae.dtype).unsqueeze(2)
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        images = vae.decode(latents, return_dict=False)[0].squeeze(2)
        block_state.images = components.image_processor.postprocess(images, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state
