# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import torch

from diffusers import AutoModel
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


# VAE spatial downsampling (8) times the transformer patch size (2)
SPATIAL_COMPRESSION = 16
LATENT_CHANNELS = 16


class MingImageTextInputStep(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def description(self) -> str:
        return (
            "Input step that determines `batch_size` and `dtype`, and repeats each prompt's conditioning "
            "`num_images_per_prompt` times."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", AutoModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt"),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=list[torch.Tensor],
                description="Query-token conditioning, one tensor per prompt. Can be generated in the text_encoder step.",
            ),
            InputParam(
                "prompt_embeds_2",
                required=True,
                type_hint=list[torch.Tensor],
                description="Text-token conditioning, one tensor per prompt. Can be generated in the text_encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=list[torch.Tensor],
                description="Unconditional query-token conditioning. Can be generated in the text_encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds_2",
                type_hint=list[torch.Tensor],
                description="Unconditional text-token conditioning. Can be generated in the text_encoder step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "batch_size",
                type_hint=int,
                description="Number of prompts; the model batch is `batch_size * num_images_per_prompt`.",
            ),
            OutputParam("dtype", type_hint=torch.dtype, description="Dtype of the transformer inputs."),
            OutputParam(
                "prompt_embeds",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Query-token conditioning repeated per image.",
            ),
            OutputParam(
                "prompt_embeds_2",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Text-token conditioning repeated per image.",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Unconditional query-token conditioning repeated per image.",
            ),
            OutputParam(
                "negative_prompt_embeds_2",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="Unconditional text-token conditioning repeated per image.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if len(block_state.prompt_embeds) != len(block_state.prompt_embeds_2):
            raise ValueError(
                f"`prompt_embeds` ({len(block_state.prompt_embeds)}) and `prompt_embeds_2` "
                f"({len(block_state.prompt_embeds_2)}) must hold one tensor per prompt"
            )

        block_state.batch_size = len(block_state.prompt_embeds)
        block_state.dtype = components.transformer.dtype

        def repeat(embeds):
            if embeds is None:
                return None
            return [e for e in embeds for _ in range(block_state.num_images_per_prompt)]

        block_state.prompt_embeds = repeat(block_state.prompt_embeds)
        block_state.prompt_embeds_2 = repeat(block_state.prompt_embeds_2)
        block_state.negative_prompt_embeds = repeat(block_state.negative_prompt_embeds)
        block_state.negative_prompt_embeds_2 = repeat(block_state.negative_prompt_embeds_2)

        self.set_block_state(state, block_state)
        return components, state


class MingImagePrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def description(self) -> str:
        return "Prepare latents step that creates the initial float32 noise for text-to-image generation."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=2048),
            InputParam.template("width", default=2048),
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Initial noise of shape `(batch_size * num_images_per_prompt, 16, height // 8, width // 8)`.",
            ),
            InputParam.template("num_images_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts. Can be generated in the input step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The initial noisy latents, float32, of shape `(batch, 16, height // 8, width // 8)`.",
            )
        ]

    @staticmethod
    def check_inputs(height, width):
        if height % SPATIAL_COMPRESSION != 0 or width % SPATIAL_COMPRESSION != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {SPATIAL_COMPRESSION} but are {height} and {width}."
            )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.height, block_state.width)

        shape = (
            block_state.batch_size * block_state.num_images_per_prompt,
            LATENT_CHANNELS,
            block_state.height // 8,
            block_state.width // 8,
        )
        if block_state.latents is None:
            block_state.latents = randn_tensor(
                shape, generator=block_state.generator, device=components._execution_device, dtype=torch.float32
            )
        elif block_state.latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {tuple(block_state.latents.shape)}, expected {shape}")
        else:
            block_state.latents = block_state.latents.to(components._execution_device, torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class MingImageSetTimestepsStep(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def description(self) -> str:
        return (
            "Step that sets the scheduler timesteps: `num_inference_steps` sigmas from 1 down to 0, shifted by the "
            "scheduler's static `shift`. The schedule does not depend on the resolution. Sets the scheduler's "
            "`sigma_min` to 0."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=12),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps of the denoising loop."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        # The original pipeline runs the schedule down to sigma 0 by setting `sigma_min` on the scheduler and letting
        # it build the sigmas; doing the same reproduces its float64 schedule exactly (custom `sigmas` would be cast to
        # float32 before shifting).
        components.scheduler.sigma_min = 0.0
        components.scheduler.set_timesteps(block_state.num_inference_steps, device=components._execution_device)
        block_state.timesteps = components.scheduler.timesteps

        self.set_block_state(state, block_state)
        return components, state
