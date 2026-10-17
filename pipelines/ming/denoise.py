# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

import torch

from diffusers import AutoModel
from diffusers.configuration_utils import FrozenDict
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class MingImageLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that prepares the transformer inputs: one `(channels, 1, height, width)` "
            "latent per image and the normalized time `(1000 - t) / 1000`."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="The latents being denoised. Can be generated in the prepare latents step.",
            ),
            InputParam(
                "dtype",
                required=True,
                type_hint=torch.dtype,
                description="Dtype of the transformer inputs. Can be generated in the input step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        latents = block_state.latents.to(block_state.dtype).unsqueeze(2)
        block_state.latent_model_input = list(latents.unbind(dim=0))
        block_state.timestep = (1000 - t.expand(latents.shape[0])) / 1000
        return components, block_state


class MingImageLoopDenoiser(ModularPipelineBlocks):
    model_name = "ming-image"

    # transformer argument -> (conditional, unconditional) block_state fields
    guider_input_fields = {
        "encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds"),
        "encoder_hidden_states_2": ("prompt_embeds_2", "negative_prompt_embeds_2"),
    }

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0, "use_original_formulation": True, "enabled": False}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", AutoModel),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that runs the transformer for each guidance batch and combines the "
            "predictions with the guider. The original pipeline's `cfg` matches `ClassifierFreeGuidance` with "
            "`use_original_formulation=True` and `guidance_scale=cfg`."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of denoising steps. Can be generated in the set timesteps step.",
            ),
            InputParam.template("denoiser_input_fields"),
            InputParam("prompt_embeds", required=True, type_hint=list[torch.Tensor]),
            InputParam("prompt_embeds_2", required=True, type_hint=list[torch.Tensor]),
            InputParam("negative_prompt_embeds", type_hint=list[torch.Tensor]),
            InputParam("negative_prompt_embeds_2", type_hint=list[torch.Tensor]),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs_from_block_state(block_state, self.guider_input_fields)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {
                name: [e.to(block_state.dtype) for e in getattr(guider_state_batch, name)]
                for name in self.guider_input_fields
            }
            model_out = components.transformer(
                hidden_states=block_state.latent_model_input,
                timestep=block_state.timestep,
                return_dict=False,
                **cond_kwargs,
            )[0]
            # the transformer predicts the negated flow-matching velocity
            guider_state_batch.noise_pred = -torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
            components.guider.cleanup_models(components.transformer)

        block_state.noise_pred = components.guider(guider_state)[0]
        return components, block_state


class MingImageLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ming-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def description(self) -> str:
        return "Step within the denoising loop that updates the float32 latents with the scheduler."

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latents = components.scheduler.step(
            block_state.noise_pred.float(), t, block_state.latents, return_dict=False
        )[0]
        return components, block_state


class MingImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "ming-image"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the latents over `timesteps`. "
            "The steps of each iteration are defined by `sub_blocks`."
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The timesteps of the denoising loop. Can be generated in the set timesteps step.",
            ),
            InputParam(
                "num_inference_steps",
                required=True,
                type_hint=int,
                description="The number of denoising steps. Can be generated in the set timesteps step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class MingImageDenoiseStep(MingImageDenoiseLoopWrapper):
    block_classes = [MingImageLoopBeforeDenoiser, MingImageLoopDenoiser, MingImageLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises the latents. At each iteration it runs, in order: "
            "`MingImageLoopBeforeDenoiser`, `MingImageLoopDenoiser`, `MingImageLoopAfterDenoiser`."
        )
