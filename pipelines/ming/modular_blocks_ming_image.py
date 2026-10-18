# Copyright 2026 inclusionAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the MIT License. See the LICENSE file in this repository.

from dataclasses import replace

from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict, OutputParam

from .before_denoise import MingImagePrepareLatentsStep, MingImageSetTimestepsStep, MingImageTextInputStep
from .connector_ming_image import MingImageConnector
from .decoders import MingImageDecodeStep
from .denoise import MingImageDenoiseStep
from .encoders import MingImageTextEncoderStep
from .text_encoder_ming_image import MingImageTextEncoder
from .transformer_ming_image import MingImageTransformer2DModel


LOCAL_COMPONENT_TYPES = {
    "transformer": MingImageTransformer2DModel,
    "connector": MingImageConnector,
    "text_encoder": MingImageTextEncoder,
}


MingImageCoreDenoiseBlocks = InsertableDict(
    [
        ("input", MingImageTextInputStep()),
        ("prepare_latents", MingImagePrepareLatentsStep()),
        ("set_timesteps", MingImageSetTimestepsStep()),
        ("denoise", MingImageDenoiseStep()),
    ]
)


class MingImageCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "ming-image"
    block_classes = MingImageCoreDenoiseBlocks.values()
    block_names = MingImageCoreDenoiseBlocks.keys()

    @property
    def description(self) -> str:
        return "Denoise block that takes the encoded conditioning and runs the text-to-image denoising process."

    @property
    def outputs(self):
        return [OutputParam("latents", description="The denoised latents.")]


MingImageBlocksDict = InsertableDict(
    [
        ("text_encoder", MingImageTextEncoderStep()),
        ("denoise", MingImageCoreDenoiseStep()),
        ("decode", MingImageDecodeStep()),
    ]
)


class MingImageBlocks(SequentialPipelineBlocks):
    """Text-to-image blocks for Ming-Image-0.1-Design."""

    model_name = "ming-image"
    block_classes = MingImageBlocksDict.values()
    block_names = MingImageBlocksDict.keys()
    _requirements = {"diffusers": ">=0.41.0.dev0", "transformers": ">=5.0.0"}

    def __init__(self, components_repo=None, components_revision=None):
        super().__init__()
        self.components_repo = components_repo
        self.components_revision = components_revision
        self.component_types = LOCAL_COMPONENT_TYPES

    @property
    def expected_components(self):
        specs = super().expected_components
        if self.components_repo is None:
            return specs
        return [
            replace(
                spec,
                type_hint=self.component_types.get(spec.name, spec.type_hint),
                pretrained_model_name_or_path=str(self.components_repo),
                subfolder=spec.name,
                revision=self.components_revision,
            )
            if spec.default_creation_method == "from_pretrained"
            else spec
            for spec in specs
        ]

    @property
    def description(self) -> str:
        return "Text-to-image pipeline for Ming-Image-0.1-Design (RGBA output)."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
