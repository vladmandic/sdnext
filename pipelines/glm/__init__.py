"""GLM-Image pipeline package.

Exports :data:`GLM_IMAGE_SPEC`. The minimum
``TransformerSpec(cls=GlmImageTransformer2DModel)`` works because
GLM-Image community files use BFL-style ``model.diffusion_model.``
prefixed keys whose names match the diffusers state_dict verbatim after
prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


GLM_IMAGE_SPEC = TransformerSpec(cls=diffusers.GlmImageTransformer2DModel)
