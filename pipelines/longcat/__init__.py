"""LongCat pipeline package.

Exports :data:`LONGCAT_SPEC`. The minimum
``TransformerSpec(cls=LongCatImageTransformer2DModel)`` works because
LongCat community files use BFL-style ``model.diffusion_model.``
prefixed keys whose names match the diffusers state_dict verbatim after
prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


LONGCAT_SPEC = TransformerSpec(cls=diffusers.LongCatImageTransformer2DModel)
