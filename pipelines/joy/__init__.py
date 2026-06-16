"""Joy-Image-Edit pipeline package.

Exports :data:`JOY_SPEC`. The minimum
``TransformerSpec(cls=JoyImageEditTransformer3DModel)`` works because
Joy community files use BFL-style ``model.diffusion_model.``-prefixed
keys whose names match the diffusers state_dict verbatim after prefix
strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


JOY_SPEC = TransformerSpec(cls=diffusers.JoyImageEditTransformer3DModel)
