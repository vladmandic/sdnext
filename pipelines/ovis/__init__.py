"""Ovis-Image pipeline package.

Exports :data:`OVIS_SPEC`. The minimum
``TransformerSpec(cls=OvisImageTransformer2DModel)`` works because
Ovis community files use BFL-style ``model.diffusion_model.``-prefixed
keys whose names match the diffusers state_dict verbatim after prefix
strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


OVIS_SPEC = TransformerSpec(cls=diffusers.OvisImageTransformer2DModel)
