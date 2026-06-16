"""Nucleus MoE-Image pipeline package.

Exports :data:`NUCLEUS_SPEC`. The minimum
``TransformerSpec(cls=NucleusMoEImageTransformer2DModel)`` works because
Nucleus community files use BFL-style ``model.diffusion_model.``
prefixed keys whose names match the diffusers state_dict verbatim after
prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


NUCLEUS_SPEC = TransformerSpec(cls=diffusers.NucleusMoEImageTransformer2DModel)
