"""PixArt pipeline package.

Exports :data:`PIXART_SPEC`. The minimum
``TransformerSpec(cls=PixArtTransformer2DModel)`` works because PixArt
community files use BFL-style ``model.diffusion_model.``-prefixed keys
whose names match the diffusers state_dict verbatim after prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


PIXART_SPEC = TransformerSpec(cls=diffusers.PixArtTransformer2DModel)
