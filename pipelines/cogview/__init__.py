"""CogView pipeline package.

Exports :data:`COGVIEW3_SPEC` and :data:`COGVIEW4_SPEC`. Both default
specs work because CogView community files use BFL-style
``model.diffusion_model.``-prefixed keys whose names match the diffusers
state_dict verbatim after prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


COGVIEW3_SPEC = TransformerSpec(cls=diffusers.CogView3PlusTransformer2DModel)
COGVIEW4_SPEC = TransformerSpec(cls=diffusers.CogView4Transformer2DModel)
