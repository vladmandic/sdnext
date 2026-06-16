"""PRX pipeline package.

Exports :data:`PRX_SPEC`. The minimum
``TransformerSpec(cls=PRXTransformer2DModel)`` works because PRX
community files use BFL-style ``model.diffusion_model.``-prefixed keys
whose names match the diffusers state_dict verbatim after prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


PRX_SPEC = TransformerSpec(cls=diffusers.PRXTransformer2DModel)
