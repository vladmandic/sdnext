"""HunyuanDiT pipeline package.

Exports :data:`HUNYUANDIT_SPEC`. The minimum
``TransformerSpec(cls=HunyuanDiT2DModel)`` works because HunyuanDiT
community files use BFL-style ``model.diffusion_model.``-prefixed keys
whose names match the diffusers state_dict verbatim after prefix strip.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


HUNYUANDIT_SPEC = TransformerSpec(cls=diffusers.HunyuanDiT2DModel)
