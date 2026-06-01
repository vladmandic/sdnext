"""HunyuanImage pipeline package.

Exports :data:`HUNYUANIMAGE_SPEC`. The minimum
``TransformerSpec(cls=HunyuanImageTransformer2DModel)`` works because
HunyuanImage 2.1 community files use BFL-style
``model.diffusion_model.``-prefixed keys whose names match the diffusers
state_dict verbatim after prefix strip.

The HunyuanImage 3 path is a transformers ``AutoModelForCausalLM`` and
does not go through the native transformer loader, so it does not need a
spec here.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


HUNYUANIMAGE_SPEC = TransformerSpec(cls=diffusers.HunyuanImageTransformer2DModel)
