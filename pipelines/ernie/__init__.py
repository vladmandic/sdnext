"""ERNIE-Image pipeline package.

Exports :data:`ERNIE_SPEC`. The minimum
``TransformerSpec(cls=ErnieImageTransformer2DModel)`` works because Ernie
trainer dumps use BFL-style ``model.diffusion_model.``-prefixed keys
whose names match the diffusers state_dict verbatim after prefix strip
(probed against a community finetune: full key overlap with zero
missing or unexpected). No siblings, no converter, no forbidden markers.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


ERNIE_SPEC = TransformerSpec(cls=diffusers.ErnieImageTransformer2DModel)
