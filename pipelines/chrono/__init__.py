"""ChronoEdit pipeline package.

Exports :data:`CHRONOEDIT_SPEC`. The minimum
``TransformerSpec(cls=ChronoEditTransformer3DModel)`` works because
ChronoEdit community files use BFL-style ``model.diffusion_model.``
prefixed keys whose names match the diffusers state_dict verbatim after
prefix strip. No siblings, no converter, no forbidden markers.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


CHRONOEDIT_SPEC = TransformerSpec(cls=diffusers.ChronoEditTransformer3DModel)
