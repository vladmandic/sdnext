"""ERNIE-Image pipeline package.

Exports :data:`ERNIE_SPEC`. The minimum
``TransformerSpec(cls=ErnieImageTransformer2DModel)`` works because Ernie
trainer dumps use BFL-style ``model.diffusion_model.``-prefixed keys
whose names match the diffusers state_dict verbatim after prefix strip
(probed against a community finetune: full key overlap with zero
missing or unexpected). No siblings, no converter, no forbidden markers.

Without this spec, selecting an Ernie file via the UNET dropdown crashes
in :func:`diffusers.loaders.ModelMixin.from_pretrained` with a
misleading ``OSError: ... is not a valid JSON file``, because
``ErnieImageTransformer2DModel`` lacks ``from_single_file`` and the
fallback treats the safetensors as a directory looking for
``config.json``.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


ERNIE_SPEC = TransformerSpec(cls=diffusers.ErnieImageTransformer2DModel)
