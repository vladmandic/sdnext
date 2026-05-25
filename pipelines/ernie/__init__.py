"""ERNIE-Image pipeline package.

Exports :data:`ERNIE_SPEC` for use by :mod:`pipelines.model_ernie` together
with :mod:`pipelines.native_transformer`. The spec captures the
ERNIE-Image-specific knobs that differ from the native-loader defaults:

- No converter is needed: community Ernie trainer dumps use BFL-style keys
  (``model.diffusion_model.``-prefixed) whose names match diffusers'
  ``ErnieImageTransformer2DModel.state_dict()`` verbatim after prefix strip.
  Probed against ``jibMixErnie_v20.safetensors`` (the upstream community
  finetune): 409/409 keys overlap with zero missing or unexpected.
- No siblings; no forbidden markers; default prefixes
  (``model.diffusion_model.``, ``diffusion_model.``, ``net.``) cover every
  exporter seen in the wild.

Before this spec, selecting an Ernie finetune via the UNET dropdown crashed
in :func:`diffusers.loaders.ModelMixin.from_pretrained` with a misleading
``OSError: ... is not a valid JSON file`` because
``ErnieImageTransformer2DModel`` lacks ``from_single_file`` support and the
fallback path treats the safetensors as a directory looking for
``config.json``.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


ERNIE_SPEC = TransformerSpec(cls=diffusers.ErnieImageTransformer2DModel)
