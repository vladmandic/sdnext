"""Chroma pipeline package.

Exports :data:`CHROMA_SPEC`. Chroma community files use BFL-style
``model.diffusion_model.``-prefixed keys that need renaming into the
diffusers naming convention, so the spec plugs in
:func:`convert_chroma_transformer_checkpoint_to_diffusers` explicitly.
"""

import diffusers
from diffusers.loaders.single_file_utils import convert_chroma_transformer_checkpoint_to_diffusers

from pipelines.native_transformer import TransformerSpec


CHROMA_SPEC = TransformerSpec(
    cls=diffusers.ChromaTransformer2DModel,
    converter=convert_chroma_transformer_checkpoint_to_diffusers,
)
