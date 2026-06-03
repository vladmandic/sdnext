"""Chroma pipeline package.

Exports :data:`CHROMA_SPEC`. Chroma community files use BFL-style
``model.diffusion_model.``-prefixed keys that need renaming into the
diffusers naming convention, so the spec plugs in
:func:`convert_chroma_transformer_checkpoint_to_diffusers` explicitly.

``required_markers`` names the two key families the converter dereferences
unconditionally (the double-stream blocks and the distilled guidance layer).
A UNET/DiT override missing them is not a Chroma checkpoint and is rejected
before the converter, rather than crashing inside it.
"""

import diffusers
from diffusers.loaders.single_file_utils import convert_chroma_transformer_checkpoint_to_diffusers

from pipelines.native_transformer import TransformerSpec


CHROMA_SPEC = TransformerSpec(
    cls=diffusers.ChromaTransformer2DModel,
    converter=convert_chroma_transformer_checkpoint_to_diffusers,
    required_markers=(
        ("double_blocks.", "Chroma double-stream transformer blocks"),
        ("distilled_guidance_layer.", "Chroma distilled guidance layer"),
    ),
)
