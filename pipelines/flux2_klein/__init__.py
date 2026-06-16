"""Flux 2 Klein pipeline package.

Exports :data:`FLUX2_KLEIN_SPEC`. Klein shares
:class:`Flux2Transformer2DModel` with full Flux 2 but uses a smaller
config (hidden_size and friends). diffusers' ``from_single_file`` picks
the class default (= Flux 2 full), so loading a Klein-shaped community
file crashes at ``load_model_dict_into_meta`` with a shape mismatch like
``expected (36864, 6144), got (24576, 4096)``.

Routing through :mod:`pipelines.native_transformer` pulls the Klein
``transformer/config.json`` from the base repo first and instantiates
``Flux2Transformer2DModel`` at the right size, then runs the diffusers
Flux 2 converter to split fused QKV blocks and rename BFL keys into the
diffusers-expected names.
"""

import diffusers
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

from pipelines.native_transformer import TransformerSpec


FLUX2_KLEIN_SPEC = TransformerSpec(
    cls=diffusers.Flux2Transformer2DModel,
    converter=convert_flux2_transformer_checkpoint_to_diffusers,
)
