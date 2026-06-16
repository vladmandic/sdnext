"""Bria pipeline package.

Exports :data:`BRIA_SPEC` and :data:`BRIA_FIBO_SPEC` for use by
:mod:`pipelines.model_bria` together with :mod:`pipelines.native_transformer`.

Bria ships two transformer variants:

- The original Bria family uses a custom :class:`BriaTransformer2DModel`
  imported from :mod:`pipelines.bria.transformer_bria`. The custom class
  has no entry in diffusers' ``SINGLE_FILE_LOADABLE_CLASSES`` table.
- Bria FIBO (and FIBO Edit) use the upstream
  :class:`diffusers.BriaFiboTransformer2DModel`, which also lacks a
  ``SINGLE_FILE_LOADABLE_CLASSES`` entry.

Both specs use the default 3-prefix detection
(``model.diffusion_model.``, ``diffusion_model.``, ``net.``), no
converter, and no siblings.
"""

import diffusers

from pipelines.bria.transformer_bria import BriaTransformer2DModel
from pipelines.native_transformer import TransformerSpec


BRIA_SPEC = TransformerSpec(cls=BriaTransformer2DModel)
BRIA_FIBO_SPEC = TransformerSpec(cls=diffusers.BriaFiboTransformer2DModel)
