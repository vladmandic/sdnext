"""Kandinsky 5.0 pipeline package.

Exports :data:`KANDINSKY5_SPEC`. Kandinsky 5 uses
:class:`Kandinsky5Transformer3DModel` and follows the default layout.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


KANDINSKY5_SPEC = TransformerSpec(cls=diffusers.Kandinsky5Transformer3DModel)
