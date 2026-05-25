"""Kandinsky pipeline package.

Exports :data:`KANDINSKY3_UNET_SPEC` and :data:`KANDINSKY5_SPEC`.

Kandinsky ships several generations under one family:

- Kandinsky 2.1 / 2.2 are unet-based and load through diffusers'
  combined pipelines without going through the native loader, so no
  spec is needed.
- Kandinsky 3 uses :class:`Kandinsky3UNet` in the ``unet`` subfolder of
  the repo, hence ``subfolder='unet'`` instead of the default
  ``'transformer'``.
- Kandinsky 5 uses the new :class:`Kandinsky5Transformer3DModel` and
  follows the default layout.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


KANDINSKY3_UNET_SPEC = TransformerSpec(cls=diffusers.Kandinsky3UNet, subfolder='unet')
KANDINSKY5_SPEC = TransformerSpec(cls=diffusers.Kandinsky5Transformer3DModel)
