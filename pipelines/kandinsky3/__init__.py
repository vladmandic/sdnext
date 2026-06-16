"""Kandinsky 3.0 pipeline package.

Exports :data:`KANDINSKY3_UNET_SPEC`. Kandinsky 3 uses
:class:`Kandinsky3UNet` in the ``unet`` subfolder of the repo, hence
``subfolder='unet'`` instead of the default ``'transformer'``.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec


KANDINSKY3_UNET_SPEC = TransformerSpec(cls=diffusers.Kandinsky3UNet, subfolder='unet')
