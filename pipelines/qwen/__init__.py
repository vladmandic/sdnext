"""Qwen-Image pipeline package.

Exports :data:`QWEN_SPEC` for use by :mod:`pipelines.model_qwen` together
with :mod:`pipelines.native_transformer`. Qwen-Image is the lone Mode C
arch: diffusers registers a no-op identity lambda as its
``SINGLE_FILE_LOADABLE_CLASSES`` converter, so ``from_single_file`` silently
accepts whatever key naming the community file uses and loads with mangled
weights instead of raising.

The spec explicitly sets ``converter=None`` to short-circuit the no-op
pickup; if a real Qwen-Image converter is needed for some trainer dump
in the wild, it can be plugged in here. Until then, validation surfaces a
clear error listing unexpected/missing keys instead of letting a malformed
file load silently.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec
from pipelines.qwen.qwen_nunchaku import load_qwen_nunchaku
from pipelines.qwen.qwen_pruning import check_qwen_pruning


QWEN_SPEC = TransformerSpec(cls=diffusers.QwenImageTransformer2DModel, converter=None)
