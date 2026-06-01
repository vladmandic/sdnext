"""Qwen-Image pipeline package.

Exports :data:`QWEN_SPEC`. diffusers registers a no-op identity lambda
for ``QwenImageTransformer2DModel`` in ``SINGLE_FILE_LOADABLE_CLASSES``,
so ``from_single_file`` silently accepts whatever key naming the file
uses and loads with mismatched weights. The spec sets ``converter=None``
explicitly to skip that no-op; validation then surfaces mismatches as
clear errors. A real converter can be plugged in here if a trainer
format that needs one is encountered.
"""

import diffusers

from pipelines.native_transformer import TransformerSpec
from pipelines.qwen.qwen_nunchaku import load_qwen_nunchaku
from pipelines.qwen.qwen_pruning import check_qwen_pruning


QWEN_SPEC = TransformerSpec(cls=diffusers.QwenImageTransformer2DModel, converter=None)
