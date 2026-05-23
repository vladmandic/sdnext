"""Lens - minimal text-to-image inference package."""

import diffusers as _diffusers
import transformers as _transformers
from .pipeline import LensPipeline, LensPipelineOutput
from .reasoner import PromptReasoner
from .resolution import RESOLUTION_BUCKETS, resolve_resolution
from .text_encoder import LensGptOssEncoder
from .transformer import LensTransformer2DModel

# ---------------------------------------------------------------------------
# Make our custom subclasses discoverable by ``diffusers.DiffusionPipeline``.
#
# When ``LensPipeline.from_pretrained`` reads ``model_index.json``, it sees
# entries like ``["transformers", "LensGptOssEncoder"]`` and runs
# ``getattr(importlib.import_module("transformers"), "LensGptOssEncoder")``.
# diffusers only allow-lists the libraries ``diffusers``, ``transformers`` and
# ``onnxruntime.training`` - any other name is interpreted as a custom .py file
# in the repo. So we inject our subclasses into those two namespaces here.
#
# Importing ``lens`` is required before calling ``LensPipeline.from_pretrained``
# (this happens automatically when the user does ``from lens import LensPipeline``).
# ---------------------------------------------------------------------------

if not hasattr(_transformers, "LensGptOssEncoder"):
    _transformers.LensGptOssEncoder = LensGptOssEncoder
if not hasattr(_diffusers, "LensTransformer2DModel"):
    _diffusers.LensTransformer2DModel = LensTransformer2DModel
if not hasattr(_diffusers, "LensPipeline"):
    _diffusers.LensPipeline = LensPipeline

del _diffusers, _transformers

__all__ = [
    "LensPipeline",
    "LensPipelineOutput",
    "LensTransformer2DModel",
    "LensGptOssEncoder",
    "PromptReasoner",
    "RESOLUTION_BUCKETS",
    "resolve_resolution",
]
