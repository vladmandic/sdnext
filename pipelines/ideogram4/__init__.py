"""Ideogram 4 architecture support for SD.Next (diffusers-native port).

Importing this package registers the ported classes under the ``diffusers``
namespace using the exact names Ideogram declared in the shipped
``model_index.json`` (``Ideogram4Pipeline`` / ``Ideogram4Transformer2DModel``),
so folder ``_class_name`` resolution and ``shared_items`` lookups find them.
"""

from __future__ import annotations

import diffusers

from pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from pipelines.ideogram4.scheduler_ideogram4 import Ideogram4Scheduler
from pipelines.ideogram4.transformer_ideogram4 import Ideogram4Transformer2DModel

for cls in (Ideogram4Transformer2DModel, Ideogram4Pipeline, Ideogram4Scheduler):
    if not hasattr(diffusers, cls.__name__):
        setattr(diffusers, cls.__name__, cls)

__all__ = ["Ideogram4Pipeline", "Ideogram4Scheduler", "Ideogram4Transformer2DModel"]
