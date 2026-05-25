"""VIBE pipeline components for SD.Next."""

from pipelines.native_transformer import TransformerSpec
from .vibe_sana_editing import VIBESanaEditingModel
from .vibe_sana_pipeline import VIBESanaEditingPipeline, VIBESanaImagePipeline


VIBE_SPEC = TransformerSpec(cls=VIBESanaEditingModel)


__all__ = ["VIBE_SPEC", "VIBESanaEditingModel", "VIBESanaEditingPipeline", "VIBESanaImagePipeline"]
