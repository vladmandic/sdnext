from pydantic.dataclasses import dataclass
from ..config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    input_key: str = "image"
