from pydantic.dataclasses import dataclass
from ...config import BaseConfig


@dataclass
class BaseConditionerConfig(BaseConfig):
    """This is the ClipEmbedderConfig class which defines all the useful parameters to instantiate the model

    Args:

        input_key (str): The key for the input. Defaults to "text".
        unconditional_conditioning_rate (float): Drops the conditioning with this probability during training. Defaults to 0.0.
    """

    input_key: str = "text"
    unconditional_conditioning_rate: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        assert (
            self.unconditional_conditioning_rate >= 0.0
            and self.unconditional_conditioning_rate <= 1.0
        ), "Unconditional conditioning rate should be between 0 and 1"
