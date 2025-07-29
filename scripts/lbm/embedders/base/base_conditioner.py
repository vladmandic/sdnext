from typing import Any, Dict
from ...base.base_model import BaseModel
from .base_conditioner_config import BaseConditionerConfig


DIM2CONDITIONING = {
    2: "vector",
    3: "crossattn",
    4: "concat",
}


class BaseConditioner(BaseModel):
    """This is the base class for all the conditioners. This absctacts the conditioning process

    Args:

        config (BaseConditionerConfig): The configuration of the conditioner

    Examples
    ########

    To use the conditioner, you can import the class and use it as follows:

    .. code-block:: python

        from cr.models.embedders import BaseConditioner, BaseConditionerConfig

        # Create the conditioner config
        config = BaseConditionerConfig(
            input_key="text", # The key for the input
            unconditional_conditioning_rate=0.3, # Drops the conditioning with 30% probability during training
        )

        # Create the conditioner
        conditioner = BaseConditioner(config)
    """

    def __init__(self, config: BaseConditionerConfig):
        BaseModel.__init__(self, config)
        self.config = config
        self.input_key = config.input_key
        self.dim2outputkey = DIM2CONDITIONING
        self.ucg_rate = config.unconditional_conditioning_rate

    def forward(
        self, batch: Dict[str, Any], force_zero_embedding: bool = False, *args, **kwargs
    ):
        """
         Forward pass of the embedder.

        Args:

            batch (Dict[str, Any]): A dictionary containing the input data.
            force_zero_embedding (bool): Whether to force zero embedding.
                This will return an embedding with all entries set to 0. Defaults to False.
        """
        raise NotImplementedError("Forward pass must be implemented in child class")
