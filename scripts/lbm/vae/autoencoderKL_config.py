from typing import Tuple
from pydantic.dataclasses import dataclass
from ..base import ModelConfig


@dataclass
class AutoencoderKLDiffusersConfig(ModelConfig):
    """This is the VAEConfig class which defines all the useful parameters to instantiate the model.

    Args:

        version (str): The version of the model. Defaults to "stabilityai/sdxl-vae".
        subfolder (str): The subfolder of the model if loaded from another model. Defaults to "".
        revision (str): The revision of the model. Defaults to "main".
        input_key (str): The key of the input data in the batch. Defaults to "image".
        tiling_size (Tuple[int, int]): The size of the tiling. Defaults to (64, 64).
        tiling_overlap (Tuple[int, int]): The overlap of the tiling. Defaults to (16, 16).
    """

    version: str = "stabilityai/sdxl-vae"
    subfolder: str = ""
    revision: str = "main"
    input_key: str = "image"
    tiling_size: Tuple[int, int] = (64, 64)
    tiling_overlap: Tuple[int, int] = (16, 16)
