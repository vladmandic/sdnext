from dataclasses import field
from typing import List, Union
from pydantic.dataclasses import dataclass
from ..base import BaseConditionerConfig


@dataclass
class LatentsConcatEmbedderConfig(BaseConditionerConfig):
    """
    Configs for the LatentsConcatEmbedder embedder

    Args:
        image_keys (Union[List[str], None]): Keys of the images to compute the VAE embeddings
        mask_keys (Union[List[str], None]): Keys of the masks to resize
    """

    image_keys: Union[List[str], None] = field(default_factory=lambda: ["image"])
    mask_keys: Union[List[str], None] = field(default_factory=lambda: ["mask"])

    def __post_init__(self):
        super().__post_init__()

        # Make sure that at least one of the image_keys or mask_keys is provided
        assert (self.image_keys is not None) or (
            self.mask_keys is not None
        ), "At least one of the image_keys or mask_keys must be provided."

        self.image_keys = self.image_keys if self.image_keys is not None else []
        self.mask_keys = self.mask_keys if self.mask_keys is not None else []
