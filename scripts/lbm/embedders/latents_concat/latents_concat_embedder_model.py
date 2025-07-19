from typing import Any, Dict
import torch
import torchvision.transforms.functional as F
from ...vae import AutoencoderKLDiffusers
from ..base import BaseConditioner
from .latents_concat_embedder_config import LatentsConcatEmbedderConfig


class LatentsConcatEmbedder(BaseConditioner):
    """
    Class computing VAE embeddings from given images and resizing the masks.
    Then outputs are then concatenated to the noise in the latent space.

    Args:
        config (LatentsConcatEmbedderConfig): Configs to create the embedder
    """

    def __init__(self, config: LatentsConcatEmbedderConfig):
        BaseConditioner.__init__(self, config)

    def forward(
        self, batch: Dict[str, Any], vae: AutoencoderKLDiffusers, *args, **kwargs
    ) -> dict:
        """
        Args:
            batch (dict): A batch of images to be processed by this embedder. In the batch,
            the images must range between [-1, 1] and the masks range between [0, 1].
            vae (AutoencoderKLDiffusers): VAE

        Returns:
            output (dict): outputs
        """

        # Check if image are of the same size
        dims_list = []
        for image_key in self.config.image_keys:
            dims_list.append(batch[image_key].shape[-2:])
        for mask_key in self.config.mask_keys:
            dims_list.append(batch[mask_key].shape[-2:])
        assert all(
            dims == dims_list[0] for dims in dims_list
        ), "All images and masks must have the same dimensions."

        # Find the latent dimensions
        if len(self.config.image_keys) > 0:
            latent_dims = (
                batch[self.config.image_keys[0]].shape[-2] // vae.downsampling_factor,
                batch[self.config.image_keys[0]].shape[-1] // vae.downsampling_factor,
            )
        else:
            latent_dims = (
                batch[self.config.mask_keys[0]].shape[-2] // vae.downsampling_factor,
                batch[self.config.mask_keys[0]].shape[-1] // vae.downsampling_factor,
            )

        outputs = []

        # Resize the masks and concat them
        for mask_key in self.config.mask_keys:
            curr_latents = F.resize(
                batch[mask_key],
                size=latent_dims,
                interpolation=F.InterpolationMode.BILINEAR,
            )
            outputs.append(curr_latents)

        # Compute VAE embeddings from the images
        for image_key in self.config.image_keys:
            vae_embs = vae.encode(batch[image_key])
            outputs.append(vae_embs)

        # Concat all the outputs
        outputs = torch.concat(outputs, dim=1)

        outputs = {self.dim2outputkey[outputs.dim()]: outputs}

        return outputs
