import torch
from diffusers.models import AutoencoderKL
from ..base.base_model import BaseModel
from ..tiler import Tiler, pad
from .autoencoderKL_config import AutoencoderKLDiffusersConfig


class AutoencoderKLDiffusers(BaseModel):
    """This is the VAE class used to work with latent models

    Args:

        config (AutoencoderKLDiffusersConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: AutoencoderKLDiffusersConfig):
        BaseModel.__init__(self, config)
        self.config = config
        self.vae_model = AutoencoderKL.from_pretrained(
            config.version,
            subfolder=config.subfolder,
            revision=config.revision,
        )
        self.tiling_size = config.tiling_size
        self.tiling_overlap = config.tiling_overlap

        # get downsampling factor
        self._get_properties()

    @torch.no_grad()
    def _get_properties(self):
        self.has_shift_factor = (
            hasattr(self.vae_model.config, "shift_factor")
            and self.vae_model.config.shift_factor is not None
        )
        self.shift_factor = (
            self.vae_model.config.shift_factor if self.has_shift_factor else 0
        )

        # set latent channels
        self.latent_channels = self.vae_model.config.latent_channels
        self.has_latents_mean = (
            hasattr(self.vae_model.config, "latents_mean")
            and self.vae_model.config.latents_mean is not None
        )
        self.has_latents_std = (
            hasattr(self.vae_model.config, "latents_std")
            and self.vae_model.config.latents_std is not None
        )
        self.latents_mean = self.vae_model.config.latents_mean
        self.latents_std = self.vae_model.config.latents_std

        x = torch.randn(1, self.vae_model.config.in_channels, 32, 32)
        z = self.encode(x)

        # set downsampling factor
        self.downsampling_factor = int(x.shape[2] / z.shape[2])

    def encode(self, x: torch.tensor, batch_size: int = 8):
        latents = []
        for i in range(0, x.shape[0], batch_size):
            latents.append(
                self.vae_model.encode(x[i : i + batch_size]).latent_dist.sample()
            )
        latents = torch.cat(latents, dim=0)
        latents = (latents - self.shift_factor) * self.vae_model.config.scaling_factor

        return latents

    def decode(self, z: torch.tensor):

        if self.has_latents_mean and self.has_latents_std:
            latents_mean = (
                torch.tensor(self.latents_mean)
                .view(1, self.latent_channels, 1, 1)
                .to(z.device, z.dtype)
            )
            latents_std = (
                torch.tensor(self.latents_std)
                .view(1, self.latent_channels, 1, 1)
                .to(z.device, z.dtype)
            )
            z = z * latents_std / self.vae_model.config.scaling_factor + latents_mean
        else:
            z = z / self.vae_model.config.scaling_factor + self.shift_factor

        use_tiling = (
            z.shape[2] > self.tiling_size[0] or z.shape[3] > self.tiling_size[1]
        )

        if use_tiling:
            samples = []
            for i in range(z.shape[0]):

                z_i = z[i].unsqueeze(0)

                tiler = Tiler()
                tiles = tiler.get_tiles(
                    input=z_i,
                    tile_size=self.tiling_size,
                    overlap_size=self.tiling_overlap,
                    scale=self.downsampling_factor,
                    out_channels=3,
                )

                for i, tile_row in enumerate(tiles):
                    for j, tile in enumerate(tile_row):
                        tile_shape = tile.shape
                        # pad tile to inference size if tile is smaller than inference size
                        tile = pad(
                            tile,
                            base_h=self.tiling_size[0],
                            base_w=self.tiling_size[1],
                        )
                        tile_decoded = self.vae_model.decode(tile).sample
                        tiles[i][j] = (
                            tile_decoded[
                                0,
                                :,
                                : int(tile_shape[2] * self.downsampling_factor),
                                : int(tile_shape[3] * self.downsampling_factor),
                            ]
                            .cpu()
                            .unsqueeze(0)
                        )

                # merge tiles
                samples.append(tiler.merge_tiles(tiles=tiles))

            samples = torch.cat(samples, dim=0)

        else:
            samples = self.vae_model.decode(z).sample

        return samples
