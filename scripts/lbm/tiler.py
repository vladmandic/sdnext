import logging
import math
from copy import deepcopy
from typing import List, Tuple
import torch
import torch.nn.functional as F


TILING_METHODS = ["average", "gaussian", "linear"]


class Tiler:
    def get_tiles(
        self,
        input: torch.Tensor,
        tile_size: tuple,
        overlap_size: tuple,
        scale: int = 1,
        out_channels: int = 3,
    ) -> List[List[torch.tensor]]:
        """Get tiles
        Args:
            input (torch.Tensor): input array of shape (batch_size, channels, height, width)
            tile_size (tuple): tile size
            overlap_size (tuple): overlap size
            scale (int): scaling factor of the output wrt input
            out_channels (int): number of output channels
        Returns:
            List[List[torch.Tensor]]: List of tiles
        """
        # assert isinstance(scale, int)
        assert (
            overlap_size[0] <= tile_size[0]
        ), f"Overlap size {overlap_size} must be smaller than tile size {tile_size}"
        assert (
            overlap_size[1] <= tile_size[1]
        ), f"Overlap size {overlap_size} must be smaller than tile size {tile_size}"

        B, C, H, W = input.shape
        tile_size_H, tile_size_W = tile_size

        # sets overlap to 0 if the input is smaller than the tile size (i.e. no overlap)
        overlap_H, overlap_W = (
            overlap_size[0] if H > tile_size_H else 0,
            overlap_size[1] if W > tile_size_W else 0,
        )

        self.output_overlap_size = (
            int(overlap_H * scale),
            int(overlap_W * scale),
        )
        self.tile_size = tile_size
        self.output_tile_size = (
            int(tile_size_H * scale),
            int(tile_size_W * scale),
        )
        self.output_shape = (
            B,
            out_channels,
            int(H * scale),
            int(W * scale),
        )
        tiles = []
        logging.debug(f"(Tiler) Input shape: {(B, C, H, W)}")
        logging.debug(f"(Tiler) Output shape: {self.output_shape}")
        logging.debug(f"(Tiler) Tile size: {(tile_size_H, tile_size_W)}")
        logging.debug(f"(Tiler) Overlap size: {(overlap_H, overlap_W)}")
        # loop over all tiles in the image with overlap
        for i in range(0, H, tile_size_H - overlap_H):
            row = []
            for j in range(0, W, tile_size_W - overlap_W):
                tile = deepcopy(
                    input[
                        :,
                        :,
                        i : i + tile_size_H,
                        j : j + tile_size_W,
                    ]
                )
                row.append(tile)
            tiles.append(row)
        return tiles

    def merge_tiles(
        self, tiles: List[List[torch.tensor]], tiling_method: str = "gaussian"
    ) -> torch.tensor:
        """Merge tiles by averaging the overlaping regions
        Args:
            tiles (Dict[str, Tile]): dictionary of processed tiles
            tiling_method (str): tiling method. Can be "average", "gaussian" or "linear"
        Returns:
            torch.tensor: output image
        """
        if tiling_method == "average":
            return self._average_merge_tiles(tiles)
        elif tiling_method == "gaussian":
            return self._gaussian_merge_tiles(tiles)
        elif tiling_method == "linear":
            return self._linear_merge_tiles(tiles)
        else:
            raise ValueError(
                f"Unknown tiling method {tiling_method}. Available methods are {TILING_METHODS}"
            )

    def _average_merge_tiles(self, tiles: List[List[torch.tensor]]) -> torch.tensor:
        """Merge tiles by averaging the overlaping regions
        Args:
            tiles (Dict[str, Tile]): dictionary of processed tiles
        Returns:
            torch.tensor: output image
        """

        output = torch.zeros(self.output_shape)

        # weights to store multiplicity
        weights = torch.zeros(self.output_shape)

        _, _, output_H, output_W = self.output_shape
        output_overlap_size_H, output_overlap_size_W = self.output_overlap_size
        output_tile_size_H, output_tile_size_W = self.output_tile_size

        for id_i, i in enumerate(
            range(
                0,
                output_H,
                output_tile_size_H - output_overlap_size_H,
            )
        ):
            for id_j, j in enumerate(
                range(
                    0,
                    output_W,
                    output_tile_size_W - output_overlap_size_W,
                )
            ):
                output[
                    :,
                    :,
                    i : i + output_tile_size_H,
                    j : j + output_tile_size_W,
                ] += (
                    tiles[id_i][id_j] * 1
                )
                weights[
                    :,
                    :,
                    i : i + output_tile_size_H,
                    j : j + output_tile_size_W,
                ] += 1

        # outputs is summed up with this multiplicity
        # so we need to divide by the weights wich is either 1, 2 or 4 depending on the region
        output = output / weights
        return output

    def _gaussian_weights(
        self, tile_width: int, tile_height: int, nbatches: int, channels: int
    ):
        """Generates a gaussian mask of weights for tile contributions.

        Args:
            tile_width (int): width of the tile
            tile_height (int): height of the tile
            nbatches (int): number of batches
            channels (int): number of channels
        Returns:
            torch.tensor: weights
        """
        import numpy as np
        from numpy import exp, pi, sqrt

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (
            latent_width - 1
        ) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(
                -(x - midpoint)
                * (x - midpoint)
                / (latent_width * latent_width)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        midpoint = latent_height / 2
        y_probs = [
            exp(
                -(y - midpoint)
                * (y - midpoint)
                / (latent_height * latent_height)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(
            torch.tensor(weights, device="cpu"), (nbatches, channels, 1, 1)
        )

    def _gaussian_merge_tiles(self, tiles: List[List[torch.tensor]]) -> torch.tensor:
        """Merge tiles by averaging the overlaping regions
        Args:
            List[List[torch.tensor]]: List of processed tiles
        Returns:
            torch.tensor: output image
        """
        B, output_C, output_H, output_W = self.output_shape
        output_overlap_size_H, output_overlap_size_W = self.output_overlap_size
        output_tile_size_H, output_tile_size_W = self.output_tile_size

        output = torch.zeros(self.output_shape)
        # weights to store multiplicity
        weights = torch.zeros(self.output_shape)

        for id_i, i in enumerate(
            range(
                0,
                output_H,
                output_tile_size_H - output_overlap_size_H,
            )
        ):
            for id_j, j in enumerate(
                range(
                    0,
                    output_W,
                    output_tile_size_W - output_overlap_size_W,
                )
            ):
                w = self._gaussian_weights(
                    tiles[id_i][id_j].shape[3],
                    tiles[id_i][id_j].shape[2],
                    B,
                    output_C,
                )
                output[
                    :,
                    :,
                    i : i + output_tile_size_H,
                    j : j + output_tile_size_W,
                ] += (
                    tiles[id_i][id_j] * w
                )
                weights[
                    :,
                    :,
                    i : i + output_tile_size_H,
                    j : j + output_tile_size_W,
                ] += w

        # outputs is summed up with this multiplicity
        output = output / weights
        return output

    def _blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, y, :
            ] * (y / blend_extent)
        return b

    def _blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    def _linear_merge_tiles(self, tiles: List[List[torch.tensor]]) -> torch.Tensor:
        """Merge tiles by blending the overlaping regions
        Args:
            tiles (List[List[torch.tensor]]): List of processed tiles
        Returns:
            torch.Tensor: output image
        """
        output_overlap_size_H, output_overlap_size_W = self.output_overlap_size
        output_tile_size_H, output_tile_size_W = self.output_tile_size

        res_rows = []
        tiles_copy = deepcopy(tiles)

        # Cut the right and bottom overlap region
        limit_i = output_tile_size_H - output_overlap_size_H
        limit_j = output_tile_size_W - output_overlap_size_W
        for i, tile_row in enumerate(tiles_copy):
            res_row = []
            for j, tile in enumerate(tile_row):
                tile_val = tile
                if j > 0:
                    tile_val = self._blend_h(
                        tile_row[j - 1], tile, output_overlap_size_W
                    )
                tiles_copy[i][j] = tile_val
                if i > 0:
                    tile_val = self._blend_v(
                        tiles_copy[i - 1][j], tile_val, output_overlap_size_H
                    )
                tiles_copy[i][j] = tile_val
                res_row.append(tile_val[:, :, :limit_i, :limit_j])
            res_rows.append(torch.cat(res_row, dim=3))
        output = torch.cat(res_rows, dim=2)
        return output


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Extracts values from a tensor into a new tensor using indices from another tensor.

    :param a: the tensor to extract values from.
    :param t: the tensor containing the indices.
    :param x_shape: the shape of the tensor to extract values into.
    :return: a new tensor containing the extracted values.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def pad(x: torch.Tensor, base_h: int, base_w: int) -> torch.Tensor:
    """
    Pads a tensor to the nearest multiple of base_h and base_w.

    :param x: the tensor to pad.
    :param base_h: the base height.
    :param base_w: the base width.
    :return: the padded tensor.
    """
    h, w = x.shape[-2:]
    h_ = math.ceil(h / base_h) * base_h
    w_ = math.ceil(w / base_w) * base_w
    if w_ != w:
        x = F.pad(x, (0, abs(w_ - w), 0, 0))
    if h_ != h:
        x = F.pad(x, (0, 0, 0, abs(h_ - h)))
    return x


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


@torch.no_grad()
def update_ema(
    target_params: List[torch.Tensor],
    source_params: List[torch.Tensor],
    rate: float = 0.99,
):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
