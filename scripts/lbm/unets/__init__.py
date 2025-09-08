"""
This module contains a collection of U-Net models.
The :mod:`cr.models.unets` module includes the following classes:

- :class:`DiffusersUNet2DWrapper`: A 2D U-Net model for diffusers.
- :class:`DiffusersUNet2DCondWrapper`: A 2D U-Net model for diffusers with conditional input.
"""

from .unet import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper


__all__ = [
    "DiffusersUNet2DWrapper",
    "DiffusersUNet2DCondWrapper",
]
