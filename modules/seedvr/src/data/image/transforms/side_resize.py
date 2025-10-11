# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Union
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


class SideResize:
    def __init__(
        self,
        size: int,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = size
        self.downsample_only = downsample_only
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        """
        Args:
            image (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        if self.downsample_only and min(width, height) < self.size:
            # keep original height and width for small pictures.
            size = min(width, height)
        else:
            size = self.size

        return TVF.resize(image, size, self.interpolation)
