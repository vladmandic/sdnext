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

import math
import random
from typing import Union
import torch
from PIL import Image
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode


class AreaResize:
    def __init__(
        self,
        max_area: float,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.max_area = max_area
        self.downsample_only = downsample_only
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image]):

        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        scale = math.sqrt(self.max_area / (height * width))

        # keep original height and width for small pictures.
        scale = 1 if scale >= 1 and self.downsample_only else scale

        resized_height, resized_width = round(height * scale), round(width * scale)

        return TVF.resize(
            image,
            size=(resized_height, resized_width),
            interpolation=self.interpolation,
        )


class AreaRandomCrop:
    def __init__(
        self,
        max_area: float,
    ):
        self.max_area = max_area

    def get_params(self, input_size, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        # w, h = _get_image_size(img)
        h, w = input_size
        th, tw = output_size
        if w <= tw and h <= th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        resized_height = math.sqrt(self.max_area / (width / height))
        resized_width = (width / height) * resized_height

        resized_height, resized_width = round(resized_height), round(resized_width)
        i, j, h, w = self.get_params((height, width), (resized_height, resized_width))
        image = TVF.crop(image, i, j, h, w)
        return image

class ScaleResize:
    def __init__(
        self,
        scale: float,
    ):
        self.scale = scale

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
            interpolation_mode = InterpolationMode.BILINEAR
            antialias = True if image.ndim == 4 else "warn"
        elif isinstance(image, Image.Image):
            width, height = image.size
            interpolation_mode = InterpolationMode.LANCZOS
            antialias = "warn"
        else:
            raise NotImplementedError

        scale = self.scale

        # keep original height and width for small pictures

        resized_height, resized_width = round(height * scale), round(width * scale)
        image = TVF.resize(
            image,
            size=(resized_height, resized_width),
            interpolation=interpolation_mode,
            antialias=antialias,
        )
        return image
