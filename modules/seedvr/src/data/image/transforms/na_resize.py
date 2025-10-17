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

from typing import Literal
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from .area_resize import AreaResize
from .side_resize import SideResize


def NaResize(
    resolution: int,
    mode: Literal["area", "side"],
    downsample_only: bool,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
):
    if mode == "area":
        return AreaResize(
            max_area=resolution**2,
            downsample_only=downsample_only,
            interpolation=interpolation,
        )
    if mode == "side":
        return SideResize(
            size=resolution,
            downsample_only=downsample_only,
            interpolation=interpolation,
        )
    if mode == "square":
        return Compose(
            [
                Resize(
                    size=resolution,
                    interpolation=interpolation,
                ),
                CenterCrop(resolution),
            ]
        )
    raise ValueError(f"Unknown resize mode: {mode}")
