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

from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from ...common.cache import Cache
from ...common.distributed.ops import gather_outputs, slice_inputs

from . import na


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        return vid


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(dim, out_channels * t * h * w)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        vid = rearrange(vid, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)
        return vid


class NaPatchIn(PatchIn):
    def forward(
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.LongTensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid, vid_shape = na.rearrange(
                vid, vid_shape, "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w
            )
        # slice vid after patching in when using sequence parallelism
        vid = slice_inputs(vid, dim=0)
        vid = self.proj(vid)
        return vid, vid_shape


class NaPatchOut(PatchOut):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
    ]:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        # gather vid before patching out when enabling sequence parallelism
        vid = gather_outputs(
            vid,
            gather_dim=0,
            padding_dim=0,
            unpad_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        if not (t == h == w == 1):
            vid, vid_shape = na.rearrange(
                vid, vid_shape, "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w
            )
        return vid, vid_shape
