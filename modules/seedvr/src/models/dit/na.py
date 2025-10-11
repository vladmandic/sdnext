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

from itertools import chain
from typing import Callable, Dict, List, Tuple
import einops
import torch


def flatten(
    hid: List[torch.FloatTensor],  # List of (*** c)
) -> Tuple[
    torch.FloatTensor,  # (L c)
    torch.LongTensor,  # (b n)
]:
    assert len(hid) > 0
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid[0].device) for x in hid])
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape


def unflatten(
    hid: torch.FloatTensor,  # (L c) or (L ... c)
    hid_shape: torch.LongTensor,  # (b n)
) -> List[torch.Tensor]:  # List of (*** c) or (*** ... c)
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    hid = [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]
    return hid


def concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> torch.FloatTensor:  # (L ... c)
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    return torch.cat(list(chain(*zip(vid, txt))))


def concat_idx(
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    Callable,
    Callable,
]:
    device = vid_len.device
    vid_idx = torch.arange(vid_len.sum(), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=device)
    tgt_idx = concat(vid_idx, txt_idx, vid_len, txt_len)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda vid, txt: torch.index_select(torch.cat([vid, txt]), 0, tgt_idx),
        lambda all: torch.index_select(all, 0, src_idx).split([len(vid_idx), len(txt_idx)]),
    )


def unconcat(
    all: torch.FloatTensor,  # (L ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    torch.FloatTensor,  # (VL ... c)
    torch.FloatTensor,  # (TL ... c)
]:
    interleave_len = list(chain(*zip(vid_len.tolist(), txt_len.tolist())))
    all = all.split(interleave_len)
    vid = torch.cat(all[0::2])
    txt = torch.cat(all[1::2])
    return vid, txt


def repeat_concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: List,  # (n)
) -> torch.FloatTensor:  # (L ... c)
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    txt = [[x] * n for x, n in zip(txt, txt_repeat)]
    txt = list(chain(*txt))
    return torch.cat(list(chain(*zip(vid, txt))))


def repeat_concat_idx(
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: torch.LongTensor,  # (n)
) -> Tuple[
    Callable,
    Callable,
]:
    device = vid_len.device
    vid_idx = torch.arange(vid_len.sum(), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=device)
    txt_repeat_list = txt_repeat.tolist()
    tgt_idx = repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    repeat_txt_len = (txt_len * txt_repeat).tolist()

    def unconcat_coalesce(all):
        """
        Un-concat vid & txt, and coalesce the repeated txt.
        e.g. vid [0 1 2 3 4 5 6 7 8] -> 3 splits -> [0 1 2] [3 4 5] [6 7 8]
             txt [9 10]
             repeat_concat ==> [0 1 2 9 10 3 4 5 9 10 6 7 8 9 10]
             1. argsort re-index ==> [0 1 2 3 4 5 6 7 8 9 9 9 10 10 10]
                           split ==> vid_out [0 1 2 3 4 5 6 7 8] txt_out [9 9 9 10 10 10]
             2. reshape & mean for each sample to coalesce the repeated txt.
        """
        vid_out, txt_out = all[src_idx].split([len(vid_idx), txt_idx_len])
        txt_out_coalesced = []
        for txt, repeat_time in zip(txt_out.split(repeat_txt_len), txt_repeat_list):
            txt = txt.reshape(-1, repeat_time, *txt.shape[1:]).mean(1)
            txt_out_coalesced.append(txt)
        return vid_out, torch.cat(txt_out_coalesced)

    # Note: Backward of torch.index_select is non-deterministic when existing repeated index,
    # the difference may cumulative like torch.repeat_interleave, so we use vanilla index here.
    return (
        lambda vid, txt: torch.cat([vid, txt])[tgt_idx],
        lambda all: unconcat_coalesce(all),
    )


def rearrange(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    return flatten([einops.rearrange(h, pattern, **kwargs) for h in unflatten(hid, hid_shape)])


def rearrange_idx(
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[Callable, Callable, torch.LongTensor]:
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape = rearrange(hid_idx, hid_shape, pattern, **kwargs)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: torch.index_select(hid, 0, src_idx),
        tgt_shape,
    )


def repeat(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, torch.LongTensor],  # (b)
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    hid = unflatten(hid, hid_shape)
    kwargs = [{k: v[i].item() for k, v in kwargs.items()} for i in range(len(hid))]
    return flatten([einops.repeat(h, pattern, **a) for h, a in zip(hid, kwargs)])


def pack(
    samples: List[torch.Tensor],  # List of (h w c).
) -> Tuple[
    List[torch.Tensor],  # groups [(b1 h1 w1 c1), (b2 h2 w2 c2)]
    List[List[int]],  # reversal indices.
]:
    batches = {}
    indices = {}
    for i, sample in enumerate(samples):
        shape = sample.shape
        batches[shape] = batches.get(shape, [])
        indices[shape] = indices.get(shape, [])
        batches[shape].append(sample)
        indices[shape].append(i)

    batches = list(map(torch.stack, batches.values()))
    indices = list(indices.values())
    return batches, indices


def unpack(
    batches: List[torch.Tensor],
    indices: List[List[int]],
) -> List[torch.Tensor]:
    samples = [None] * (max(chain(*indices)) + 1)
    for batch, index in zip(batches, indices):
        for sample, i in zip(batch.unbind(), index):
            samples[i] = sample
    return samples


def window(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    hid = unflatten(hid, hid_shape)
    hid = list(map(window_fn, hid))
    hid_windows = torch.tensor(list(map(len, hid)), device=hid_shape.device)
    hid, hid_shape = flatten(list(chain(*hid)))
    return hid, hid_shape, hid_windows


def window_idx(
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows = window(hid_idx, hid_shape, window_fn)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: torch.index_select(hid, 0, src_idx),
        tgt_shape,
        tgt_windows,
    )
