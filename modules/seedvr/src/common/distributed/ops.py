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

"""
Distributed ops for supporting sequence parallel.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import Tensor

from ..cache import Cache
from .advanced import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)

from .basic import get_device

_SEQ_DATA_BUF = defaultdict(lambda: [None, None, None])
_SEQ_DATA_META_SHAPES = defaultdict()
_SEQ_DATA_META_DTYPES = defaultdict()
_SEQ_DATA_ASYNC_COMMS = defaultdict(list)
_SYNC_BUFFER = defaultdict(dict)


def single_all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    """
    A function to do all-to-all on a tensor
    """
    seq_world_size = dist.get_world_size(group)
    prev_scatter_dim = scatter_dim
    if scatter_dim != 0:
        local_input = local_input.transpose(0, scatter_dim)
        if gather_dim == 0:
            gather_dim = scatter_dim
        scatter_dim = 0

    inp_shape = list(local_input.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    input_t = local_input.reshape(
        [seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]
    ).contiguous()
    output = torch.empty_like(input_t)
    comm = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
    if async_op:
        # let user's code transpose & reshape
        return output, comm, prev_scatter_dim

    # first dim is seq_world_size, so we can split it directly
    output = torch.cat(output.split(1), dim=gather_dim + 1).squeeze(0)
    if prev_scatter_dim:
        output = output.transpose(0, prev_scatter_dim).contiguous()
    return output


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
):
    seq_world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        if async_op:
            output, comm, prev_scatter_dim = single_all_to_all(
                local_input, scatter_dim, gather_dim, group, async_op=async_op
            )
            ctx.prev_scatter_dim = prev_scatter_dim
            return output, comm

        return _all_to_all(local_input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[0].split(1), dim=ctx.gather_dim + 1).squeeze(0)
            if ctx.prev_scatter_dim:
                input_t = input_t.transpose(0, ctx.prev_scatter_dim)
        else:
            input_t = grad_output[0]
        return (
            None,
            _all_to_all(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None,
            None,
            None,
        )


class Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        ctx.dim = dim
        dim_size = local_input.shape[dim]
        return local_input.split(dim_size // seq_world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        dim_size = list(grad_output.size())
        split_size = dim_size[0]
        dim_size[0] = dim_size[0] * ctx.seq_world_size
        output = torch.empty(dim_size, dtype=grad_output.dtype, device=torch.cuda.current_device())
        dist._all_gather_base(output, grad_output, group=ctx.group)
        return (None, torch.cat(output.split(split_size), dim=ctx.dim), None)


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        dim: int,
        grad_scale: Optional[bool] = False,
    ) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        dim_size = list(local_input.size())
        split_size = dim_size[0]
        ctx.part_size = dim_size[dim]
        dim_size[0] = dim_size[0] * seq_world_size
        output = torch.empty(dim_size, dtype=local_input.dtype, device=torch.cuda.current_device())
        dist._all_gather_base(output, local_input.contiguous(), group=ctx.group)
        return torch.cat(output.split(split_size), dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        if ctx.grad_scale:
            grad_output = grad_output * ctx.seq_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.dim)[ctx.rank].contiguous(),
            None,
            None,
        )


def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor,
    *,
    seq_dim: int,
    qkv_shape: Optional[Tensor] = None,
    cache: Cache = Cache(disable=True),
    restore_shape: bool = True,
):
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    group = get_sequence_parallel_group()
    if not group:
        return qkv_tensor
    world = get_sequence_parallel_world_size()
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    qkv_tensor = SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, False)
    if restore_shape:
        out_shape = list(orig_shape)
        out_shape[seq_dim] *= world
        out_shape[-1] = qkv_proj_dim // world
        qkv_tensor = qkv_tensor.view(out_shape)

    # remove padding
    if qkv_shape is not None:
        unpad_dim_size = cache(
            "unpad_dim_size", lambda: torch.sum(torch.prod(qkv_shape, dim=-1)).item()
        )
        if unpad_dim_size % world != 0:
            padding_size = qkv_tensor.size(seq_dim) - unpad_dim_size
            qkv_tensor = _unpad_tensor(qkv_tensor, seq_dim, padding_size)
    return qkv_tensor


def slice_inputs(x: Tensor, dim: int, padding: bool = True):
    """
    A func to slice the input sequence in sequence parallel
    """
    group = get_sequence_parallel_group()
    if group is None:
        return x
    sp_rank = get_sequence_parallel_rank()
    sp_world = get_sequence_parallel_world_size()
    dim_size = x.shape[dim]
    unit = (dim_size + sp_world - 1) // sp_world
    if padding and dim_size % sp_world:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, dim, padding_size)
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(unit * sp_rank, unit * (sp_rank + 1))
    return x[slc]


def remove_seqeunce_parallel_padding(x: Tensor, dim: int, unpad_dim_size: int):
    """
    A func to remove the padding part of the tensor based on its original shape
    """
    group = get_sequence_parallel_group()
    if group is None:
        return x
    sp_world = get_sequence_parallel_world_size()
    if unpad_dim_size % sp_world == 0:
        return x
    padding_size = sp_world - (unpad_dim_size % sp_world)
    assert (padding_size + unpad_dim_size) % sp_world == 0
    return _unpad_tensor(x, dim=dim, padding_size=padding_size)


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_sequence_parallel_world_size()
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def gather_seq_scatter_heads(x: Tensor, seq_dim: int, head_dim: int) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return SeqAllToAll.apply(group, x, head_dim, seq_dim, False)


def scatter_heads(x: Tensor, dim: int) -> Tensor:
    """
    A func to split heads before attention in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return Slice.apply(group, x, dim)


def gather_heads(x: Tensor, dim: int, grad_scale: Optional[bool] = False) -> Tensor:
    """
    A func to gather heads for the attention result in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return Gather.apply(group, x, dim, grad_scale)


def gather_outputs(
    x: Tensor,
    *,
    gather_dim: int,
    padding_dim: Optional[int] = None,
    unpad_shape: Optional[Tensor] = None,
    cache: Cache = Cache(disable=True),
    scale_grad=True,
):
    """
    A func to gather the outputs for the model result in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    x = Gather.apply(group, x, gather_dim, scale_grad)
    if padding_dim is not None:
        unpad_dim_size = cache(
            "unpad_dim_size", lambda: torch.sum(torch.prod(unpad_shape, dim=1)).item()
        )
        x = remove_seqeunce_parallel_padding(x, padding_dim, unpad_dim_size)
    return x


def _pad_tensor(x: Tensor, dim: int, padding_size: int):
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size):
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def _broadcast_data(data, shape, dtype, src, group, async_op):
    comms = []
    if isinstance(data, (list, tuple)):
        for i, sub_shape in enumerate(shape):
            comms += _broadcast_data(data[i], sub_shape, dtype[i], src, group, async_op)
    elif isinstance(data, dict):
        for key, sub_data in data.items():
            comms += _broadcast_data(sub_data, shape[key], dtype[key], src, group, async_op)
    elif isinstance(data, Tensor):
        comms.append(dist.broadcast(data, src=src, group=group, async_op=async_op))
    return comms


def _traverse(data: Any, op: Callable) -> Union[None, List, Dict, Any]:
    if isinstance(data, (list, tuple)):
        return [_traverse(sub_data, op) for sub_data in data]
    elif isinstance(data, dict):
        return {key: _traverse(sub_data, op) for key, sub_data in data.items()}
    elif isinstance(data, Tensor):
        return op(data)
    else:
        return None


def _get_shapes(data):
    return _traverse(data, op=lambda x: x.shape)


def _get_dtypes(data):
    return _traverse(data, op=lambda x: x.dtype)


def _construct_broadcast_buffer(shapes, dtypes, device):
    if isinstance(shapes, torch.Size):
        return torch.empty(shapes, dtype=dtypes, device=device)

    if isinstance(shapes, (list, tuple)):
        buffer = []
        for i, sub_shape in enumerate(shapes):
            buffer.append(_construct_broadcast_buffer(sub_shape, dtypes[i], device))
    elif isinstance(shapes, dict):
        buffer = {}
        for key, sub_shape in shapes.items():
            buffer[key] = _construct_broadcast_buffer(sub_shape, dtypes[key], device)
    else:
        return None
    return buffer


class SPDistForward:
    """A forward tool to sync different result across sp group

    Args:
        module: a function or module to process users input
        sp_step: current training step to judge which rank to broadcast its result to all
        name: a distinct str to save meta and async comm
        comm_shape: if different ranks have different shape, mark this arg to True
        device: the device for current rank, can be empty
    """

    def __init__(
        self,
        name: str,
        comm_shape: bool,
        device: torch.device = None,
    ):
        self.name = name
        self.comm_shape = comm_shape
        if device:
            self.device = device
        else:
            self.device = get_device()

    def __call__(self, inputs) -> Any:
        group = get_sequence_parallel_group()
        if not group:
            yield inputs
        else:
            device = self.device
            sp_world = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            for local_step in range(sp_world):
                src_rank = dist.get_global_rank(group, local_step)
                is_src = sp_rank == local_step
                local_shapes = []
                local_dtypes = []
                if local_step == 0:
                    local_result = inputs
                    _SEQ_DATA_BUF[self.name][-1] = local_result
                    local_shapes = _get_shapes(local_result)
                    local_dtypes = _get_dtypes(local_result)
                    if self.comm_shape:
                        group_shapes_lists = [None] * sp_world
                        dist.all_gather_object(group_shapes_lists, local_shapes, group=group)
                        _SEQ_DATA_META_SHAPES[self.name] = group_shapes_lists
                    else:
                        _SEQ_DATA_META_SHAPES[self.name] = [local_shapes] * sp_world
                    _SEQ_DATA_META_DTYPES[self.name] = local_dtypes
                shapes = _SEQ_DATA_META_SHAPES[self.name][local_step]
                dtypes = _SEQ_DATA_META_DTYPES[self.name]
                buf_id = local_step % 2
                if local_step == 0:
                    sync_data = (
                        local_result
                        if is_src
                        else _construct_broadcast_buffer(shapes, dtypes, device)
                    )
                    _broadcast_data(sync_data, shapes, dtypes, src_rank, group, False)
                    _SEQ_DATA_BUF[self.name][buf_id] = sync_data

                # wait for async comm ops
                if _SEQ_DATA_ASYNC_COMMS[self.name]:
                    for comm in _SEQ_DATA_ASYNC_COMMS[self.name]:
                        comm.wait()
                # before return the sync result, do async broadcast for next batch
                if local_step < sp_world - 1:
                    next_buf_id = 1 - buf_id
                    shapes = _SEQ_DATA_META_SHAPES[self.name][local_step + 1]
                    src_rank = dist.get_global_rank(group, local_step + 1)
                    is_src = sp_rank == local_step + 1
                    next_sync_data = (
                        _SEQ_DATA_BUF[self.name][-1]
                        if is_src
                        else _construct_broadcast_buffer(shapes, dtypes, device)
                    )
                    _SEQ_DATA_ASYNC_COMMS[self.name] = _broadcast_data(
                        next_sync_data, shapes, dtypes, src_rank, group, True
                    )
                    _SEQ_DATA_BUF[self.name][next_buf_id] = next_sync_data
                yield _SEQ_DATA_BUF[self.name][buf_id]


sync_inputs = SPDistForward(name="bef_fwd", comm_shape=True)


def sync_data(data, sp_idx, name="tmp"):
    group = get_sequence_parallel_group()
    if group is None:
        return data
    # if sp_idx in _SYNC_BUFFER[name]:
    #     return _SYNC_BUFFER[name][sp_idx]
    sp_rank = get_sequence_parallel_rank()
    src_rank = dist.get_global_rank(group, sp_idx)
    objects = [data] if sp_rank == sp_idx else [None]
    dist.broadcast_object_list(objects, src=src_rank, group=group)
    # _SYNC_BUFFER[name] = {sp_idx: objects[0]}
    return objects[0]
