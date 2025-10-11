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
Advanced distributed functions for sequence parallel.
"""

from typing import Optional, List
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import ShardingStrategy

from .basic import get_global_rank, get_world_size


_DATA_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_CPU_GROUP = None
_MODEL_SHARD_CPU_INTER_GROUP = None
_MODEL_SHARD_CPU_INTRA_GROUP = None
_MODEL_SHARD_INTER_GROUP = None
_MODEL_SHARD_INTRA_GROUP = None
_SEQUENCE_PARALLEL_GLOBAL_RANKS = None


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel process group.
    """
    return _DATA_PARALLEL_GROUP


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel process group.
    """
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel CPU process group.
    """
    return _SEQUENCE_PARALLEL_CPU_GROUP


def get_data_parallel_rank() -> int:
    """
    Get data parallel rank.
    """
    group = get_data_parallel_group()
    return dist.get_rank(group) if group else get_global_rank()


def get_data_parallel_world_size() -> int:
    """
    Get data parallel world size.
    """
    group = get_data_parallel_group()
    return dist.get_world_size(group) if group else get_world_size()


def get_sequence_parallel_rank() -> int:
    """
    Get sequence parallel rank.
    """
    group = get_sequence_parallel_group()
    return dist.get_rank(group) if group else 0


def get_sequence_parallel_world_size() -> int:
    """
    Get sequence parallel world size.
    """
    group = get_sequence_parallel_group()
    return dist.get_world_size(group) if group else 1


def get_model_shard_cpu_intra_group() -> Optional[dist.ProcessGroup]:
    """
    Get the CPU intra process group of model sharding.
    """
    return _MODEL_SHARD_CPU_INTRA_GROUP


def get_model_shard_cpu_inter_group() -> Optional[dist.ProcessGroup]:
    """
    Get the CPU inter process group of model sharding.
    """
    return _MODEL_SHARD_CPU_INTER_GROUP


def get_model_shard_intra_group() -> Optional[dist.ProcessGroup]:
    """
    Get the GPU intra process group of model sharding.
    """
    return _MODEL_SHARD_INTRA_GROUP


def get_model_shard_inter_group() -> Optional[dist.ProcessGroup]:
    """
    Get the GPU inter process group of model sharding.
    """
    return _MODEL_SHARD_INTER_GROUP


def init_sequence_parallel(sequence_parallel_size: int):
    """
    Initialize sequence parallel.
    """
    global _DATA_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_CPU_GROUP
    global _SEQUENCE_PARALLEL_GLOBAL_RANKS
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    data_parallel_size = world_size // sequence_parallel_size
    for i in range(data_parallel_size):
        start_rank = i * sequence_parallel_size
        end_rank = (i + 1) * sequence_parallel_size
        ranks = range(start_rank, end_rank)
        group = dist.new_group(ranks)
        cpu_group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_CPU_GROUP = cpu_group
            _SEQUENCE_PARALLEL_GLOBAL_RANKS = list(ranks)


def init_model_shard_group(
    *,
    sharding_strategy: ShardingStrategy,
    device_mesh: Optional[DeviceMesh] = None,
):
    """
    Initialize process group of model sharding.
    """
    global _MODEL_SHARD_INTER_GROUP
    global _MODEL_SHARD_INTRA_GROUP
    global _MODEL_SHARD_CPU_INTER_GROUP
    global _MODEL_SHARD_CPU_INTRA_GROUP
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    if device_mesh is not None:
        num_shards_per_group = device_mesh.shape[1]
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        num_shards_per_group = 1
    elif sharding_strategy in [
        ShardingStrategy.HYBRID_SHARD,
        ShardingStrategy._HYBRID_SHARD_ZERO2,
    ]:
        num_shards_per_group = torch.cuda.device_count()
    else:
        num_shards_per_group = world_size
    num_groups = world_size // num_shards_per_group
    device_mesh = (num_groups, num_shards_per_group)

    gpu_mesh_2d = init_device_mesh("cuda", device_mesh, mesh_dim_names=("inter", "intra"))
    cpu_mesh_2d = init_device_mesh("cpu", device_mesh, mesh_dim_names=("inter", "intra"))

    _MODEL_SHARD_INTER_GROUP = gpu_mesh_2d.get_group("inter")
    _MODEL_SHARD_INTRA_GROUP = gpu_mesh_2d.get_group("intra")
    _MODEL_SHARD_CPU_INTER_GROUP = cpu_mesh_2d.get_group("inter")
    _MODEL_SHARD_CPU_INTRA_GROUP = cpu_mesh_2d.get_group("intra")

def get_sequence_parallel_global_ranks() -> List[int]:
    """
    Get all global ranks of the sequence parallel process group
    that the caller rank belongs to.
    """
    if _SEQUENCE_PARALLEL_GLOBAL_RANKS is None:
        return [dist.get_rank()]
    return _SEQUENCE_PARALLEL_GLOBAL_RANKS


def get_next_sequence_parallel_rank() -> int:
    """
    Get the next global rank of the sequence parallel process group
    that the caller rank belongs to.
    """
    sp_global_ranks = get_sequence_parallel_global_ranks()
    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()
    return sp_global_ranks[(sp_rank + 1) % sp_size]


def get_prev_sequence_parallel_rank() -> int:
    """
    Get the previous global rank of the sequence parallel process group
    that the caller rank belongs to.
    """
    sp_global_ranks = get_sequence_parallel_global_ranks()
    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()
    return sp_global_ranks[(sp_rank + sp_size - 1) % sp_size]
