# pylint: disable=redefined-builtin,no-member,protected-access

from typing import Optional

import torch

from .common import dtype_dict


def pack_int_symetric(tensor: torch.CharTensor, weights_dtype: str) -> torch.ByteTensor:
    return packed_int_function_dict[weights_dtype]["pack"](tensor.sub_(dtype_dict[weights_dtype]["min"]).to(dtype=dtype_dict[weights_dtype]["storage_dtype"]))


def pack_int_asymetric(tensor: torch.CharTensor, weights_dtype: str) -> torch.ByteTensor:
    return packed_int_function_dict[weights_dtype]["pack"](tensor.to(dtype=dtype_dict[weights_dtype]["storage_dtype"]))


def unpack_int_symetric(packed_tensor: torch.ByteTensor, shape: torch.Size, weights_dtype: str, dtype: Optional[torch.dtype] = None) -> torch.CharTensor:
    if dtype is None:
        dtype = dtype_dict[weights_dtype]["torch_dtype"]
    return packed_int_function_dict[weights_dtype]["unpack"](packed_tensor, shape).to(dtype=dtype).add_(dtype_dict[weights_dtype]["min"])


def unpack_int_asymetric(packed_tensor: torch.ByteTensor, shape: torch.Size, weights_dtype: str) -> torch.CharTensor:
    return packed_int_function_dict[weights_dtype]["unpack"](packed_tensor, shape)


def pack_uint7(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :7],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 7], 1),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 3),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 5),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 7),
                ),
                dim=-1
            ),
            128
        ),
    )
    return packed_tensor


def pack_uint6(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(
                packed_tensor[:, :2],
                torch.bitwise_and(
                    torch.stack(
                        (
                            torch.bitwise_left_shift(packed_tensor[:, 3], 2),
                            torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                        ),
                        dim=-1
                    ),
                    192
                )
            ),
            torch.bitwise_or(packed_tensor[:, 2], torch.bitwise_left_shift(packed_tensor[:, 3], 6)).unsqueeze(-1),
        ),
        dim=-1
    )
    return packed_tensor


def pack_uint5(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 5)),
            torch.bitwise_or(
                packed_tensor[:, 3],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 5], 2), 96),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 3), 128),
                ),
            ).unsqueeze(-1),
            torch.bitwise_or(
                packed_tensor[:, 4],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 2), 96),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 4), 128),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    )
    return packed_tensor


def pack_uint4(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 2)
    packed_tensor = torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 4))
    return packed_tensor


def pack_uint3(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 3:6], 3)),
        torch.cat(
            (
                torch.bitwise_left_shift(packed_tensor[:, 6:8], 6),
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 4), 64),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 5), 128),
                ).unsqueeze(-1),
            ),
            dim=-1
        )
    )
    return packed_tensor


def pack_uint2(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 2)),
        torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 4), torch.bitwise_left_shift(packed_tensor[:, 3], 6)),
    )
    return packed_tensor


def pack_uint1(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 1)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 2), torch.bitwise_left_shift(packed_tensor[:, 3], 3))
        ),
        torch.bitwise_or(
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 4], 4), torch.bitwise_left_shift(packed_tensor[:, 5], 5)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 6], 6), torch.bitwise_left_shift(packed_tensor[:, 7], 7))
        ),
    )
    return packed_tensor


def unpack_uint7(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :7], 127),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 1), 64),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 2), 32),
                    ),
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 2], 3), 16),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3], 4), 8),
                    ),
                ),
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 4], 5), 4),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 5], 6), 2),
                    ),
                    torch.bitwise_right_shift(packed_tensor[:, 6], 7),
                ),
            ).unsqueeze(-1)
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint6(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, 0:3], 63),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 2), 48),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 4), 12),
                ),
                torch.bitwise_right_shift(packed_tensor[:, 2], 6)
            ).unsqueeze(-1)
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint5(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result_bitwise_right_shift = torch.bitwise_right_shift(packed_tensor[:, :3], 5)
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :5], 31),
            torch.bitwise_or(
                result_bitwise_right_shift[:, :2],
                torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3:5], 2), 24),
            ),
            torch.bitwise_or(
                result_bitwise_right_shift[:, 2],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3], 3), 16),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 4], 4), 8),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint4(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1).view(shape)
    return result


def unpack_uint3(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.bitwise_and(
        torch.cat(
            (
                packed_tensor[:, :3],
                torch.bitwise_right_shift(packed_tensor[:, :3], 3),
                torch.bitwise_or(
                    torch.bitwise_right_shift(packed_tensor[:, :2], 6),
                    torch.bitwise_and(
                        torch.stack(
                            (
                                torch.bitwise_right_shift(packed_tensor[:, 2], 4),
                                torch.bitwise_right_shift(packed_tensor[:, 2], 5),
                            ),
                            dim=-1
                        ),
                        4
                    ),
                ),
            ),
            dim=-1
        ),
        7
    ).view(shape)
    return result


def unpack_uint2(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.bitwise_and(
        torch.stack(
            (
                packed_tensor,
                torch.bitwise_right_shift(packed_tensor, 2),
                torch.bitwise_right_shift(packed_tensor, 4),
                torch.bitwise_right_shift(packed_tensor, 6),
            ),
            dim=-1
        ),
        3
    ).view(shape)
    return result


def unpack_uint1(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.bitwise_and(
        torch.stack(
            (
                packed_tensor,
                torch.bitwise_right_shift(packed_tensor, 1),
                torch.bitwise_right_shift(packed_tensor, 2),
                torch.bitwise_right_shift(packed_tensor, 3),
                torch.bitwise_right_shift(packed_tensor, 4),
                torch.bitwise_right_shift(packed_tensor, 5),
                torch.bitwise_right_shift(packed_tensor, 6),
                torch.bitwise_right_shift(packed_tensor, 7),
            ),
            dim=-1
        ),
        1
    ).view(shape)
    return result


packed_int_function_dict = {
    "int7": {"pack": pack_uint7, "unpack": unpack_uint7},
    "int6": {"pack": pack_uint6, "unpack": unpack_uint6},
    "int5": {"pack": pack_uint5, "unpack": unpack_uint5},
    "int4": {"pack": pack_uint4, "unpack": unpack_uint4},
    "int3": {"pack": pack_uint3, "unpack": unpack_uint3},
    "int2": {"pack": pack_uint2, "unpack": unpack_uint2},
    "uint7": {"pack": pack_uint7, "unpack": unpack_uint7},
    "uint6": {"pack": pack_uint6, "unpack": unpack_uint6},
    "uint5": {"pack": pack_uint5, "unpack": unpack_uint5},
    "uint4": {"pack": pack_uint4, "unpack": unpack_uint4},
    "uint3": {"pack": pack_uint3, "unpack": unpack_uint3},
    "uint2": {"pack": pack_uint2, "unpack": unpack_uint2},
    "uint1": {"pack": pack_uint1, "unpack": unpack_uint1},
    "bool": {"pack": pack_uint1, "unpack": unpack_uint1},
}
