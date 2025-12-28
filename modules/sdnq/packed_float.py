import torch

from .common import dtype_dict
from .packed_int import pack_int_asymetric, unpack_int_asymetric


float_bits_to_uint_dict = {
    1: "uint1",
    2: "uint2",
    3: "uint3",
    4: "uint4",
    5: "uint5",
    6: "uint6",
    7: "uint7",
}


def pack_float(x: torch.FloatTensor, weights_dtype: str) -> torch.Tensor:
    exponent_bits = dtype_dict[weights_dtype]["exponent"]
    mantissa_bits = dtype_dict[weights_dtype]["mantissa"]
    total_bits = dtype_dict[weights_dtype]["num_bits"]

    if dtype_dict[weights_dtype]["is_unsigned"]:
        sign_mask = (1 << (total_bits-1))
    else:
        sign_mask = (1 << (total_bits-1)) + (1 << (total_bits-2))

    mantissa_difference = 23 - mantissa_bits
    exponent_difference = 8 - exponent_bits
    mantissa_mask = (1 << mantissa_difference)

    x = x.to(dtype=torch.float32).view(torch.int32)

    x = torch.where(
        torch.gt(
            torch.bitwise_and(x, -(1 << (mantissa_difference-4)) & ~(-mantissa_mask)),
            (1 << (mantissa_difference-1)),
        ),
        torch.add(x, mantissa_mask),
        x,
    )

    x = torch.where(torch.lt(x.view(torch.float32).abs(), dtype_dict[weights_dtype]["min_normal"]), 0, x)

    x = torch.bitwise_right_shift(x, mantissa_difference)
    x = torch.bitwise_and(
        torch.bitwise_or(
            torch.bitwise_and(torch.bitwise_right_shift(x, exponent_difference), sign_mask),
            torch.bitwise_and(x, ~sign_mask),
        ),
        ~(-(1 << total_bits)),
    ).view(torch.uint32)

    if total_bits < 8:
        x = pack_int_asymetric(x, float_bits_to_uint_dict[total_bits])
    else:
        x = x.to(dtype=dtype_dict[weights_dtype]["storage_dtype"])

    return x


def unpack_float(x: torch.Tensor, shape: torch.Size, weights_dtype: str) -> torch.FloatTensor:
    exponent_bits = dtype_dict[weights_dtype]["exponent"]
    mantissa_bits = dtype_dict[weights_dtype]["mantissa"]
    total_bits = dtype_dict[weights_dtype]["num_bits"]

    if dtype_dict[weights_dtype]["is_unsigned"]:
        sign_mask = (1 << (total_bits-1))
    else:
        sign_mask = (1 << (total_bits-1)) + (1 << (total_bits-2))

    mantissa_difference = 23 - mantissa_bits
    exponent_difference = 8 - exponent_bits

    if total_bits < 8:
        x = unpack_int_asymetric(x, shape, float_bits_to_uint_dict[total_bits])

    x = x.to(dtype=torch.uint32).view(torch.int32)
    x = torch.bitwise_left_shift(
        torch.bitwise_or(
            torch.bitwise_left_shift(torch.bitwise_and(x, sign_mask), exponent_difference),
            torch.bitwise_and(x, ~sign_mask),
        ),
        mantissa_difference,
    )

    x = torch.bitwise_or(
        x,
        torch.bitwise_and(
            torch.bitwise_right_shift(
                -torch.bitwise_and(torch.bitwise_not(x),  1073741824),
                exponent_difference,
            ),
            1065353216,
        ),
    )

    overflow_mask = (~(-(1 << (22 + exponent_bits))) | -1073741824)
    x = torch.where(torch.bitwise_and(x, overflow_mask).to(dtype=torch.bool), x, 0)
    x = x.view(torch.float32)

    return x
