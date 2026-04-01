import torch


def pack_uint15(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 16)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :15],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 15], 1),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 3),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 5),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 7),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 8),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 9),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 10),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 11),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 12),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 13),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 14),
                    torch.bitwise_left_shift(packed_tensor[:, 15], 15),
                ),
                dim=-1
            ),
            32768
        ),
    )
    return packed_tensor


def pack_uint14(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :7],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 7], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 8),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 10),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 12),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 14),
                ),
                dim=-1
            ),
            49152
        ),
    )
    return packed_tensor


def pack_uint13(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 16)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :13],
        torch.bitwise_and(
            torch.cat(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 13:], 13),
                    torch.bitwise_left_shift(packed_tensor[:, 13:], 10),
                    torch.bitwise_left_shift(packed_tensor[:, 13:], 7),
                    torch.bitwise_left_shift(packed_tensor[:, 13:], 4),
                    torch.bitwise_or(
                        torch.bitwise_or(
                            torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 13], 1), 8192),
                            torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 14], 2), 16384),
                        ),
                        torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 15], 3), 32768),
                    ).unsqueeze(-1),
                ),
                dim=-1,
            ),
            57344,
        ),
    )
    return packed_tensor


def pack_uint12(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :3],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 8),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 12),
                ),
                dim=-1
            ),
            61440
        )
    )
    return packed_tensor


def pack_uint11(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 16)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :8], torch.bitwise_left_shift(packed_tensor[:, 8:], 11)),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 8:11], 5), 63),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 11:14], 1), 4032),
                ),
                torch.cat(
                    (
                        torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 14:], 7), -4096),
                        torch.bitwise_or(
                            torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 14], 3), 12288),
                            torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 15], 5), -16384),
                        ).unsqueeze(-1),
                    ),
                    dim=-1,
                ),
            ),
        ),
        dim=-1,
    )
    return packed_tensor


def pack_uint10(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 10)),
            torch.bitwise_or(
                packed_tensor[:, 3:5],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 5:7], 4), 15360),
                    torch.bitwise_and(
                        torch.stack(
                            (
                                torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                                torch.bitwise_left_shift(packed_tensor[:, 7], 8),
                            ),
                            dim=-1,
                        ),
                        49152
                    ),
                ),
            ),
        ),
        dim=-1
    )
    return packed_tensor


def pack_uint9(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 16)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :8], torch.bitwise_left_shift(packed_tensor[:, 8:], 9)),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 8], 7), 3),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 9], 5), 12),
                    ),
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 10], 3), 48),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 11], 1), 192),
                    ),
                ),
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 12], 1), 768),
                        torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 13], 3), 3072),
                    ),
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 14], 5), 12288),
                        torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 15], 7), 49152),
                    ),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    return packed_tensor


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
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :3],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 3], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 6),
                ),
                dim=-1
            ),
            192
        )
    )
    return packed_tensor


def pack_uint5(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 5)),
            torch.bitwise_or(
                packed_tensor[:, 3:5],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 5:7], 2), 96),
                    torch.bitwise_and(
                        torch.stack(
                            (
                                torch.bitwise_left_shift(packed_tensor[:, 7], 3),
                                torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                            ),
                            dim=-1,
                        ),
                        128,
                    ),
                ),
            ),
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
