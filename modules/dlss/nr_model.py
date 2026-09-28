from __future__ import annotations

import os
import math
import torch
from torch import nn
from torch.nn import functional

COSINE_NORM_FLOOR = 0.00006198883056640625
CHUNK_TOKENS = 131072 # 0=disable chunking, otherwise use 32k-256k


def recovered_window_origin(block_index: int) -> tuple[int, int]:
    """Return the vendor window origin as ``(y, x)`` for one graph block."""
    if block_index == 0:
        phase = 0
    elif 1 <= block_index <= 4:
        phase = block_index - 1
    elif 5 <= block_index <= 8:
        phase = block_index - 5
    elif 9 <= block_index <= 14:
        phase = block_index - 9
    elif 15 <= block_index <= 22:
        phase = block_index - 15
    elif 23 <= block_index <= 30:
        phase = block_index - 23
    elif 40 <= block_index <= 55:
        phase = block_index - (40 if block_index < 48 else 48)
    elif 56 <= block_index <= 61:
        phase = block_index - 54
    elif 62 <= block_index <= 69:
        phase = block_index - (62 if block_index < 66 else 66)
    elif block_index == 70:
        phase = 1
    else:
        return (0, 0)
    return ((0, -4, 0, -4)[phase % 4], (0, -4, -4, 0)[phase % 4])


def quadratic_gate(value: torch.Tensor) -> torch.Tensor:
    clamped = value.clamp(-4.0, 4.0)
    linear = torch.abs(clamped) * -0.055908203125 + 0.447265625
    return clamped * linear + 0.89453125


def quadratic_gate_activation(value: torch.Tensor) -> torch.Tensor:
    if value.numel() > 2 * 1024 * 1024:
        if not value.is_contiguous():
            value = value.contiguous()
        out = torch.empty_like(value)
        flat = value.view(-1, value.shape[-1])
        step = 524288
        for start in range(0, flat.shape[0], step):
            chunk = flat[start : start + step]
            cl = chunk.clamp(-4.0, 4.0)
            lin = torch.abs(cl) * -0.055908203125 + 0.447265625
            out.view(-1, value.shape[-1])[start : start + step] = chunk * (cl * lin + 0.89453125)
        return out
    clamped = value.clamp(-4.0, 4.0)
    linear = torch.abs(clamped) * -0.055908203125 + 0.447265625
    return value * (clamped * linear + 0.89453125)


def e4m3_round_trip(value: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E4M3 value (round-half-even, saturating at 448).
    CPU and CUDA use PyTorch's float8 conversion; tracing (Core ML export) and
    devices without float8 support (MPS) use an exact bit-level equivalent.
    """
    if torch.jit.is_tracing():
        magnitude = torch.minimum(value.abs(), torch.full_like(value, 448.0))
        normal_floor = torch.full_like(magnitude, 2**-6)
        exponent = torch.floor(torch.log2(torch.maximum(magnitude, normal_floor)))
        normal_step = torch.pow(torch.full_like(magnitude, 2.0), exponent - 3)
        step = torch.where(
            magnitude < normal_floor,
            torch.full_like(magnitude, 2**-9),
            normal_step,
        )
        rounded = torch.round(magnitude / step) * step
        return torch.where(value < 0, -rounded, rounded)
    if value.device.type in _FLOAT8_CAST_DEVICES and value.dtype in (torch.float32, torch.float16):
        if value.numel() > 2 * 1024 * 1024:
            if not value.is_contiguous():
                value = value.contiguous()
            flat = value.view(-1)
            step = 524288 * 8
            for s in range(0, flat.shape[0], step):
                chunk = flat[s : s + step]
                flat[s : s + step] = chunk.clamp(-448, 448).to(torch.float8_e4m3fn).to(value.dtype)
            return value
        return value.clamp(-448, 448).to(torch.float8_e4m3fn).to(value.dtype)
    if CHUNK_TOKENS > 0 and value.numel() > 8 * CHUNK_TOKENS:
        # The bit-level path builds several int32/float32 temporaries per element;
        # on a whole frame that is a multi-GB spike, so publish in slices.
        flat = value.reshape(-1)
        out = torch.empty_like(flat)
        step = 8 * CHUNK_TOKENS
        for start in range(0, flat.shape[0], step):
            out[start : start + step] = e4m3_round_trip_bitwise(flat[start : start + step])
        return out.reshape(value.shape)
    return e4m3_round_trip_bitwise(value)


_FLOAT8_CAST_DEVICES = frozenset({"cpu", "cuda"})


def e4m3_round_trip_bitwise(value: torch.Tensor) -> torch.Tensor:
    """Exact E4M3 rounding without float8 tensors (any device, any float dtype).
    The step below a value is 2^(floor(log2 |v|) - 3) for normals and 2^-9 in
    the subnormal range; floor(log2) comes from the float32 exponent bits, so
    no transcendental is involved and the result is bit-identical to the
    float8 cast (round-half-even via ``torch.round``).
    """
    single = value.to(torch.float32)
    magnitude = single.abs().clamp(max=448.0)
    normal_floor = torch.full_like(magnitude, 2**-6)
    normal_magnitude = torch.maximum(magnitude, normal_floor)
    exponent_bits = (normal_magnitude.view(torch.int32) >> 23) & 0xFF
    step_bits = (exponent_bits - 3) << 23
    normal_step = step_bits.view(torch.float32)
    step = torch.where(
        magnitude < normal_floor, torch.full_like(magnitude, 2**-9), normal_step
    )
    rounded = _round_half_even(magnitude / step) * step
    return torch.where(single < 0, -rounded, rounded).to(value.dtype)


def _round_half_even(value: torch.Tensor) -> torch.Tensor:
    """Round-half-even for non-negative inputs (MPS rounds ties away from zero)."""
    floor = torch.floor(value)
    fraction = value - floor
    odd = torch.remainder(floor, 2) == 1
    return torch.where(
        fraction > 0.5, floor + 1, torch.where(fraction < 0.5, floor, torch.where(odd, floor + 1, floor))
    )


# The global (`vit_1d`) attention kernels saturate their logits at +3. Verified
# teacher-forced on every global block with the 2026-09-02 per-launch captures
# (standard controls): rel 0.144/0.125/0.106/0.073/0.088/0.071/0.046/0.058 ->
# 0.032/0.093/0.052/0.041/0.037/0.035/0.033/0.034 for blocks 31-38, optimum 3.0
# on each block (2.5 and 3.5 are worse); window-block kernels show no cap.
GLOBAL_ATTENTION_LOGIT_CAP = 3.0
EXPERIMENTAL_GLOBAL_ATTENTION_LOGIT_CAP = GLOBAL_ATTENTION_LOGIT_CAP

def _per_token(function, value: torch.Tensor) -> torch.Tensor:
    """Apply a token-local ``function`` to ``[..., C]`` in chunks of ``CHUNK_TOKENS`` tokens."""
    channels = value.shape[-1]
    flat = value.reshape(-1, channels)
    if CHUNK_TOKENS <= 0 or flat.shape[0] <= CHUNK_TOKENS:
        return function(value)
    parts = [function(part) for part in flat.split(CHUNK_TOKENS, dim=0)]
    return torch.cat(parts, dim=0).reshape(*value.shape[:-1], parts[0].shape[-1])


def vendor_approximate_softmax(value: torch.Tensor) -> torch.Tensor:
    """Reproduce the fused kernels' half bit-affine softmax approximation."""
    if value.ndim == 0 or value.shape[-1] <= 0:
        raise ValueError("vendor softmax expects a non-empty row")
    if value.shape[-1] % 2:
        raise ValueError("vendor softmax expects an even token count")
    if torch.jit.is_tracing():
        return e4m3_round_trip(value.softmax(dim=-1))

    if value.numel() > 2 * 1024 * 1024:
        shape = value.shape
        flat = value.view(-1, shape[-1])
        out = torch.empty_like(flat)
        step = 262144
        for start in range(0, flat.shape[0], step):
            chunk = flat[start : start + step]
            affine = (chunk.to(torch.float32) * 0.044921875 + 1.30078125).clamp(1.03125, 1.5693359375).to(torch.float16)
            bits = affine.view(torch.int16).to(torch.int32) & 0xFFFF
            pairs = bits.view(*bits.shape[:-1], bits.shape[-1] // 2, 2)
            packed = pairs[..., 0] | (pairs[..., 1] << 16)
            transformed = (packed << 5) + 0x7FF88000

            w = torch.empty_like(bits, dtype=torch.int16)
            w[..., 0::2] = (transformed & 0xFFFF).to(torch.int16)
            w[..., 1::2] = (transformed >> 16).to(torch.int16)
            weights = w.view(torch.float16)

            totals = weights.sum(dim=-1, keepdim=True, dtype=torch.float16)
            reciprocal = totals.to(torch.float32).reciprocal().to(torch.float16)
            out[start : start + step] = (weights * reciprocal).to(value.dtype)
        return e4m3_round_trip(out.view(shape))

    affine = (value.to(torch.float16).to(torch.float32) * 0.044921875 + 1.30078125).to(
        torch.float16
    ).clamp(1.03125, 1.5693359375)
    bits = affine.view(torch.int16).to(torch.int32) & 0xFFFF
    pairs = bits.view(*bits.shape[:-1], bits.shape[-1] // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 16)
    transformed = (packed << 5) + 0x7FF88000
    out = torch.empty_like(bits, dtype=torch.int16)
    out[..., 0::2] = (transformed & 0xFFFF).to(torch.int16)
    out[..., 1::2] = (transformed >> 16).to(torch.int16)
    weights = out.view(torch.float16)
    totals = weights.sum(dim=-1, keepdim=True, dtype=torch.float16)
    reciprocal = totals.to(torch.float32).reciprocal().to(torch.float16)
    probabilities = (weights * reciprocal).to(value.dtype)
    return e4m3_round_trip(probabilities)


def cosine_residual(
    skip: torch.Tensor,
    branch: torch.Tensor,
    cosine: torch.Tensor,
) -> torch.Tensor:
    return branch + skip * cosine


def _half_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.to(torch.float32) * right.to(torch.float32)).to(torch.float16)


def _half_add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.to(torch.float32) + right.to(torch.float32)).to(torch.float16)


def _half_fma(
    left: torch.Tensor,
    right: torch.Tensor,
    accumulator: torch.Tensor,
) -> torch.Tensor:
    return (
        left.to(torch.float32) * right.to(torch.float32)
        + accumulator.to(torch.float32)
    ).to(torch.float16)


def vendor_cosine_normalize(value: torch.Tensor) -> torch.Tensor:
    """Apply cosine normalization."""
    half = value.to(torch.float16)
    squared_norm = half.square().sum(
        dim=-1,
        keepdim=True,
        dtype=torch.float16,
    )
    return (
        half * squared_norm.clamp_min(COSINE_NORM_FLOOR).rsqrt()
    ).to(value.dtype)


def vendor_cosine_publish(
    value: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = vendor_cosine_normalize(value).to(torch.float16)
    if scale is not None:
        normalized = _half_multiply(
            normalized,
            scale.to(torch.float16).reshape(1, scale.shape[0], 1, 1),
        )
    return e4m3_round_trip(normalized.to(value.dtype))


def _fragment_swizzle_indices() -> list[int]:
    """Stored offset of every logical ``(query, key)`` window-bias entry."""
    indices: list[int] = []
    for entry in range(64 * 64):
        query, key = divmod(entry, 64)
        query_y, query_x = divmod(query, 8)
        key_y, key_x = divmod(key, 8)

        def bit(value: int, position: int) -> int:
            return (value >> position) & 1

        indices.append(
            (bit(query_y, 2) << 11)
            | (bit(query_x, 2) << 10)
            | (bit(key_y, 2) << 9)
            | (bit(key_x, 2) << 8)
            | (bit(query_y, 0) << 7)
            | (bit(query_x, 1) << 6)
            | (bit(query_x, 0) << 5)
            | (bit(key_y, 0) << 4)
            | (bit(key_x, 1) << 3)
            | (bit(key_y, 1) << 2)
            | (bit(query_y, 1) << 1)
            | bit(key_x, 0)
        )
    return indices


FRAGMENT_SWIZZLE_INDICES = _fragment_swizzle_indices()
_SWIZZLE_INDEX_TENSOR: dict[torch.device, torch.Tensor] = {}


def _get_swizzle_index_tensor(device: torch.device) -> torch.Tensor:
    if device not in _SWIZZLE_INDEX_TENSOR:
        _SWIZZLE_INDEX_TENSOR[device] = torch.tensor(
            FRAGMENT_SWIZZLE_INDICES, dtype=torch.long, device=device
        )
    return _SWIZZLE_INDEX_TENSOR[device]


def recover_attention_bias_layout(bias: torch.Tensor) -> torch.Tensor:
    """Undo the fused kernel's ``mma`` fragment order of a stored ``[heads, 64, 64]`` bias.

    The single-head window blocks (1-4 and 66-70) store their relative bias in
    fragment order; read row-major it mixes unrelated token pairs. The map is a
    pure permutation of the twelve ``(query, key)`` token-index bits, fitted on
    vendor stage captures of blocks 1 and 3 and confirmed on blocks 2 and 4.
    The 2/4/8-head window blocks are already logical while the 16-head split
    blocks use the fragment order again, so ``uses_fragment_swizzle`` gates
    the call. The index table is a
    Python constant so tracing (Core ML conversion) sees a plain gather.
    """
    if bias.ndim != 3 or bias.shape[1] != 64 or bias.shape[2] != 64:
        raise ValueError("attention bias must be [heads, 64, 64]")
    heads = bias.shape[0]
    flat = bias.reshape(heads, 64 * 64)
    index = _get_swizzle_index_tensor(bias.device)
    return flat.index_select(1, index).reshape(heads, 64, 64)


def uses_fragment_swizzle(block_index: int, head_count: int) -> bool: # pylint: disable=unused-argument
    """Single-head window blocks (block 0 included) and the 16-head split blocks.

    The 2/4/8-head window blocks store a logical bias (remapping them puts the
    teacher-forced error at 0.15-0.33); the 16-head split blocks 23-30 and
    40-47 store the fragment order like the single-head blocks (0.001-0.01
    remapped versus 0.03-0.30 raw on DLL captures, sharp-attention blocks
    worst).
    """
    return head_count in (1, 16)


def cosine_attention(
    value: torch.Tensor,
    *,
    qkv_weight: torch.Tensor,
    attention_scale: torch.Tensor,
    attention_bias: torch.Tensor,
    projection_weight: torch.Tensor,
    head_count: int,
    logit_cap: float | None = None,
    symmetric_logit_cap: bool = False,
) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("attention input must be rank 3")
    batch_count, token_count, channels = value.shape
    if head_count <= 0 or channels % head_count:
        raise ValueError("channels must be divisible by a positive head count")

    # 2D GEMM for QKV projection
    flat_value = value.view(-1, channels)
    projected = (flat_value @ qkv_weight).view(batch_count, token_count, -1)
    query, key, projected_value = projected.chunk(3, dim=-1)
    head_channels = channels // head_count
    head_shape = (batch_count, token_count, head_count, head_channels)
    query = query.reshape(head_shape).permute(0, 2, 1, 3)
    key = key.reshape(head_shape).permute(0, 2, 1, 3)
    projected_value = projected_value.reshape(head_shape).permute(0, 2, 1, 3)

    query = vendor_cosine_publish(query, attention_scale)
    key = vendor_cosine_publish(key)
    projected_value = e4m3_round_trip(projected_value)
    scores = query @ key.transpose(-2, -1) + attention_bias.reshape(
        1, head_count, token_count, token_count
    )
    if logit_cap is not None:
        # The global ``vit_1d`` attention kernels clamp their logits to
        # ``[-cap, cap]``: on DLL captures the symmetric clamp puts every
        # global block at 0.017-0.024 where the one-sided cap leaves
        # 0.019-0.087 (window blocks must not be clamped at all).
        scores = (
            scores.clamp(-logit_cap, logit_cap)
            if symmetric_logit_cap
            else scores.clamp(max=logit_cap)
        )
    probabilities = vendor_approximate_softmax(scores)
    attended = (probabilities @ projected_value).permute(0, 2, 1, 3)
    attended = e4m3_round_trip(attended.reshape(batch_count * token_count, channels))
    return (attended @ projection_weight).view(batch_count, token_count, channels)


def partition_windows(value: torch.Tensor, window_size: int) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError("window input must be rank-4 NHWC")
    batch_count, height, width, channels = value.shape
    if height % window_size or width % window_size:
        raise ValueError("spatial dimensions must be divisible by window size")
    return (
        value.reshape(
            batch_count,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, channels)
    )


def reverse_windows(
    windows: torch.Tensor,
    *,
    batch_count: int,
    height: int,
    width: int,
    window_size: int,
) -> torch.Tensor:
    channels = windows.shape[-1]
    return (
        windows.reshape(
            batch_count,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch_count, height, width, channels)
    )


def window_attention(
    value: torch.Tensor,
    *,
    qkv_weight: torch.Tensor,
    attention_scale: torch.Tensor,
    attention_bias: torch.Tensor,
    projection_weight: torch.Tensor,
    head_count: int,
    window_size: int,
    window_origin: tuple[int, int] = (0, 0),
    logit_cap: float | None = None,
) -> torch.Tensor:
    if len(window_origin) != 2 or any(
        origin > 0 or origin <= -window_size for origin in window_origin
    ):
        raise ValueError("window origin must lie within one non-positive window")
    pad_top, pad_left = (-window_origin[0], -window_origin[1])
    pad_bottom = (-(value.shape[1] + pad_top)) % window_size
    pad_right = (-(value.shape[2] + pad_left)) % window_size
    padded_value = (
        value
        if not (pad_top or pad_left or pad_bottom or pad_right)
        else functional.pad(value, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
    )
    def attend(strip: torch.Tensor) -> torch.Tensor:
        """Window attention of a strip of whole window rows, back in NHWC."""
        windows = partition_windows(strip, window_size)
        attended = cosine_attention(
            windows,
            qkv_weight=qkv_weight,
            attention_scale=attention_scale,
            attention_bias=attention_bias,
            projection_weight=projection_weight,
            head_count=head_count,
            logit_cap=logit_cap,
        )
        return reverse_windows(
            attended,
            batch_count=strip.shape[0],
            height=strip.shape[1],
            width=strip.shape[2],
            window_size=window_size,
        )

    # Strips of whole window rows keep the temporaries (the window copy, the
    # scores, the attended windows) bounded by CHUNK_TOKENS instead of the frame.
    padded_height, padded_width = padded_value.shape[1], padded_value.shape[2]
    window_rows = padded_height // window_size
    tokens_per_window_row = padded_width * window_size
    rows_per_strip = max(1, CHUNK_TOKENS // tokens_per_window_row) if CHUNK_TOKENS > 0 else window_rows
    if window_rows <= rows_per_strip:
        attention = attend(padded_value)
    else:
        attention = torch.empty_like(padded_value)
        for first in range(0, window_rows, rows_per_strip):
            y0 = first * window_size
            y1 = min(first + rows_per_strip, window_rows) * window_size
            attention[:, y0:y1] = attend(padded_value[:, y0:y1])
    return attention[
        :,
        pad_top : pad_top + value.shape[1],
        pad_left : pad_left + value.shape[2],
        :,
    ]


def window_block(
    value: torch.Tensor,
    *,
    expansion_weight: torch.Tensor,
    feed_forward_projection_weight: torch.Tensor,
    feed_forward_cosine: torch.Tensor,
    qkv_weight: torch.Tensor,
    attention_scale: torch.Tensor,
    attention_bias: torch.Tensor,
    attention_projection_weight: torch.Tensor,
    attention_cosine: torch.Tensor,
    head_count: int,
    window_size: int,
    window_origin: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    shape = value.shape
    channels = shape[-1]
    flat_tokens = value.view(-1, channels)

    if flat_tokens.shape[0] > 2 * 1024 * 1024:
        out_br = torch.empty_like(flat_tokens)
        step = 524288
        for s in range(0, flat_tokens.shape[0], step):
            chunk = flat_tokens[s : s + step]
            exp = chunk @ expansion_weight
            act = quadratic_gate_activation(exp)
            e4 = e4m3_round_trip(act)
            out_br[s : s + step] = e4 @ feed_forward_projection_weight
        branch = out_br.view(shape)
    else:
        expanded = flat_tokens @ expansion_weight
        act = quadratic_gate_activation(expanded)
        branch = (e4m3_round_trip(act) @ feed_forward_projection_weight).view(shape)

    feed_forward_output = cosine_residual(value, branch, feed_forward_cosine)
    del branch

    attention_branch = window_attention(
        feed_forward_output,
        qkv_weight=qkv_weight,
        attention_scale=attention_scale,
        attention_bias=attention_bias,
        projection_weight=attention_projection_weight,
        head_count=head_count,
        window_size=window_size,
        window_origin=window_origin,
    )
    return cosine_residual(feed_forward_output, attention_branch, attention_cosine)


def _rows(function, *tensors: torch.Tensor) -> torch.Tensor | None:
    """Apply a token-local ``function`` of several NHWC tensors in row chunks, into one output."""
    lead = tensors[0]
    if CHUNK_TOKENS <= 0 or lead.ndim != 4 or lead.shape[0] * lead.shape[1] * lead.shape[2] <= CHUNK_TOKENS:
        return function(*tensors)
    rows = max(1, CHUNK_TOKENS // (lead.shape[0] * lead.shape[2]))
    out = None
    for y0 in range(0, lead.shape[1], rows):
        part = function(*(t[:, y0 : y0 + rows] for t in tensors))
        if out is None:
            out = torch.empty((lead.shape[0], lead.shape[1], lead.shape[2], part.shape[-1]), dtype=part.dtype, device=part.device)
        out[:, y0 : y0 + rows] = part
    return out


def _residual_per_token(value: torch.Tensor, branch: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
    """``cosine_residual`` in row chunks, written into ``branch`` (a fresh attention output the block owns)."""
    if CHUNK_TOKENS <= 0 or branch.ndim != 4 or branch.shape != value.shape:
        return cosine_residual(value, branch, cosine)
    rows = max(1, CHUNK_TOKENS // (branch.shape[0] * branch.shape[2]))
    for y0 in range(0, branch.shape[1], rows):
        branch[:, y0 : y0 + rows] = cosine_residual(value[:, y0 : y0 + rows], branch[:, y0 : y0 + rows], cosine)
    return branch


def branched_feed_forward(
    value: torch.Tensor,
    *,
    expansion_weight: torch.Tensor,
    branch_projection_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
) -> torch.Tensor:
    channels = value.shape[-1]
    if channels < 64 or channels % 32:
        raise ValueError("branched feed-forward input must have 32-aligned channels")
    channel_groups = channels // 32
    if expansion_weight.shape != (
        channel_groups,
        4,
        channel_groups,
        32,
        32,
    ):
        raise ValueError("invalid branched expansion weight shape")
    if branch_projection_weight.shape != (channel_groups, 4, 32, 32):
        raise ValueError("invalid branch projection weight shape")
    if output_projection_weight.shape != (channels, channels):
        raise ValueError("invalid branched output projection weight shape")

    shape = value.shape
    val_g = value.view(*shape[:-1], channel_groups, 32)
    # o: out_groups, k: branch (0..3), i: in_groups, c: in_dim (32), d: out_dim (32)
    expanded = torch.einsum('...ic,okicd->...okd', val_g, expansion_weight)
    act = e4m3_round_trip(quadratic_gate_activation(expanded))
    branches = torch.einsum('...okc,okcd->...okd', act, branch_projection_weight)
    summed = e4m3_round_trip(branches.sum(dim=-2))
    flat = summed.reshape(*shape[:-1], channels)
    return flat @ output_projection_weight


def split_group_feed_forward(
    value: torch.Tensor,
    *,
    first_projection_weight: torch.Tensor,
    expand_weight: torch.Tensor,
    project_weight: torch.Tensor,
) -> torch.Tensor:
    """Split-family feed-forward core: ``e4m3(x @ first)`` followed by a per
    64-channel-group ``64 -> 256 -> 64`` MLP with the quadratic-gate activation.

    The vendor ``ffwd_512_chained`` kernel publishes both the first projection
    and the concatenated group outputs as E4M3; ``weight3`` and the cosine
    residual are applied by the following ``ffwd_proj`` kernel.
    """
    channels = value.shape[-1]
    if channels % 64:
        raise ValueError("split feed-forward channels must be divisible by 64")
    groups = channels // 64
    if first_projection_weight.shape != (channels, channels):
        raise ValueError("invalid split first projection shape")
    if expand_weight.shape != (groups, 64, 256):
        raise ValueError("invalid split group expansion shape")
    if project_weight.shape != (groups, 256, 64):
        raise ValueError("invalid split group projection shape")

    hidden = e4m3_round_trip(value @ first_projection_weight)
    shape = hidden.shape
    hidden_g = hidden.view(*shape[:-1], groups, 64)
    # expand: (..., groups, 64) x (groups, 64, 256) -> (..., groups, 256)
    expanded = torch.einsum('...gc,gcd->...gd', hidden_g, expand_weight)
    act = quadratic_gate_activation(expanded)
    # project: (..., groups, 256) x (groups, 256, 64) -> (..., groups, 64)
    projected = torch.einsum('...gc,gcd->...gd', act, project_weight)
    outputs = projected.reshape(*shape[:-1], channels)
    return e4m3_round_trip(outputs)


def branched_window_block(
    value: torch.Tensor,
    *,
    expansion_weight: torch.Tensor,
    branch_projection_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    feed_forward_cosine: torch.Tensor,
    qkv_weight: torch.Tensor,
    attention_scale: torch.Tensor,
    attention_bias: torch.Tensor,
    attention_projection_weight: torch.Tensor,
    attention_cosine: torch.Tensor,
    head_count: int,
    window_size: int,
    window_origin: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    branch = branched_feed_forward(
        value,
        expansion_weight=expansion_weight,
        branch_projection_weight=branch_projection_weight,
        output_projection_weight=output_projection_weight,
    )
    feed_forward_output = e4m3_round_trip(cosine_residual(value, branch, feed_forward_cosine))

    attention_branch = window_attention(
        feed_forward_output,
        qkv_weight=qkv_weight,
        attention_scale=attention_scale,
        attention_bias=attention_bias,
        projection_weight=attention_projection_weight,
        head_count=head_count,
        window_size=window_size,
        window_origin=window_origin,
    )
    return cosine_residual(feed_forward_output, attention_branch, attention_cosine)


def split_window_block(
    value: torch.Tensor,
    *,
    first_projection_weight: torch.Tensor,
    expand_weight: torch.Tensor,
    project_weight: torch.Tensor,
    feed_forward_projection_weight: torch.Tensor,
    feed_forward_cosine: torch.Tensor,
    qkv_weight: torch.Tensor,
    attention_scale: torch.Tensor,
    attention_bias: torch.Tensor,
    attention_projection_weight: torch.Tensor,
    attention_cosine: torch.Tensor,
    head_count: int,
    window_size: int,
    window_origin: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    branch_ff = split_group_feed_forward(
        value,
        first_projection_weight=first_projection_weight,
        expand_weight=expand_weight,
        project_weight=project_weight,
    )
    shape = branch_ff.shape
    channels = shape[-1]
    branch = (branch_ff.view(-1, channels) @ feed_forward_projection_weight).view(shape)
    feed_forward_output = cosine_residual(value, branch, feed_forward_cosine)

    attention_branch = window_attention(
        feed_forward_output,
        qkv_weight=qkv_weight,
        attention_scale=attention_scale,
        attention_bias=attention_bias,
        projection_weight=attention_projection_weight,
        head_count=head_count,
        window_size=window_size,
        window_origin=window_origin,
    )
    return cosine_residual(feed_forward_output, attention_branch, attention_cosine)


def global_block(
    value: torch.Tensor,
    *,
    expansion_weight: torch.Tensor,
    feed_forward_projection_weight: torch.Tensor,
    feed_forward_cosine: torch.Tensor,
    qkv_weight: torch.Tensor,
    attention_scale: torch.Tensor,
    attention_projection_weight: torch.Tensor,
    attention_cosine: torch.Tensor,
    head_count: int,
    logit_cap: float | None = GLOBAL_ATTENTION_LOGIT_CAP,
) -> torch.Tensor:
    shape = value.shape
    tokens = value.reshape(shape[0], shape[1] * shape[2], shape[3])
    feed_forward_branch = (
        e4m3_round_trip(quadratic_gate_activation(tokens @ expansion_weight))
        @ feed_forward_projection_weight
    )
    feed_forward_output = cosine_residual(
        tokens,
        feed_forward_branch,
        feed_forward_cosine,
    )

    batch_count, token_count, channels = feed_forward_output.shape
    projected = feed_forward_output @ qkv_weight
    query, key, projected_value = projected.chunk(3, dim=-1)
    head_channels = channels // head_count
    head_shape = (batch_count, token_count, head_count, head_channels)
    query = query.reshape(head_shape).permute(0, 2, 1, 3)
    key = key.reshape(head_shape).permute(0, 2, 1, 3)
    projected_value = projected_value.reshape(head_shape).permute(0, 2, 1, 3)

    scale = attention_scale * math.sqrt(channels // head_count)
    query = vendor_cosine_publish(query, scale)
    key = vendor_cosine_publish(key)
    projected_value = e4m3_round_trip(projected_value)

    # Compute attention in head chunks to minimize peak VRAM for large spatial inputs
    if token_count > 8192:
        head_chunk = head_count // 8
    elif token_count > 2048:
        head_chunk = head_count // 4
    elif token_count > 1024:
        head_chunk = head_count // 2
    else:
        head_chunk = head_count
    attended_heads = []
    for h_start in range(0, head_count, head_chunk):
        h_end = min(h_start + head_chunk, head_count)
        q_h = query[:, h_start:h_end]
        k_h = key[:, h_start:h_end]
        v_h = projected_value[:, h_start:h_end]
        scores = q_h @ k_h.transpose(-2, -1)
        if logit_cap is not None:
            scores = scores.clamp(-logit_cap, logit_cap)
        probs = vendor_approximate_softmax(scores)
        att = probs @ v_h
        attended_heads.append(att)

    attended = torch.cat(attended_heads, dim=1).permute(0, 2, 1, 3) if len(attended_heads) > 1 else attended_heads[0].permute(0, 2, 1, 3)
    attended = e4m3_round_trip(attended.reshape(batch_count, token_count, channels))
    attention_branch = attended @ attention_projection_weight

    return cosine_residual(
        feed_forward_output,
        attention_branch,
        attention_cosine,
    ).reshape(shape)


def downsample(value: torch.Tensor, *, weight: torch.Tensor) -> torch.Tensor:
    return average_pool2(value) @ weight


def average_pool2(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] % 2 or value.shape[2] % 2:
        raise ValueError("average pool expects even rank-4 NHWC input")
    return (
        value[:, 0::2, 0::2, :]
        + value[:, 1::2, 0::2, :]
        + value[:, 0::2, 1::2, :]
        + value[:, 1::2, 1::2, :]
    ) * 0.25


def pad_spatial_end(value: torch.Tensor, multiple: int) -> torch.Tensor:
    if value.ndim != 4 or multiple <= 0:
        raise ValueError("spatial padding expects rank-4 NHWC and a positive multiple")
    pad_bottom = (-value.shape[1]) % multiple
    pad_right = (-value.shape[2]) % multiple
    return functional.pad(value, (0, 0, 0, pad_right, 0, pad_bottom))


def learned_upsample2(
    value: torch.Tensor,
    *,
    interpolation: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 4 or interpolation.shape != (value.shape[-1],):
        raise ValueError(
            "upsample expects NHWC input and one interpolation per channel"
        )
    batch_count, height, width, channels = value.shape
    right = torch.cat((value[:, :, 1:, :], value[:, :, -1:, :]), dim=2)
    horizontal_midpoint = value * (1 - interpolation) + right * interpolation
    horizontal = torch.stack((value, horizontal_midpoint), dim=3).reshape(
        batch_count, height, width * 2, channels
    )
    below = torch.cat((horizontal[:, 1:, :, :], horizontal[:, -1:, :, :]), dim=1)
    vertical_midpoint = horizontal * (1 - interpolation) + below * interpolation
    return torch.stack((horizontal, vertical_midpoint), dim=2).reshape(
        batch_count, height * 2, width * 2, channels
    )


def decoder_input_merge(
    value: torch.Tensor,
    *,
    skip: torch.Tensor,
    skip_sine: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 4 or skip.ndim != 4 or value.shape[0] != skip.shape[0]:
        raise ValueError("decoder input merge expects compatible NHWC tensors")
    if value.shape[-1] != skip.shape[-1] or skip_sine.shape != (value.shape[-1],):
        raise ValueError("decoder input merge channel mismatch")
    upsampled = nearest_upsample2_crop(
        value,
        height=skip.shape[1],
        width=skip.shape[2],
    )
    return upsampled + skip * skip_sine


def nearest_upsample2_crop(
    value: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if value.ndim != 4 or height <= 0 or width <= 0:
        raise ValueError("nearest upsample expects rank-4 NHWC input and target extent")
    upsampled = value.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    if height > upsampled.shape[1] or width > upsampled.shape[2]:
        raise ValueError("target exceeds the doubled latent extent")
    return upsampled[:, :height, :width, :]


class NeuralRenderingModel(nn.Module):
    """Fixed recovered 71-block graph with external logical weights."""

    def __init__(self, weights: dict[str, torch.Tensor], chunk: int = 131072):
        super().__init__()
        self._weight_attributes: dict[str, str] = {}
        self._chunk_size = chunk
        global CHUNK_TOKENS # pylint: disable=global-statement
        CHUNK_TOKENS = self._chunk_size
        for index, (name, value) in enumerate(sorted(weights.items())):
            attribute = f"weight_{index}"
            self.register_buffer(attribute, value.detach().to(dtype=torch.float32))
            self._weight_attributes[name] = attribute

    def weight(self, name: str) -> torch.Tensor:
        try:
            return getattr(self, self._weight_attributes[name])
        except KeyError as error:
            raise ValueError(f"missing logical weight: {name}") from error

    def _window(
        self,
        value: torch.Tensor,
        index: int,
        *,
        head_count: int,
        publish: bool = True,
    ) -> torch.Tensor:
        prefix = f"block{index}.layer0"
        attention_bias = self.weight(f"{prefix}.attn_bias")
        if uses_fragment_swizzle(index, head_count):
            attention_bias = recover_attention_bias_layout(attention_bias)
        if f"{prefix}.ffn_expand_weight" in self._weight_attributes:
            output = branched_window_block(
                value,
                expansion_weight=self.weight(f"{prefix}.ffn_expand_weight"),
                branch_projection_weight=self.weight(
                    f"{prefix}.ffn_branch_projection_weight"
                ),
                output_projection_weight=self.weight(
                    f"{prefix}.ffn_output_projection_weight"
                ),
                feed_forward_cosine=self.weight(f"{prefix}.ffn_cos_skip"),
                qkv_weight=self.weight(f"{prefix}.qkv_weight"),
                attention_scale=self.weight(f"{prefix}.attn_scale"),
                attention_bias=attention_bias,
                attention_projection_weight=self.weight(f"{prefix}.projection_weight"),
                attention_cosine=self.weight(f"{prefix}.attn_cos_skip"),
                head_count=head_count,
                window_size=8,
                window_origin=recovered_window_origin(index),
            )
        else:
            output = window_block(
                value,
                expansion_weight=self.weight(f"{prefix}.weight1"),
                feed_forward_projection_weight=self.weight(f"{prefix}.weight2"),
                feed_forward_cosine=self.weight(f"{prefix}.ffn_cos_skip"),
                qkv_weight=self.weight(f"{prefix}.qkv_weight"),
                attention_scale=self.weight(f"{prefix}.attn_scale"),
                attention_bias=attention_bias,
                attention_projection_weight=self.weight(f"{prefix}.projection_weight"),
                attention_cosine=self.weight(f"{prefix}.attn_cos_skip"),
                head_count=head_count,
                window_size=8,
                window_origin=recovered_window_origin(index),
            )
        # Block 0 is pooled before its publication and block 70 feeds the head;
        # ``publish=False`` hands the half-precision output to a fused transition.
        return output if index in (0, 70) or not publish else e4m3_round_trip(output)

    def _split_window(self, value: torch.Tensor, index: int) -> torch.Tensor:
        prefix = f"block{index}"
        return e4m3_round_trip(
            split_window_block(
                value,
                first_projection_weight=self.weight(
                    f"{prefix}.layer0.first_projection_weight"
                ),
                expand_weight=self.weight(f"{prefix}.layer0.group_expand_weight"),
                project_weight=self.weight(f"{prefix}.layer0.group_project_weight"),
                feed_forward_projection_weight=self.weight(f"{prefix}.layer1.weight3"),
                feed_forward_cosine=self.weight(f"{prefix}.layer1.ffn_cos_skip"),
                qkv_weight=self.weight(f"{prefix}.layer2.qkv_weight"),
                attention_scale=self.weight(f"{prefix}.layer2.attn_scale"),
                attention_bias=self._split_attention_bias(index),
                attention_projection_weight=self.weight(
                    f"{prefix}.layer3.projection_weight"
                ),
                attention_cosine=self.weight(f"{prefix}.layer3.attn_cos_skip"),
                head_count=16,
                window_size=8,
                window_origin=recovered_window_origin(index),
            )
        )

    def _split_attention_bias(self, index: int) -> torch.Tensor:
        attention_bias = self.weight(f"block{index}.layer2.attn_bias")
        if uses_fragment_swizzle(index, 16):
            attention_bias = recover_attention_bias_layout(attention_bias)
        return attention_bias

    def _global(self, value: torch.Tensor, index: int) -> torch.Tensor:
        prefix = f"block{index}"
        return e4m3_round_trip(
            global_block(
                value,
                expansion_weight=self.weight(f"{prefix}.layer0.weight"),
                feed_forward_projection_weight=self.weight(f"{prefix}.layer1.weight"),
                feed_forward_cosine=self.weight(f"{prefix}.layer1.ffn_cos_skip"),
                qkv_weight=self.weight(f"{prefix}.layer2.qkv_weight"),
                attention_scale=self.weight(f"{prefix}.layer2.attn_scale"),
                attention_projection_weight=self.weight(
                    f"{prefix}.layer4.projection_weight"
                ),
                attention_cosine=self.weight(f"{prefix}.layer4.attn_cos_skip"),
                head_count=32,
            )
        )

    def _downsample_window(
        self,
        value: torch.Tensor,
        index: int,
        *,
        head_count: int,
    ) -> torch.Tensor:
        # The fused ``ds`` kernels pool the block's half-precision output before
        # its E4M3 publication (vendor transition captures: 0.025-0.028 versus
        # 0.026-0.031 when pooling the published tensor).
        transformed = self._window(
            value, index, head_count=head_count, publish=False
        )
        if index == 22:
            transformed = pad_spatial_end(transformed, 8)
        # The pooled tensor is published as E4M3 before the QMMA projection
        # (vendor transition captures: 0.010-0.021 versus 0.025-0.028 without).
        return e4m3_round_trip(
            e4m3_round_trip(average_pool2(transformed))
            @ self.weight(f"block{index}.layer0.weight0")
        )

    def _upsample_window(
        self,
        value: torch.Tensor,
        skip: torch.Tensor,
        index: int,
        *,
        head_count: int,
    ) -> torch.Tensor:
        prefix = f"block{index}.layer0"
        projected = value @ self.weight(f"{prefix}.weight0")
        skip_path = skip * self.weight(f"{prefix}.sin")
        upsampled = nearest_upsample2_crop(
            projected,
            height=skip.shape[1],
            width=skip.shape[2],
        )
        # The fused ``upsample`` kernels read the merged tensor as E4M3 (vendor
        # decoder captures: blocks 48/56/62 move from 0.037-0.041 to 0.019-0.022).
        return self._window(
            e4m3_round_trip(upsampled + skip_path), index, head_count=head_count
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        adapter = self.weight("block0.layer0.input_adapter_weight")
        value = input_value @ adapter
        # The fused pre kernel runs the block-0 window attention on the full
        # adapter output, then 2x2-average-pools and publishes once (vendor
        # block-0 capture: 0.0254 -> 0.0075 MAE versus pool-then-block).
        # The full-resolution skip the post block merges is that block-0
        # output (published E4M3), not the raw adapter output: the adapter
        # output carries the noise channels straight into the head and shows
        # up as pixel-level grain (four DLL goldens at 256: MAE 0.0118 -> 0.0055,
        # high-pass correlation 0.52 -> 0.94, spectral bands within 8%).
        block0_output = self._window(value, 0, head_count=1)
        del value
        full_resolution_skip = e4m3_round_trip(block0_output)
        value = e4m3_round_trip(average_pool2(block0_output))
        del block0_output
        for index in range(1, 4):
            value = self._window(value, index, head_count=1)
        skips = [value]
        value = self._downsample_window(value, 4, head_count=1)

        for regular, transition, head_count in (
            (range(5, 8), 8, 2),
            (range(9, 14), 14, 4),
            (range(15, 22), 22, 8),
        ):
            for index in regular:
                value = self._window(value, index, head_count=head_count)
            skips.append(value)
            value = self._downsample_window(
                value,
                transition,
                head_count=head_count,
            )

        for index in range(23, 30):
            value = self._split_window(value, index)
        value = self._split_window(value, 30)
        split_skip = value
        value = pad_spatial_end(value, 8)
        value = downsample(
            value,
            weight=self.weight("block30.layer4.weight"),
        )
        value = e4m3_round_trip(value)
        for index in range(31, 39):
            value = self._global(value, index)

        value = decoder_input_merge(
            value @ self.weight("block39.layer0.conv_weight"),
            skip=split_skip,
            skip_sine=self.weight("block39.layer0.inp_upsample_sin"),
        )
        value = e4m3_round_trip(value)
        for index in range(40, 48):
            value = self._split_window(value, index)
        value = self._upsample_window(value, skips[3], 48, head_count=8)
        for index in range(49, 56):
            value = self._window(value, index, head_count=8)

        for transition, regular, skip_index, head_count in (
            (56, range(57, 62), 2, 4),
            (62, range(63, 66), 1, 2),
            (66, range(67, 70), 0, 1),
        ):
            value = self._upsample_window(
                value,
                skips[skip_index],
                transition,
                head_count=head_count,
            )
            for index in regular:
                value = self._window(value, index, head_count=head_count)

        value = nearest_upsample2_crop(
            value,
            height=full_resolution_skip.shape[1],
            width=full_resolution_skip.shape[2],
        )
        merge_sin = self.weight("block70.layer0.inp_merge_sin")
        merge_cos = self.weight("block70.layer0.inp_merge_cos")
        value = value * merge_sin + full_resolution_skip * merge_cos
        del full_resolution_skip
        value = self._window(value, 70, head_count=1)
        out_gain = self.weight("block70.layer0.out_gain")
        out_conv = self.weight("block70.layer0.out_conv_weight")
        return value[..., :16] @ out_gain + value[..., 16:] @ out_conv


def load_model(path: str) -> NeuralRenderingModel:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if metadata.get("format") not in {
            "dlssnr-logical-v8",
            "dlssnr-logical-v9",
            "dlssnr-logical-v10",
            "dlssnr-logical-v11",
            "dlssnr-logical-v12",
            "dlssnr-logical-v13",
            "dlssnr-logical-v14",
            "dlssnr-logical-v15",
            "dlssnr-logical-v16",
            "dlssnr-logical-v17",
            "dlssnr-logical-v18",
        }:
            raise ValueError("source format must be dlssnr-logical-v8 through v18")
        if metadata.get("fully_logical") != "true":
            raise ValueError("source must declare fully_logical=true")
        names = source.keys()
        weights = {name: source.get_tensor(name) for name in names}
    return NeuralRenderingModel(weights)
