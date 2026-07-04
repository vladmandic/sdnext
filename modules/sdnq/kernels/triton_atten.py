import os
import math
import torch
import triton
import triton.language as tl

from ..common import compile_func # pylint: disable=relative-beyond-top-level
from ..quant_utils import quantize_int_mm, quantize_fp_mm, get_hadamard, apply_hadamard # pylint: disable=relative-beyond-top-level


min_block_size = int(os.environ.get("SDNQ_TRITON_ATTEN_MIN_BLOCK_SIZE", "256"))
matmul_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN}, num_warps=w, num_stages=s)
    for BM in [int(BM) for BM in os.environ.get("SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST", "64,128").replace(" ","").split(",")]
    for BN in [int(BN) for BN in os.environ.get("SDNQ_TRITON_ATTEN_BLOCK_SIZE_N_LIST", "32,64").replace(" ","").split(",")]
    for w in [int(w) for w in os.environ.get("SDNQ_TRITON_ATTEN_NUM_WARPS_LIST", "4,8").replace(" ","").split(",")]
    for s in [int(s) for s in os.environ.get("SDNQ_TRITON_ATTEN_NUM_STAGES_LIST", "1,2,4").replace(" ","").split(",")]
]


@triton.autotune(
    configs=matmul_configs,
    key=[
        "is_causal", "do_mask",
        "QZ", "QH", "QN_AT", "QHD",
        "KZ", "KH", "KN_AT", "KHD",
        "VZ", "VH", "VN_AT", "VHD",
        "qk_is_quantized",
        "pv_is_quantized",
        "q_dtype", "v_dtype",
        "out_dtype", "mask_dtype",
    ],
    cache_results=True,
)
@triton.jit
def sdnq_attn_kernel(
    q_ptr, k_ptr, v_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr,
    out_ptr, mask_ptr, is_causal: tl.constexpr, do_mask: tl.constexpr,
    QZ: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr, QHD: tl.constexpr,
    KZ: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr, KHD: tl.constexpr,
    VZ: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, VHD: tl.constexpr,
    OZ: tl.constexpr, OH: tl.constexpr, ON: tl.constexpr, OHD: tl.constexpr,
    MZ: tl.constexpr, MH: tl.constexpr, MQN: tl.constexpr, MKN: tl.constexpr,
    QN_AT: tl.constexpr, KN_AT: tl.constexpr, VN_AT: tl.constexpr,
    qk_is_quantized: tl.constexpr, pv_is_quantized: tl.constexpr,
    q_dtype: tl.constexpr, v_dtype: tl.constexpr,
    out_dtype: tl.constexpr, mask_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
) -> None: # pylint: disable=unused-argument
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    tl.assume(QZ > 0)
    tl.assume(QH > 0)
    tl.assume(QN > 0)
    tl.assume(QHD > 0)
    tl.assume(KZ > 0)
    tl.assume(KH > 0)
    tl.assume(KN > 0)
    tl.assume(KHD > 0)
    tl.assume(VZ > 0)
    tl.assume(VH > 0)
    tl.assume(VN > 0)
    tl.assume(VHD > 0)
    tl.assume(OZ > 0)
    tl.assume(OH > 0)
    tl.assume(ON > 0)
    tl.assume(OHD > 0)
    tl.assume(MZ >= 0)
    tl.assume(MH >= 0)
    tl.assume(MQN >= 0)
    tl.assume(MKN >= 0)
    tl.assume(off_h >= 0)
    tl.assume(off_z >= 0)
    tl.assume(start_m >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(do_mask == 0 or do_mask == 1)
    tl.assume(is_causal == 0 or is_causal == 1)
    tl.assume(qk_is_quantized == 0 or qk_is_quantized == 1)
    tl.assume(pv_is_quantized == 0 or pv_is_quantized == 1)

    do_k_mask: tl.constexpr = KN % BLOCK_SIZE_N != 0
    start_m_block = start_m * BLOCK_SIZE_M
    offs_m = start_m_block + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offset_y = off_z * (QN * QH) + off_h * QN
    offset_y_k = off_z * (KN * KH) + ((off_h * KH) // QH) * KN
    offset_y_v = off_z * (VN * VH) + ((off_h * VH) // QH) * VN

    q_desc = tl.make_tensor_descriptor(q_ptr + offset_y * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, QHD])
    k_desc = tl.make_tensor_descriptor(k_ptr + offset_y_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, KHD])
    v_desc = tl.make_tensor_descriptor(v_ptr + offset_y_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, VHD])

    if qk_is_quantized:
        q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + offset_y, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + offset_y_k, shape=[KN], strides=[1,], block_shape=[BLOCK_SIZE_N])
        q_scale = q_scale_desc.load([start_m_block])[:, None]
    if pv_is_quantized:
        v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + offset_y_v, shape=[VN], strides=[1,], block_shape=[BLOCK_SIZE_N])
    if do_mask:
        mask_desc = tl.make_tensor_descriptor(mask_ptr + offset_y * MKN, shape=[MQN, MKN],  strides=[MKN, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    m_i = tl.full([BLOCK_SIZE_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_SIZE_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, VHD], dtype=tl.float32)
    q = q_desc.load([start_m_block, 0])

    for start_n_idx in tl.range(0, tl.cdiv(KN, BLOCK_SIZE_N)):
        start_n = start_n_idx * BLOCK_SIZE_N
        skip = False
        if is_causal and ((start_m_block + BLOCK_SIZE_M) <= start_n):
            skip = True
        if do_mask:
            mask = mask_desc.load([start_m_block, start_n])
            mask_max = tl.max(mask)
            if mask.dtype == tl.int8:
                mask = mask.to(tl.int1)
            else:
                mask = mask.to(tl.float32)
            if mask.dtype == tl.int1:
                if mask_max == 0:
                    skip = True
            elif mask_max == float("-inf"):
                skip = True
        if not skip:
            k = k_desc.load([start_n, 0]).T
            if qk_is_quantized:
                k_scale = k_scale_desc.load([start_n])[None, :]
                if q.dtype == tl.int8:
                    qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * q_scale * k_scale
                else:
                    qk = tl.dot(q, k, out_dtype=tl.float32) * q_scale * k_scale
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32)

            if is_causal:
                qk = tl.where((offs_m[:, None] >= (start_n + offs_n[None, :])) & (offs_m[:, None] < QN), qk, float("-inf"))
            if do_mask:
                if mask.dtype == tl.int1:
                    qk = tl.where(mask, qk, float("-inf"))
                else:
                    qk += mask
            if do_k_mask and (start_n + BLOCK_SIZE_N) > KN:
                qk = tl.where(offs_n[None, :] < (KN - start_n), qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            if do_mask:
                alpha = tl.exp2(tl.where((m_i == float("-inf")) & (m_ij == float("-inf")), 0.0, (m_i - m_ij)))
                qk -= tl.where(m_ij == float("-inf"), 0.0, m_ij)[:, None]
            else:
                alpha = tl.exp2(m_i - m_ij)
                qk = qk - m_ij[:, None]
            p = tl.exp2(qk)
            l_i = tl.fma(l_i, alpha, tl.sum(p, 1))
            acc *= alpha[:, None]

            v = v_desc.load([start_n, 0])
            if pv_is_quantized:
                v_scale = v_scale_desc.load([start_n])[None, :]
                p *= v_scale
                if v.dtype == tl.int8:
                    p_scale = tl.max(p, 1)[:, None] * (1 / 127.0)
                    p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                    p = tl.floor(p * (1 / p_scale) + 0.5).to(tl.int8)
                    acc += tl.dot(p, v, out_dtype=tl.int32).to(tl.float32) * p_scale
                else:
                    p_scale = tl.max(p, 1)[:, None] * (1 / (65504.0 if v.dtype == tl.float16 else 448.0))
                    p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                    p = (p * (1 / p_scale)).to(v.dtype)
                    acc += tl.dot(p, v, out_dtype=tl.float32) * p_scale
            else:
                p = p.to(v.dtype)
                acc += tl.dot(p, v, out_dtype=tl.float32)
            m_i = m_ij

    l_i = 1 / l_i[:, None]
    acc *= l_i
    acc = acc.to(out_ptr.type.element_ty)
    out_desc = tl.make_tensor_descriptor(out_ptr + offset_y * OHD, shape=[ON, OHD], strides=[OHD, 1], block_shape=[BLOCK_SIZE_M, OHD])
    out_desc.store([start_m_block, 0], acc)


def quantize_attn(
    q, k, v,
    scale: float | None = None,
    smooth_k: bool = False,
    hadamard: torch.FloatTensor | None = None,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
) -> tuple[torch.Tensor]:
    if matmul_dtype in {"auto", "uint8"}:
        matmul_dtype = "int8"
    if pv_matmul_dtype == "uint8":
        pv_matmul_dtype = "int8"
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if smooth_k:
        if k.dtype != torch.float32:
            k = k.to(dtype=torch.float32)
            k = k.sub_(k.mean(dim=2, keepdim=True))
        else:
            k = k.sub(k.mean(dim=2, keepdim=True))
    if matmul_dtype not in {None, "none", "no"}:
        if hadamard is not None:
            q, use_hadamard, hadamard_group_size = apply_hadamard(q, group_size=hadamard_group_size, hadamard=hadamard, layer_class_name="Linear")
            if use_hadamard:
                k = apply_hadamard(k.to(dtype=hadamard.dtype), group_size=hadamard_group_size, hadamard=hadamard, layer_class_name="Linear")[0]
        quantize_mm_func = quantize_int_mm if matmul_dtype.startswith("int") else quantize_fp_mm
        q_q, q_scale = quantize_mm_func(q.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
        k_q, k_scale = quantize_mm_func(k.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
        q_scale = q_scale.squeeze(-1).mul_(scale * 1.4426950408889634)
        k_scale = k_scale.squeeze(-1)
    else:
        q_q = q.contiguous().mul(scale * 1.4426950408889634)
        k_q = k.contiguous().to(dtype=q.dtype)
        q_scale = None
        k_scale = None
    if pv_matmul_dtype not in {None, "auto", "none", "no"}:
        quantize_mm_func_pv = quantize_int_mm if pv_matmul_dtype.startswith("int") else quantize_fp_mm
        v_q, v_scale = quantize_mm_func_pv(v.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=pv_matmul_dtype)
        v_scale = v_scale.squeeze(-1)
    else:
        v_q = v.contiguous()
        v_scale = None
    return q_q, q_scale, k_q, k_scale, v_q, v_scale


def get_attn_inputs(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    hadamard: torch.FloatTensor | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0, # pylint: disable=unused-argument
    is_causal: bool = False, # pylint: disable=unused-argument
    scale: float | None = None,
    enable_gqa: bool = False, # pylint: disable=unused-argument
    smooth_k: bool = False,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, float, torch.dtype]:
    QZ, QH, QN, QHD = query.shape
    _, _, KN, KHD = key.shape
    _, _, _, VHD = value.shape
    if out_dtype is None:
        out_dtype = query.dtype
    if scale is None:
        scale = QHD ** -0.5
    if not math.log(QHD, 2).is_integer():
        query = torch.nn.functional.pad(query, (0, triton.next_power_of_2(QHD) - QHD))
        key = torch.nn.functional.pad(key, (0, triton.next_power_of_2(KHD) - KHD))
        value = torch.nn.functional.pad(value, (0, triton.next_power_of_2(VHD) - VHD))
    if attn_mask is not None:
        attn_mask = attn_mask.expand((QZ, QH, QN, KN))
        if not math.log(KN, 2).is_integer():
            pad_value = float("-inf") if torch.is_floating_point(attn_mask) else 0
            attn_mask = torch.nn.functional.pad(attn_mask, (0, triton.next_power_of_2(KN) - KN), value=pad_value)
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(dtype=torch.int8)
        attn_mask = attn_mask.contiguous()
    query, query_scale, key, key_scale, value, value_scale = quantize_attn(
        query, key, value,
        scale=scale,
        smooth_k=smooth_k,
        hadamard=hadamard,
        hadamard_group_size=hadamard_group_size,
        matmul_dtype=matmul_dtype if do_quantize else "no",
        pv_matmul_dtype=pv_matmul_dtype if do_quantize else "no",
    )
    return query, query_scale, key, key_scale, value, value_scale, attn_mask, scale, out_dtype


def sdnq_triton_atten(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0, # pylint: disable=unused-argument
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False, # pylint: disable=unused-argument
    smooth_k: bool = False,
    use_hadamard: bool = False,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    out_dtype: torch.dtype | None = None,
) -> torch.FloatTensor:
    QZ, QH, QN, _ = query.shape
    _, _, KN, _ = key.shape
    _, _, VN, VHD = value.shape

    if use_hadamard and matmul_dtype not in {None, "none", "no"}:
        hadamard = get_hadamard(min(hadamard_group_size, query.shape[-1], key.shape[-1]), dtype=query.dtype, device=query.device)
    else:
        hadamard = None
    (
        query, query_scale,
        key, key_scale,
        value, value_scale,
        attn_mask, scale, out_dtype,
    ) = get_attn_inputs(
        query=query, key=key, value=value,
        hadamard=hadamard, attn_mask=attn_mask,
        dropout_p=dropout_p, is_causal=is_causal,
        scale=scale, enable_gqa=enable_gqa,
        smooth_k=smooth_k, hadamard_group_size=hadamard_group_size,
        matmul_dtype=matmul_dtype, pv_matmul_dtype=pv_matmul_dtype,
        do_quantize=do_quantize, out_dtype=out_dtype,
    )
    def grid(META):
        return (triton.cdiv(QN, META["BLOCK_SIZE_M"]), QH, QZ)
    out = torch.empty((QZ, QH, QN, value.shape[-1]), dtype=out_dtype, device=query.device)
    sdnq_attn_kernel[grid](
        query, key, value,
        query_scale, key_scale, value_scale,
        out, attn_mask,
        (1 if is_causal else 0),
        (1 if attn_mask is not None else 0),
        *query.shape, *key.shape, *value.shape, *out.shape,
        *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
        math.ceil(QN / min_block_size),
        math.ceil(KN / min_block_size),
        math.ceil(VN / min_block_size),
        (1 if query_scale is not None else 0),
        (1 if value_scale is not None else 0),
        str(query.dtype), str(value.dtype), str(out.dtype),
        str(attn_mask.dtype if attn_mask is not None else None),
    )
    return out[..., :VHD]


get_attn_inputs = compile_func(get_attn_inputs, dynamic=True)
