import os
import math
import torch
import triton
import triton.language as tl

from ..common import compile_func # pylint: disable=relative-beyond-top-level
from ..quant_utils import quantize_int_mm, quantize_fp_mm, get_hadamard, apply_hadamard # pylint: disable=relative-beyond-top-level


min_block_size = int(os.environ.get("SDNQ_TRITON_ATTEN_MIN_BLOCK_SIZE", "32"))
matmul_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN}, num_warps=w, num_stages=s)
    for BM in [int(BM) for BM in os.environ.get("SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST", "64").replace(" ","").split(",")]
    for BN in [int(BN) for BN in os.environ.get("SDNQ_TRITON_ATTEN_BLOCK_SIZE_N_LIST", "32").replace(" ","").split(",")]
    for w in [int(w) for w in os.environ.get("SDNQ_TRITON_ATTEN_NUM_WARPS_LIST", "2,4,8").replace(" ","").split(",")]
    for s in [int(s) for s in os.environ.get("SDNQ_TRITON_ATTEN_NUM_STAGES_LIST", "1,2,4").replace(" ","").split(",")]
]


def quantize_attn(
    q, k, v,
    hadamard: torch.FloatTensor | None = None,
    hadamard_group_size: int = 256,
    scale: float | None = None,
    smooth_k: bool = False,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
):
    if matmul_dtype == "auto":
        matmul_dtype = "int8"
    if scale is None:
        scale = q.shape[-1]**-0.5
    if smooth_k:
        if k.dtype != torch.float32:
            k = k.to(dtype=torch.float32)
            k = k.sub_(k.mean(dim=2, keepdim=True))
        else:
            k = k.sub(k.mean(dim=2, keepdim=True))
    if hadamard is not None:
        q, use_hadamard, hadamard_group_size = apply_hadamard(q, group_size=hadamard_group_size, hadamard=hadamard, layer_class_name="Linear")
        if use_hadamard:
            k = apply_hadamard(k.to(dtype=hadamard.dtype), group_size=hadamard_group_size, hadamard=hadamard, layer_class_name="Linear")[0]
    if matmul_dtype not in {None, "none", "no"}:
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


@triton.autotune(configs=matmul_configs, key=["QN_AT", "QHD", "VN_AT", "VHD", "qk_is_quantized", "pv_is_quantized", "q_dtype", "v_dtype", "out_dtype", "mask_dtype"], cache_results=True)
@triton.jit
def sdnq_attn_kernel(
    q_ptr, k_ptr, v_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, out_ptr, mask_ptr,
    QZ: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr, QHD: tl.constexpr,
    KZ: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr, KHD: tl.constexpr,
    VZ: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, VHD: tl.constexpr,
    OZ: tl.constexpr, OH: tl.constexpr, ON: tl.constexpr, OHD: tl.constexpr,
    MZ: tl.constexpr, MH: tl.constexpr, MQN: tl.constexpr, MKN: tl.constexpr,
    stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr, stride_qhd: tl.constexpr,
    stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_khd: tl.constexpr,
    stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vhd: tl.constexpr,
    stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_on: tl.constexpr, stride_ohd: tl.constexpr,
    stride_sqz: tl.constexpr, stride_sqh: tl.constexpr, stride_sqn: tl.constexpr,
    stride_skz: tl.constexpr, stride_skh: tl.constexpr, stride_skn: tl.constexpr,
    stride_svz: tl.constexpr, stride_svh: tl.constexpr, stride_svn: tl.constexpr,
    stride_mz: tl.constexpr, stride_mh: tl.constexpr, stride_mqn: tl.constexpr, stride_mkn: tl.constexpr,
    QN_AT: tl.constexpr, VN_AT: tl.constexpr,
    qk_is_quantized: tl.constexpr, pv_is_quantized: tl.constexpr,
    q_dtype: tl.constexpr, v_dtype: tl.constexpr,
    out_dtype: tl.constexpr, mask_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
): # pylint: disable=unused-argument
    start_m = tl.program_id(0)
    off_z = tl.program_id(2)
    off_h = tl.program_id(1)
    num_kv_groups = QH // VH
    off_h_kv = off_h // num_kv_groups

    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_SIZE_M, VHD], dtype=tl.float32)

    Q_desc = tl.make_tensor_descriptor(q_ptr + off_z * stride_qz + off_h * stride_qh, shape=[QN, QHD], strides=[stride_qn, stride_qhd], block_shape=[BLOCK_SIZE_M, QHD])
    K_desc = tl.make_tensor_descriptor(k_ptr + off_z * stride_kz + off_h_kv * stride_kh, shape=[KN, KHD], strides=[stride_kn, stride_khd], block_shape=[BLOCK_SIZE_N, KHD])
    V_desc = tl.make_tensor_descriptor(v_ptr + off_z * stride_vz + off_h_kv * stride_vh, shape=[VN, VHD], strides=[stride_vn, stride_vhd], block_shape=[BLOCK_SIZE_N, VHD])

    if q_scale_ptr is not None:
        q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + off_z * stride_sqz + off_h * stride_sqh, shape=[QN], strides=[stride_sqn], block_shape=[BLOCK_SIZE_M])
        k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + off_z * stride_skz + off_h_kv * stride_skh, shape=[KN], strides=[stride_skn], block_shape=[BLOCK_SIZE_N])
        q_scale = q_scale_desc.load([start_m * BLOCK_SIZE_M])[:, None]
    if v_scale_ptr is not None:
        v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + off_z * stride_svz + off_h_kv * stride_svh, shape=[VN], strides=[stride_svn], block_shape=[BLOCK_SIZE_N])
    if mask_ptr is not None:
        mask_desc = tl.make_tensor_descriptor(mask_ptr + (off_z * stride_mz + off_h * stride_mh), shape=[MQN, MKN],  strides=[stride_mqn, stride_mkn], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    q = Q_desc.load([start_m * BLOCK_SIZE_M, 0])

    lo, hi = 0, KN
    for start_n in range(lo, hi, BLOCK_SIZE_N):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)
        skip = False
        if mask_ptr is not None:
            mask = mask_desc.load([start_m * BLOCK_SIZE_M, start_n])
            if mask.dtype == tl.int8:
                mask = mask.to(tl.int1)
            if mask.dtype == tl.int1 and tl.max(mask) == 0:
                skip = True
        if not skip:
            k = tl.trans(K_desc.load([start_n, 0]))
            if q_scale_ptr is not None:
                k_scale = k_scale_desc.load([start_n])[None, :]
                if q.dtype == tl.int8:
                    qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * q_scale * k_scale
                else:
                    qk = tl.dot(q, k, out_dtype=tl.float32) * q_scale * k_scale
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32)

            if mask_ptr is not None:
                if mask.dtype == tl.int1:
                    qk = tl.where(mask, qk, -float('inf'))
                else:
                    qk = qk + mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            if mask_ptr is not None:
                m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            qk = qk - m_ij[:, None]
            p = tl.exp2(qk)
            l_ij = tl.sum(p, 1)
            if mask_ptr is not None:
                l_ij = tl.where(l_ij == 0.0, 1.0, l_ij)
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = V_desc.load([start_n, 0])
            if v_scale_ptr is not None:
                v_scale = v_scale_desc.load([start_n])[None, :]
                p *= v_scale
                if v.dtype == tl.int8:
                    p_scale = tl.max(p, 1)[:, None] / 127.0
                    p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                    p = tl.floor(p / p_scale + 0.5).to(tl.int8)
                    acc += tl.dot(p, v, out_dtype=tl.int32).to(tl.float32) * p_scale
                else:
                    p_scale = tl.max(p, 1)[:, None] / (65504.0 if v.dtype == tl.float16 else 448.0)
                    p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                    p = (p / p_scale).to(v.dtype)
                    acc += tl.dot(p, v, out_dtype=tl.float32) * p_scale
            else:
                p = p.to(v.dtype)
                acc += tl.dot(p, v, out_dtype=tl.float32)
            m_i = m_ij

    acc = (acc / l_i[:, None]).to(out_ptr.type.element_ty)
    out_desc = tl.make_tensor_descriptor(out_ptr + off_z * stride_oz + off_h * stride_oh, shape=[ON, OHD], strides=[stride_on, stride_ohd], block_shape=[BLOCK_SIZE_M, OHD])
    out_desc.store([start_m * BLOCK_SIZE_M, 0], acc)


def sdnq_triton_atten_forward(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    hadamard: torch.FloatTensor | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0, # pylint: disable=unused-argument
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False, # pylint: disable=unused-argument
    smooth_k: bool = False,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    out_dtype: torch.dtype | None = None,
) -> torch.FloatTensor:
    assert not is_causal
    if out_dtype is None:
        out_dtype = query.dtype
    QZ, QH, QN, QHD = query.shape
    _, _, KN, KHD = key.shape
    _, _, _, VHD = value.shape
    if not math.log(QHD, 2).is_integer():
        query = torch.nn.functional.pad(query, (0, triton.next_power_of_2(QHD) - QHD))
        key = torch.nn.functional.pad(key, (0, triton.next_power_of_2(KHD) - KHD))
        value = torch.nn.functional.pad(value, (0, triton.next_power_of_2(VHD) - VHD))
    if scale is None:
        scale = QHD ** -0.5
    if attn_mask is not None:
        attn_mask = attn_mask.expand((QZ, QH, QN, KN)).contiguous()
        if not math.log(KN, 2).is_integer():
            pad_value = -float('inf') if torch.is_floating_point(attn_mask) else 0
            attn_mask = torch.nn.functional.pad(attn_mask, (0, triton.next_power_of_2(KN) - KN), value=pad_value)
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(dtype=torch.int8)
    def grid(META):
        return (triton.cdiv(QN, META["BLOCK_SIZE_M"]), QH, QZ)
    out = torch.empty((QZ, QH, QN, value.shape[-1]), dtype=out_dtype, device=query.device)
    query, query_scale, key, key_scale, value, value_scale = quantize_attn(
        query, key, value,
        hadamard=hadamard,
        hadamard_group_size=hadamard_group_size,
        scale=scale, smooth_k=smooth_k,
        matmul_dtype=matmul_dtype if do_quantize else "no",
        pv_matmul_dtype=pv_matmul_dtype if do_quantize else "no",
    )
    sdnq_attn_kernel[grid](
        query, key, value, query_scale, key_scale, value_scale, out, attn_mask,
        *query.shape, *key.shape, *value.shape, *out.shape,
        *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
        *query.stride(), *key.stride(), *value.stride(), *out.stride(),
        *(query_scale.stride() if query_scale is not None else (0, 0, 0)),
        *(key_scale.stride() if key_scale is not None else (0, 0, 0)),
        *(value_scale.stride() if value_scale is not None else (0, 0, 0)),
        *(attn_mask.stride() if attn_mask is not None else (0, 0, 0, 0)),
        math.ceil(QN / min_block_size), math.ceil(value.shape[-2] / min_block_size),
        bool(query_scale is not None), bool(value_scale is not None),
        str(query.dtype), str(value.dtype), str(out.dtype),
        str(attn_mask.dtype if attn_mask is not None else None),
    )
    return out[..., :VHD]


def sdnq_triton_atten(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    smooth_k: bool = False,
    use_hadamard: bool = False,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    out_dtype: torch.dtype | None = None,
) -> torch.FloatTensor:
    if use_hadamard:
        hadamard = get_hadamard(min(hadamard_group_size, query.shape[-1], key.shape[-1]), dtype=query.dtype, device=query.device)
    else:
        hadamard = None
    return sdnq_triton_atten_forward(
        query, key, value,
        hadamard=hadamard,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        smooth_k=smooth_k,
        hadamard_group_size=hadamard_group_size,
        matmul_dtype=matmul_dtype,
        pv_matmul_dtype=pv_matmul_dtype,
        do_quantize=do_quantize,
        out_dtype=out_dtype,
    )


sdnq_triton_atten_forward = compile_func(sdnq_triton_atten_forward)
