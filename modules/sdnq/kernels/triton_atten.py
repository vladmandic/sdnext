import os
import math
import torch
import triton
import triton.language as tl

from ..common import compile_func # pylint: disable=relative-beyond-top-level
from ..quant_utils import quantize_int_mm, quantize_fp_mm, get_hadamard, apply_hadamard # pylint: disable=relative-beyond-top-level


matmul_configs = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in [int(w) for w in os.environ.get("SDNQ_TRITON_ATTEN_NUM_WARPS_LIST", "2,4,8,16").replace(" ","").split(",")]
    for s in [int(s) for s in os.environ.get("SDNQ_TRITON_ATTEN_NUM_STAGES_LIST", "1,2,4" if torch.version.hip else "2,4,8,16").replace(" ","").split(",")]
]


def quantize_tensor(tensor: torch.FloatTensor, group_size: int = 128, matmul_dtype: str = "int8") -> tuple[torch.Tensor, torch.FloatTensor]:
    quantize_mm_func = quantize_int_mm if matmul_dtype.startswith("int") else quantize_fp_mm
    quant_dim = tensor.shape[-2]
    if quant_dim < group_size:
        padding = group_size - quant_dim
    else:
        padding = quant_dim % group_size
        if padding != 0:
            padding = group_size - padding
    if padding != 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding), value=0)
    tensor = tensor.unflatten(-2, (-1, group_size))
    tensor, scale = quantize_mm_func(tensor.to(dtype=torch.float32), dim=(-1,-2), matmul_dtype=matmul_dtype)
    scale = scale.squeeze(-2,-1).contiguous()
    tensor = tensor.flatten(-3,-2)
    if padding != 0:
        tensor = tensor[..., :-padding, :]
    tensor = tensor.contiguous()
    return tensor, scale


def quantize_attn(
    q, k, v,
    hadamard: torch.FloatTensor | None = None,
    group_size: int = 128,
    group_size_kv: int = 32,
    scale: float | None = None,
    smooth_k: bool = False,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = "float16",
):
    if pv_matmul_dtype is None:
        pv_matmul_dtype = matmul_dtype
    if scale is None:
        scale = q.shape[-1]**-0.5
    if smooth_k:
        if k.dtype != torch.float32:
            k = k.to(dtype=torch.float32)
            k = k.sub_(k.mean(dim=2, keepdim=True))
        else:
            k = k.sub(k.mean(dim=2, keepdim=True))
    if hadamard is not None:
        q, use_hadamard, hadamard_group_size = apply_hadamard(q, group_size=group_size, hadamard=hadamard, layer_class_name="Linear")
        if use_hadamard:
            k = apply_hadamard(k.to(dtype=hadamard.dtype), group_size=hadamard_group_size, hadamard=hadamard, layer_class_name="Linear")[0]
    q_q, q_scale = quantize_tensor(q, group_size=group_size, matmul_dtype=matmul_dtype)
    k_q, k_scale = quantize_tensor(k, group_size=group_size_kv, matmul_dtype=matmul_dtype)
    v_q, v_scale = quantize_tensor(v, group_size=group_size_kv, matmul_dtype=pv_matmul_dtype)
    q_scale = q_scale.mul_(scale * 1.4426950408889634)
    return q_q, q_scale, k_q, k_scale, v_q, v_scale


@triton.autotune(configs=matmul_configs, key=["BLOCK_M", "BLOCK_N", "qz", "qh", "qn", "qhd", "q_dtype", "v_dtype", "out_dtype"], cache_results=True)
@triton.jit
def sdnq_attn_kernel(
    Q, K, V, Q_scale, K_scale, V_scale, out, mask,
    s_qz: tl.constexpr, s_qh: tl.constexpr, s_qn: tl.constexpr, s_qhd: tl.constexpr,
    s_kz: tl.constexpr, s_kh: tl.constexpr, s_kn: tl.constexpr, s_khd: tl.constexpr,
    s_vz: tl.constexpr, s_vh: tl.constexpr, s_vn: tl.constexpr, s_vhd: tl.constexpr,
    s_oz: tl.constexpr, s_oh: tl.constexpr, s_on: tl.constexpr, s_ohd: tl.constexpr,
    s_mz: tl.constexpr, s_mh: tl.constexpr, s_mqn: tl.constexpr, s_mkn: tl.constexpr,
    qz: tl.constexpr, qh: tl.constexpr, qn: tl.constexpr, qhd: tl.constexpr,
    kz: tl.constexpr, kh: tl.constexpr, kn: tl.constexpr, khd: tl.constexpr,
    vz: tl.constexpr, vh: tl.constexpr, vn: tl.constexpr, vhd: tl.constexpr,
    oz: tl.constexpr, oh: tl.constexpr, on: tl.constexpr, ohd: tl.constexpr,
    mz: tl.constexpr, mh: tl.constexpr, mqn: tl.constexpr, mkn: tl.constexpr,
    q_dtype: tl.constexpr, v_dtype: tl.constexpr, out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
): # pylint: disable=unused-argument
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    num_kv_groups = qh // vh

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, khd], dtype=tl.float32)

    Q_desc = tl.make_tensor_descriptor(Q + off_z * s_qz + off_h * s_qh, shape=[qn, qhd], strides=[s_qn, s_qhd], block_shape=[BLOCK_M, qhd])
    K_desc = tl.make_tensor_descriptor(K + off_z * s_kz + (off_h // num_kv_groups) * s_kh, shape=[kn, khd], strides=[s_kn, s_khd], block_shape=[BLOCK_N, khd])
    V_desc = tl.make_tensor_descriptor(V + off_z * s_vz + (off_h // num_kv_groups) * s_vh, shape=[vn, vhd], strides=[s_vn, s_vhd], block_shape=[BLOCK_N, vhd])

    q_scale_offset = (off_z * qh + off_h) * tl.cdiv(qn, BLOCK_M)
    k_scale_offset = (off_z * (kh // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kn, BLOCK_N)
    v_scale_offset = (off_z * (vh // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(vn, BLOCK_N)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_scale_ptr = K_scale + k_scale_offset
    V_scale_ptr = V_scale + v_scale_offset

    q = Q_desc.load([start_m * BLOCK_M, 0])
    q_scale = tl.load(Q_scale_ptr)

    if mask is not None:
        mask_desc = tl.make_tensor_descriptor(mask + (off_z * s_mz + off_h * s_mh), shape=[mqn, mkn],  strides=[s_mqn, s_mkn], block_shape=[BLOCK_M, BLOCK_N])

    lo, hi = 0, kn
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask is not None:
            mask_block = mask_desc.load([start_m * BLOCK_M, start_n])
            if mask_block.dtype == tl.int8:
                mask_block = mask_block.to(tl.int1)
            if mask_block.dtype == tl.int1 and tl.max(mask_block) == 0:
                skip = True
        if not skip:
            k = tl.trans(K_desc.load([start_n, 0]))
            k_scale = tl.load(K_scale_ptr)
            if q.dtype == tl.int8:
                qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * (q_scale * k_scale)
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32) * (q_scale * k_scale)

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = tl.where(mask_block, qk, -float('inf'))
                else:
                    qk = qk + mask_block

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            if mask_block is not None:
                m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            if mask_block is not None:
                l_ij = tl.where(l_ij == 0.0, 1.0, l_ij)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = V_desc.load([start_n, 0])
            v_scale = tl.load(V_scale_ptr)
            if v.dtype == tl.int8:
                p = tl.floor(p * 127.0 + 0.5).to(tl.int8)
                acc += tl.dot(p, v, out_dtype=tl.int32).to(tl.float32) * (v_scale / 127.0)
            else:
                p_scale = 65504.0 if v.dtype == tl.float16 else 448.0
                p = (p * p_scale).to(v.dtype)
                acc += tl.dot(p, v, out_dtype=tl.float32) * (v_scale / p_scale)
            m_i = m_ij
        K_scale_ptr += 1
        V_scale_ptr += 1

    acc = (acc / l_i[:, None]).to(out.type.element_ty)
    O_desc = tl.make_tensor_descriptor(out + off_z * s_oz + off_h * s_oh, shape=[on, ohd], strides=[s_on, s_ohd], block_shape=[BLOCK_M, ohd])
    O_desc.store([start_m * BLOCK_M, 0], acc)


@triton.autotune(configs=matmul_configs, key=["BLOCK_M", "BLOCK_N", "qz", "qh", "qn", "qhd", "q_dtype", "v_dtype", "out_dtype"], cache_results=True)
@triton.jit
def triton_attn_kernel(
    Q, K, V, out, mask,
    s_qz: tl.constexpr, s_qh: tl.constexpr, s_qn: tl.constexpr, s_qhd: tl.constexpr,
    s_kz: tl.constexpr, s_kh: tl.constexpr, s_kn: tl.constexpr, s_khd: tl.constexpr,
    s_vz: tl.constexpr, s_vh: tl.constexpr, s_vn: tl.constexpr, s_vhd: tl.constexpr,
    s_oz: tl.constexpr, s_oh: tl.constexpr, s_on: tl.constexpr, s_ohd: tl.constexpr,
    s_mz: tl.constexpr, s_mh: tl.constexpr, s_mqn: tl.constexpr, s_mkn: tl.constexpr,
    qz: tl.constexpr, qh: tl.constexpr, qn: tl.constexpr, qhd: tl.constexpr,
    kz: tl.constexpr, kh: tl.constexpr, kn: tl.constexpr, khd: tl.constexpr,
    vz: tl.constexpr, vh: tl.constexpr, vn: tl.constexpr, vhd: tl.constexpr,
    oz: tl.constexpr, oh: tl.constexpr, on: tl.constexpr, ohd: tl.constexpr,
    mz: tl.constexpr, mh: tl.constexpr, mqn: tl.constexpr, mkn: tl.constexpr,
    q_dtype: tl.constexpr, v_dtype: tl.constexpr, out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
): # pylint: disable=unused-argument
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    num_kv_groups = qh // vh

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, khd], dtype=tl.float32)

    Q_desc = tl.make_tensor_descriptor(Q + off_z * s_qz + off_h * s_qh, shape=[qn, qhd], strides=[s_qn, s_qhd], block_shape=[BLOCK_M, qhd])
    K_desc = tl.make_tensor_descriptor(K + off_z * s_kz + (off_h // num_kv_groups) * s_kh, shape=[kn, khd], strides=[s_kn, s_khd], block_shape=[BLOCK_N, khd])
    V_desc = tl.make_tensor_descriptor(V + off_z * s_vz + (off_h // num_kv_groups) * s_vh, shape=[vn, vhd], strides=[s_vn, s_vhd], block_shape=[BLOCK_N, vhd])
    if mask is not None:
        mask_desc = tl.make_tensor_descriptor(mask + (off_z * s_mz + off_h * s_mh), shape=[mqn, mkn],  strides=[s_mqn, s_mkn], block_shape=[BLOCK_M, BLOCK_N])
    q = Q_desc.load([start_m * BLOCK_M, 0])

    lo, hi = 0, kn
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask is not None:
            mask_block = mask_desc.load([start_m * BLOCK_M, start_n])
            if mask_block.dtype == tl.int8:
                mask_block = mask_block.to(tl.int1)
            if mask_block.dtype == tl.int1 and tl.max(mask_block) == 0:
                skip = True
        if not skip:
            k = tl.trans(K_desc.load([start_n, 0]))
            qk = tl.dot(q, k, out_dtype=tl.float32)

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = tl.where(mask_block, qk, -float('inf'))
                else:
                    qk = qk + mask_block

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            if mask_block is not None:
                m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            if mask_block is not None:
                l_ij = tl.where(l_ij == 0.0, 1.0, l_ij)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = V_desc.load([start_n, 0])
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_ij

    acc = (acc / l_i[:, None]).to(out.type.element_ty)
    O_desc = tl.make_tensor_descriptor(out + off_z * s_oz + off_h * s_oh, shape=[on, ohd], strides=[s_on, s_ohd], block_shape=[BLOCK_M, ohd])
    O_desc.store([start_m * BLOCK_M, 0], acc)


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
    quant_group_size: int = 128,
    quant_group_size_kv: int = 32,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = "float16",
    do_quantize: bool = True,
) -> torch.FloatTensor:
    assert not is_causal
    out_dtype = query.dtype
    qz, qh, qn, qhd = query.shape
    if not math.log(qhd, 2).is_integer():
        head_dim_pow2 = triton.next_power_of_2(qhd)
        query = torch.nn.functional.pad(query, (0, head_dim_pow2 - qhd))
        key = torch.nn.functional.pad(key, (0, head_dim_pow2 - qhd))
        value = torch.nn.functional.pad(value, (0, head_dim_pow2 - qhd))
    if scale is None:
        scale = qhd ** -0.5
    if attn_mask is not None:
        attn_mask = attn_mask.expand((qz, qh, qn, key.shape[-2])).contiguous()
        if not math.log(key.shape[-2], 2).is_integer():
            pad_value = -float('inf') if torch.is_floating_point(attn_mask) else 0
            attn_mask = torch.nn.functional.pad(attn_mask, (0, triton.next_power_of_2(key.shape[-2]) - key.shape[-2]), value=pad_value)
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(dtype=torch.int8)
    out = torch.empty(query.shape, dtype=out_dtype, device=query.device)
    grid = (triton.cdiv(qn, quant_group_size), qh, qz)
    if do_quantize:
        query, query_scale, key, key_scale, value, value_scale = quantize_attn(
            query, key, value,
            hadamard=hadamard,
            group_size=quant_group_size,
            group_size_kv=quant_group_size_kv,
            scale=scale, smooth_k=smooth_k,
            matmul_dtype=matmul_dtype,
            pv_matmul_dtype=pv_matmul_dtype,
        )
        sdnq_attn_kernel[grid](
            query, key, value, query_scale, key_scale, value_scale, out, attn_mask,
            *query.stride(), *key.stride(), *value.stride(), *out.stride(),
            *(attn_mask.stride() if attn_mask is not None else (0, 0, 0, 0)),
            *query.shape, *key.shape, *value.shape, *out.shape,
            *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
            str(query.dtype), str(value.dtype), str(out.dtype),
            BLOCK_M=quant_group_size, BLOCK_N=quant_group_size_kv,
        )
    else:
        triton_attn_kernel[grid](
            query.mul(scale * 1.4426950408889634), key, value, out, attn_mask,
            *query.stride(), *key.stride(), *value.stride(), *out.stride(),
            *(attn_mask.stride() if attn_mask is not None else (0, 0, 0, 0)),
            *query.shape, *key.shape, *value.shape, *out.shape,
            *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
            str(query.dtype), str(value.dtype), str(out.dtype),
            BLOCK_M=quant_group_size, BLOCK_N=quant_group_size_kv,
        )
    return out[..., :qhd]


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
    quant_group_size: int = 128,
    quant_group_size_kv: int = 32,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = "float16",
    do_quantize: bool = True,
) -> torch.FloatTensor:
    if use_hadamard:
        hadamard = get_hadamard(min(quant_group_size, query.shape[-1], key.shape[-1]), dtype=query.dtype, device=query.device)
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
        quant_group_size=quant_group_size,
        quant_group_size_kv=quant_group_size_kv,
        matmul_dtype=matmul_dtype,
        pv_matmul_dtype=pv_matmul_dtype,
        do_quantize=do_quantize,
    )


sdnq_triton_atten_forward = compile_func(sdnq_triton_atten_forward)
