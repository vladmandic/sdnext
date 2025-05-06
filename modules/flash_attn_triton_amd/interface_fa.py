import torch
from modules.flash_attn_triton_amd.fwd_prefill import attention_prefill_forward_triton_impl
from modules.flash_attn_triton_amd.utils import MetaData


def fwd(q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool
):
    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    metadata.layout = "bshd"

    if causal:
        metadata.need_causal(True)

    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p)

    # check arguments
    metadata.check_args(q, k, v, out)

    # call implementation
    attention_prefill_forward_triton_impl(
                                        q,
                                        k,
                                        v,
                                        out,
                                        metadata.sm_scale,
                                        metadata.alibi_slopes,
                                        metadata.causal,
                                        None,
                                        metadata.layout,
                                        metadata.cu_seqlens_q,
                                        metadata.cu_seqlens_k,
                                        metadata.max_seqlens_q,
                                        metadata.max_seqlens_k,
                                        metadata.cache_seqlens,
                                        metadata.cache_batch_idx,
                                        metadata.dropout_p,
                                        metadata.philox_seed,
                                        metadata.philox_offset,
                                        False,
                                        metadata.use_exp2)

# varlen
