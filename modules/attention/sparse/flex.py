"""Turn a BlockSelection into the BlockMask FlexAttention consumes, and call it so the mask is honored."""
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention, _dense_to_ordered
from modules.attention.sparse.selector import BlockSelection


compiled_flex = None


def to_block_mask(selection: BlockSelection, device=None) -> BlockMask:
    """All selected tiles go in the full slots, so mask_mod is never invoked and no dense S squared mask is built."""
    keep = selection.keep
    if device is not None and keep.device != device:
        keep = keep.to(device)
    if keep.dim() != 4:
        raise ValueError(f'block selection must be 4d, got {tuple(keep.shape)}')
    # the partial slots stay empty by construction, so build them directly rather than sorting a mask of zeros
    empty_num = torch.zeros(keep.shape[:-1], dtype=torch.int32, device=keep.device)
    empty_indices = torch.zeros(keep.shape, dtype=torch.int32, device=keep.device)
    full_num, full_indices = _dense_to_ordered(keep)
    return BlockMask.from_kv_blocks(
        empty_num, empty_indices,
        full_kv_num_blocks=full_num, full_kv_indices=full_indices,
        BLOCK_SIZE=(selection.block_q, selection.block_kv),
        seq_lengths=(selection.seq_q, selection.seq_kv), # exact lengths, so a ragged tail is handled rather than rounded up
        compute_q_blocks=False, # backward only metadata, and inference never reads it
    )


def flex_call():
    """flex_attention reads the block lists only when compiled; called eagerly it evaluates mask_mod instead and a block-only mask is silently dense."""
    global compiled_flex # pylint: disable=global-statement
    if compiled_flex is None:
        compiled_flex = torch.compile(flex_attention, dynamic=False)
    return compiled_flex


def attend(query, key, value, selection: BlockSelection, scale=None, enable_gqa=False):
    return flex_call()(query, key, value, block_mask=to_block_mask(selection, device=query.device), scale=scale, enable_gqa=enable_gqa)
