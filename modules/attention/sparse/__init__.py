"""Block-sparse attention: the selector, the token layout it respects, and the consumers that apply it."""
from modules.attention.sparse.selector import BlockSelection, SparseSpec, block_count, radial_blocks, schedule, select_blocks
from modules.attention.sparse.layout import PACKINGS, Packing, Span, Stream, TokenLayout, block_pins, layout_from_index_kwargs, layout_from_prefix, layout_from_segments, layouts_from_kwargs, layouts_from_packing, publish_segments, segments_from_live

__all__ = [
    'BlockSelection', 'SparseSpec', 'block_count', 'radial_blocks', 'schedule', 'select_blocks',
    'PACKINGS', 'Packing', 'Span', 'Stream', 'TokenLayout', 'block_pins', 'layout_from_index_kwargs', 'layout_from_prefix', 'layout_from_segments', 'layouts_from_kwargs', 'layouts_from_packing', 'publish_segments', 'segments_from_live',
]
