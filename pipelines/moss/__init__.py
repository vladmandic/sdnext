"""MOSS-Image: single-stream DiT over image then text tokens, Qwen3.5 text encoder, FLUX.2 VAE."""

import re

from pipelines.moss.moss_image import MOSSImageTransformer
from pipelines.native_transformer import TransformerSpec


MOSS_QKV_RE = re.compile(r"^(layers\.\d+)\.proj\.qkv\.(weight|bias)$")


def convert_moss_transformer_state_dict(state_dict: dict) -> dict:
    """Split fused outer-block QKV weights into the transformer's q/k/v projections."""
    converted = {}
    for key, value in state_dict.items():
        match = MOSS_QKV_RE.match(key)
        if match is None:
            converted[key] = value
            continue
        prefix, suffix = match.groups()
        if suffix != 'weight':
            raise ValueError(f'MOSS outer-block fused QKV bias is unsupported: {key}')
        if value.ndim != 2 or value.shape[0] % 3 != 0:
            raise ValueError(f'MOSS outer-block fused QKV weight must have three equal row groups: {key} {tuple(value.shape)}')
        query, key_weight, value_weight = value.chunk(3, dim=0)
        converted[f'{prefix}.proj.q.weight'] = query
        converted[f'{prefix}.proj.k.weight'] = key_weight
        converted[f'{prefix}.proj.v.weight'] = value_weight
    return converted


MOSS_SPEC = TransformerSpec(
    cls=MOSSImageTransformer,
    converter=convert_moss_transformer_state_dict,
)
