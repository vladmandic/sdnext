"""MOSS-Image: single-stream DiT over image then text tokens, Qwen3.5 text encoder, FLUX.2 VAE."""

import re

from pipelines.moss.moss_image import MOSSImageTransformer
from pipelines.moss.moss_transformer_2d import MOSSImageTransformer2DModel
from pipelines.native_transformer import TransformerSpec, split_fused_rows


MOSS_QKV_RE = re.compile(r"^(layers\.\d+\.proj)\.qkv\.(.+)$")
MOSS_QKV_TARGETS = ('q', 'k', 'v')


def convert_moss_transformer_state_dict(state_dict: dict) -> dict:
    """Split fused outer-block QKV layers, comfy_quant sidecars included, into the transformer's q/k/v projections."""
    converted = {}
    for key, value in state_dict.items():
        match = MOSS_QKV_RE.match(key)
        if match is None:
            converted[key] = value
            continue
        prefix, suffix = match.groups()
        if suffix == 'bias':
            raise ValueError(f'MOSS outer-block fused QKV bias is unsupported: {key}')
        if suffix == 'weight' and (value.ndim != 2 or value.shape[0] % 3 != 0):
            raise ValueError(f'MOSS outer-block fused QKV weight must have three equal row groups: {key} {tuple(value.shape)}')
        converted.update(split_fused_rows(state_dict, f'{prefix}.qkv', suffix, tuple(f'{prefix}.{name}' for name in MOSS_QKV_TARGETS)))
    return converted


MOSS_SPEC = TransformerSpec(
    cls=MOSSImageTransformer,
    converter=convert_moss_transformer_state_dict,
    converter_handles_quant=True,
)

MOSS_SDNQ_SPEC = TransformerSpec(
    cls=MOSSImageTransformer2DModel
)
