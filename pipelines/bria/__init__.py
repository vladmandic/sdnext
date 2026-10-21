"""Bria pipeline package.

Exports transformer specs for use by
:mod:`pipelines.model_bria` together with :mod:`pipelines.native_transformer`.

Bria ships three transformer variants:

- The original Bria family uses a custom :class:`BriaTransformer2DModel`
  imported from :mod:`pipelines.bria.transformer_bria`. The custom class
  has no entry in diffusers' ``SINGLE_FILE_LOADABLE_CLASSES`` table.
- Bria FIBO (and FIBO Edit) use the upstream
  :class:`diffusers.BriaFiboTransformer2DModel`, which also lacks a
  ``SINGLE_FILE_LOADABLE_CLASSES`` entry.
- Bria FIBO 2.0 uses the upstream
  :class:`diffusers.BriaFibo2Transformer2DModel`. Its spec converts the
  ComfyUI single-file layout (fused ``qkv``, ``to_kv`` and ``w13`` with the
  FFN width zero-padded to a multiple of 256, ``out`` and ``q_norm`` names)
  back to the diffusers module names; diffusers-layout files pass through.

All specs use the default 3-prefix detection
(``model.diffusion_model.``, ``diffusion_model.``, ``net.``) and no siblings.
"""

import re

import diffusers

from pipelines.bria.transformer_bria import BriaTransformer2DModel
from pipelines.native_transformer import TransformerSpec


FIBO2_FUSED_QKV_RE = re.compile(r"^(.*\.attention)\.qkv\.weight$")
FIBO2_FUSED_KV_RE = re.compile(r"^(.*\.(?:cross_attention|self_attention))\.to_kv\.weight$")
FIBO2_FUSED_W13_RE = re.compile(r"^(.*\.(?:feed_forward|ffn))\.w13\.weight$")
FIBO2_W2_RE = re.compile(r"^.*\.(?:feed_forward|ffn)\.w2\.weight$")
FIBO2_RENAMES = (
    (re.compile(r"\.attention\.out\.(weight|bias)$"), r".attention.to_out.0.\1"),
    (re.compile(r"\.(cross_attention|self_attention)\.to_out\.(weight|bias)$"), r".\1.to_out.0.\2"),
    (re.compile(r"\.attention\.q_norm\."), ".attention.norm_q."),
    (re.compile(r"\.attention\.k_norm\."), ".attention.norm_k."),
)


def fibo2_ffn_width(dim: int) -> int:
    """FFN hidden width of a FIBO 2 block of width ``dim``, as BriaFibo2TransformerBlock sizes it."""
    return (dim * 8 // 3) // 64 * 64


def unpad_fibo2(value, hidden: int, dim: int, key: str):
    """Drop the zero padding past ``hidden`` along ``dim``; a padded file carries only zeros there."""
    if value.shape[dim] == hidden:
        return value
    if value.shape[dim] < hidden:
        raise ValueError(f'FIBO 2 projection is narrower than the FFN width: {key} {tuple(value.shape)} width={hidden}')
    kept, padding = value.split([hidden, value.shape[dim] - hidden], dim=dim)
    if padding.count_nonzero() != 0:
        raise ValueError(f'FIBO 2 projection padding is not zero: {key} {tuple(value.shape)} width={hidden}')
    return kept


def convert_fibo2_transformer_state_dict(state_dict: dict) -> dict:
    """Rename ComfyUI single-file keys to the diffusers layout: split fused qkv, to_kv and w13, drop the FFN padding."""
    converted = {}
    for key, value in state_dict.items():
        match = FIBO2_FUSED_QKV_RE.match(key)
        if match is not None:
            if value.ndim != 2 or value.shape[0] % 3 != 0:
                raise ValueError(f'FIBO 2 fused QKV weight must have three equal row groups: {key} {tuple(value.shape)}')
            for name, part in zip(('to_q', 'to_k', 'to_v'), value.chunk(3, dim=0)):
                converted[f'{match.group(1)}.{name}.weight'] = part
            continue
        match = FIBO2_FUSED_KV_RE.match(key)
        if match is not None:
            if value.ndim != 2 or value.shape[0] % 2 != 0:
                raise ValueError(f'FIBO 2 fused KV weight must have two equal row groups: {key} {tuple(value.shape)}')
            for name, part in zip(('to_k', 'to_v'), value.chunk(2, dim=0)):
                converted[f'{match.group(1)}.{name}.weight'] = part
            continue
        match = FIBO2_FUSED_W13_RE.match(key)
        if match is not None:
            if value.ndim != 2 or value.shape[0] % 2 != 0:
                raise ValueError(f'FIBO 2 fused W13 weight must have two equal row groups: {key} {tuple(value.shape)}')
            hidden = fibo2_ffn_width(value.shape[1])
            for name, part in zip(('w1', 'w3'), value.chunk(2, dim=0)):
                converted[f'{match.group(1)}.{name}.weight'] = unpad_fibo2(part, hidden, 0, key)
            continue
        if FIBO2_W2_RE.match(key) is not None and value.ndim == 2:
            converted[key] = unpad_fibo2(value, fibo2_ffn_width(value.shape[0]), 1, key)
            continue
        for pattern, replacement in FIBO2_RENAMES:
            key = pattern.sub(replacement, key)
        converted[key] = value
    return converted


BRIA_SPEC = TransformerSpec(cls=BriaTransformer2DModel)
BRIA_FIBO_SPEC = TransformerSpec(cls=diffusers.BriaFiboTransformer2DModel)
BRIA_FIBO2_SPEC = TransformerSpec(cls=diffusers.BriaFibo2Transformer2DModel, converter=convert_fibo2_transformer_state_dict)
