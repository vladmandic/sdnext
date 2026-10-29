"""Ideogram 4 native loader spec.

Kept import-light: only the converter and spec live here so the native
loader can import them without pulling the pipeline or the qwen patch.
"""

import re

import diffusers

from pipelines.native_transformer import TransformerSpec, split_fused_rows


QKV_TARGETS = ("to_q", "to_k", "to_v")
QKV_RE = re.compile(r"^(layers\.\d+\.attention)\.qkv\.(.+)$")
OUT_RE = re.compile(r"^(layers\.\d+\.attention)\.o\.(.+)$")


def convert_ideogram4_transformer_checkpoint(state_dict: dict, **kwargs) -> dict:  # pylint: disable=unused-argument
    """Fused community layout to ``Ideogram4Transformer2DModel`` layout.

    ``layers.N.attention.qkv.<suffix>`` splits into ``to_q/to_k/to_v.<suffix>``
    through :func:`split_fused_rows`. ``attention.o.<suffix>`` renames to
    ``attention.to_out.0.<suffix>``, everything else passes through. Returns a
    new dict; the input (which may be the cached state dict) is not mutated.
    """
    converted: dict = {}
    for key, value in state_dict.items():
        fused_match = QKV_RE.match(key)
        if fused_match is not None:
            prefix, suffix = fused_match.group(1), fused_match.group(2)
            converted.update(split_fused_rows(state_dict, f"{prefix}.qkv", suffix, tuple(f"{prefix}.{target}" for target in QKV_TARGETS)))
            continue
        out_match = OUT_RE.match(key)
        if out_match is not None:
            converted[f"{out_match.group(1)}.to_out.0.{out_match.group(2)}"] = value
            continue
        converted[key] = value
    return converted


IDEOGRAM4_SPEC = TransformerSpec(
    cls=diffusers.Ideogram4Transformer2DModel,
    converter=convert_ideogram4_transformer_checkpoint,
    converter_handles_quant=True,
)
