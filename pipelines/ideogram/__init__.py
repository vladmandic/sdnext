"""Ideogram 4 native loader spec.

Kept import-light: only the converter and spec live here so the native
loader can import them without pulling the pipeline or the qwen patch.
"""

import re

import torch
import diffusers

from pipelines.native_transformer import TransformerSpec


QKV_TARGETS = ("to_q", "to_k", "to_v")
QKV_RE = re.compile(r"^(layers\.\d+\.attention)\.qkv\.(.+)$")
OUT_RE = re.compile(r"^(layers\.\d+\.attention)\.o\.(.+)$")


def convert_ideogram4_transformer_checkpoint(state_dict: dict, **kwargs) -> dict:  # pylint: disable=unused-argument
    """Fused community layout to ``Ideogram4Transformer2DModel`` layout.

    ``layers.N.attention.qkv.<suffix>`` splits into ``to_q/to_k/to_v.<suffix>``:
    row-stacked tensors (weight, row-wise scale, bias) are dim-0 sliced into
    thirds, non-row-stacked sidecars (markers, scalar scales) copied verbatim
    to all three. ``attention.o.<suffix>`` renames to ``attention.to_out.0.<suffix>``,
    everything else passes through. Returns a new dict; the input (which may
    be the cached state dict) is not mutated.
    """
    converted: dict = {}
    for key, value in state_dict.items():
        fused_match = QKV_RE.match(key)
        if fused_match is not None:
            prefix, suffix = fused_match.group(1), fused_match.group(2)
            fused_weight = state_dict.get(f"{prefix}.qkv.weight")
            fused_rows = fused_weight.shape[0] if torch.is_tensor(fused_weight) and fused_weight.ndim >= 1 else None
            row_stacked = (
                suffix != "comfy_quant"
                and torch.is_tensor(value) and value.ndim >= 1
                and fused_rows is not None and value.shape[0] == fused_rows and fused_rows % 3 == 0
            )
            if row_stacked:
                third = fused_rows // 3
                for i, target in enumerate(QKV_TARGETS):
                    converted[f"{prefix}.{target}.{suffix}"] = value[i * third:(i + 1) * third]
            else:
                for target in QKV_TARGETS:
                    converted[f"{prefix}.{target}.{suffix}"] = value
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
