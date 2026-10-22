"""SD3 and SD3.5 native adapter loader.

Recognized layouts:

- kohya (``lora_unet_joint_blocks_0_context_block_attn_qkv``) and the original
  dotted names (``joint_blocks.0.x_block.attn.qkv``, bare or under
  ``diffusion_model.``): ``context_block`` carries the text tokens and maps to
  the ``add_*`` projections, ``x_block`` the image tokens and maps to
  ``to_q/k/v``; fused ``qkv`` splits into three targets. SD3.5 dual attention
  (``x_block.attn2``) maps to ``attn2``. The pre-only last block's context
  modulation and the final layer's modulation are stored (shift, scale) and read
  (scale, shift) by ``AdaLayerNormContinuous``, so their rows swap halves.
- diffusers PEFT (``transformer.``, bare diffusers paths, ``base_model.model.``),
  OneTrainer (``lora_transformer_``) and LyCORIS (``lycoris_``): bound verbatim.
- text encoders (kohya ``lora_te1_`` CLIP-L, ``lora_te2_`` CLIP-G, ``lora_te3_``
  T5; diffusers ``text_encoder.``/``text_encoder_2.``/``text_encoder_3.``):
  routed by encoder role; the T5 is not stamped, so its keys count as unmapped.
"""

import functools
import re

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


TE_PREFIXES = ("lora_te_", "lora_te1_", "lora_te2_", "lora_te3_", "text_encoder.", "text_encoder_2.", "text_encoder_3.", "clip_l.", "clip_g.", "t5.")
KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT + TE_PREFIXES

STREAM_TARGETS = { # (stream, underscored suffix) -> path under transformer_blocks.N
    ("context_block", "attn_proj"): "attn.to_add_out",
    ("context_block", "mlp_fc1"): "ff_context.net.0.proj",
    ("context_block", "mlp_fc2"): "ff_context.net.2",
    ("context_block", "adaLN_modulation_1"): "norm1_context.linear",
    ("x_block", "attn_proj"): "attn.to_out.0",
    ("x_block", "attn2_proj"): "attn2.to_out.0",
    ("x_block", "mlp_fc1"): "ff.net.0.proj",
    ("x_block", "mlp_fc2"): "ff.net.2",
    ("x_block", "adaLN_modulation_1"): "norm1.linear",
}
STREAM_QKV = {
    ("context_block", "attn_qkv"): ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
    ("x_block", "attn_qkv"): ("attn.to_q", "attn.to_k", "attn.to_v"),
    ("x_block", "attn2_qkv"): ("attn2.to_q", "attn2.to_k", "attn2.to_v"),
}
EXTRA_TARGETS = { # underscored original name -> (diffusers path, chunk)
    "x_embedder_proj": ("pos_embed.proj", None),
    "context_embedder": ("context_embedder", None),
    "t_embedder_mlp_0": ("time_text_embed.timestep_embedder.linear_1", None),
    "t_embedder_mlp_2": ("time_text_embed.timestep_embedder.linear_2", None),
    "y_embedder_mlp_0": ("time_text_embed.text_embedder.linear_1", None),
    "y_embedder_mlp_2": ("time_text_embed.text_embedder.linear_2", None),
    "final_layer_linear": ("proj_out", None),
    "final_layer_adaLN_modulation_1": ("norm_out.linear", ChunkSpec(reorder=(1, 0))),
}
BLOCK = re.compile(r"^joint_blocks_(\d+)_(context_block|x_block)_(.+)$")


class Resolver:
    """Per-load target resolution against the loaded SD3 model."""

    def __init__(self, pipe, mapping):
        self.mapping = mapping
        self.te = native_adapter.TextEncoderRouter(pipe, mapping)
        transformer = getattr(pipe, "transformer", None)
        blocks = getattr(transformer, "transformer_blocks", None) or []
        # pre-only blocks normalise their context with AdaLayerNormContinuous, which reads (scale, shift)
        self.pre_only = {i for i, block in enumerate(blocks) if type(getattr(block, "norm1_context", None)).__name__ == "AdaLayerNormContinuous"}

    def network_prefix(self, prefix_used):
        if prefix_used in TE_PREFIXES:
            return self.te.prefix(prefix_used) or prefix_used
        return "lora_transformer_"

    def original(self, flat):
        """Targets for an underscored original-layout name; None when the name is not in that layout."""
        extra = EXTRA_TARGETS.get(flat)
        if extra is not None:
            return [extra]
        m = BLOCK.match(flat)
        if m is None:
            return None
        idx, stream, suffix = m.groups()
        parts = STREAM_QKV.get((stream, suffix))
        if parts is not None:
            return [(f"transformer_blocks.{idx}.{p}", ChunkSpec(idx=i, total=len(parts))) for i, p in enumerate(parts)]
        target = STREAM_TARGETS.get((stream, suffix))
        if target is None:
            return [(f"transformer_blocks.{idx}.{stream}.{suffix}", None)] # not a module: unmapped
        chunk = ChunkSpec(reorder=(1, 0)) if (stream, suffix) == ("context_block", "adaLN_modulation_1") and int(idx) in self.pre_only else None
        return [(f"transformer_blocks.{idx}.{target}", chunk)]

    def resolve_targets(self, prefix_used, base):
        if prefix_used in TE_PREFIXES:
            path = self.te.path(prefix_used, base)
            return [(path if path is not None else base.replace(".", "_"), None)]
        if prefix_used == "lora_unet_":
            return self.original(base) or [(base, None)]
        if prefix_used in ("diffusion_model.", native_adapter.BARE_DIFFUSERS_PREFIX_USED):
            targets = self.original(base.replace(".", "_"))
            if targets is not None:
                return targets
            return [(base, None)] if prefix_used == "diffusion_model." else []
        return []


def try_load(name, network_on_disk, lora_scale):
    """Run every family loader against the loaded SD3 model and merge any that match."""
    from modules import shared
    mapping = native_adapter.resolve_mapping()
    resolver = Resolver(getattr(shared.sd_model, "pipe", shared.sd_model), mapping)
    kwargs = dict(resolve_targets=resolver.resolve_targets, prefixes=KNOWN_PREFIXES, network_prefix=resolver.network_prefix, arch_name="sd3")
    loaders = (
        native_adapter.try_load_lora, native_adapter.try_load_lokr, native_adapter.try_load_loha, native_adapter.try_load_oft,
        native_adapter.try_load_ia3, native_adapter.try_load_glora, native_adapter.try_load_norm, native_adapter.try_load_full,
    )
    return native_adapter.try_load_chain(name, network_on_disk, lora_scale, family_loaders=tuple(functools.partial(loader, **kwargs) for loader in loaders))
