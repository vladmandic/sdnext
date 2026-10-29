"""Flux.1 native adapter loader.

Serves every pipeline sdnext types as ``f1``: FLUX.1 dev and schnell, Fill,
Control, Kontext and Flex. Their transformers share the
``FluxTransformer2DModel`` layout; Fill and Control widen ``x_embedder``, which
:func:`modules.lora.native_adapter.widen_input` covers.

Recognized layouts:

- kohya (``lora_unet_double_blocks_0_img_attn_qkv``) and BFL dotted
  (``double_blocks.0.img_attn.qkv``, bare or under ``diffusion_model.``):
  renamed to diffusers paths, fused ``qkv`` split into three targets and
  single-block ``linear1`` into Q/K/V/MLP targets at the widths of the loaded
  block; the final layer's modulation is stored (shift, scale) and read
  (scale, shift) by ``AdaLayerNormContinuous``, so its rows swap halves.
- diffusers PEFT (``transformer.``, bare diffusers paths, ``base_model.model.``),
  OneTrainer (``lora_transformer_``) and LyCORIS (``lycoris_``): already in the
  diffusers layout, bound verbatim.
- XLabs (``double_blocks.0.processor.qkv_lora1.down.weight``): rewritten to the
  BFL names before grouping; ``qkv_lora1``/``proj_lora1`` are the image stream,
  ``qkv_lora2``/``proj_lora2`` the text stream, the single-block ``qkv_lora``
  covers the Q/K/V rows of ``linear1`` and ``proj_lora`` is ``proj_out``.
- text encoders (kohya ``lora_te1_`` CLIP-L and ``lora_te3_`` T5, sdnext
  extracts' ``lora_te2_`` T5, diffusers ``text_encoder.``/``text_encoder_2.``):
  routed by encoder role through :class:`modules.lora.native_adapter.TextEncoderRouter`.

A target the loaded model lacks (the guidance embedder of a dev LoRA on
schnell) counts as unmapped.
"""

import functools
import re

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


TE_PREFIXES = ("lora_te_", "lora_te1_", "lora_te2_", "lora_te3_", "text_encoder.", "text_encoder_2.", "clip_l.", "t5.")
KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT + TE_PREFIXES

DOUBLE_TARGETS = { # underscored BFL suffix -> diffusers path under transformer_blocks.N
    "img_attn_proj": "attn.to_out.0",
    "txt_attn_proj": "attn.to_add_out",
    "img_mlp_0": "ff.net.0.proj",
    "img_mlp_2": "ff.net.2",
    "txt_mlp_0": "ff_context.net.0.proj",
    "txt_mlp_2": "ff_context.net.2",
    "img_mod_lin": "norm1.linear",
    "txt_mod_lin": "norm1_context.linear",
}
DOUBLE_QKV = {
    "img_attn_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn_qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
SINGLE_TARGETS = { # underscored BFL suffix -> diffusers path under single_transformer_blocks.N
    "linear2": "proj_out",
    "modulation_lin": "norm.linear",
}
SINGLE_LINEAR1 = ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp")
EXTRA_TARGETS = { # underscored BFL name -> (diffusers path, chunk)
    "img_in": ("x_embedder", None),
    "txt_in": ("context_embedder", None),
    "time_in_in_layer": ("time_text_embed.timestep_embedder.linear_1", None),
    "time_in_out_layer": ("time_text_embed.timestep_embedder.linear_2", None),
    "vector_in_in_layer": ("time_text_embed.text_embedder.linear_1", None),
    "vector_in_out_layer": ("time_text_embed.text_embedder.linear_2", None),
    "guidance_in_in_layer": ("time_text_embed.guidance_embedder.linear_1", None),
    "guidance_in_out_layer": ("time_text_embed.guidance_embedder.linear_2", None),
    "final_layer_linear": ("proj_out", None),
    "final_layer_adaLN_modulation_1": ("norm_out.linear", ChunkSpec(reorder=(1, 0))),
}
BLOCK = re.compile(r"^(double|single)_blocks_(\d+)_(.+)$")
XLABS = re.compile(r"^(diffusion_model\.)?(double|single)_blocks\.(\d+)\.processor\.(qkv_lora1|qkv_lora2|proj_lora1|proj_lora2|qkv_lora|proj_lora)\.(down|up)\.weight$")
XLABS_NAMES = {
    ("double", "qkv_lora1"): "img_attn.qkv",
    ("double", "qkv_lora2"): "txt_attn.qkv",
    ("double", "proj_lora1"): "img_attn.proj",
    ("double", "proj_lora2"): "txt_attn.proj",
    ("single", "qkv_lora"): "linear1_qkv", # the Q/K/V rows of linear1 only
    ("single", "proj_lora"): "linear2",
}


def xlabs_to_bfl(key):
    """BFL-named equivalent of an XLabs processor key, or the key unchanged."""
    m = XLABS.match(key)
    if m is None:
        return key
    _, kind, idx, name, side = m.groups()
    return f"{kind}_blocks.{idx}.{XLABS_NAMES[(kind, name)]}.lora_{side}.weight"


def group_by_suffixes(state_dict, suffixes, *, prefixes=KNOWN_PREFIXES):
    """:func:`native_adapter.group_by_suffixes` after renaming XLabs keys to the BFL layout."""
    if any(".processor." in k for k in state_dict):
        state_dict = {xlabs_to_bfl(k): v for k, v in state_dict.items()}
    return native_adapter.group_by_suffixes(state_dict, suffixes, prefixes=prefixes)


class Resolver:
    """Per-load target resolution; reads fused widths and text-encoder names from the loaded model."""

    def __init__(self, pipe, mapping):
        self.mapping = mapping
        self.te = native_adapter.TextEncoderRouter(pipe, mapping)

    def network_prefix(self, prefix_used):
        if prefix_used in TE_PREFIXES:
            return self.te.prefix(prefix_used) or prefix_used # an encoder the pipeline lacks: the key stays unmapped
        return "lora_transformer_"

    def width(self, path):
        module = self.mapping.get("lora_transformer_" + path.replace(".", "_"))
        shape = native_adapter.module_shape(module) if module is not None else None
        return shape[0] if shape else None

    def single_linear1(self, idx, parts):
        targets, start = [], 0
        for part in parts:
            path = f"single_transformer_blocks.{idx}.{part}"
            width = self.width(path)
            if width is None:
                return [(path, None)] # the block is missing: unmapped
            targets.append((path, ChunkSpec(start=start, end=start + width)))
            start += width
        return targets

    def bfl(self, flat):
        """Targets for an underscored BFL name; None when the name is not BFL."""
        extra = EXTRA_TARGETS.get(flat)
        if extra is not None:
            return [extra]
        m = BLOCK.match(flat)
        if m is None:
            return None
        kind, idx, suffix = m.groups()
        if kind == "double":
            if suffix in DOUBLE_QKV:
                parts = DOUBLE_QKV[suffix]
                return [(f"transformer_blocks.{idx}.{p}", ChunkSpec(idx=i, total=len(parts))) for i, p in enumerate(parts)]
            if suffix in DOUBLE_TARGETS:
                return [(f"transformer_blocks.{idx}.{DOUBLE_TARGETS[suffix]}", None)]
            return [(f"transformer_blocks.{idx}.{suffix}", None)]
        if suffix == "linear1":
            return self.single_linear1(idx, SINGLE_LINEAR1)
        if suffix == "linear1_qkv":
            return self.single_linear1(idx, SINGLE_LINEAR1[:3])
        if suffix in SINGLE_TARGETS:
            return [(f"single_transformer_blocks.{idx}.{SINGLE_TARGETS[suffix]}", None)]
        return [(f"single_transformer_blocks.{idx}.{suffix}", None)]

    def resolve_targets(self, prefix_used, base):
        if prefix_used in TE_PREFIXES:
            path = self.te.path(prefix_used, base)
            return [(path if path is not None else base.replace(".", "_"), None)]
        if prefix_used == "lora_unet_":
            return self.bfl(base) or [(base, None)]
        if prefix_used in ("diffusion_model.", native_adapter.BARE_DIFFUSERS_PREFIX_USED):
            targets = self.bfl(base.replace(".", "_"))
            if targets is not None:
                return targets
            return [(base, None)] if prefix_used == "diffusion_model." else [] # bare diffusers paths bind verbatim upstream
        return [] # transformer., lora_transformer_ and lycoris_ are already in the diffusers layout


def bind(resolver):
    return dict(
        resolve_targets=resolver.resolve_targets,
        prefixes=KNOWN_PREFIXES,
        network_prefix=resolver.network_prefix,
        group_by_suffixes_fn=group_by_suffixes,
        arch_name="f1",
    )


def try_load(name, network_on_disk, lora_scale):
    """Run every family loader against the loaded Flux.1 model and merge any that match."""
    from modules import shared
    mapping = native_adapter.resolve_mapping()
    resolver = Resolver(getattr(shared.sd_model, "pipe", shared.sd_model), mapping)
    kwargs = bind(resolver)
    loaders = (
        native_adapter.try_load_lora, native_adapter.try_load_lokr, native_adapter.try_load_loha, native_adapter.try_load_oft,
        native_adapter.try_load_ia3, native_adapter.try_load_glora, native_adapter.try_load_norm, native_adapter.try_load_full,
    )
    family_loaders = tuple(functools.partial(loader, **kwargs) for loader in loaders)
    return native_adapter.try_load_chain(name, network_on_disk, lora_scale, family_loaders=family_loaders)
