"""SD 1.x, SD 2.x and SDXL native adapter loader.

Serves every pipeline sdnext types as ``sd`` or ``sdxl``: the UNet is a
``UNet2DConditionModel`` (inpaint and pix2pix variants widen ``conv_in``, which
:func:`modules.lora.native_adapter.widen_input` covers) with one or two CLIP
text encoders.

Recognized layouts:

- kohya ``lora_unet_`` with diffusers-flat names, as sdnext's own extracts
  write: bound verbatim.
- kohya ``lora_unet_`` and ``diffusion_model_`` with CompVis block names
  (``input_blocks_4_1_proj_in``): renamed through
  :func:`modules.lora.lora_convert.make_unet_conversion_map`. The table is
  SDXL-shaped at ``output_blocks.{3i+2}.1``, which SD 1.x holds as the upsampler
  of ``up_blocks.i``; the loaded tree decides.
- kohya OFT ``oft_unet_``, LyCORIS ``lycoris_``, diffusers ``unet.`` (training
  scripts' ``.lora.down.weight`` included) and the ComfyUI ``diffusion_model.``
  dotted form.
- the 2023 diffusers attention-processor save
  (``attn1.processor.to_k_lora.down.weight``): renamed to the module before
  grouping.
- text encoders (kohya ``lora_te_``/``lora_te1_``/``lora_te2_``, diffusers
  ``text_encoder.``/``text_encoder_2.``, OMI ``clip_l.``/``clip_g.``): routed by
  encoder role.
"""

import functools
import re

from modules.lora import lora_convert, native_adapter


TE_PREFIXES = ("lora_te_", "lora_te1_", "lora_te2_", "text_encoder.", "text_encoder_2.", "clip_l.", "clip_g.")
UNET_PREFIXES = ("lora_unet_", "oft_unet_", "lycoris_", "unet.", "diffusion_model_", "diffusion_model.")
KNOWN_PREFIXES = UNET_PREFIXES + TE_PREFIXES
COMPVIS = re.compile(r"^(input_blocks|middle_block|output_blocks|time_embed|label_emb|out)_")
UPSAMPLER_SLOT = re.compile(r"^output_blocks_(\d+)_1$")
PROCESSOR = re.compile(r"(\.processor)?\.(to_q|to_k|to_v|to_out)_lora\.(down|up)\.weight$")
PROCESSOR_UNET = {"to_q": "to_q", "to_k": "to_k", "to_v": "to_v", "to_out": "to_out.0"}
PROCESSOR_TE = {"to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out": "out_proj"}


def processor_to_module(key):
    """Module-named equivalent of a 2023 attention-processor key, or the key unchanged."""
    m = PROCESSOR.search(key)
    if m is None:
        return key
    names = PROCESSOR_TE if ".self_attn." in key else PROCESSOR_UNET
    return f"{key[:m.start()]}.{names[m.group(2)]}.lora_{m.group(3)}.weight"


def group_by_suffixes(state_dict, suffixes, *, prefixes=None):
    """:func:`native_adapter.group_by_suffixes` after renaming attention-processor keys."""
    if any("_lora.down." in k for k in state_dict):
        state_dict = {processor_to_module(k): v for k, v in state_dict.items()}
    return native_adapter.group_by_suffixes(state_dict, suffixes, prefixes=prefixes or KNOWN_PREFIXES)


@functools.cache
def compvis_map():
    """CompVis underscored prefix -> diffusers underscored prefix, longest first."""
    table = lora_convert.make_unet_conversion_map()
    return sorted(table.items(), key=lambda kv: -len(kv[0]))


class Resolver:
    """Per-load target resolution against the loaded UNet and text encoders."""

    def __init__(self, pipe, mapping):
        self.mapping = mapping
        self.te = native_adapter.TextEncoderRouter(pipe, mapping)

    def network_prefix(self, prefix_used):
        if prefix_used in TE_PREFIXES:
            return self.te.prefix(prefix_used) or prefix_used
        return "lora_unet_"

    def has(self, flat):
        return "lora_unet_" + flat in self.mapping

    def from_compvis(self, flat):
        """Diffusers name for a CompVis-named module, or None when no table prefix matches."""
        for sd_prefix, hf_prefix in compvis_map():
            if flat == sd_prefix or flat.startswith(sd_prefix + "_"):
                tail = flat[len(sd_prefix):]
                m = UPSAMPLER_SLOT.match(sd_prefix)
                if m is not None and not self.has(hf_prefix + tail):
                    n = int(m.group(1))
                    upsampler = f"up_blocks_{n // 3}_upsamplers_0{tail}"
                    if n % 3 == 2 and self.has(upsampler): # SD 1.x keeps the upsampler where SDXL keeps the third attention
                        return upsampler
                return hf_prefix + tail
        return None

    def resolve_targets(self, prefix_used, base):
        if prefix_used in TE_PREFIXES:
            path = self.te.path(prefix_used, base)
            return [(path if path is not None else base.replace(".", "_"), None)]
        flat = base.replace(".", "_")
        if COMPVIS.match(flat):
            converted = self.from_compvis(flat)
            if converted is not None:
                return [(converted, None)]
        return [(flat, None)]


def try_load(name, network_on_disk, lora_scale):
    """Run every family loader against the loaded UNet pipeline and merge any that match."""
    from modules import shared
    mapping = native_adapter.resolve_mapping()
    resolver = Resolver(getattr(shared.sd_model, "pipe", shared.sd_model), mapping)
    kwargs = dict(resolve_targets=resolver.resolve_targets, prefixes=KNOWN_PREFIXES, network_prefix=resolver.network_prefix, group_by_suffixes_fn=group_by_suffixes, arch_name="sd")
    loaders = (
        native_adapter.try_load_lora, native_adapter.try_load_lokr, native_adapter.try_load_loha, native_adapter.try_load_oft,
        native_adapter.try_load_ia3, native_adapter.try_load_glora, native_adapter.try_load_norm, native_adapter.try_load_full,
    )
    return native_adapter.try_load_chain(name, network_on_disk, lora_scale, family_loaders=tuple(functools.partial(loader, **kwargs) for loader in loaders))
