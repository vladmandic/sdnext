"""Flux 3 pipeline package.

Exports :data:`FLUX3_SPEC` for the native single-file loader. The BFL export
(``model.safetensors`` of the ``flux-3-image-klein`` repos and the ComfyUI
repackages) is renamed to diffusers keys by the diffusers converter, which
passes diffusers-named files through unchanged.

The step-distilled checkpoints carry a second timestep embedder
(``ref_time_in``) that only a ``flow_map`` transformer instantiates, and the
distilled blocks tree is the one that feeds it. The flag is therefore read from
the file rather than from the base repo config, so a distilled file loads
against the base repo and the other way round.
"""

import diffusers
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers

from pipelines.native_transformer import TransformerSpec


FLOW_MAP_KEYS = ('ref_time_in.in_layer.weight', 'target_timestep_embedder.linear_1.weight') # BFL and diffusers naming


def infer_config(state_dict: dict) -> dict:
    """``flow_map`` from the presence of the target-timestep embedder."""
    return {'flow_map': any(key in state_dict for key in FLOW_MAP_KEYS)}


FLUX3_SPEC = TransformerSpec(
    cls=diffusers.Flux3Transformer2DModel,
    converter=convert_flux3_transformer_checkpoint_to_diffusers,
    infer_config=infer_config,
)
