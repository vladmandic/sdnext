"""Anima pipeline package.

Exports :data:`ANIMA_SPEC` for use by :mod:`pipelines.model_anima` together
with :mod:`pipelines.native_transformer`. The spec captures the Anima-specific
knobs that differ from the native-loader defaults:

- The bundled ``llm_adapter`` sibling: Anima community files frequently inline
  the custom AnimaLLMAdapter weights in the same safetensors as the
  transformer. The resolved adapter class is supplied at load time via
  ``sibling_classes`` because AnimaLLMAdapter is loaded dynamically through
  ``trust_remote_code`` and is not available at import time.
- Cosmos 1.0 structural marker: any community file whose state dict contains
  a Cosmos 1.0 nested key (``net.blocks.block1.*``) is rejected with a clear
  error since Anima is Cosmos 2.0 only.
- ``num_layers`` comes from the block indices in the file, so depth-expanded
  finetunes (Anima-2.9B: 40 blocks) load against the 28-layer base config.
- All other knobs (prefixes, ``acceptable_missing`` buffers) use the defaults
  from :mod:`pipelines.native_transformer`.
"""

import re

import diffusers
from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

from pipelines.native_transformer import TransformerSpec, SiblingSpec


BLOCK_INDEX = re.compile(r'^(?:transformer_)?blocks\.(\d+)\.')


def infer_config(state_dict: dict) -> dict:
    """Derive ``num_layers`` from the highest transformer block index in the file."""
    indices = [int(m.group(1)) for key in state_dict if (m := BLOCK_INDEX.match(key))]
    if not indices:
        return {}
    return {'num_layers': max(indices) + 1}


ANIMA_SPEC = TransformerSpec(
    cls=diffusers.CosmosTransformer3DModel,
    converter=convert_cosmos_transformer_checkpoint_to_diffusers,
    siblings={
        'llm_adapter': SiblingSpec(
            subfolder='llm_adapter',
            inline_prefix='llm_adapter.',
        ),
    },
    forbidden_markers=(
        (
            'net.blocks.block1.blocks.0.block.attn.to_q.0.weight',
            'unsupported Cosmos 1.0 structure',
        ),
    ),
    infer_config=infer_config,
)
