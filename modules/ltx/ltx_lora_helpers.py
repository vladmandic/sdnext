"""Helpers for routing the LTX-2.x canonical stage-2 distilled LoRA through the
shared :mod:`modules.lora` pipeline.

The LTX-2.x Dev refine recipe (huggingface/diffusers#13217) composes a small
distilled LoRA on top of the base transformer for stage 2 only. Historically
this LoRA was loaded via ``pipe.load_lora_weights(repo_id) + set_adapters`` in
the stage-2 block, which (a) bypassed ``lora_force_diffusers`` (the diffusers
path ran unconditionally), (b) dropped any user-supplied LoRAs from stage 2
because ``set_adapters`` overwrote the active-adapter list, and (c) added a
third LoRA loading mechanism that diverged from every other adapter in sdnext.

This helper materializes the distilled LoRA as a synthetic
:class:`modules.lora.network.NetworkOnDisk` registered under a reserved name in
:data:`modules.lora.lora_load.available_network_aliases`. The standard
``extra_networks.activate(p, augmented_network_data)`` call then composes it
with whatever user LoRAs the prompt already activated, using whichever path
(``native`` or ``diffusers``) :func:`modules.lora.lora_overrides.get_method`
picks. Registration is idempotent (first run downloads via ``hf_hub_download``
and caches the constructed ``NetworkOnDisk``; subsequent runs reuse it).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING

from modules import shared
from modules.extra_networks import ExtraNetworkParams
from modules.logger import log
from modules.lora import lora_load, network

if TYPE_CHECKING:
    from modules.ltx.ltx_capabilities import LTXCaps


# Reserved name format; double-underscore wrapper plus an ``sdnext_`` prefix so
# the synthetic name cannot collide with a user file even if the user creates a
# LoRA whose basename happens to read ``ltx2_stage2_distilled_23``.
_NAME_TEMPLATE = "__sdnext_ltx2_stage2_distilled_{variant_slug}__"

# Standard PEFT / diffusers save filename. Both
# ``CalamitousFelicitousness/LTX-2.[03]-distilled-lora-384-Diffusers`` ship the
# weights under this name (created by ``pipe.save_lora_adapter``).
_LORA_FILENAME = "pytorch_lora_weights.safetensors"

# Process-local cache: repo_id -> resolved NetworkOnDisk. Survives across
# multiple runs in the same session so the hf_hub_download lookup and the
# NetworkOnDisk metadata/hash work only happen once.
_cache: dict[str, network.NetworkOnDisk] = {}


def synthetic_name_for(caps: "LTXCaps") -> str:
    """Return the reserved registry name for the caps' stage-2 distilled LoRA."""
    return _NAME_TEMPLATE.format(variant_slug=caps.variant.replace(".", "_"))


def ensure_registered(caps: "LTXCaps") -> str | None:
    """Download (if needed), build, and register the stage-2 distilled NetworkOnDisk.

    Returns the synthetic name on success, or ``None`` if the caps don't define
    a stage-2 distilled LoRA (Distilled variants are already at refine identity
    natively) or the download fails. Idempotent across calls.
    """
    repo_id = caps.stage2_dev_lora_repo
    if not repo_id:
        return None
    name = synthetic_name_for(caps)

    cached = _cache.get(repo_id)
    if cached is not None:
        lora_load.available_network_aliases.setdefault(name, cached)
        lora_load.available_networks.setdefault(name, cached)
        return name

    try:
        from huggingface_hub import hf_hub_download
        offline_args = {"local_files_only": True} if shared.opts.offline_mode else {}
        filename = hf_hub_download(
            repo_id=repo_id,
            filename=_LORA_FILENAME,
            cache_dir=shared.opts.hfcache_dir,
            **offline_args,
        )
    except Exception as e:
        log.error(f'LTX: stage-2 distilled LoRA download failed repo="{repo_id}" {e}')
        return None

    if not os.path.isfile(filename):
        log.error(f'LTX: stage-2 distilled LoRA file missing path="{filename}"')
        return None

    try:
        entry = network.NetworkOnDisk(name, filename)
    except Exception as e:
        log.error(f'LTX: stage-2 distilled LoRA registration failed repo="{repo_id}" {e}')
        return None

    _cache[repo_id] = entry
    lora_load.available_networks[name] = entry
    lora_load.available_network_aliases[name] = entry
    log.debug(f'LTX: registered stage-2 distilled LoRA name="{name}" repo="{repo_id}" file="{filename}"')
    return name


def augment_network_data(
    network_data: defaultdict[str, list[ExtraNetworkParams]] | None,
    synthetic_name: str,
    scale: float = 1.0,
) -> defaultdict[str, list[ExtraNetworkParams]]:
    """Return a copy of ``network_data`` with the synthetic LoRA appended.

    The original mapping is not mutated; the synthetic entry is appended so that
    any user LoRAs already in ``['lora']`` precede it in the resulting list.
    Order is semantically irrelevant for both native (per-module additive delta)
    and diffusers (PEFT additive composition) paths.
    """
    augmented: defaultdict[str, list[ExtraNetworkParams]] = defaultdict(list)
    if network_data:
        for key, params_list in network_data.items():
            augmented[key].extend(params_list)
    augmented["lora"].append(ExtraNetworkParams(items=[synthetic_name, str(scale)]))
    return augmented
