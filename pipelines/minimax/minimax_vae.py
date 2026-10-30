import torch
import diffusers
from modules.logger import log


def load_vae(repo_id: str, cache_dir: str, revision: str | None = None, offline_args: dict | None = None):
    """The video VAE through diffusers' own class at float16, whatever class the repo's config names.

    The quantized OzzyGT repos ship MiniMaxH3Fp16DecoderVae, which keeps post_quant_conv in float32 next to a float16
    decoder and relies on a float16 autocast that diffusers' decode no longer applies, so the decoder receives float32
    input. diffusers' class loaded at float16 holds the stored float16 decoder exactly and keeps its own float32 modules."""
    vae = diffusers.AutoencoderKLMiniMaxH3.from_pretrained(repo_id, subfolder='vae', torch_dtype=torch.float16, cache_dir=cache_dir, revision=revision, **(offline_args or {}))
    log.debug(f'Load model: type=MiniMaxH3 vae repo="{repo_id}" cls={vae.__class__.__name__} dtype={vae.dtype}')
    return vae
