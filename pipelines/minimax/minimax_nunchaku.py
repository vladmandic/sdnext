import diffusers
from modules import shared, devices
from modules.logger import log


def load_nunchaku(repo_id, load_config=None):
    load_config = load_config or {}

    from modules.attention import hijack_kernels
    hijack_kernels()

    ## naive approach using diffusers class
    cls_name = diffusers.MiniMaxH3Transformer3DModel
    log.debug(f'Load model: transformer="{repo_id}" subfolder="calibrated-8x20" cls={cls_name.__name__} loader="nunchaku-lite" patch=False args={load_config}')
    transformer = cls_name.from_pretrained(
        repo_id,
        subfolder="calibrated-8x20",
        cache_dir=shared.opts.hfcache_dir,
        **load_config,
    )
    return transformer


def load_nunchaku_patched(repo_id, load_config=None): # pylint: disable=unused-argument
    import torch
    from huggingface_hub import hf_hub_download
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    from modules.attention import hijack_kernels
    hijack_kernels()

    from installer import install
    install('git+https://github.com/rootonchair/nunchaku-lite', 'nunchaku-lite')
    from nunchaku_lite import core

    filename = "svdq-int4_r32-minimax-h3-t2va.safetensors"
    orig_repo = "MiniMaxAI/MiniMax-H3"
    cls_name = diffusers.MiniMaxH3Transformer3DModel

    log.debug(f'Load model: transformer="{repo_id}" fn="{filename}" cls={cls_name.__name__} loader="nunchaku-lite" patch=True args={load_config}')
    ckpt = hf_hub_download(repo_id,
                           filename=filename,
                           cache_dir=shared.opts.diffusers_dir
                          )
    config = diffusers.MiniMaxH3Transformer3DModel.load_config(orig_repo, subfolder="transformer")
    with torch.device("meta"):
        transformer: diffusers.MiniMaxH3Transformer3DModel = diffusers.MiniMaxH3Transformer3DModel.from_config(config)

    # rope.inv_freq is a non-persistent buffer (absent from the checkpoint), so the meta-assign load would leave it on the meta device — rebuild it off-meta first.
    transformer.rope.inv_freq = MiniMaxH3RotaryPosEmbed(rope_freq_dim=transformer.config.rope_freq_dim, rope_theta=transformer.config.rope_theta).inv_freq # pylint: disable=no-member
    core._patch_component(transformer, # pylint: disable=protected-access
                          ckpt,
                          target="manifest",
                          precision="int4",
                          torch_dtype=devices.dtype,
                          device=devices.device,
                          strict=True,
                          adapter_options=None,
                          assign=True,
                         )
    return transformer
