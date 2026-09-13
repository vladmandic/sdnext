import diffusers
from modules import shared
from modules.logger import log


def load_nunchaku(repo_id, load_config=None):
    load_config = load_config or {}

    from modules.attention import hijack_kernels
    hijack_kernels()

    cls_name = diffusers.MiniMaxH3Transformer3DModel
    log.debug(f'Load model: transformer="{repo_id}" subfolder="calibrated-8x20" cls={cls_name.__name__} loader="nunchaku-lite" args={load_config}')
    transformer = cls_name.from_pretrained(
        repo_id,
        subfolder="calibrated-8x20",
        cache_dir=shared.opts.hfcache_dir,
        **load_config,
    )
    return transformer
