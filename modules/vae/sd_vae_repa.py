import diffusers
from modules import shared
from modules.logger import log


models = {
    'sd': { 'repo_id': 'REPA-E/e2e-sdvae-hf', 'cls': 'AutoencoderKL' },
    'sdxl': { 'repo_id': 'REPA-E/e2e-sdvae-hf', 'cls': 'AutoencoderKL' },
    'sd3': { 'repo_id': 'REPA-E/e2e-sd3.5-vae', 'cls': 'AutoencoderKL' },
    'f1': { 'repo_id': 'REPA-E/e2e-flux-vae', 'cls': 'AutoencoderKL' },
    'qwen': { 'repo_id': 'REPA-E/e2e-qwenimage-vae', 'cls': 'AutoencoderKLQwenImage' },
}
loaded_cls = None
loaded_vae = None


def repa_load(latents):
    global loaded_cls, loaded_vae # pylint: disable=global-statement
    config = models.get(shared.sd_model_type, None)
    if config is None:
        log.error(f'Decode: type="repa" model={shared.sd_model_type} not supported')
        return latents

    cls = getattr(diffusers, config['cls'])
    if (cls != loaded_cls) or (loaded_vae is None):
        log.info(f'RePA VAE load: {config["repo_id"]} cls={config["cls"]}')
        loaded_vae = cls.from_pretrained(
            config['repo_id'],
            torch_dtype=latents.dtype,
            cache_dir=shared.opts.hfcache_dir,
            )
        loaded_cls = cls
    return loaded_vae
