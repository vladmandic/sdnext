import os
from modules import shared, model_quant
from modules.logger import log


debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


def load_vae_override(pipe, load_config=None, override_cls=None, override_args={}):
    if shared.state.interrupted:
        return
    if (shared.opts.sd_vae in [None, 'None', 'Default', 'Automatic']):
        return
    if (pipe is None) or (getattr(pipe, 'vae', None) is None):
        return
    if load_config is None:
        load_config = {}

    cls = override_cls or pipe.vae.__class__
    if not hasattr(cls, 'from_single_file'):
        log.error(f'Load model: vae="{shared.opts.sd_vae}" cls={cls.__name__} safetensors=unsupported')
        return
    load_args, quant_args = model_quant.get_dit_args(load_config, module='VAE')
    log.info(f'Load model: vae="{shared.opts.sd_vae}" cls={cls.__name__} args={load_args} quant={quant_args}')
    try:
        fn = os.path.join(shared.opts.vae_dir, shared.opts.sd_vae)
        vae = cls.from_single_file(
            fn,
            cache_dir=shared.opts.hfcache_dir,
            **override_args,
            **load_args,
            **quant_args,
        )
        if vae is not None:
            pipe.vae = vae
    except Exception as e:
        log.error(f'Load model: vae="{shared.opts.sd_vae}" cls={cls.__name__} {e}')
        # errors.display(e, 'Load')
