import sys
from modules import shared


def get_loader(component):
    """Return loader type for log messages."""
    if sys.platform != 'linux':
        return 'default'
    if component == 'diffusers':
        return 'runai' if shared.opts.runai_streamer_diffusers else 'default'
    return 'runai' if shared.opts.runai_streamer_transformers else 'default'


def set_pipeline(name, cls):
    """Set pipeline class in shared_items.pipelines with logging."""
    import diffusers
    from modules.logger import log
    from modules import shared_items
    if shared_items.pipelines.get(name, None) is not None:
        return
    shared_items.pipelines[name] = cls
    setattr(diffusers, name, cls)
    setattr(diffusers.pipelines, name, cls)
    log.debug(f'Pipeline: {name}={cls} initialized')
