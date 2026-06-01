import sys
from modules import shared


def get_loader(component):
    """Return loader type for log messages."""
    if sys.platform != 'linux':
        return 'default'
    if component == 'diffusers':
        return 'runai' if shared.opts.runai_streamer_diffusers else 'default'
    return 'runai' if shared.opts.runai_streamer_transformers else 'default'
