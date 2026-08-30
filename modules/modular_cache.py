from modules import processing
from modules.logger import log


def set_cache(p: processing.StableDiffusionProcessing, phase: str | None = None): # pylint: disable=unused-argument
    import modules.ui_cache
    inputs = modules.ui_cache.get_modular_args()
    method = inputs.get('cache_method', 'None')
    if method == 'None':
        return
    args = {}
    log.debug(f'Pipeline: cache={method} args={args}')
