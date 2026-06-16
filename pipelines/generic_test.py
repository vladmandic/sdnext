import os
import importlib

from scipy import stats
from installer import log


def test_pipelines():
    from modules.sd_checkpoint import CheckpointInfo

    log.info('Pipelines test...')
    pipelines = os.listdir("pipelines")
    pipelines.sort()
    for filename in pipelines:
        if filename.startswith("model_") and filename.endswith(".py"):
            module_name = filename[:-3]
            module = importlib.import_module("pipelines." + module_name)
            for attr in dir(module):
                if attr.startswith("load_"):
                    load_func = getattr(module, attr)
                    log.debug(f"Test: {module_name}.{attr}()")
                    ckpt = CheckpointInfo(filename = 'none')
                    try:
                        load_func(ckpt)
                    except Exception as e:
                        log.error(f"Error: {module_name}.{attr}(): {e}")

    log.info('Pipelines verify...')
    from modules.shared_items import get_pipelines
    pipelines = get_pipelines()
    stats_diffusers = 0
    stats_transformers = 0
    stats_custom = 0
    stats_deprecated = 0
    stats_fallback = 0
    stats_online = 0
    for name, cls in pipelines.items():
        if name == 'Autodetect' or name == 'AutoPipeline' or name == 'Diffusion' or name.startswith('ONNX'):
            continue
        elif cls is None:
            log.warning(f"Pipeline: {name} not available")
        elif cls.__name__ == 'DiffusionPipeline':
            log.warning(f"Pipeline: {name} using fallback")
        else:
            if 'deprecated.' in str(cls):
                log.warning(f"Pipeline: {name}={cls} deprecated")
                stats_deprecated += 1
            if 'diffusers.pipelines.' in str(cls):
                stats_diffusers += 1
            elif 'pipelines.' in str(cls):
                stats_custom += 1
            elif 'transformers.' in str(cls):
                stats_transformers += 1
            elif 'OnlinePipeline' in str(cls):
                stats_online += 1
            else:
                stats_fallback += 1
                log.warning(f"Pipeline: {name}={cls} not recognized")
    log.info(f"Pipelines test: diffusers={stats_diffusers} transformers={stats_transformers} custom={stats_custom} deprecated={stats_deprecated} online={stats_online} fallback={stats_fallback}")
