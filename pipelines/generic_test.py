import os
import importlib
from installer import log


def test_pipelines():
    from modules.sd_checkpoint import CheckpointInfo
    log.info('Test pipelines...')
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
                    load_func(ckpt)
