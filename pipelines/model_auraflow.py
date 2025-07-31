import os
import torch
import diffusers
from modules import shared, sd_models, devices


debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_auraflow(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = torch.float16
    debug(f'Load model: type=AuraFlow repo="{repo_id}" config={diffusers_load_config}')
    pipe = diffusers.AuraFlowPipeline.from_pretrained(
        repo_id,
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    devices.torch_gc(force=True, reason='load')
    return pipe
