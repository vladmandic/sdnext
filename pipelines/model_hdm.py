import sys
import torch
import diffusers
from modules import shared, devices, sd_models, errors


def load_hdm(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    try:
        devices.dtype = torch.float16
        diffusers_load_config['torch_dtype'] = torch.float16
        torch.set_float32_matmul_precision("high")
        from pipelines.hdm import hdm
        sys.modules['hdm'] = hdm
        from pipelines.hdm.hdm.pipeline import HDMXUTPipeline
        diffusers.HDMXUTPipeline = HDMXUTPipeline
        pipe = diffusers.HDMXUTPipeline.from_pretrained(
            repo_id,
            cache_dir=shared.opts.diffusers_dir,
            trust_remote_code=True,
            **diffusers_load_config,
        ).to(devices.device)
    except Exception as e:
        shared.log.error(f'Load HDM-XUT: path="{checkpoint_info.path}" {e}')
        errors.display(e, 'hdm')
        return None

    devices.torch_gc(force=True, reason='load')
    return pipe
