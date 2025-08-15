import torch
import diffusers
from modules import shared, devices, sd_hijack_te


def load_kolors(_checkpoint_info, diffusers_load_config={}):
    diffusers_load_config['variant'] = "fp16"
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = torch.float16

    repo_id = 'Kwai-Kolors/Kolors-diffusers'
    shared.log.debug(f'Load model: type=Kolors repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')
    pipe = diffusers.KolorsPipeline.from_pretrained(
        repo_id,
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    pipe.vae.config.force_upcast = True
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
