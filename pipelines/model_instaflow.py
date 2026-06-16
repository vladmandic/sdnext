import diffusers
from modules.logger import log
from modules import shared, devices, sd_models, model_quant
from pipelines import generic


def load_instaflow(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=InstaFlow repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    if repo_id is None or repo_id.lower() == 'none':
        return None

    pipeline = diffusers.utils.get_class_from_dynamic_module('instaflow_one_step', module_file='pipeline.py')
    generic.set_pipeline('InstaFlow', pipeline)
    sd_model = pipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
    return sd_model
