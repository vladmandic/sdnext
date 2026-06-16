from modules.logger import log
from modules import shared, devices, sd_models, model_quant
from pipelines import generic


def load_segmoe(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=SegMoE repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.segmoe.segmoe_model import SegMoEPipeline
    generic.set_pipeline('SegMoE', SegMoEPipeline)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    sd_model = SegMoEPipeline(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
    sd_model = sd_model.pipe # segmoe pipe does its stuff in __init__ and __call__ is the original pipeline
    return sd_model
