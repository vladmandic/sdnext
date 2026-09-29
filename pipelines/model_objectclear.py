import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_objectclear(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=ObjectClear repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.objectclear.pipeline_objectclear import ObjectClearPipeline

    if repo_id is None or repo_id.lower() == 'none':
        return None
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        repo_id,
        apply_attention_guided_fusion=True,
        variant='fp16',
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'strength': 1.0,
    }
    pipe.no_task_switch = True
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["objectclear"] = ObjectClearPipeline

    generic.load_vae_override(pipe, diffusers_load_config)
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
