import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_mageflow(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}

    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=MageFlow repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.mageflow import MageFlowPipeline

    if repo_id is None or repo_id.lower() == 'none':
        return None

    generic.set_pipeline('MageFlow', MageFlowPipeline)
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['mageflow'] = MageFlowPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['mageflow'] = MageFlowPipeline

    pipe = MageFlowPipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    pipe.task_args = {
        'output_type': 'pil',
        'device': devices.device,
    }

    generic.load_vae_override(pipe, diffusers_load_config)

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
