import sys
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_boogu(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}

    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Boogu repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.boogu.pipeline_boogu import BooguImagePipeline
    from pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
    from pipelines.boogu.transformer_boogu import BooguImageTransformer2DModel
    from pipelines.boogu import transformer_boogu, scheduling_flow_match_euler_discrete_time_shifting
    sys.modules['transformer_boogu'] = transformer_boogu  # for loading custom code from HF repo
    sys.modules['scheduling_flow_match_euler_discrete_time_shifting'] = scheduling_flow_match_euler_discrete_time_shifting  # for loading custom code from HF repo
    scheduler = scheduling_flow_match_euler_discrete_time_shifting.FlowMatchEulerDiscreteScheduler.from_pretrained(repo_id, subfolder='scheduler', cache_dir=shared.opts.diffusers_dir)

    if repo_id is None or repo_id.lower() == 'none':
        return None

    mllm = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3VLForConditionalGeneration, load_config=diffusers_load_config, subfolder='mllm')
    transformer = generic.load_transformer(repo_id, cls_name=BooguImageTransformer2DModel, load_config=diffusers_load_config)

    if 'turbo' in repo_id.lower():
        cls = BooguImageTurboPipeline
    else:
        cls = BooguImagePipeline

    generic.set_pipeline('Boogu', cls)
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['boogu'] = cls
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['boogu'] = cls

    pipe = cls.from_pretrained(
        repo_id,
        transformer=transformer,
        mllm=mllm,
        scheduler=scheduler,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    scheduler.__class__.__name__ = 'BooguFlowMatchEulerScheduler' # its not same as normal euler
    pipe.default_scheduler = scheduler
    pipe.scheduler = scheduler

    pipe.task_args = {
        'output_type': 'np',
    }

    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer
    del mllm
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
