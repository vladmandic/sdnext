import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_step1x_edit(checkpoint_info, diffusers_load_config=None):
    from pipelines.step1x.pipeline_step1x_edit import Step1XEditPipeline
    from pipelines.step1x.transformer_step1x_edit import Step1XEditTransformer2DModel

    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Step1XEdit repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    diffusers.Step1XEditPipeline = Step1XEditPipeline
    diffusers.Step1XEditTransformer2DModel = Step1XEditTransformer2DModel

    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config)
    processor = transformers.Qwen2_5_VLProcessor.from_pretrained(repo_id, cache_dir=shared.opts.hfcache_dir, subfolder='processor')
    transformer = generic.load_transformer(repo_id, cls_name=Step1XEditTransformer2DModel, load_config=diffusers_load_config)

    pipe = Step1XEditPipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        processor=processor,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'pil', # step1x is buggy with np
    }

    del text_encoder
    del processor
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
