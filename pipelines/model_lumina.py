import transformers
import diffusers
from modules import shared, sd_models, sd_hijack_te, devices, model_quant
from modules.logger import log
from pipelines import generic


def load_lumina(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_config, _quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=LuminaSFT repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')

    if repo_id is None or repo_id.lower() == 'none':
        return None

    pipe = diffusers.LuminaText2ImgPipeline.from_pretrained(
        repo_id,
        cache_dir = shared.opts.diffusers_dir,
        **load_config,
    )

    generic.load_vae_override(pipe, diffusers_load_config)

    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe


def load_lumina2(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    if shared.opts.teacache_enabled:
        from modules import teacache
        log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.Lumina2Transformer2DModel.__name__}')
        diffusers.Lumina2Transformer2DModel.forward = teacache.teacache_lumina2_forward # patch must be done before transformer is loaded

    log.debug(f'Load model: type=Lumina2 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Lumina2Transformer2DModel, load_config=diffusers_load_config)
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Gemma2Model, load_config=diffusers_load_config)

    if repo_id is None or repo_id.lower() == 'none':
        return None

    load_config, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.Lumina2Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        text_encoder=text_encoder,
        transformer=transformer,
        **load_config,
    )

    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe


def load_lumina_dimoo(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_config, quant_args = model_quant.get_dit_args(diffusers_load_config)
    log.debug(f'Load model: type=LuminaDiMOO repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_config}')

    from pipelines.lumina_dimmo.pipelines import LuminaDiMOOTextPipeline, LuminaDiMOOImagePipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["luminadimoo"] = LuminaDiMOOTextPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["luminadimoo"] = LuminaDiMOOImagePipeline
    generic.set_pipeline('LuminaDiMOO', LuminaDiMOOTextPipeline)

    if repo_id is None or repo_id.lower() == 'none':
        return None

    # Force slow tokenizer path for Lumina-DiMOO to avoid fast-tokenizer conversion failures.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        use_fast=False,
    )

    pipe = LuminaDiMOOTextPipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        tokenizer=tokenizer,
        **load_config,
        **quant_args,
    )
    pipe.skip_processing = True
    pipe.task_args = {'output_type': 'np'}

    del tokenizer
    devices.torch_gc(force=True, reason='load')
    return pipe
