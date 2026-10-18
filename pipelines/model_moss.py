import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic
from pipelines.generic_text_encoder import check_linear_attention_kernels


def load_moss_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=MOSSImage repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.moss.moss_image import MOSSImagePipeline, MOSSImageTransformer, MOSSImageFlowMatchScheduler
    generic.set_pipeline('MOSSImage', MOSSImagePipeline)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    check_linear_attention_kernels('Qwen3_5Model') # before the transformer load, so an unsupported setup fails fast
    # the repo's model_index names classes from its remote-code file, so the pipeline is assembled here instead of through from_pretrained
    transformer = generic.load_transformer(repo_id, cls_name=MOSSImageTransformer, load_config=diffusers_load_config, modules_to_not_convert=['video_embedder', '.text_embedder.proj', 'time_embedding', 'shared_block_modulation', 'video_head_modulation', 'final_linear_video'])
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3_5Model, load_config=diffusers_load_config)
    pipe = MOSSImagePipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=transformers.Qwen2Tokenizer.from_pretrained(repo_id, subfolder='tokenizer', cache_dir=shared.opts.diffusers_dir),
        vae=diffusers.AutoencoderKLFlux2.from_pretrained(repo_id, subfolder='vae', cache_dir=shared.opts.diffusers_dir, **load_args),
        scheduler=MOSSImageFlowMatchScheduler.from_pretrained(repo_id, subfolder='scheduler', cache_dir=shared.opts.diffusers_dir),
    )
    pipe.register_to_config(_name_or_path=repo_id)
    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer, text_encoder
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
