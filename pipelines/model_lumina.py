import transformers
import diffusers
from modules import shared, sd_models, sd_hijack_te, devices, modelloader, model_quant
from pipelines import generic


def load_lumina(_checkpoint_info, diffusers_load_config={}):
    modelloader.hf_login()
    load_config, _quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.LuminaText2ImgPipeline.from_pretrained(
        'Alpha-VLLM/Lumina-Next-SFT-diffusers',
        cache_dir = shared.opts.diffusers_dir,
        **load_config,
    )
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe


def load_lumina2(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)

    if shared.opts.teacache_enabled:
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.Lumina2Transformer2DModel.__name__}')
        diffusers.Lumina2Transformer2DModel.forward = teacache.teacache_lumina2_forward # patch must be done before transformer is loaded

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Lumina2Transformer2DModel, load_config=diffusers_load_config)
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Gemma2Model, load_config=diffusers_load_config)

    load_config, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.Lumina2Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        text_encoder=text_encoder,
        transformer=transformer,
        **load_config,
    )

    del transformer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
