import os
import transformers
import diffusers
from huggingface_hub import repo_exists


def load_lumina(_checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader, model_quant
    modelloader.hf_login()
    load_config, _quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.LuminaText2ImgPipeline.from_pretrained(
        'Alpha-VLLM/Lumina-Next-SFT-diffusers',
        cache_dir = shared.opts.diffusers_dir,
        **load_config,
    )
    devices.torch_gc(force=True)
    return pipe


def load_lumina2(checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, sd_models, model_quant
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    if os.path.isdir(checkpoint_info.filename) and not repo_exists(repo_id):
        repo_id = checkpoint_info.filename

    if shared.opts.teacache_enabled:
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.Lumina2Transformer2DModel.__name__}')
        diffusers.Lumina2Transformer2DModel.forward = teacache.teacache_lumina2_forward # patch must be done before transformer is loaded

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='Transformer')
    transformer = diffusers.Lumina2Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    text_encoder = transformers.AutoModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.Lumina2Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        text_encoder=text_encoder,
        transformer=transformer,
        **load_config,
    )

    devices.torch_gc(force=True)
    return pipe
