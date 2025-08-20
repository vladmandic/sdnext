import os
import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from pipelines import generic


def load_flux(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    if 'Fill' in repo_id:
        cls_name = diffusers.FluxFillPipeline
    elif 'Canny' in repo_id:
        cls_name = diffusers.FluxControlPipeline
    elif 'Depth' in repo_id:
        cls_name = diffusers.FluxControlPipeline
    elif 'Kontext' in repo_id:
        cls_name = diffusers.FluxKontextPipeline
        diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["flux1kontext"] = diffusers.FluxKontextPipeline
        diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["flux1kontext"] = diffusers.FluxKontextPipeline
        diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["flux1kontext"] = diffusers.FluxKontextInpaintPipeline
    else:
        cls_name = diffusers.FluxPipeline

    from pipelines.flux import flux_lora
    flux_lora.apply_patch()

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=Flux repo="{repo_id}" cls={cls_name.__name__} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # optional teacache patch
    if shared.opts.teacache_enabled and not model_quant.check_nunchaku('Model'):
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.FluxTransformer2DModel.__name__}')
        diffusers.FluxTransformer2DModel.forward = teacache.teacache_flux_forward # patch must be done before transformer is loaded

    transformer = None
    text_encoder_2 = None

    # handle transformer svdquant if available, t5 is handled inside load_text_encoder
    prequantized = model_quant.get_quant(checkpoint_info.path)
    if model_quant.check_nunchaku('Model'):
        from pipelines.flux.flux_nunchaku import load_flux_nunchaku
        transformer = load_flux_nunchaku(repo_id)
    # handle prequantized models
    elif prequantized == 'nf4':
        from pipelines.flux.flux_nf4 import load_flux_nf4
        transformer, text_encoder_2 = load_flux_nf4(checkpoint_info)
    elif prequantized == 'qint8' or prequantized == 'qint4':
        from pipelines.flux.flux_quanto import load_flux_quanto
        transformer, text_encoder_2 = load_flux_quanto(checkpoint_info)
    elif prequantized == 'fp4' or prequantized == 'fp8':
        from pipelines.flux.flux_bnb import load_flux_bnb
        transformer = load_flux_bnb(checkpoint_info, diffusers_load_config)

    # finally load transformer and text encoder if not already loaded
    if transformer is None:
        transformer = generic.load_transformer(repo_id, cls_name=diffusers.FluxTransformer2DModel, load_config=diffusers_load_config)
    if text_encoder_2 is None:
        text_encoder_2 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config)

    pipe = cls_name.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    if os.environ.get('SD_REMOTE_T5', None) is not None:
        from modules import sd_te_remote
        shared.log.warning('Remote-TE: applying patch')
        pipe._get_t5_prompt_embeds = sd_te_remote.get_t5_prompt_embeds # pylint: disable=protected-access
        pipe.text_encoder_2 = None

    del text_encoder_2
    del transformer

    # optional first-block patch
    if shared.opts.teacache_enabled and model_quant.check_nunchaku('Model'):
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
        apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)

    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
