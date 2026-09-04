import os
import importlib.util
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae, errors
from modules.logger import log
from pipelines import generic
from pipelines.generic_map import transformers_map


def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_transformer_component(repo_id, diffusers_load_config, adapter_cls, local_file=None):
    """Load (transformer, llm_adapter_or_none).

    A UNET dropdown selection, else ``local_file`` (a single-file checkpoint),
    goes through :mod:`pipelines.native_transformer`, which also extracts a
    bundled ``llm_adapter``. Without either, the transformer comes from the base
    repo and the adapter is ``None`` for the caller to load.
    """
    from modules import sd_unet
    from pipelines import native_transformer
    override = native_transformer.resolve_path()
    local_file = override or local_file
    if local_file is not None:
        from pipelines.anima import ANIMA_SPEC
        try:
            transformer, siblings = native_transformer.load(
                local_file, repo_id, ANIMA_SPEC, diffusers_load_config,
                sibling_classes={'llm_adapter': adapter_cls},
            )
        except Exception as e:
            log.error(f'Load model: type=Anima custom transformer="{local_file}": {e}')
            errors.display(e, 'Load')
            return None, None
        if override is not None:
            sd_unet.loaded_unet = shared.opts.sd_unet
        return transformer, siblings.get('llm_adapter')
    transformer = generic.load_transformer(
        repo_id,
        cls_name=diffusers.CosmosTransformer3DModel,
        load_config=diffusers_load_config,
        subfolder="transformer"
    )
    return transformer, None


def load_anima(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    # single-file checkpoint: transformer (and bundled llm_adapter) from the file, everything else from the base repo
    local_file = None
    if repo_id is not None and os.path.isfile(repo_id) and repo_id.lower().endswith('.safetensors'):
        local_file = repo_id
        repo_id = transformers_map['AnimaTextToImagePipeline']

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    load_args.pop('cache_dir', None)
    log.debug(f'Load model: type=Anima repo="{repo_id}" file="{local_file}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    if repo_id is None or repo_id.lower() == 'none':
        return None

    import sys
    from pipelines.anima import modeling_llm_adapter
    sys.modules['modeling_llm_adapter'] = modeling_llm_adapter
    from pipelines.anima.pipeline import AnimaTextToImagePipeline
    from pipelines.anima.anima_image import build_anima_pipeline_classes
    AnimaImageToImagePipeline, AnimaInpaintPipeline = build_anima_pipeline_classes(AnimaTextToImagePipeline)
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["anima"] = AnimaTextToImagePipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["anima"] = AnimaImageToImagePipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["anima"] = AnimaInpaintPipeline
    generic.set_pipeline('Anima', AnimaTextToImagePipeline)

    # UNET dropdown or single-file checkpoint may bundle transformer and llm_adapter
    transformer, llm_adapter = init_transformer_component(repo_id, diffusers_load_config, modeling_llm_adapter.AnimaLLMAdapter, local_file=local_file)
    if transformer is None:
        return None
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.Qwen3Model,
        load_config=diffusers_load_config,
        subfolder="text_encoder"
    )

    if llm_adapter is None:
        shared.state.begin('Load adapter')
        try:
            llm_adapter = modeling_llm_adapter.AnimaLLMAdapter.from_pretrained(
                repo_id,
                subfolder="llm_adapter",
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
            )
        except Exception as e:
            log.error(f'Load model: type=Anima adapter: {e}')
            return None
        finally:
            shared.state.end()

    # assemble pipeline
    pipe = AnimaTextToImagePipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        llm_adapter=llm_adapter,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_args,
    )

    generic.load_vae_override(pipe, diffusers_load_config)

    del text_encoder
    del transformer
    del llm_adapter

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
