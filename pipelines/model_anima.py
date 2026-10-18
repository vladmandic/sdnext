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


# diffusers AnimaTextConditioner config keys that AnimaLLMAdapter names differently; the weights are identical
TEXT_CONDITIONER_KEYS = {'num_attention_heads': 'num_heads', 'use_self_attention': 'use_self_attn', 'target_vocab_size': 'vocab_size'}


def is_modular_layout(path: str | None) -> bool:
    """True for a diffusers modular repo folder, which has modular_model_index.json and no model_index.json."""
    return path is not None and os.path.isdir(path) and os.path.isfile(os.path.join(path, 'modular_model_index.json')) and not os.path.isfile(os.path.join(path, 'model_index.json'))


def _load_text_conditioner(repo_id: str, adapter_cls):
    """Load the text_conditioner of a diffusers modular Anima repo as an AnimaLLMAdapter."""
    config = adapter_cls.load_config(repo_id, subfolder='text_conditioner', cache_dir=shared.opts.hfcache_dir)
    if config.get('use_layer_norm', False) or config.get('min_sequence_length', 512) != 512:
        raise ValueError(f'text_conditioner use_layer_norm={config.get("use_layer_norm")} min_sequence_length={config.get("min_sequence_length")} not supported')
    renamed = {new: config[old] for old, new in TEXT_CONDITIONER_KEYS.items() if old in config}
    return adapter_cls.from_pretrained(repo_id, subfolder='text_conditioner', cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype, **renamed)


def build_modular_pipeline(cls, repo_id: str, load_args: dict, **components):
    """Assemble the pipeline from a modular repo, which has no model_index.json for from_pretrained to read."""
    pipe = cls(
        tokenizer=transformers.Qwen2Tokenizer.from_pretrained(repo_id, subfolder='tokenizer', cache_dir=shared.opts.diffusers_dir),
        t5_tokenizer=transformers.T5Tokenizer.from_pretrained(repo_id, subfolder='t5_tokenizer', cache_dir=shared.opts.diffusers_dir),
        vae=diffusers.AutoencoderKLWan.from_pretrained(repo_id, subfolder='vae', cache_dir=shared.opts.diffusers_dir, **load_args),
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(repo_id, subfolder='scheduler', cache_dir=shared.opts.diffusers_dir),
        **components,
    )
    pipe.register_to_config(_name_or_path=repo_id)
    return pipe


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
    if (repo_id is not None) and os.path.isfile(repo_id) and repo_id.lower().endswith('.safetensors'):
        local_file = repo_id
        repo_id = transformers_map['AnimaTextToImagePipeline']

    modular = local_file is None and is_modular_layout(checkpoint_info.path)
    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    load_args.pop('cache_dir', None)
    log.debug(f'Load model: type=Anima repo="{repo_id}" file="{local_file}" modular={modular} config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

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

    if repo_id is None or repo_id.lower() == 'none':
        return None

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
            if modular:
                llm_adapter = _load_text_conditioner(repo_id, modeling_llm_adapter.AnimaLLMAdapter)
            else:
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
    if modular:
        pipe = build_modular_pipeline(AnimaTextToImagePipeline, repo_id, load_args, transformer=transformer, text_encoder=text_encoder, llm_adapter=llm_adapter)
    else:
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
