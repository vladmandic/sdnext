import os
import sys
import importlib.util
import transformers
import diffusers
import huggingface_hub as hf
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae, errors
from modules.logger import log
from pipelines import generic


def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_custom_transformer_path():
    """Return an absolute path if the user selected a transformer in the UNET
    dropdown and the file is resolvable, else ``None``.
    """
    sel = shared.opts.sd_unet
    if sel is None or sel in ('Default', 'None'):
        return None
    from modules import sd_unet
    if sel not in list(sd_unet.unet_dict):
        log.error(f'Load module: type=transformer file="{sel}" not found')
        return None
    path = sd_unet.unet_dict[sel]
    if not os.path.exists(path):
        log.error(f'Load module: type=transformer path="{path}" does not exist')
        return None
    return path


def load_transformer_components(repo_id, diffusers_load_config, adapter_cls):
    """Load (transformer, llm_adapter_or_none).

    If the UNET dropdown points at a valid safetensors, route through the
    custom-transformer helper, which also extracts the bundled adapter
    weights. Otherwise fall back to ``generic.load_transformer`` and return
    ``None`` for the adapter so the caller loads it from the base repo.
    """
    local_file = resolve_custom_transformer_path()
    if local_file is not None:
        from pipelines.anima import anima_transformer
        try:
            return anima_transformer.load_custom_transformer(repo_id, local_file, diffusers_load_config, adapter_cls)
        except Exception as e:
            log.error(f'Load model: type=Anima custom transformer="{local_file}": {e}')
            errors.display(e, 'Load')
            return None, None
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.CosmosTransformer3DModel, load_config=diffusers_load_config, subfolder="transformer")
    return transformer, None


def load_anima(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Anima repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # download custom pipeline modules from repo
    try:
        pipeline_file = hf.hf_hub_download(repo_id, filename='pipeline.py', cache_dir=shared.opts.diffusers_dir)
        adapter_file = hf.hf_hub_download(repo_id, filename='llm_adapter/modeling_llm_adapter.py', cache_dir=shared.opts.diffusers_dir)
    except Exception as e:
        log.error(f'Load model: type=Anima failed to download custom modules: {e}')
        return None

    # dynamically import custom classes and register in sys.modules so
    # Diffusers' from_pretrained can resolve them via trust_remote_code
    adapter_mod = _import_from_file('modeling_llm_adapter', adapter_file)
    sys.modules['modeling_llm_adapter'] = adapter_mod
    pipeline_mod = _import_from_file('pipeline', pipeline_file)
    sys.modules['pipeline'] = pipeline_mod
    AnimaTextToImagePipeline = pipeline_mod.AnimaTextToImagePipeline
    AnimaLLMAdapter = adapter_mod.AnimaLLMAdapter

    # UNET dropdown (shared.opts.sd_unet) may redirect the transformer to a
    # community file that bundles both the transformer and the llm_adapter.
    transformer, llm_adapter = load_transformer_components(repo_id, diffusers_load_config, AnimaLLMAdapter)
    if transformer is None:
        return None
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3Model, load_config=diffusers_load_config, subfolder="text_encoder", allow_shared=False)

    if llm_adapter is None:
        shared.state.begin('Load adapter')
        try:
            llm_adapter = AnimaLLMAdapter.from_pretrained(
                repo_id,
                subfolder="llm_adapter",
                cache_dir=shared.opts.diffusers_dir,
                torch_dtype=devices.dtype,
            )
        except Exception as e:
            log.error(f'Load model: type=Anima adapter: {e}')
            return None
        finally:
            shared.state.end()

    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer", cache_dir=shared.opts.diffusers_dir)
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, subfolder="t5_tokenizer", cache_dir=shared.opts.diffusers_dir)

    # assemble pipeline
    pipe = AnimaTextToImagePipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        llm_adapter=llm_adapter,
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_args,
    )

    del text_encoder
    del transformer
    del llm_adapter

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
