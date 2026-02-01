import os
import importlib.util
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from pipelines import generic


def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_anima(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=Anima repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # resolve local path for custom pipeline modules
    local_path = sd_models.path_to_repo(checkpoint_info, local=True)
    pipeline_file = os.path.join(local_path, 'pipeline.py')
    adapter_file = os.path.join(local_path, 'llm_adapter', 'modeling_llm_adapter.py')
    if not os.path.isfile(pipeline_file):
        shared.log.error(f'Load model: type=Anima missing pipeline.py in "{local_path}"')
        return None
    if not os.path.isfile(adapter_file):
        shared.log.error(f'Load model: type=Anima missing llm_adapter/modeling_llm_adapter.py in "{local_path}"')
        return None

    # dynamically import custom classes from the model repo
    pipeline_mod = _import_from_file('anima_pipeline', pipeline_file)
    adapter_mod = _import_from_file('anima_llm_adapter', adapter_file)
    AnimaTextToImagePipeline = pipeline_mod.AnimaTextToImagePipeline
    AnimaLLMAdapter = adapter_mod.AnimaLLMAdapter

    # load components
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.CosmosTransformer3DModel, load_config=diffusers_load_config, subfolder="transformer")
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3Model, load_config=diffusers_load_config, subfolder="text_encoder", allow_shared=False)

    shared.state.begin('Load adapter')
    try:
        llm_adapter = AnimaLLMAdapter.from_pretrained(
            repo_id,
            subfolder="llm_adapter",
            cache_dir=shared.opts.diffusers_dir,
            torch_dtype=devices.dtype,
        )
    except Exception as e:
        shared.log.error(f'Load model: type=Anima adapter: {e}')
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
        **load_args,
    )

    del text_encoder
    del transformer
    del llm_adapter

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe
