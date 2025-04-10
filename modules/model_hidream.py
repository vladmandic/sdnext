import time
import transformers
import diffusers
from modules import shared, devices, sd_models, timer


llama_repo = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    # shared.log.debug(f'Hijack: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


def get_args(diffusers_load_config={}, module=None):
    from modules import model_quant, modelloader
    modelloader.hf_login()
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = devices.dtype
    if 'low_cpu_mem_usage' in diffusers_load_config:
        del diffusers_load_config['low_cpu_mem_usage']
    if 'load_connected_pipeline' in diffusers_load_config:
        del diffusers_load_config['load_connected_pipeline']
    if 'safety_checker' in diffusers_load_config:
        del diffusers_load_config['safety_checker']
    if 'requires_safety_checker' in diffusers_load_config:
        del diffusers_load_config['requires_safety_checker']
    quant_args = model_quant.create_config(module=module)
    quant_type = model_quant.get_quant_type(quant_args)
    if quant_type:
        shared.log.debug(f'Load model: type=HiDream quantization module="{module}" {quant_type}')
    return diffusers_load_config, quant_args


def load_hidream(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    shared.log.debug(f'Load model: type=HiDream model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    load_args, quant_args = get_args(diffusers_load_config, module='Transformer')
    transformer = diffusers.HiDreamImageTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )

    load_args, quant_args = get_args(diffusers_load_config, module='TE')
    text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder_3",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )

    load_args, quant_args = get_args(diffusers_load_config, module='LLM')
    tokenizer_4 = transformers.PreTrainedTokenizerFast.from_pretrained(
        llama_repo,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
    )
    text_encoder_4 = transformers.LlamaForCausalLM.from_pretrained(
        llama_repo,
        output_hidden_states=True,
        output_attentions=True,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )

    load_args, quant_args = get_args(diffusers_load_config, module='Model')
    pipe = diffusers.HiDreamImagePipeline.from_pretrained(
        repo_id,
        text_encoder_3=text_encoder_3,
        text_encoder_4=text_encoder_4,
        tokenizer_4=tokenizer_4,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    pipe.orig_encode_prompt = pipe.encode_prompt
    pipe.encode_prompt = hijack_encode_prompt

    devices.torch_gc()
    return pipe
