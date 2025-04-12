import os
import time
import transformers
import diffusers
from modules import shared, devices, sd_models, timer, model_quant, modelloader


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    # shared.log.debug(f'Hijack: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


def get_args(load_config:dict={}, module:str=None, device_map:bool=False):
    config = load_config.copy()
    modelloader.hf_login()
    if 'torch_dtype' not in config:
        config['torch_dtype'] = devices.dtype
    if 'low_cpu_mem_usage' in config:
        del config['low_cpu_mem_usage']
    if 'load_connected_pipeline' in config:
        del config['load_connected_pipeline']
    if 'safety_checker' in config:
        del config['safety_checker']
    if 'requires_safety_checker' in config:
        del config['requires_safety_checker']
    if device_map:
        if shared.opts.device_map == 'cpu':
            config['device_map'] = 'cpu'
        if shared.opts.device_map == 'gpu':
            config['device_map'] = devices.device
        if devices.backend == "ipex" and os.environ.get('UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS', '0') != '1' and module in {'TE', 'LLM'}:
            # Alchemis GPUs hits the 4GB allocation limit with transformers
            # UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS emulates above 4GB allocations
            config['device_map'] = 'cpu'
    quant_args = model_quant.create_config(module=module)
    return config, quant_args


def load_hidream(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    shared.log.debug(f'Load model: type=HiDream model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    load_args, quant_args = get_args(diffusers_load_config, module='Transformer', device_map=True)
    shared.log.debug(f'Load model: type=HiDream transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    transformer = diffusers.HiDreamImageTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none':
        transformer = transformer.to(devices.cpu)

    load_args, quant_args = get_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=HiDream te3="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder_3",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none':
        text_encoder_3 = text_encoder_3.to(devices.cpu)

    load_args, quant_args = get_args(diffusers_load_config, module='LLM', device_map=True)
    shared.log.debug(f'Load model: type=HiDream te4="{shared.opts.model_h1_llama_repo}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    tokenizer_4 = transformers.PreTrainedTokenizerFast.from_pretrained(
        shared.opts.model_h1_llama_repo,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
    )
    text_encoder_4 = transformers.LlamaForCausalLM.from_pretrained(
        shared.opts.model_h1_llama_repo,
        output_hidden_states=True,
        output_attentions=True,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none':
        text_encoder_4 = text_encoder_4.to(devices.cpu)

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
