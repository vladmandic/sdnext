import os
import time
import transformers
import diffusers
from modules import shared, devices, sd_models, timer, model_quant, modelloader


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    if 'max_sequence_length' in kwargs:
        kwargs['max_sequence_length'] = os.environ.get('HIDREAM_MAX_SEQUENCE_LENGTH', 256)
    res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    # shared.log.debug(f'Hijack: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


def load_hidream(checkpoint_info, diffusers_load_config={}):
    login = modelloader.hf_login()
    repo_id = sd_models.path_to_repo(checkpoint_info.name)

    from huggingface_hub import auth_check
    try:
        auth_check(shared.opts.model_h1_llama_repo)
    except Exception as e:
        shared.log.error(f'Load model: type=HiDream te4="{shared.opts.model_h1_llama_repo}" login={login} {e}')
        return False

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Transformer', device_map=True)
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

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
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

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='LLM', device_map=True)
    shared.log.debug(f'Load model: type=HiDream te4="{shared.opts.model_h1_llama_repo}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')

    text_encoder_4 = transformers.LlamaForCausalLM.from_pretrained(
        shared.opts.model_h1_llama_repo,
        output_hidden_states=True,
        output_attentions=True,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    tokenizer_4 = transformers.PreTrainedTokenizerFast.from_pretrained(
        shared.opts.model_h1_llama_repo,
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
    )
    if shared.opts.diffusers_offload_mode != 'none':
        text_encoder_4 = text_encoder_4.to(devices.cpu)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=HiDream model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')
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
