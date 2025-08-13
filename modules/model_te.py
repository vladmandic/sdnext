import os
import json
import torch
import transformers
from safetensors.torch import load_file
from modules import shared, devices, files_cache, errors, model_quant


te_dict = {}
debug = os.environ.get('SD_LOAD_DEBUG', None) is not None
loaded_te = None


def load_t5(name=None, cache_dir=None):
    global loaded_te # pylint: disable=global-statement
    if name is None:
        return None
    cache_dir = cache_dir or shared.opts.hfcache_dir
    from modules import modelloader
    modelloader.hf_login()
    repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    if os.path.exists(name):
        fn = name
    else:
        fn = te_dict.get(name) if name in te_dict else None

    if fn is not None and name.lower().endswith('gguf'):
        from modules import ggml
        ggml.install_gguf()
        with open(os.path.join('configs', 'flux', 'text_encoder_2', 'config.json'), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        t5 = transformers.T5EncoderModel.from_pretrained(None, gguf_file=fn, config=t5_config, device_map="auto", cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif fn is not None and 'fp8' in name.lower():
        from accelerate.utils import set_module_tensor_to_device
        with open(os.path.join('configs', 'flux', 'text_encoder_2', 'config.json'), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        state_dict = load_file(fn)
        dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
        with torch.device("meta"):
            t5 = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
        for param_name, param in state_dict.items():
            is_param_float8_e4m3fn = hasattr(torch, "float8_e4m3fn") and param.dtype == torch.float8_e4m3fn
            if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
                param = param.to(devices.dtype)
                set_module_tensor_to_device(t5, param_name, device=0, value=param)
        if t5.dtype != devices.dtype:
            try:
                t5 = t5.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"T5: Failed to cast text encoder to {devices.dtype}, set dtype to {t5.dtype}")
                raise
        del state_dict

    elif fn is not None:
        with open(os.path.join('configs', 'flux', 'text_encoder_2', 'config.json'), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        state_dict = load_file(fn)
        t5 = transformers.T5EncoderModel.from_pretrained(None, state_dict=state_dict, config=t5_config, torch_dtype=devices.dtype)

    elif 'fp16' in name.lower():
        t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif 'fp4' in name.lower():
        model_quant.load_bnb('Load model: type=T5')
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
        t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', quantization_config=quantization_config, cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif 'fp8' in name.lower():
        model_quant.load_bnb('Load model: type=T5')
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', quantization_config=quantization_config, cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif 'int8' in name.lower():
        from modules.model_quant import create_sdnq_config
        quantization_config = create_sdnq_config(kwargs=None, allow=True, module='any', weights_dtype='int8')
        if quantization_config is not None:
            t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', quantization_config=quantization_config, cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif 'uint4' in name.lower():
        from modules.model_quant import create_sdnq_config
        quantization_config = create_sdnq_config(kwargs=None, allow=True, module='any', weights_dtype='uint4')
        if quantization_config is not None:
            t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', quantization_config=quantization_config, cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif 'qint4' in name.lower():
        model_quant.load_quanto('Load model: type=T5')
        quantization_config = transformers.QuantoConfig(weights='int4')
        if quantization_config is not None:
            t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', quantization_config=quantization_config, cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif 'qint8' in name.lower():
        model_quant.load_quanto('Load model: type=T5')
        quantization_config = transformers.QuantoConfig(weights='int8')
        if quantization_config is not None:
            t5 = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder='text_encoder_3', quantization_config=quantization_config, cache_dir=cache_dir, torch_dtype=devices.dtype)

    elif '/' in name:
        shared.log.debug(f'Load model: type=T5 repo={name}')
        quant_config = model_quant.create_config(module='TE')
        if quantization_config is not None:
            t5 = transformers.T5EncoderModel.from_pretrained(name, cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_config)

    else:
        t5 = None

    if t5 is not None:
        loaded_te = name
    return t5


def set_t5(pipe, module, t5=None, cache_dir=None):
    global loaded_te # pylint: disable=global-statement
    if loaded_te == shared.opts.sd_text_encoder:
        return pipe
    if pipe is None or not hasattr(pipe, module):
        return pipe
    try:
        t5 = load_t5(name=t5, cache_dir=cache_dir)
    except Exception as e:
        shared.log.error(f'Load module: type={module} class="T5" file="{shared.opts.sd_text_encoder}" {e}')
        if debug:
            errors.display(e, 'TE:')
        t5 = None
    if t5 is None:
        return pipe
    loaded_te = shared.opts.sd_text_encoder
    setattr(pipe, module, t5)
    if shared.opts.diffusers_offload_mode == "sequential":
        from accelerate import cpu_offload
        getattr(pipe, module).to("cpu")
        cpu_offload(getattr(pipe, module), devices.device, offload_buffers=len(getattr(pipe, module)._parameters) > 0) # pylint: disable=protected-access
    elif shared.opts.diffusers_offload_mode == "model":
        if not hasattr(pipe, "_all_hooks") or len(pipe._all_hooks) == 0: # pylint: disable=protected-access
            pipe.enable_model_cpu_offload(device=devices.device)
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    devices.torch_gc()
    return pipe


def load_vit_l():
    global loaded_te # pylint: disable=global-statement
    config = transformers.PretrainedConfig.from_json_file('configs/sdxl/text_encoder/config.json')
    state_dict = load_file(os.path.join(shared.opts.te_dir, f'{shared.opts.sd_text_encoder}.safetensors'))
    te = transformers.CLIPTextModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=state_dict, config=config)
    te = te.to(dtype=devices.dtype)
    loaded_te = shared.opts.sd_text_encoder
    del state_dict
    return te


def load_vit_g():
    global loaded_te # pylint: disable=global-statement
    config = transformers.PretrainedConfig.from_json_file('configs/sdxl/text_encoder_2/config.json')
    state_dict = load_file(os.path.join(shared.opts.te_dir, f'{shared.opts.sd_text_encoder}.safetensors'))
    te = transformers.CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=None, state_dict=state_dict, config=config)
    te = te.to(dtype=devices.dtype)
    loaded_te = shared.opts.sd_text_encoder
    del state_dict
    return te


def set_clip(pipe):
    if loaded_te == shared.opts.sd_text_encoder:
        return
    from modules.sd_models import move_model
    if 'vit-l' in shared.opts.sd_text_encoder.lower() and hasattr(shared.sd_model, 'text_encoder') and shared.sd_model.text_encoder.__class__.__name__ == 'CLIPTextModel':
        try:
            te = load_vit_l()
        except Exception as e:
            shared.log.error(f'Load module: type="text_encoder" class="ViT-L" file="{shared.opts.sd_text_encoder}" {e}')
            if debug:
                errors.display(e, 'TE:')
            te = None
        if te is not None:
            pipe.text_encoder = te
            shared.log.info(f'Load module: type="text_encoder" class="ViT-L" file="{shared.opts.sd_text_encoder}"')
            import modules.prompt_parser_diffusers
            modules.prompt_parser_diffusers.cache.clear()
            move_model(pipe.text_encoder, devices.device)
            devices.torch_gc()
    if 'vit-g' in shared.opts.sd_text_encoder.lower() and hasattr(shared.sd_model, 'text_encoder_2') and shared.sd_model.text_encoder_2.__class__.__name__ == 'CLIPTextModelWithProjection':
        try:
            te = load_vit_g()
        except Exception as e:
            shared.log.error(f'Load module: type module="text_encoder_2" class="ViT-G" file="{shared.opts.sd_text_encoder}" {e}')
            if debug:
                errors.display(e, 'TE:')
            te = None
        if te is not None:
            pipe.text_encoder_2 = te
            shared.log.info(f'Load module: type="text_encoder_2" class="ViT-G" file="{shared.opts.sd_text_encoder}"')
            import modules.prompt_parser_diffusers
            modules.prompt_parser_diffusers.cache.clear()
            move_model(pipe.text_encoder_2, devices.device)
            devices.torch_gc()


def refresh_te_list():
    te_dict.clear()
    for file in files_cache.list_files(shared.opts.te_dir, ext_filter=['.safetensors', '.gguf']):
        basename = os.path.basename(file)
        name = os.path.splitext(basename)[0] if '.safetensors' in basename else basename
        te_dict[name] = file
    shared.log.info(f'Available TEs: path="{shared.opts.te_dir}" items={len(te_dict)}')
