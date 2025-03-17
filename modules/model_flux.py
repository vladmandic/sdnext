import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules import shared, errors, devices, modelloader, sd_models, sd_unet, model_te, model_quant


debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_flux_quanto(checkpoint_info):
    transformer, text_encoder_2 = None, None
    quanto = model_quant.load_quanto('Load model: type=FLUX')
    quanto.tensor.qbits.QBitsTensor.create = lambda *args, **kwargs: quanto.tensor.qbits.QBitsTensor(*args, **kwargs)

    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path

    try:
        quantization_map = os.path.join(repo_path, "transformer", "quantization_map.json")
        debug(f'Load model: type=FLUX quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="transformer"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info.name)
            quantization_map = hf_hub_download(repo_id, subfolder='transformer', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        state_dict = load_file(os.path.join(repo_path, "transformer", "diffusion_pytorch_model.safetensors"))
        dtype = state_dict['context_embedder.bias'].dtype
        with torch.device("meta"):
            transformer = diffusers.FluxTransformer2DModel.from_config(os.path.join(repo_path, "transformer", "config.json")).to(dtype=dtype)
        quanto.requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
        if shared.opts.diffusers_eval:
            transformer.eval()
        if transformer.dtype != devices.dtype:
            try:
                transformer = transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=FLUX Failed to cast transformer to {devices.dtype}, set dtype to {transformer.dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=FLUX failed to load Quanto transformer: {e}")
        if debug:
            errors.display(e, 'FLUX Quanto:')

    try:
        quantization_map = os.path.join(repo_path, "text_encoder_2", "quantization_map.json")
        debug(f'Load model: type=FLUX quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="text_encoder_2"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info.name)
            quantization_map = hf_hub_download(repo_id, subfolder='text_encoder_2', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        with open(os.path.join(repo_path, "text_encoder_2", "config.json"), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        state_dict = load_file(os.path.join(repo_path, "text_encoder_2", "model.safetensors"))
        dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
        with torch.device("meta"):
            text_encoder_2 = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
        quanto.requantize(text_encoder_2, state_dict, quantization_map, device=torch.device("cpu"))
        if shared.opts.diffusers_eval:
            text_encoder_2.eval()
        if text_encoder_2.dtype != devices.dtype:
            try:
                text_encoder_2 = text_encoder_2.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=FLUX Failed to cast text encoder to {devices.dtype}, set dtype to {text_encoder_2.dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=FLUX failed to load Quanto text encoder: {e}")
        if debug:
            errors.display(e, 'FLUX Quanto:')

    return transformer, text_encoder_2


def load_flux_bnb(checkpoint_info, diffusers_load_config): # pylint: disable=unused-argument
    transformer, text_encoder_2 = None, None
    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path
    model_quant.load_bnb('Load model: type=FLUX')
    quant = model_quant.get_quant(repo_path)
    try:
        if quant == 'fp8':
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=devices.dtype)
            debug(f'Quantization: {quantization_config}')
            transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
        elif quant == 'fp4':
            quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=devices.dtype, bnb_4bit_quant_type= 'fp4')
            debug(f'Quantization: {quantization_config}')
            transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
        elif quant == 'nf4':
            quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=devices.dtype, bnb_4bit_quant_type= 'nf4')
            debug(f'Quantization: {quantization_config}')
            transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
        else:
            transformer = diffusers.FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config)
    except Exception as e:
        shared.log.error(f"Load model: type=FLUX failed to load BnB transformer: {e}")
        transformer, text_encoder_2 = None, None
        if debug:
            errors.display(e, 'FLUX:')
    return transformer, text_encoder_2


def load_quants(kwargs, repo_id, cache_dir, allow_quant):
    try:
        quant_args = model_quant.create_config(allow=allow_quant)
        if not quant_args:
            return kwargs
        if 'transformer' not in kwargs and ('Model' in shared.opts.bnb_quantization or 'Model' in shared.opts.torchao_quantization or 'Model' in shared.opts.quanto_quantization):
            kwargs['transformer'] = diffusers.FluxTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
        quant_args = model_quant.create_config(allow=allow_quant, module='Text Encoder')
        if not quant_args:
            return kwargs
        if 'text_encoder_2' not in kwargs and ('Text Encoder' in shared.opts.bnb_quantization or 'Text Encoder' in shared.opts.torchao_quantization or 'Text Encoder' in shared.opts.quanto_quantization):
            kwargs['text_encoder_2'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_2", cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
    except Exception as e:
        shared.log.error(f'Quantization: {e}')
        errors.display(e, 'Quantization:')
    return kwargs


"""
def load_flux_gguf(file_path):
    transformer = None
    ggml.install_gguf()
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    from modules import ggml, sd_hijack_accelerate
    with init_empty_weights():
        config = diffusers.FluxTransformer2DModel.load_config(os.path.join('configs', 'flux'), subfolder="transformer")
        transformer = diffusers.FluxTransformer2DModel.from_config(config).to(devices.dtype)
        expected_state_dict_keys = list(transformer.state_dict().keys())
    state_dict, stats = ggml.load_gguf_state_dict(file_path, devices.dtype)
    state_dict = convert_flux_transformer_checkpoint_to_diffusers(state_dict)
    applied, skipped = 0, 0
    for param_name, param in state_dict.items():
        if param_name not in expected_state_dict_keys:
            # shared.log.warning(f'Load model: type=Unet/Transformer param={param_name} unexpected')
            skipped += 1
            continue
        applied += 1
        sd_hijack_accelerate.hijack_set_module_tensor_simple(transformer, tensor_name=param_name, value=param, device=0)
        transformer.gguf = 'gguf'
        state_dict[param_name] = None
    shared.log.debug(f'Load model: type=Unet/Transformer applied={applied} skipped={skipped} stats={stats}')
    return transformer, None
"""


def load_transformer(file_path): # triggered by opts.sd_unet change
    if file_path is None or not os.path.exists(file_path):
        return None
    transformer = None
    quant = model_quant.get_quant(file_path)
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "cache_dir": shared.opts.hfcache_dir,
    }
    if quant is not None and quant != 'none':
        shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} prequant={quant} dtype={devices.dtype}')
    if 'gguf' in file_path.lower():
        # _transformer, _text_encoder_2 = load_flux_gguf(file_path)
        from modules import ggml
        _transformer = ggml.load_gguf(file_path, cls=diffusers.FluxTransformer2DModel, compute_dtype=devices.dtype)
        if _transformer is not None:
            transformer = _transformer
    elif quant == 'qint8' or quant == 'qint4':
        _transformer, _text_encoder_2 = load_flux_quanto(file_path)
        if _transformer is not None:
            transformer = _transformer
    elif quant == 'fp8' or quant == 'fp4' or quant == 'nf4':
        _transformer, _text_encoder_2 = load_flux_bnb(file_path, diffusers_load_config)
        if _transformer is not None:
            transformer = _transformer
    elif 'nf4' in quant: # TODO flux: loader for civitai nf4 models
        from modules.model_flux_nf4 import load_flux_nf4
        _transformer, _text_encoder_2 = load_flux_nf4(file_path, prequantized=True)
        if _transformer is not None:
            transformer = _transformer
    else:
        quant_args = model_quant.create_bnb_config({})
        if quant_args:
            shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant=bnb dtype={devices.dtype}')
            from modules.model_flux_nf4 import load_flux_nf4
            transformer, _text_encoder_2 = load_flux_nf4(file_path, prequantized=False)
            if transformer is not None:
                return transformer
        quant_args = model_quant.create_config()
        if quant_args:
            shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant=torchao dtype={devices.dtype}')
            transformer = diffusers.FluxTransformer2DModel.from_single_file(file_path, **diffusers_load_config, **quant_args)
            if transformer is not None:
                return transformer
        shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant=none dtype={devices.dtype}')
        # TODO flux transformer from-single-file with quant
        # shared.log.warning('Load module: type=UNet/Transformer does not support load-time quantization')
        transformer = diffusers.FluxTransformer2DModel.from_single_file(file_path, **diffusers_load_config)
    if transformer is None:
        shared.log.error('Failed to load UNet model')
        shared.opts.sd_unet = 'Default'
    return transformer


def load_flux(checkpoint_info, diffusers_load_config): # triggered by opts.sd_checkpoint change
    quant = model_quant.get_quant(checkpoint_info.path)
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    shared.log.debug(f'Load model: type=FLUX model="{checkpoint_info.name}" repo="{repo_id}" unet="{shared.opts.sd_unet}" te="{shared.opts.sd_text_encoder}" vae="{shared.opts.sd_vae}" quant={quant} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')
    debug(f'Load model: type=FLUX config={diffusers_load_config}')
    modelloader.hf_login()

    transformer = None
    text_encoder_1 = None
    text_encoder_2 = None
    vae = None

    # unload current model
    sd_models.unload_model_weights()
    shared.sd_model = None
    devices.torch_gc(force=True)

    # load overrides if any
    if shared.opts.sd_unet != 'Default':
        try:
            debug(f'Load model: type=FLUX unet="{shared.opts.sd_unet}"')
            transformer = load_transformer(sd_unet.unet_dict[shared.opts.sd_unet])
            if transformer is None:
                shared.opts.sd_unet = 'Default'
                sd_unet.failed_unet.append(shared.opts.sd_unet)
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load UNet: {e}")
            shared.opts.sd_unet = 'Default'
            if debug:
                errors.display(e, 'FLUX UNet:')
    if shared.opts.sd_text_encoder != 'Default':
        try:
            debug(f'Load model: type=FLUX te="{shared.opts.sd_text_encoder}"')
            from modules.model_te import load_t5, load_vit_l
            if 'vit-l' in shared.opts.sd_text_encoder.lower():
                text_encoder_1 = load_vit_l()
            else:
                text_encoder_2 = load_t5(name=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load T5: {e}")
            shared.opts.sd_text_encoder = 'Default'
            if debug:
                errors.display(e, 'FLUX T5:')
    if shared.opts.sd_vae != 'Default' and shared.opts.sd_vae != 'Automatic':
        try:
            debug(f'Load model: type=FLUX vae="{shared.opts.sd_vae}"')
            from modules import sd_vae
            # vae = sd_vae.load_vae_diffusers(None, sd_vae.vae_dict[shared.opts.sd_vae], 'override')
            vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
            if os.path.exists(vae_file):
                vae_config = os.path.join('configs', 'flux', 'vae', 'config.json')
                vae = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, **diffusers_load_config)
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load VAE: {e}")
            shared.opts.sd_vae = 'Default'
            if debug:
                errors.display(e, 'FLUX VAE:')

    # load quantized components if any
    if quant == 'nf4':
        try:
            from modules.model_flux_nf4 import load_flux_nf4
            _transformer, _text_encoder = load_flux_nf4(checkpoint_info)
            if _transformer is not None:
                transformer = _transformer
            if _text_encoder is not None:
                text_encoder_2 = _text_encoder
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load NF4 components: {e}")
            if debug:
                errors.display(e, 'FLUX NF4:')
    if quant == 'qint8' or quant == 'qint4':
        try:
            _transformer, _text_encoder = load_flux_quanto(checkpoint_info)
            if _transformer is not None:
                transformer = _transformer
            if _text_encoder is not None:
                text_encoder_2 = _text_encoder
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load Quanto components: {e}")
            if debug:
                errors.display(e, 'FLUX Quanto:')

    # initialize pipeline with pre-loaded components
    kwargs = {}
    if transformer is not None:
        kwargs['transformer'] = transformer
        sd_unet.loaded_unet = shared.opts.sd_unet
    if text_encoder_1 is not None:
        kwargs['text_encoder'] = text_encoder_1
        model_te.loaded_te = shared.opts.sd_text_encoder
    if text_encoder_2 is not None:
        kwargs['text_encoder_2'] = text_encoder_2
        model_te.loaded_te = shared.opts.sd_text_encoder
    if vae is not None:
        kwargs['vae'] = vae
    if repo_id == 'sayakpaul/flux.1-dev-nf4':
        repo_id = 'black-forest-labs/FLUX.1-dev' # workaround since sayakpaul model is missing model_index.json
    if 'Fill' in repo_id:
        cls = diffusers.FluxFillPipeline
    elif 'Canny' in repo_id:
        cls = diffusers.FluxControlPipeline
    elif 'Depth' in repo_id:
        cls = diffusers.FluxControlPipeline
    else:
        cls = diffusers.FluxPipeline
    shared.log.debug(f'Load model: type=FLUX cls={cls.__name__} preloaded={list(kwargs)} revision={diffusers_load_config.get("revision", None)}')
    for c in kwargs:
        if getattr(kwargs[c], 'quantization_method', None) is not None or getattr(kwargs[c], 'gguf', None) is not None:
            shared.log.debug(f'Load model: type=FLUX component={c} dtype={kwargs[c].dtype} quant={getattr(kwargs[c], "quantization_method", None) or getattr(kwargs[c], "gguf", None)}')
        if kwargs[c].dtype == torch.float32 and devices.dtype != torch.float32:
            try:
                kwargs[c] = kwargs[c].to(dtype=devices.dtype)
                shared.log.warning(f'Load model: type=FLUX component={c} dtype={kwargs[c].dtype} cast dtype={devices.dtype} recast')
            except Exception:
                pass

    allow_quant = 'gguf' not in (sd_unet.loaded_unet or '') and (quant is None or quant == 'none')
    fn = checkpoint_info.path
    if (fn is None) or (not os.path.exists(fn) or os.path.isdir(fn)):
        kwargs = load_quants(kwargs, repo_id, cache_dir=shared.opts.diffusers_dir, allow_quant=allow_quant)
    # kwargs = model_quant.create_config(kwargs, allow_quant)
    if fn.endswith('.safetensors') and os.path.isfile(fn):
        pipe = diffusers.FluxPipeline.from_single_file(fn, cache_dir=shared.opts.diffusers_dir, **kwargs, **diffusers_load_config)
    else:
        pipe = cls.from_pretrained(repo_id, cache_dir=shared.opts.diffusers_dir, **kwargs, **diffusers_load_config)

    # release memory
    transformer = None
    text_encoder_1 = None
    text_encoder_2 = None
    vae = None
    for k in kwargs.keys():
        kwargs[k] = None
    devices.torch_gc()

    return pipe
