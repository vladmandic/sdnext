import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, auth_check
from modules import shared, errors, devices, modelloader, sd_models, sd_unet, model_te, model_quant, sd_hijack_te


debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_chroma_quanto(checkpoint_info):
    transformer, text_encoder = None, None
    quanto = model_quant.load_quanto('Load model: type=Chroma')

    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path

    try:
        quantization_map = os.path.join(repo_path, "transformer", "quantization_map.json")
        debug(f'Load model: type=Chroma quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="transformer"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info.name)
            quantization_map = hf_hub_download(repo_id, subfolder='transformer', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        state_dict = load_file(os.path.join(repo_path, "transformer", "diffusion_pytorch_model.safetensors"))
        dtype = state_dict['context_embedder.bias'].dtype
        with torch.device("meta"):
            transformer = diffusers.ChromaTransformer2DModel.from_config(os.path.join(repo_path, "transformer", "config.json")).to(dtype=dtype)
        quanto.requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
        if shared.opts.diffusers_eval:
            transformer.eval()
        transformer_dtype = transformer.dtype
        if transformer_dtype != devices.dtype:
            try:
                transformer = transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=Chroma Failed to cast transformer to {devices.dtype}, set dtype to {transformer_dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=Chroma failed to load Quanto transformer: {e}")
        if debug:
            errors.display(e, 'Chroma Quanto:')

    try:
        quantization_map = os.path.join(repo_path, "text_encoder", "quantization_map.json")
        debug(f'Load model: type=Chroma quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="text_encoder"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info.name)
            quantization_map = hf_hub_download(repo_id, subfolder='text_encoder', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        with open(os.path.join(repo_path, "text_encoder", "config.json"), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        state_dict = load_file(os.path.join(repo_path, "text_encoder", "model.safetensors"))
        dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
        with torch.device("meta"):
            text_encoder = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
        quanto.requantize(text_encoder, state_dict, quantization_map, device=torch.device("cpu"))
        if shared.opts.diffusers_eval:
            text_encoder.eval()
        text_encoder_dtype = text_encoder.dtype
        if text_encoder_dtype != devices.dtype:
            try:
                text_encoder = text_encoder.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=Chroma Failed to cast text encoder to {devices.dtype}, set dtype to {text_encoder_dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=Chroma failed to load Quanto text encoder: {e}")
        if debug:
            errors.display(e, 'Chroma Quanto:')

    return transformer, text_encoder


def load_chroma_bnb(checkpoint_info, diffusers_load_config): # pylint: disable=unused-argument
    transformer, text_encoder = None, None
    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path
    model_quant.load_bnb('Load model: type=Chroma')
    quant = model_quant.get_quant(repo_path)
    try:
        # we ignore the distilled guidance layer because it degrades quality too much
        # see: https://github.com/huggingface/diffusers/pull/11698#issuecomment-2969717180 for more details
        if quant == 'fp8':
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["distilled_guidance_layer"], bnb_4bit_compute_dtype=devices.dtype)
            debug(f'Quantization: {quantization_config}')
            transformer = diffusers.ChromaTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
        elif quant == 'fp4':
            quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["distilled_guidance_layer"], bnb_4bit_compute_dtype=devices.dtype, bnb_4bit_quant_type= 'fp4')
            debug(f'Quantization: {quantization_config}')
            transformer = diffusers.ChromaTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
        elif quant == 'nf4':
            quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["distilled_guidance_layer"], bnb_4bit_compute_dtype=devices.dtype, bnb_4bit_quant_type= 'nf4')
            debug(f'Quantization: {quantization_config}')
            transformer = diffusers.ChromaTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
        else:
            transformer = diffusers.ChromaTransformer2DModel.from_single_file(repo_path, **diffusers_load_config)
    except Exception as e:
        shared.log.error(f"Load model: type=Chroma failed to load BnB transformer: {e}")
        transformer, text_encoder = None, None
        if debug:
            errors.display(e, 'Chroma:')
    return transformer, text_encoder


def load_quants(kwargs, pretrained_model_name_or_path, cache_dir, allow_quant):
    try:
        if 'transformer' not in kwargs and model_quant.check_nunchaku('Model'):
            raise NotImplementedError('Nunchaku does not support Chroma Model yet. See https://github.com/mit-han-lab/nunchaku/issues/167')
        elif 'transformer' not in kwargs and model_quant.check_quant('Model'):
            quant_args = model_quant.create_config(allow=allow_quant, module='Model', modules_to_not_convert=["distilled_guidance_layer"])
            if quant_args:
                if os.path.isfile(pretrained_model_name_or_path):
                    kwargs['transformer'] = diffusers.ChromaTransformer2DModel.from_single_file(pretrained_model_name_or_path, cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
                else:
                    kwargs['transformer'] = diffusers.ChromaTransformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
        if 'text_encoder' not in kwargs and model_quant.check_nunchaku('TE'):
            import nunchaku
            nunchaku_precision = nunchaku.utils.get_precision()
            nunchaku_repo = 'mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors'
            shared.log.debug(f'Load module: quant=Nunchaku module=t5 repo="{nunchaku_repo}" precision={nunchaku_precision}')
            kwargs['text_encoder'] = nunchaku.NunchakuT5EncoderModel.from_pretrained(nunchaku_repo, torch_dtype=devices.dtype)
        elif 'text_encoder' not in kwargs and model_quant.check_quant('TE'):
            quant_args = model_quant.create_config(allow=allow_quant, module='TE')
            if quant_args:
                if os.path.isfile(pretrained_model_name_or_path):
                    kwargs['text_encoder'] = transformers.T5EncoderModel.from_single_file(pretrained_model_name_or_path, cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
                else:
                    kwargs['text_encoder'] = transformers.T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
    except Exception as e:
        shared.log.error(f'Quantization: {e}')
        errors.display(e, 'Quantization:')
    return kwargs


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
        from modules import ggml
        _transformer = ggml.load_gguf(file_path, cls=diffusers.ChromaTransformer2DModel, compute_dtype=devices.dtype)
        if _transformer is not None:
            transformer = _transformer
    elif quant == 'qint8' or quant == 'qint4':
        _transformer, _text_encoder = load_chroma_quanto(file_path)
        if _transformer is not None:
            transformer = _transformer
    elif quant == 'fp8' or quant == 'fp4' or quant == 'nf4':
        _transformer, _text_encoder = load_chroma_bnb(file_path, diffusers_load_config)
        if _transformer is not None:
            transformer = _transformer
    else:
        quant_args = model_quant.create_config(module='Model', modules_to_not_convert=["distilled_guidance_layer"])
        if quant_args:
            shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant=torchao dtype={devices.dtype}')
            transformer = diffusers.ChromaTransformer2DModel.from_single_file(file_path, **diffusers_load_config, **quant_args)
            if transformer is not None:
                return transformer
        shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant=none dtype={devices.dtype}')
        # TODO model load: chroma transformer from-single-file with quant
        # shared.log.warning('Load module: type=UNet/Transformer does not support load-time quantization')
        # transformer = diffusers.ChromaTransformer2DModel.from_single_file(file_path, **diffusers_load_config)
    if transformer is None:
        shared.log.error('Failed to load UNet model')
        shared.opts.sd_unet = 'Default'
    return transformer


def load_chroma(checkpoint_info, diffusers_load_config): # triggered by opts.sd_checkpoint change
    fn = checkpoint_info.path
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    login = modelloader.hf_login()
    try:
        auth_check(repo_id)
    except Exception as e:
        repo_id = None
        if not os.path.exists(fn):
            shared.log.error(f'Load model: repo="{repo_id}" login={login} {e}')
            return None

    prequantized = model_quant.get_quant(checkpoint_info.path)
    shared.log.debug(f'Load model: type=Chroma model="{checkpoint_info.name}" repo={repo_id or "none"} unet="{shared.opts.sd_unet}" te="{shared.opts.sd_text_encoder}" vae="{shared.opts.sd_vae}" quant={prequantized} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')
    debug(f'Load model: type=Chroma config={diffusers_load_config}')

    transformer = None
    text_encoder = None
    vae = None

    # unload current model
    sd_models.unload_model_weights()
    shared.sd_model = None
    devices.torch_gc(force=True)

    if shared.opts.teacache_enabled:
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.ChromaTransformer2DModel.__name__}')
        diffusers.ChromaTransformer2DModel.forward = teacache.teacache_chroma_forward # patch must be done before transformer is loaded

    # load overrides if any
    if shared.opts.sd_unet != 'Default':
        try:
            debug(f'Load model: type=Chroma unet="{shared.opts.sd_unet}"')
            transformer = load_transformer(sd_unet.unet_dict[shared.opts.sd_unet])
            if transformer is None:
                shared.opts.sd_unet = 'Default'
                sd_unet.failed_unet.append(shared.opts.sd_unet)
        except Exception as e:
            shared.log.error(f"Load model: type=Chroma failed to load UNet: {e}")
            shared.opts.sd_unet = 'Default'
            if debug:
                errors.display(e, 'Chroma UNet:')
    if shared.opts.sd_text_encoder != 'Default':
        try:
            debug(f'Load model: type=Chroma te="{shared.opts.sd_text_encoder}"')
            from modules.model_te import load_t5
            text_encoder = load_t5(name=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
        except Exception as e:
            shared.log.error(f"Load model: type=Chroma failed to load T5: {e}")
            shared.opts.sd_text_encoder = 'Default'
            if debug:
                errors.display(e, 'Chroma T5:')
    if shared.opts.sd_vae != 'Default' and shared.opts.sd_vae != 'Automatic':
        try:
            debug(f'Load model: type=Chroma vae="{shared.opts.sd_vae}"')
            from modules import sd_vae
            # vae = sd_vae.load_vae_diffusers(None, sd_vae.vae_dict[shared.opts.sd_vae], 'override')
            vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
            if os.path.exists(vae_file):
                vae_config = os.path.join('configs', 'chroma', 'vae', 'config.json')
                vae = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, **diffusers_load_config)
        except Exception as e:
            shared.log.error(f"Load model: type=Chroma failed to load VAE: {e}")
            shared.opts.sd_vae = 'Default'
            if debug:
                errors.display(e, 'Chroma VAE:')

    # initialize pipeline with pre-loaded components
    kwargs = {}
    if transformer is not None:
        kwargs['transformer'] = transformer
        sd_unet.loaded_unet = shared.opts.sd_unet
    if text_encoder is not None:
        kwargs['text_encoder'] = text_encoder
        model_te.loaded_te = shared.opts.sd_text_encoder
    if vae is not None:
        kwargs['vae'] = vae

    # TODO model load: add ChromaFillPipeline, ChromaControlPipeline, ChromaImg2ImgPipeline etc when available
    # Chroma will support inpainting *after* its training has finished: https://huggingface.co/lodestones/Chroma/discussions/28#6826dd2ed86f53ff983add5c
    cls = diffusers.ChromaPipeline
    shared.log.debug(f'Load model: type=Chroma cls={cls.__name__} preloaded={list(kwargs)} revision={diffusers_load_config.get("revision", None)}')
    for c in kwargs:
        if getattr(kwargs[c], 'quantization_method', None) is not None or getattr(kwargs[c], 'gguf', None) is not None:
            shared.log.debug(f'Load model: type=Chroma component={c} dtype={kwargs[c].dtype} quant={getattr(kwargs[c], "quantization_method", None) or getattr(kwargs[c], "gguf", None)}')
        if kwargs[c].dtype == torch.float32 and devices.dtype != torch.float32:
            try:
                kwargs[c] = kwargs[c].to(dtype=devices.dtype)
                shared.log.warning(f'Load model: type=Chroma component={c} dtype={kwargs[c].dtype} cast dtype={devices.dtype} recast')
            except Exception:
                pass

    allow_quant = 'gguf' not in (sd_unet.loaded_unet or '') and (prequantized is None or prequantized == 'none')
    if (fn is None) or (not os.path.exists(fn) or os.path.isdir(fn)):
        kwargs = load_quants(kwargs, repo_id or fn, cache_dir=shared.opts.diffusers_dir, allow_quant=allow_quant)
    # kwargs = model_quant.create_config(kwargs, allow_quant, modules_to_not_convert=["distilled_guidance_layer"])
    if fn.endswith('.safetensors') and os.path.isfile(fn):
        pipe = diffusers.ChromaPipeline.from_single_file(fn, cache_dir=shared.opts.diffusers_dir, **kwargs, **diffusers_load_config)
    else:
        pipe = cls.from_pretrained(repo_id or fn, cache_dir=shared.opts.diffusers_dir, **kwargs, **diffusers_load_config)

    if shared.opts.teacache_enabled and model_quant.check_nunchaku('Model'):
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
        apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)

    # release memory
    transformer = None
    text_encoder = None
    vae = None
    for k in kwargs.keys():
        kwargs[k] = None
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True)
    return pipe
