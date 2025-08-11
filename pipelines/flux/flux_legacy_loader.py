import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules import shared, errors, devices, sd_models, sd_unet, model_te, model_quant, sd_hijack_te


debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_flux_quanto(checkpoint_info):
    transformer, text_encoder_2 = None, None
    quanto = model_quant.load_quanto('Load model: type=FLUX')

    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path

    try:
        quantization_map = os.path.join(repo_path, "transformer", "quantization_map.json")
        debug(f'Load model: type=FLUX quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="transformer"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info)
            quantization_map = hf_hub_download(repo_id, subfolder='transformer', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        state_dict = load_file(os.path.join(repo_path, "transformer", "diffusion_pytorch_model.safetensors"))
        dtype = state_dict['context_embedder.bias'].dtype
        with torch.device("meta"):
            transformer = diffusers.FluxTransformer2DModel.from_config(os.path.join(repo_path, "transformer", "config.json")).to(dtype=dtype)
        quanto.requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
        transformer_dtype = transformer.dtype
        if transformer_dtype != devices.dtype:
            try:
                transformer = transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=FLUX Failed to cast transformer to {devices.dtype}, set dtype to {transformer_dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=FLUX failed to load Quanto transformer: {e}")
        if debug:
            errors.display(e, 'FLUX Quanto:')

    try:
        quantization_map = os.path.join(repo_path, "text_encoder_2", "quantization_map.json")
        debug(f'Load model: type=FLUX quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="text_encoder_2"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info)
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
        text_encoder_2_dtype = text_encoder_2.dtype
        if text_encoder_2_dtype != devices.dtype:
            try:
                text_encoder_2 = text_encoder_2.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=FLUX Failed to cast text encoder to {devices.dtype}, set dtype to {text_encoder_2_dtype}")
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


def load_quants(kwargs, repo_id, cache_dir, allow_quant): # pylint: disable=unused-argument
    try:
        diffusers_load_config = {
            "torch_dtype": devices.dtype,
            "cache_dir": cache_dir,
        }
        if 'transformer' not in kwargs and model_quant.check_nunchaku('Model'):
            import nunchaku
            nunchaku_precision = nunchaku.utils.get_precision()
            nunchaku_repo = None
            if 'flux.1-kontext' in repo_id.lower():
                nunchaku_repo = f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{nunchaku_precision}_r32-flux.1-kontext-dev.safetensors"
            elif 'flux.1-dev' in repo_id.lower():
                nunchaku_repo = f"mit-han-lab/nunchaku-flux.1-dev/svdq-{nunchaku_precision}_r32-flux.1-dev.safetensors"
            elif 'flux.1-schnell' in repo_id.lower():
                nunchaku_repo = f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{nunchaku_precision}_r32-flux.1-schnell.safetensors"
            elif 'flux.1-fill' in repo_id.lower():
                nunchaku_repo = f"mit-han-lab/svdq-fp4-flux.1-fill-dev/svdq-{nunchaku_precision}_r32-flux.1-schnell.safetensors"
            elif 'flux.1-depth' in repo_id.lower():
                nunchaku_repo = f"mit-han-lab/svdq-int4-flux.1-depth-dev/svdq-{nunchaku_precision}_r32-flux.1-schnell.safetensors"
            elif 'shuttle' in repo_id.lower():
                nunchaku_repo = f"mit-han-lab/nunchaku-shuttle-jaguar/svdq-{nunchaku_precision}_r32-shuttle-jaguar.safetensors"
            else:
                shared.log.error(f'Load module: quant=Nunchaku module=transformer repo="{repo_id}" unsupported')
            if nunchaku_repo is not None:
                shared.log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" precision={nunchaku_precision} offload={shared.opts.nunchaku_offload} attention={shared.opts.nunchaku_attention}')
                kwargs['transformer'] = nunchaku.NunchakuFluxTransformer2dModel.from_pretrained(nunchaku_repo, offload=shared.opts.nunchaku_offload, torch_dtype=devices.dtype)
                kwargs['transformer'].quantization_method = 'SVDQuant'
                if shared.opts.nunchaku_attention:
                    kwargs['transformer'].set_attention_impl("nunchaku-fp16")
        if 'transformer' not in kwargs and model_quant.check_quant('Model'):
            load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
            kwargs['transformer'] = diffusers.FluxTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", **load_args, **quant_args)
        if 'text_encoder_2' not in kwargs and model_quant.check_nunchaku('TE'):
            import nunchaku
            nunchaku_precision = nunchaku.utils.get_precision()
            nunchaku_repo = 'mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors'
            shared.log.debug(f'Load module: quant=Nunchaku module=t5 repo="{nunchaku_repo}" precision={nunchaku_precision}')
            kwargs['text_encoder_2'] = nunchaku.NunchakuT5EncoderModel.from_pretrained(nunchaku_repo, torch_dtype=devices.dtype)
            kwargs['text_encoder_2'].quantization_method = 'SVDQuant'
        if 'text_encoder_2' not in kwargs and model_quant.check_quant('TE'):
            load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
            kwargs['text_encoder_2'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_2", **load_args, **quant_args)
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
        "torch_dtype": devices.dtype,
        "cache_dir": shared.opts.hfcache_dir,
    }
    if quant is not None and quant != 'none':
        shared.log.info(f'Load module: type=UNet/Transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} prequant={quant} dtype={devices.dtype}')
    if 'gguf' in file_path.lower():
        from modules import ggml
        _transformer = ggml.load_gguf(file_path, cls=diffusers.FluxTransformer2DModel, compute_dtype=devices.dtype)
        if _transformer is not None:
            transformer = _transformer
    elif quant == "fp8":
        _transformer = model_quant.load_fp8_model_layerwise(file_path, diffusers.FluxTransformer2DModel.from_single_file, diffusers_load_config)
        if _transformer is not None:
            transformer = _transformer
    elif quant in {'qint8', 'qint4'}:
        _transformer, _text_encoder_2 = load_flux_quanto(file_path)
        if _transformer is not None:
            transformer = _transformer
    elif quant in {'fp8', 'fp4', 'nf4'}:
        _transformer, _text_encoder_2 = load_flux_bnb(file_path, diffusers_load_config)
        if _transformer is not None:
            transformer = _transformer
    elif 'nf4' in quant:
        from pipelines.flux.flux_nf4 import load_flux_nf4
        _transformer, _text_encoder_2 = load_flux_nf4(file_path, prequantized=True)
        if _transformer is not None:
            transformer = _transformer
    else:
        quant_args = model_quant.create_bnb_config({})
        if quant_args:
            shared.log.info(f'Load module: type=Flux transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant=bnb dtype={devices.dtype}')
            from pipelines.flux.flux_nf4 import load_flux_nf4
            transformer, _text_encoder_2 = load_flux_nf4(file_path, prequantized=False)
            if transformer is not None:
                return transformer
        load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
        shared.log.debug(f'Load model: type=Flux transformer file="{file_path}" offload={shared.opts.diffusers_offload_mode} args={load_args}')
        transformer = diffusers.FluxTransformer2DModel.from_single_file(file_path, **load_args, **quant_args)
    if transformer is None:
        shared.log.error('Failed to load UNet model')
        shared.opts.sd_unet = 'Default'
    return transformer


def load_flux(checkpoint_info, diffusers_load_config): # triggered by opts.sd_checkpoint change
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    allow_post_quant = False

    prequantized = model_quant.get_quant(checkpoint_info.path)
    shared.log.debug(f'Load model: type=FLUX model="{checkpoint_info.name}" repo="{repo_id}" unet="{shared.opts.sd_unet}" te="{shared.opts.sd_text_encoder}" vae="{shared.opts.sd_vae}" quant={prequantized} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')
    debug(f'Load model: type=FLUX config={diffusers_load_config}')

    transformer = None
    text_encoder_1 = None
    text_encoder_2 = None
    vae = None

    # unload current model
    sd_models.unload_model_weights()
    shared.sd_model = None
    devices.torch_gc(force=True, reason='load')

    if shared.opts.teacache_enabled:
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.FluxTransformer2DModel.__name__}')
        diffusers.FluxTransformer2DModel.forward = teacache.teacache_flux_forward # patch must be done before transformer is loaded

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
    if prequantized == 'nf4':
        try:
            from pipelines.flux.flux_nf4 import load_flux_nf4
            _transformer, _text_encoder = load_flux_nf4(checkpoint_info)
            if _transformer is not None:
                transformer = _transformer
            if _text_encoder is not None:
                text_encoder_2 = _text_encoder
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load NF4 components: {e}")
            if debug:
                errors.display(e, 'FLUX NF4:')
    if prequantized == 'qint8' or prequantized == 'qint4':
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
    elif 'Kontext' in repo_id:
        cls = diffusers.FluxKontextPipeline
        from diffusers import pipelines
        pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["flux1kontext"] = diffusers.FluxKontextPipeline
        pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["flux1kontext"] = diffusers.FluxKontextPipeline
        pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["flux1kontext"] = diffusers.FluxKontextInpaintPipeline

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

    allow_quant = 'gguf' not in (sd_unet.loaded_unet or '') and (prequantized is None or prequantized == 'none')
    fn = checkpoint_info.path
    if (fn is None) or (not os.path.exists(fn) or os.path.isdir(fn)):
        kwargs = load_quants(kwargs, repo_id, cache_dir=shared.opts.diffusers_dir, allow_quant=allow_quant)
    if fn.endswith('.safetensors') and os.path.isfile(fn):
        pipe = cls.from_single_file(fn, cache_dir=shared.opts.diffusers_dir, **kwargs, **diffusers_load_config)
        allow_post_quant = True
    else:
        pipe = cls.from_pretrained(repo_id, cache_dir=shared.opts.diffusers_dir, **kwargs, **diffusers_load_config)

    if shared.opts.teacache_enabled and model_quant.check_nunchaku('Model'):
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
        apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)

    # release memory
    transformer = None
    text_encoder_1 = None
    text_encoder_2 = None
    vae = None
    for k in kwargs.keys():
        kwargs[k] = None
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe, allow_post_quant
