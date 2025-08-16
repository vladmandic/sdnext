import os
import json
import diffusers
import transformers
from modules import shared, devices, errors, sd_models, model_quant


debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


def load_transformer(repo_id, cls_name, load_config={}, subfolder="transformer", allow_quant=True, variant=None, dtype=None, modules_to_not_convert=[], modules_dtype_dict={}):
    transformer = None
    try:
        load_args, quant_args = model_quant.get_dit_args(load_config, module='Model', device_map=True, allow_quant=allow_quant, modules_to_not_convert=modules_to_not_convert, modules_dtype_dict=modules_dtype_dict)
        quant_type = model_quant.get_quant_type(quant_args)
        dtype = dtype or devices.dtype

        local_file = None
        if shared.opts.sd_unet is not None and shared.opts.sd_unet != 'Default':
            from modules import sd_unet
            if shared.opts.sd_unet not in list(sd_unet.unet_dict):
                shared.log.error(f'Load module: type=transformer file="{shared.opts.sd_unet}" not found')
            elif os.path.exists(sd_unet.unet_dict[shared.opts.sd_unet]):
                local_file = sd_unet.unet_dict[shared.opts.sd_unet]

        if local_file is not None and local_file.lower().endswith('.gguf'):
            shared.log.debug(f'Load model: transformer="{local_file}" cls={cls_name.__name__} quant="{quant_type}" args={load_args}')
            from modules import ggml
            ggml.install_gguf()
            loader = cls_name.from_single_file if hasattr(cls_name, 'from_single_file') else cls_name.from_pretrained
            transformer = loader(
                local_file,
                quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=dtype),
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
            )
            transformer = model_quant.do_post_load_quant(transformer, allow=quant_type is not None)
        elif local_file is not None and local_file.lower().endswith('.safetensors'):
            shared.log.debug(f'Load model: transformer="{local_file}" cls={cls_name.__name__} quant="{quant_type}" args={load_args}')
            if dtype is not None:
                load_args['torch_dtype'] = dtype
            loader = cls_name.from_single_file if hasattr(cls_name, 'from_single_file') else cls_name.from_pretrained
            transformer = loader(
                local_file,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
            )
        else:
            shared.log.debug(f'Load model: transformer="{repo_id}" cls={cls_name.__name__} quant="{quant_type}" args={load_args}')
            if dtype is not None:
                load_args['torch_dtype'] = dtype
            if subfolder is not None:
                load_args['subfolder'] = subfolder
            if variant is not None:
                load_args['variant'] = variant
            transformer = cls_name.from_pretrained(
                repo_id,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
            )
        sd_models.allow_post_quant = False # we already handled it
        if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
            sd_models.move_model(transformer, devices.cpu)
    except Exception as e:
        shared.log.error(f'Load model: transformer="{repo_id}" cls={cls_name.__name__} {e}')
        errors.display(e, 'Load:')
        raise
    devices.torch_gc()
    return transformer


def load_text_encoder(repo_id, cls_name, load_config={}, subfolder="text_encoder", allow_quant=True, allow_shared=True, variant=None, dtype=None, modules_to_not_convert=[], modules_dtype_dict={}):
    text_encoder = None
    try:
        load_args, quant_args = model_quant.get_dit_args(load_config, module='TE', device_map=True, allow_quant=allow_quant, modules_to_not_convert=modules_to_not_convert, modules_dtype_dict=modules_dtype_dict)
        quant_type = model_quant.get_quant_type(quant_args)
        dtype = dtype or devices.dtype

        # load from local file if specified
        local_file = None
        if shared.opts.sd_text_encoder is not None and shared.opts.sd_text_encoder != 'Default':
            from modules import model_te
            if shared.opts.sd_text_encoder not in list(model_te.te_dict):
                shared.log.error(f'Load module: type=te file="{shared.opts.sd_text_encoder}" not found')
            elif os.path.exists(model_te.te_dict[shared.opts.sd_text_encoder]):
                local_file = model_te.te_dict[shared.opts.sd_text_encoder]

        # load from local file gguf
        if local_file is not None and local_file.lower().endswith('.gguf'):
            shared.log.debug(f'Load model: text_encoder="{local_file}" cls={cls_name.__name__} quant="{quant_type}"')
            """
            from modules import ggml
            ggml.install_gguf()
            text_encoder = cls_name.from_pretrained(
                gguf_file=local_file,
                quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=dtype),
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
            )
            text_encoder = model_quant.do_post_load_quant(text_encoder, allow=quant_type is not None)
            """
            text_encoder = model_te.load_t5(local_file)
            text_encoder = model_quant.do_post_load_quant(text_encoder, allow=quant_type is not None)
        # load from local file safetensors
        elif local_file is not None and local_file.lower().endswith('.safetensors'):
            shared.log.debug(f'Load model: text_encoder="{local_file}" cls={cls_name.__name__} quant="{quant_type}"')
            from modules import model_te
            text_encoder = model_te.load_t5(local_file)
            text_encoder = model_quant.do_post_load_quant(text_encoder, allow=quant_type is not None)
        # use shared t5 if possible
        elif cls_name == transformers.T5EncoderModel and allow_shared:
            with open(os.path.join('configs', 'flux', 'text_encoder_2', 'config.json'), encoding='utf8') as f:
                load_args['config'] = transformers.T5Config(**json.load(f))
            if model_quant.check_nunchaku('TE'):
                import nunchaku
                repo_id = 'nunchaku-tech/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors'
                cls_name = nunchaku.NunchakuT5EncoderModel
                shared.log.debug(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} quant="SVDQuant"')
                text_encoder = nunchaku.NunchakuT5EncoderModel.from_pretrained(
                    repo_id,
                    torch_dtype=dtype,
                )
                text_encoder.quantization_method = 'SVDQuant'
            elif shared.opts.te_shared_t5:
                repo_id = 'Disty0/t5-xxl'
                shared.log.debug(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} quant="{quant_type}" shared={shared.opts.te_shared_t5}')
                if dtype is not None:
                    load_args['torch_dtype'] = dtype
                text_encoder = cls_name.from_pretrained(
                    repo_id,
                    cache_dir=shared.opts.hfcache_dir,
                    **load_args,
                    **quant_args,
                )

        # load from repo
        if text_encoder is None:
            shared.log.debug(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} quant="{quant_type}" shared={shared.opts.te_shared_t5}')
            if dtype is not None:
                load_args['torch_dtype'] = dtype
            if subfolder is not None:
                load_args['subfolder'] = subfolder
            if variant is not None:
                load_args['variant'] = variant
            text_encoder = cls_name.from_pretrained(
                repo_id,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
            )

        sd_models.allow_post_quant = False # we already handled it
        if shared.opts.diffusers_offload_mode != 'none' and text_encoder is not None:
            sd_models.move_model(text_encoder, devices.cpu)
    except Exception as e:
        shared.log.error(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} {e}')
        errors.display(e, 'Load:')
        raise
    devices.torch_gc()
    return text_encoder
