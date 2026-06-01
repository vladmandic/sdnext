import os
import json
import transformers
from modules import shared, devices, errors, sd_models, model_quant
from modules.logger import log
from pipelines.generic_util import get_loader
from pipelines.generic_shared import shared_te_map


debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


def get_shared(cls, repo_id, subfolder=None, variant=None):
    args = {}
    if variant is not None:
        args['variant'] = variant
    for name, item in shared_te_map.items():
        if item['cls'] == cls and (item['identifier'] is None or item['identifier'].lower() in repo_id.lower()):
            if item.get('config_class', None) is not None and item.get('config_path', None) is not None:
                with open(item['config_path'], encoding='utf8') as f:
                    args['config'] = item['config_class'](**json.load(f))
            if item.get('target_subfolder', None) is not None:
                args['subfolder'] = item['target_subfolder']
            log.debug(f'Load model: text_encoder="{repo_id}" cls={cls.__name__} target="{item["target_repo"]}" args={args} shared="{name}"')
            return item['target_repo'], args
    if subfolder is not None: # use default provided subfolder
        args['subfolder'] = subfolder
    return repo_id, args


def load_local_file(local_file, cls_name, quant_type): # t5-only
    from modules import model_te
    text_encoder = None

    # 1. load from local file gguf
    if local_file.lower().endswith('.gguf'):
        log.debug(f'Load model: text_encoder="{local_file}" cls={cls_name.__name__} quant="{quant_type}" loader={get_loader("transformers")} file=gguf')
        text_encoder = model_te.load_t5(local_file)
        text_encoder = model_quant.do_post_load_quant(text_encoder, allow=quant_type is not None)

    # 2. t5 - load from local file safetensors
    elif local_file.lower().endswith('.safetensors'):
        log.debug(f'Load model: text_encoder="{local_file}" cls={cls_name.__name__} quant="{quant_type}" loader={get_loader("transformers")} file=safetensors')
        text_encoder = model_te.load_t5(local_file)
        text_encoder = model_quant.do_post_load_quant(text_encoder, allow=quant_type is not None)

    return text_encoder


def load_text_encoder(repo_id, cls_name, load_config=None, subfolder="text_encoder", allow_quant=True, allow_shared=True, variant=None, dtype=None, modules_to_not_convert=None, modules_dtype_dict=None, **kwargs):
    if shared.state.interrupted:
        return None
    text_encoder = None
    allow_shared = allow_shared and shared.opts.te_shared_te
    if load_config is None:
        load_config = {}
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    jobid = shared.state.begin('Load TE')
    try:
        load_args, quant_args = model_quant.get_dit_args(load_config, module='TE', device_map=True, allow_quant=allow_quant, modules_to_not_convert=modules_to_not_convert, modules_dtype_dict=modules_dtype_dict)
        quant_type = model_quant.get_quant_type(quant_args)
        load_args.pop('torch_dtype', None)
        dtype = dtype or devices.dtype
        load_args['dtype'] = dtype

        # 1. load override from local file
        if (shared.opts.sd_text_encoder is not None) and (shared.opts.sd_text_encoder != 'Default') and (text_encoder is None):
            local_file = None
            from modules import model_te
            if shared.opts.sd_text_encoder not in list(model_te.te_dict):
                log.error(f'Load module: type=te file="{shared.opts.sd_text_encoder}" not found')
            elif os.path.exists(model_te.te_dict[shared.opts.sd_text_encoder]):
                local_file = model_te.te_dict[shared.opts.sd_text_encoder]
            if local_file is not None:
                text_encoder = load_local_file(local_file, cls_name, quant_type)

        # 2. load override from repo
        if (shared.opts.sd_text_encoder is not None) and (shared.opts.sd_text_encoder != 'Default') and (text_encoder is None):
            repo_id = shared.opts.sd_text_encoder
            if '/' in repo_id: # shared.opts.sd_text_encoder can be in format org/repo or org/repo/subfolder
                parts = repo_id.split('/')
                if len(parts) >= 3:
                    repo_id = '/'.join(parts[:2])
                    load_args['subfolder'] = '/'.join(parts[2:])
            log.debug(f'Load model: text_encoder="{repo_id}" quant="{quant_type}" loader={get_loader("transformers")} type=override')
            text_encoder = transformers.AutoModel.from_pretrained(
                repo_id,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
                **kwargs,
            )

        # 3. load shared from repo
        if allow_shared and (text_encoder is None):
            log.debug(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} quant="{quant_type}" loader={get_loader("transformers")}')
            target_repo, extra_args = get_shared(cls_name, repo_id, subfolder=subfolder, variant=variant)
            text_encoder = cls_name.from_pretrained(
                target_repo,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
                **extra_args,
            )

        # 4. load default from repo
        if text_encoder is None:
            log.debug(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} quant="{quant_type}" loader={get_loader("transformers")}')
            if subfolder is not None:
                load_args['subfolder'] = subfolder
            if variant is not None:
                load_args['variant'] = variant
            text_encoder = cls_name.from_pretrained(
                repo_id,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
                **kwargs,
            )

        sd_models.allow_post_quant = False # we already handled it
        if shared.opts.diffusers_offload_mode != 'none' and text_encoder is not None:
            sd_models.move_model(text_encoder, devices.cpu)

        if text_encoder is not None and not hasattr(text_encoder, 'quantization_config'): # attach quantization_config
            if hasattr(text_encoder, 'config') and hasattr(text_encoder.config, 'quantization_config'):
                text_encoder.quantization_config = text_encoder.config.quantization_config
            elif (quant_type is not None) and (quant_args.get('quantization_config', None) is not None):
                text_encoder.quantization_config = quant_args.get('quantization_config', None)

    except Exception as e:
        log.error(f'Load model: text_encoder="{repo_id}" cls={cls_name.__name__} {e}')
        errors.display(e, 'Load')
        raise

    devices.torch_gc()
    shared.state.end(jobid)
    return text_encoder
