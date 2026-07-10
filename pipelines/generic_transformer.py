import os
import torch
from modules import shared, devices, errors, sd_models, sd_offload, model_quant
from modules.logger import log
from pipelines.generic_util import get_loader
from pipelines.generic_map import transformers_map


debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


def load_transformer(
        repo_id,
        cls_name,
        load_config=None,
        subfolder="transformer",
        allow_quant=True,
        variant=None,
        dtype=None,
        modules_to_not_convert=None,
        modules_dtype_dict=None,
        use_safetensors=True,
        native_spec=None,
        **kwargs):

    """Load a DiT transformer from the base repo, or from a user-selected
    single file when the UNET dropdown (``shared.opts.sd_unet``) is set.

    With ``native_spec`` set and a .safetensors override selected, dispatches
    to :func:`pipelines.native_transformer.load`. Without a spec, a single-file
    override falls back to ``from_single_file``.
    """
    if repo_id is None or repo_id.lower() == 'none':
        return None
    if shared.state.interrupted:
        return None
    transformer = None
    if load_config is None:
        load_config = {}
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    jobid = shared.state.begin('Load DiT')
    try:
        load_args, quant_args = model_quant.get_dit_args(load_config, module='Model', device_map=True, allow_quant=allow_quant, modules_to_not_convert=modules_to_not_convert, modules_dtype_dict=modules_dtype_dict)
        quant_type = model_quant.get_quant_type(quant_args)
        dtype = dtype or devices.dtype

        def load_from_repo():
            nonlocal quant_args
            log.debug(f'Load model: transformer="{repo_id}" cls={cls_name.__name__} subfolder={subfolder} loader={get_loader("diffusers")} args={load_args}')
            if 'sdnq-' in repo_id.lower():
                quant_args = {}
            if dtype is not None:
                load_args['torch_dtype'] = dtype
            if subfolder is not None:
                load_args['subfolder'] = subfolder
            if variant is not None:
                load_args['variant'] = variant
            if use_safetensors:
                load_args['use_safetensors'] = True
            return cls_name.from_pretrained(
                repo_id,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
                **kwargs,
            )

        local_file = None
        override_name = None
        fallback = True

        from modules import sd_unet
        if shared.opts.sd_unet is not None and shared.opts.sd_unet != 'Default':
            if shared.opts.sd_unet not in list(sd_unet.unet_dict):
                log.error(f'Load module: type=transformer file="{shared.opts.sd_unet}" not found')
            elif os.path.exists(sd_unet.unet_dict[shared.opts.sd_unet]):
                local_file = sd_unet.unet_dict[shared.opts.sd_unet]
                override_name = shared.opts.sd_unet

        if repo_id.startswith(shared.opts.ckpt_dir) and os.path.exists(repo_id):
            log.error(f'Load model: transformer="{repo_id}" is incorrectly placed in the checkpoints folder')
            local_file = repo_id
            if shared.opts.allow_incomplete_model:
                log.warning(f'Load model: transformer="{repo_id}" is a local path, attempting to map to a HuggingFace for config fetch')
                repo_id = transformers_map.get(cls_name.__name__, repo_id)
                log.warning(f'Load model: transformer="{repo_id}" repo="{repo_id}" attempting to load...')
            else:
                return None
            fallback = False

        # 1. load gguf
        if local_file is not None and local_file.lower().endswith('.gguf'):
            log.debug(f'Load model: transformer="{local_file}" cls={cls_name.__name__} quant="{quant_type}" loader={get_loader("diffusers")} args={load_args}')
            from modules import ggml
            ggml.load_gguf_diffusers(local_file, cls=cls_name, compute_dtype=dtype, config=repo_id, subfolder=subfolder, variant=variant)
            # transformer = model_quant.do_post_load_quant(transformer, allow=quant_type is not None)

        # 2. load safetensors with native loader if spec is available
        elif local_file is not None and local_file.lower().endswith('.safetensors') and native_spec is not None:
            from pipelines import native_transformer
            log.debug(f'Load model: transformer="{local_file}" cls={cls_name.__name__} quant="{quant_type}" loader=native args={load_args}')
            try:
                transformer, _ = native_transformer.load(
                    local_file,
                    repo_id,
                    native_spec,
                    load_config,
                    allow_quant=allow_quant,
                    dtype=dtype,
                    modules_to_not_convert=modules_to_not_convert,
                    modules_dtype_dict=modules_dtype_dict,
                    quant_args=quant_args,
                    quant_type=quant_type,
                    **kwargs,
                )
            except native_transformer.OverrideArchMismatch as e:
                log.warning(f'Load model: transformer="{local_file}" override incompatible with cls={cls_name.__name__} ({e})')
                if fallback:
                    log.warning(f'Load model: transformer="{local_file}" ignoring override and loading base transformer')
                    shared.opts.data['sd_unet'] = 'Default'
                    sd_unet.loaded_unet = None
                    transformer = load_from_repo()

        # 3. load safetensors with diffusers loader
        elif local_file is not None and local_file.lower().endswith('.safetensors'):
            if dtype is not None:
                load_args['torch_dtype'] = dtype
            load_args.pop('device_map', None) # single-file uses different syntax
            loader = cls_name.from_single_file if hasattr(cls_name, 'from_single_file') else cls_name.from_pretrained
            log.debug(f'Load model: transformer="{local_file}" cls={cls_name.__name__} quant="{quant_type}" loader={get_loader("diffusers")} method={loader.__name__} args={load_args}')
            transformer = loader(
                local_file,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
                **kwargs,
            )

        # 4. default loading from diffusers repo (also the fallback when an
        # incompatible override is dropped above)
        else:
            transformer = load_from_repo()

        # mark the dropdown selection as loaded so the sd_unet onchange callback
        # does not force a redundant full reload for an already-consumed override
        if transformer is not None and override_name is not None and shared.opts.sd_unet == override_name:
            sd_unet.loaded_unet = override_name

        sd_models.allow_post_quant = False # we already handled it
        if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
            sd_models.move_model(transformer, devices.cpu)

        if transformer is not None and not hasattr(transformer, 'quantization_config'): # attach quantization_config
            if hasattr(transformer, 'config') and hasattr(transformer.config, 'quantization_config'):
                transformer.quantization_config = transformer.config.quantization_config
            elif (quant_type is not None) and (quant_args.get('quantization_config', None) is not None):
                transformer.quantization_config = quant_args.get('quantization_config', None)

    except Exception as e:
        log.error(f'Load model: transformer="{repo_id}" cls={cls_name.__name__} {e}')
        errors.display(e, 'Load')
        raise

    devices.torch_gc()
    shared.state.end(jobid)

    if transformer is not None:
        module_size, param_num = sd_offload.get_module_size(transformer)
        module_memory = sd_offload.get_module_memory(transformer)
        log.debug(f'Load model: transformer="{repo_id}" quant="{quant_type}" size={module_size:.3f} params={param_num:.3f} memory={module_memory}')

    try:
        actual_dtype = transformer.dtype
        if isinstance(actual_dtype, torch.dtype) and isinstance(dtype, torch.dtype) and actual_dtype != dtype:
            force = shared.opts.force_dtype
            log.warning(f'Load model: transformer="{repo_id}" dtype desired={dtype} actual={actual_dtype} force={force}')
            if force:
                transformer = transformer.to(dtype)
    except Exception:
        pass

    return transformer
