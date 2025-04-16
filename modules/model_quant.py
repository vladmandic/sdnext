import os
import sys
import copy
import time
import diffusers
import transformers
from installer import installed, install, log, setup_logging


ao = None
bnb = None
intel_nncf = None
optimum_quanto = None
quant_last_model_name = None
quant_last_model_device = None
debug = os.environ.get('SD_QUANT_DEBUG', None) is not None


def get_quant_type(args):
    if args is not None and "quantization_config" in args:
        return args['quantization_config'].__class__.__name__
    return None


def get_quant(name):
    if "qint8" in name.lower():
        return 'qint8'
    if "qint4" in name.lower():
        return 'qint4'
    if "fp8" in name.lower():
        return 'fp8'
    if "fp4" in name.lower():
        return 'fp4'
    if "nf4" in name.lower():
        return 'nf4'
    if name.endswith('.gguf'):
        return 'gguf'
    return 'none'


def create_bnb_config(kwargs = None, allow_bnb: bool = True, module: str = 'Model'):
    from modules import shared, devices
    if len(shared.opts.bnb_quantization) > 0 and allow_bnb:
        if 'Model' in shared.opts.bnb_quantization or (module is not None and module in shared.opts.bnb_quantization) or module == 'any':
            load_bnb()
            if bnb is None:
                return kwargs
            bnb_config = diffusers.BitsAndBytesConfig(
                load_in_8bit=shared.opts.bnb_quantization_type in ['fp8'],
                load_in_4bit=shared.opts.bnb_quantization_type in ['nf4', 'fp4'],
                bnb_4bit_quant_storage=shared.opts.bnb_quantization_storage,
                bnb_4bit_quant_type=shared.opts.bnb_quantization_type,
                bnb_4bit_compute_dtype=devices.dtype
            )
            log.debug(f'Quantization: module="{module}" type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
            if kwargs is None:
                return bnb_config
            else:
                kwargs['quantization_config'] = bnb_config
                return kwargs
    return kwargs


def create_ao_config(kwargs = None, allow_ao: bool = True, module: str = 'Model'):
    from modules import shared
    if len(shared.opts.torchao_quantization) > 0 and (shared.opts.torchao_quantization_mode == 'pre') and allow_ao:
        if 'Model' in shared.opts.torchao_quantization or (module is not None and module in shared.opts.torchao_quantization) or module == 'any':
            torchao = load_torchao()
            if torchao is None:
                return kwargs
            if module in {'TE', 'LLM'}:
                ao_config = transformers.TorchAoConfig(quant_type=shared.opts.torchao_quantization_type)
            else:
                ao_config = diffusers.TorchAoConfig(shared.opts.torchao_quantization_type)
            log.debug(f'Quantization: module="{module}" type=torchao dtype={shared.opts.torchao_quantization_type}')
            if kwargs is None:
                return ao_config
            else:
                kwargs['quantization_config'] = ao_config
                return kwargs
    return kwargs


def create_quanto_config(kwargs = None, allow_quanto: bool = True, module: str = 'Model'):
    from modules import shared
    if len(shared.opts.quanto_quantization) > 0 and allow_quanto:
        if 'Model' in shared.opts.quanto_quantization or (module is not None and module in shared.opts.quanto_quantization) or module == 'any':
            load_quanto(silent=True)
            if optimum_quanto is None:
                return kwargs
            if module in {'TE', 'LLM'}:
                quanto_config = transformers.QuantoConfig(weights=shared.opts.quanto_quantization_type)
                quanto_config.weights_dtype = quanto_config.weights
            else:
                quanto_config = diffusers.QuantoConfig(weights_dtype=shared.opts.quanto_quantization_type)
                quanto_config.activations = None # patch so it works with transformers
                quanto_config.weights = quanto_config.weights_dtype
            log.debug(f'Quantization: module="{module}" type=quanto dtype={shared.opts.quanto_quantization_type}')
            if kwargs is None:
                return quanto_config
            else:
                kwargs['quantization_config'] = quanto_config
                return kwargs
    return kwargs


def check_quant(module: str = ''):
    from modules import shared
    if 'Model' in shared.opts.bnb_quantization or 'Model' in shared.opts.torchao_quantization or 'Model' in shared.opts.quanto_quantization:
        return True
    if module in shared.opts.bnb_quantization or module in shared.opts.torchao_quantization or module in shared.opts.quanto_quantization:
        return True
    return False


def check_nunchaku(module: str = ''):
    from modules import shared
    if 'Model' not in shared.opts.nunchaku_quantization and module not in shared.opts.nunchaku_quantization:
        return False
    from modules import mit_nunchaku
    mit_nunchaku.install_nunchaku()
    if not mit_nunchaku.ok:
        return False
    return True


def create_config(kwargs = None, allow: bool = True, module: str = 'Model'):
    if kwargs is None:
        kwargs = {}
    kwargs = create_bnb_config(kwargs, allow_bnb=allow, module=module)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=bnb config={kwargs.get("quantization_config", None)}')
        return kwargs
    kwargs = create_ao_config(kwargs, allow_ao=allow, module=module)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=torchao config={kwargs.get("quantization_config", None)}')
        return kwargs
    kwargs = create_quanto_config(kwargs, allow_quanto=allow, module=module)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=quanto config={kwargs.get("quantization_config", None)}')
        return kwargs
    return kwargs


def load_torchao(msg='', silent=False):
    global ao # pylint: disable=global-statement
    if ao is not None:
        return ao
    if not installed('torchao'):
        install('torchao==0.10.0', quiet=True)
        log.warning('Quantization: torchao installed please restart')
    try:
        import torchao
        ao = torchao
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=torchao version={ao.__version__} fn={fn}') # pylint: disable=protected-access
        from diffusers.utils import import_utils
        import_utils.is_torchao_available = lambda: True
        import_utils._torchao_available = True # pylint: disable=protected-access
        return ao
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import torchao: {e}")
        ao = None
        if not silent:
            raise
    return None


def load_bnb(msg='', silent=False):
    from modules import devices
    global bnb # pylint: disable=global-statement
    if bnb is not None:
        return bnb
    if not installed('bitsandbytes'):
        if devices.backend == 'cuda':
            # forcing a version will uninstall the multi-backend-refactor branch of bnb
            install('bitsandbytes==0.45.1', quiet=True)
            log.warning('Quantization: bitsandbytes installed please restart')
    try:
        import bitsandbytes
        bnb = bitsandbytes
        from diffusers.utils import import_utils
        import_utils._bitsandbytes_available = True # pylint: disable=protected-access
        import_utils._bitsandbytes_version = '0.43.3' # pylint: disable=protected-access
        fn = f'{sys._getframe(3).f_code.co_name}:{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=bitsandbytes version={bnb.__version__} fn={fn}') # pylint: disable=protected-access
        return bnb
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import bitsandbytes: {e}")
        bnb = None
        if not silent:
            raise
    return None


def load_quanto(msg='', silent=False):
    global optimum_quanto # pylint: disable=global-statement
    if optimum_quanto is not None:
        return optimum_quanto
    if not installed('optimum-quanto'):
        install('optimum-quanto==0.2.7', quiet=True)
        log.warning('Quantization: optimum-quanto installed please restart')
    try:
        from optimum import quanto # pylint: disable=no-name-in-module
        # disable device specific tensors because the model can't be moved between cpu and gpu with them
        quanto.tensor.weights.qbits.WeightQBitsTensor.create = lambda *args, **kwargs: quanto.tensor.weights.qbits.WeightQBitsTensor(*args, **kwargs)
        optimum_quanto = quanto
        fn = f'{sys._getframe(3).f_code.co_name}:{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=quanto version={quanto.__version__} fn={fn}') # pylint: disable=protected-access
        from diffusers.utils import import_utils
        import_utils.is_optimum_quanto_available = lambda: True
        import_utils._optimum_quanto_available = True # pylint: disable=protected-access
        import_utils._optimum_quanto_version = quanto.__version__ # pylint: disable=protected-access
        import_utils._replace_with_quanto_layers = diffusers.quantizers.quanto.utils._replace_with_quanto_layers # pylint: disable=protected-access
        return optimum_quanto
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import optimum.quanto: {e}")
        optimum_quanto = None
        if not silent:
            raise
    return None


def load_nncf(msg='', silent=False):
    global intel_nncf # pylint: disable=global-statement
    if intel_nncf is not None:
        return intel_nncf
    if not installed('nncf'):
        install('nncf==2.7.0', quiet=True)
        log.warning('Quantization: nncf installed please restart')
    try:
        import nncf
        intel_nncf = nncf
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=nncf version={nncf.__version__} fn={fn}') # pylint: disable=protected-access
        return intel_nncf
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import nncf: {e}")
        intel_nncf = None
        if not silent:
            raise
    return None


def apply_layerwise(sd_model, quiet:bool=False):
    import torch
    from diffusers.quantizers import quantization_config
    from modules import shared, devices, sd_models
    if shared.opts.layerwise_quantization_storage == 'float8_e4m3fn' and hasattr(torch, 'float8_e4m3fn'):
        storage_dtype = torch.float8_e4m3fn
    elif shared.opts.layerwise_quantization_storage == 'float8_e5m2' and hasattr(torch, 'float8_e5m2'):
        storage_dtype = torch.float8_e5m2
    else:
        storage_dtype = None
        log.warning(f'Quantization: type=layerwise storage={shared.opts.layerwise_quantization_storage} not supported')
        return
    non_blocking = False
    if not hasattr(quantization_config.QuantizationMethod, 'LAYERWISE'):
        setattr(quantization_config.QuantizationMethod, 'LAYERWISE', 'layerwise') # noqa: B010
    for module in sd_models.get_signature(sd_model).keys():
        if not hasattr(sd_model, module):
            continue
        try:
            cls = getattr(sd_model, module).__class__.__name__
            if module.startswith('unet') and ('Model' in shared.opts.layerwise_quantization):
                m = getattr(sd_model, module)
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=non_blocking)
                    m.quantization_method = 'LayerWise'
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not non_blocking}')
            if module.startswith('transformer') and ('Model' in shared.opts.layerwise_quantization or 'Transformer' in shared.opts.layerwise_quantization):
                m = getattr(sd_model, module)
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=non_blocking)
                    m.quantization_method = 'LayerWise'
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not non_blocking}')
            if module.startswith('text_encoder') and ('Model' in shared.opts.layerwise_quantization or 'TE' in shared.opts.layerwise_quantization) and ('clip' not in cls.lower()):
                m = getattr(sd_model, module)
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=non_blocking)
                    m.quantization_method = quantization_config.QuantizationMethod.LAYERWISE # pylint: disable=no-member
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not non_blocking}')
        except Exception as e:
            log.error(f'Quantization: type=layerwise {e}')


def nncf_send_to_device(model, device):
    for child in model.children():
        if child.__class__.__name__ == "WeightsDecompressor":
            child.scale = child.scale.to(device)
            child.zero_point = child.zero_point.to(device)
        nncf_send_to_device(child, device)


def nncf_compress_model(model, op=None, sd_model=None):
    from modules import devices, shared
    global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
    nncf = load_nncf('Quantize model: type=NNCF')
    model.eval()
    backup_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        backup_embeddings = copy.deepcopy(model.get_input_embeddings())
    model = nncf.compress_weights(model)
    nncf_send_to_device(model, devices.device)
    if hasattr(model, "set_input_embeddings") and backup_embeddings is not None:
        model.set_input_embeddings(backup_embeddings)
    if op is not None and shared.opts.nncf_quantize_shuffle_weights:
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram or shared.opts.diffusers_offload_mode != "none":
            quant_last_model_name = op
            quant_last_model_device = model.device
        else:
            quant_last_model_name = None
            quant_last_model_device = None
        model.to(devices.device)
    devices.torch_gc(force=True)
    return model


def nncf_compress_weights(sd_model):
    try:
        t0 = time.time()
        from modules import shared, devices, sd_models
        log.info(f"Quantization: type=NNCF modules={shared.opts.nncf_compress_weights}")
        global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement

        sd_model = sd_models.apply_function_to_model(sd_model, nncf_compress_model, shared.opts.nncf_compress_weights, op="nncf")
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        quant_last_model_name = None
        quant_last_model_device = None

        t1 = time.time()
        log.info(f"Quantization: type=NNCF time={t1-t0:.2f}")
    except Exception as e:
        log.warning(f"Quantization: type=NNCF {e}")
    return sd_model


def optimum_quanto_model(model, op=None, sd_model=None, weights=None, activations=None):
    from modules import devices, shared
    quanto = load_quanto('Quantize model: type=Optimum Quanto')
    global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
    if sd_model is not None and "Flux" in sd_model.__class__.__name__: # LayerNorm is not supported
        exclude_list = ["transformer_blocks.*.norm1.norm", "transformer_blocks.*.norm2", "transformer_blocks.*.norm1_context.norm", "transformer_blocks.*.norm2_context", "single_transformer_blocks.*.norm.norm", "norm_out.norm"]
    else:
        exclude_list = None
    weights = getattr(quanto, weights) if weights is not None else getattr(quanto, shared.opts.optimum_quanto_weights_type)
    if activations is not None:
        activations = getattr(quanto, activations) if activations != 'none' else None
    elif shared.opts.optimum_quanto_activations_type != 'none':
        activations = getattr(quanto, shared.opts.optimum_quanto_activations_type)
    else:
        activations = None
    model.eval()
    backup_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        backup_embeddings = copy.deepcopy(model.get_input_embeddings())
    quanto.quantize(model, weights=weights, activations=activations, exclude=exclude_list)
    quanto.freeze(model)
    if hasattr(model, "set_input_embeddings") and backup_embeddings is not None:
        model.set_input_embeddings(backup_embeddings)
    if op is not None and shared.opts.optimum_quanto_shuffle_weights:
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram or shared.opts.diffusers_offload_mode != "none":
            quant_last_model_name = op
            quant_last_model_device = model.device
        else:
            quant_last_model_name = None
            quant_last_model_device = None
        model.to(devices.device)
    devices.torch_gc(force=True)
    return model


def optimum_quanto_weights(sd_model):
    try:
        t0 = time.time()
        from modules import shared, devices, sd_models
        if shared.opts.diffusers_offload_mode in {"balanced", "sequential"}:
            log.warning(f"Quantization: type=Optimum.quanto offload={shared.opts.diffusers_offload_mode} not compatible")
            return sd_model
        log.info(f"Quantization: type=Optimum.quanto: modules={shared.opts.optimum_quanto_weights}")
        global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
        quanto = load_quanto()

        sd_model = sd_models.apply_function_to_model(sd_model, optimum_quanto_model, shared.opts.optimum_quanto_weights, op="optimum-quanto")
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        quant_last_model_name = None
        quant_last_model_device = None

        if shared.opts.optimum_quanto_activations_type != 'none':
            activations = getattr(quanto, shared.opts.optimum_quanto_activations_type)
        else:
            activations = None

        if activations is not None:
            def optimum_quanto_freeze(model, op=None, sd_model=None): # pylint: disable=unused-argument
                quanto.freeze(model)
                return model
            if shared.opts.diffusers_offload_mode == "model":
                sd_model.enable_model_cpu_offload(device=devices.device)
                if hasattr(sd_model, "encode_prompt"):
                    original_encode_prompt = sd_model.encode_prompt
                    def encode_prompt(*args, **kwargs):
                        embeds = original_encode_prompt(*args, **kwargs)
                        sd_model.maybe_free_model_hooks() # Diffusers keeps the TE on VRAM
                        return embeds
                    sd_model.encode_prompt = encode_prompt
            else:
                sd_models.move_model(sd_model, devices.device)
            with quanto.Calibration(momentum=0.9):
                sd_model(prompt="dummy prompt", num_inference_steps=10)
            sd_model = sd_models.apply_function_to_model(sd_model, optimum_quanto_freeze, shared.opts.optimum_quanto_weights, op="optimum-quanto-freeze")
            if shared.opts.diffusers_offload_mode == "model":
                sd_models.disable_offload(sd_model)
                sd_models.move_model(sd_model, devices.cpu)
                if hasattr(sd_model, "encode_prompt"):
                    sd_model.encode_prompt = original_encode_prompt
            devices.torch_gc(force=True)

        t1 = time.time()
        log.info(f"Quantization: type=Optimum.quanto time={t1-t0:.2f}")
    except Exception as e:
        log.warning(f"Quantization: type=Optimum.quanto {e}")
    return sd_model


def torchao_quantization(sd_model):
    from modules import shared, devices, sd_models
    torchao = load_torchao()
    q = torchao.quantization

    fn = getattr(q, shared.opts.torchao_quantization_type, None)
    if fn is None:
        log.error(f"Quantization: type=TorchAO type={shared.opts.torchao_quantization_type} not supported")
        return sd_model
    def torchao_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
        q.quantize_(model, fn(), device=devices.device)
        return model

    log.info(f"Quantization: type=TorchAO pipe={sd_model.__class__.__name__} quant={shared.opts.torchao_quantization_type} fn={fn} targets={shared.opts.torchao_quantization}")
    try:
        t0 = time.time()
        sd_models.apply_function_to_model(sd_model, torchao_model, shared.opts.torchao_quantization, op="torchao")
        t1 = time.time()
        log.info(f"Quantization: type=TorchAO time={t1-t0:.2f}")
    except Exception as e:
        log.error(f"Quantization: type=TorchAO {e}")
    setup_logging() # torchao uses dynamo which messes with logging so reset is needed
    return sd_model


def get_dit_args(load_config:dict={}, module:str=None, device_map:bool=False, allow_quant:bool=True):
    from modules import shared, devices
    config = load_config.copy()
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
    if 'variant' in config:
        del config['variant']
    if device_map:
        if shared.opts.device_map == 'cpu':
            config['device_map'] = 'cpu'
        if shared.opts.device_map == 'gpu':
            config['device_map'] = devices.device
        if devices.backend == "ipex" and os.environ.get('UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS', '0') != '1' and module in {'TE', 'LLM'}:
            config['device_map'] = 'cpu' # alchemist gpus hits the 4GB allocation limit with transformers, UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS emulates above 4GB allocations
    if allow_quant:
        quant_args = create_config(module=module)
    else:
        quant_args = {}
    return config, quant_args
