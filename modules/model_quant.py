import os
import re
import sys
import copy
import json
import time
import diffusers
import transformers
from installer import installed, install, log, setup_logging


ao = None
bnb = None
optimum_quanto = None
trt = None
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


def dont_quant():
    from modules import shared
    models_list = re.split(r'[ ,]+', shared.opts.models_not_to_quant)
    models_list = [m.lower().strip() for m in models_list]
    if shared.sd_model_type.lower() in models_list:
        shared.log.debug(f'Quantization: model={shared.sd_model_type} skip')
        return True
    return False


def create_bnb_config(kwargs = None, allow: bool = True, module: str = 'Model', modules_to_not_convert: list = None):
    from modules import shared, devices
    if allow and (module == 'any' or module in shared.opts.bnb_quantization):
        load_bnb()
        if bnb is None:
            return kwargs
        bnb_config = diffusers.BitsAndBytesConfig(
            load_in_8bit=shared.opts.bnb_quantization_type in ['fp8'],
            load_in_4bit=shared.opts.bnb_quantization_type in ['nf4', 'fp4'],
            bnb_4bit_quant_storage=shared.opts.bnb_quantization_storage,
            bnb_4bit_quant_type=shared.opts.bnb_quantization_type,
            bnb_4bit_compute_dtype=devices.dtype,
            llm_int8_skip_modules=modules_to_not_convert,
        )
        log.debug(f'Quantization: module={module} type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
        if kwargs is None:
            return bnb_config
        else:
            kwargs['quantization_config'] = bnb_config
            return kwargs
    return kwargs


def create_ao_config(kwargs = None, allow: bool = True, module: str = 'Model', modules_to_not_convert: list = None):
    from modules import shared
    if allow and (shared.opts.torchao_quantization_mode in {'pre', 'auto'}) and (module == 'any' or module in shared.opts.torchao_quantization):
        torchao = load_torchao()
        if torchao is None:
            return kwargs
        if module in {'TE', 'LLM'}:
            ao_config = transformers.TorchAoConfig(quant_type=shared.opts.torchao_quantization_type, modules_to_not_convert=modules_to_not_convert)
        else:
            ao_config = diffusers.TorchAoConfig(shared.opts.torchao_quantization_type, modules_to_not_convert=modules_to_not_convert)
        log.debug(f'Quantization: module={module} type=torchao dtype={shared.opts.torchao_quantization_type}')
        if kwargs is None:
            return ao_config
        else:
            kwargs['quantization_config'] = ao_config
            return kwargs
    return kwargs


def create_quanto_config(kwargs = None, allow: bool = True, module: str = 'Model', modules_to_not_convert: list = None):
    from modules import shared
    if allow and (module == 'any' or module in shared.opts.quanto_quantization):
        load_quanto(silent=True)
        if optimum_quanto is None:
            return kwargs
        if module in {'TE', 'LLM'}:
            quanto_config = transformers.QuantoConfig(weights=shared.opts.quanto_quantization_type, modules_to_not_convert=modules_to_not_convert)
            quanto_config.weights_dtype = quanto_config.weights
        else:
            quanto_config = diffusers.QuantoConfig(weights_dtype=shared.opts.quanto_quantization_type, modules_to_not_convert=modules_to_not_convert)
            quanto_config.activations = None # patch so it works with transformers
            quanto_config.weights = quanto_config.weights_dtype
        log.debug(f'Quantization: module={module} type=quanto dtype={shared.opts.quanto_quantization_type}')
        if kwargs is None:
            return quanto_config
        else:
            kwargs['quantization_config'] = quanto_config
            return kwargs
    return kwargs


def create_trt_config(kwargs = None, allow: bool = True, module: str = 'Model', modules_to_not_convert: list = None):
    from modules import shared
    if allow and (module == 'any' or module in shared.opts.trt_quantization):
        load_trt()
        if trt is None:
            return kwargs
        trt_config_data = {
            "int8": {"quant_type": "INT8", "quant_method": "modelopt", "modules_to_not_convert": []},
            "int4": {"quant_type": "INT4", "quant_method": "modelopt", "block_quantize": 128, "channel_quantize": -1, "modules_to_not_convert": ["conv", "patch_embed"]},
            "fp8": {"quant_type": "FP8", "quant_method": "modelopt", "modules_to_not_convert": []},
            "nf4": {"quant_type": "NF4", "quant_method": "modelopt", "block_quantize": 128, "channel_quantize": -1, "scale_block_quantize": 8, "scale_channel_quantize": -1, "modules_to_not_convert": ["conv"]},
            "nvfp4": {"quant_type": "NVFP4", "quant_method": "modelopt", "block_quantize": 128, "channel_quantize": -1, "modules_to_not_convert": ["conv"]},
        }
        trt_quant_config = trt_config_data[shared.opts.trt_quantization_type].copy()
        if modules_to_not_convert is not None:
            for m in modules_to_not_convert:
                if m not in trt_quant_config['modules_to_not_convert']:
                    trt_quant_config['modules_to_not_convert'].append(m)
        trt_config = diffusers.quantizers.quantization_config.NVIDIAModelOptConfig(**trt_quant_config)
        log.debug(f'Quantization: module={module} type=tensorrt dtype={shared.opts.trt_quantization_type}')
        if kwargs is None:
            return trt_config
        else:
            kwargs['quantization_config'] = trt_config
            return kwargs
    return kwargs


def get_sdnq_devices(mode="pre"):
    from modules import devices, shared
    if shared.opts.device_map == "gpu":
        quantization_device = devices.device
        return_device = devices.device
    elif shared.opts.device_map == "cpu":
        quantization_device = devices.cpu
        return_device = devices.cpu
    elif shared.opts.diffusers_offload_mode in {"none", "model"} or (mode == "post" and shared.opts.sdnq_quantize_shuffle_weights):
        quantization_device = devices.device if shared.opts.sdnq_quantize_with_gpu else devices.cpu
        return_device = devices.device
    elif shared.opts.sdnq_quantize_with_gpu:
        quantization_device = devices.device
        return_device = devices.device if shared.opts.diffusers_to_gpu else devices.cpu
    else:
        quantization_device = None
        return_device = None
    return quantization_device, return_device


def create_sdnq_config(kwargs = None, allow: bool = True, module: str = 'Model', weights_dtype: str = None, quantized_matmul_dtype: str = None, modules_to_not_convert: list = None, modules_dtype_dict: dict = None):
    from modules import shared
    if allow and (shared.opts.sdnq_quantize_mode in {'pre', 'auto'}) and (module == 'any' or module in shared.opts.sdnq_quantize_weights):
        from modules.sdnq import SDNQConfig
        from modules.sdnq.common import use_torch_compile as sdnq_use_torch_compile

        if shared.opts.sdnq_use_quantized_matmul and not sdnq_use_torch_compile:
            shared.log.warning('SDNQ Quantized MatMul requires a working Triton install. Disabling Quantized MatMul.')
            shared.opts.sdnq_use_quantized_matmul = False

        if weights_dtype is None:
            if module in {"TE", "LLM"} and shared.opts.sdnq_quantize_weights_mode_te not in {"Same as model", "default"}:
                weights_dtype = shared.opts.sdnq_quantize_weights_mode_te
            else:
                weights_dtype = shared.opts.sdnq_quantize_weights_mode
        if weights_dtype is None or weights_dtype == 'none':
            return kwargs

        if quantized_matmul_dtype is None:
            if module in {"TE", "LLM"} and shared.opts.sdnq_quantize_matmul_mode_te not in {"Same as model", "default"}:
                quantized_matmul_dtype = shared.opts.sdnq_quantize_matmul_mode_te
            else:
                quantized_matmul_dtype = shared.opts.sdnq_quantize_matmul_mode
        if quantized_matmul_dtype == "auto":
            quantized_matmul_dtype = None

        if modules_to_not_convert is None:
            modules_to_not_convert = []
        if modules_dtype_dict is None:
            modules_dtype_dict = {}

        sdnq_modules_to_not_convert = [m.strip() for m in re.split(';|,| ', shared.opts.sdnq_modules_to_not_convert) if len(m.strip()) > 1]
        if len(sdnq_modules_to_not_convert) > 0:
            modules_to_not_convert.extend(sdnq_modules_to_not_convert)

        try:
            if len(shared.opts.sdnq_modules_dtype_dict) > 2:
                sdnq_modules_dtype_dict = shared.opts.sdnq_modules_dtype_dict
                if "{" not in sdnq_modules_dtype_dict:
                    sdnq_modules_dtype_dict = "{" + sdnq_modules_dtype_dict + "}"
                sdnq_modules_dtype_dict = json.loads(bytes(sdnq_modules_dtype_dict, 'utf-8'))
                for key, value in sdnq_modules_dtype_dict.items():
                    if isinstance(value, str):
                        value = [m.strip() for m in re.split(';|,| ', value) if len(m.strip()) > 1]
                    if key not in modules_dtype_dict.keys():
                        modules_dtype_dict[key] = value
                    else:
                        modules_dtype_dict[key].extend(value)
        except Exception as e:
            log.warning(f'Quantization: SDNQ failed to parse sdnq_modules_dtype_dict: {e}')

        quantization_device, return_device = get_sdnq_devices(mode="pre")

        sdnq_config = SDNQConfig(
            weights_dtype=weights_dtype,
            quantized_matmul_dtype=quantized_matmul_dtype,
            group_size=shared.opts.sdnq_quantize_weights_group_size,
            svd_rank=shared.opts.sdnq_svd_rank,
            svd_steps=shared.opts.sdnq_svd_steps,
            dynamic_loss_threshold=shared.opts.sdnq_dynamic_loss_threshold,
            use_svd=shared.opts.sdnq_use_svd,
            quant_conv=shared.opts.sdnq_quantize_conv_layers,
            use_quantized_matmul=shared.opts.sdnq_use_quantized_matmul,
            use_quantized_matmul_conv=shared.opts.sdnq_use_quantized_matmul_conv,
            use_dynamic_quantization=shared.opts.sdnq_use_dynamic_quantization,
            dequantize_fp32=shared.opts.sdnq_dequantize_fp32,
            non_blocking=shared.opts.diffusers_offload_nonblocking,
            quantization_device=quantization_device,
            return_device=return_device,
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict.copy(),
        )
        if quantized_matmul_dtype is None:
            quantized_matmul_dtype = "auto" # set for logging
        svd = f'{shared.opts.sdnq_use_svd} rank={shared.opts.sdnq_svd_rank} steps={shared.opts.sdnq_svd_steps}' if shared.opts.sdnq_use_svd else f'{shared.opts.sdnq_use_svd}'
        log.debug(f'Quantization: module="{module}" type=sdnq mode=pre dtype={weights_dtype} svd={svd} dynamic={shared.opts.sdnq_use_dynamic_quantization} group={shared.opts.sdnq_quantize_weights_group_size} loss={shared.opts.sdnq_dynamic_loss_threshold} matmul_dtype={quantized_matmul_dtype} matmul_quant={shared.opts.sdnq_use_quantized_matmul} matmul_conv={shared.opts.sdnq_use_quantized_matmul_conv}  quant_conv={shared.opts.sdnq_quantize_conv_layers} fp32={shared.opts.sdnq_dequantize_fp32} device={quantization_device} return={return_device} use_gpu={shared.opts.sdnq_quantize_with_gpu} map={shared.opts.device_map} offload={shared.opts.diffusers_offload_mode} non_blocking={shared.opts.diffusers_offload_nonblocking} skip_modules={modules_to_not_convert} dict={modules_dtype_dict}')
        if kwargs is None:
            return sdnq_config
        else:
            kwargs['quantization_config'] = sdnq_config
            return kwargs
    return kwargs


def check_quant(module: str = ''):
    from modules import shared
    if module in shared.opts.sdnq_quantize_weights or module in shared.opts.bnb_quantization or module in shared.opts.torchao_quantization or module in shared.opts.quanto_quantization:
        return True
    return False


def check_nunchaku(module: str = ''):
    from modules import shared
    model_name = getattr(shared.opts, 'sd_model_checkpoint', '')
    if '+nunchaku' not in model_name:
        return False
    base_path = model_name.split('+')[0]
    for v in shared.reference_models.values():
        if v.get('path', '') != base_path:
            continue
        nunchaku_modules = v.get('nunchaku', None)
        if nunchaku_modules is None:
            continue
        if isinstance(nunchaku_modules, bool) and nunchaku_modules:
            nunchaku_modules = ['Model', 'TE']
        if not isinstance(nunchaku_modules, list):
            continue
        if module in nunchaku_modules:
            from modules import mit_nunchaku
            mit_nunchaku.install_nunchaku()
            return mit_nunchaku.ok
    return False


def create_config(kwargs = None, allow: bool = True, module: str = 'Model', modules_to_not_convert: list = None, modules_dtype_dict: dict = None):
    if kwargs is None:
        kwargs = {}
    if module == 'Model' and dont_quant():
        return kwargs
    kwargs = create_sdnq_config(kwargs, allow=allow, module=module, modules_to_not_convert=modules_to_not_convert, modules_dtype_dict=modules_dtype_dict)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=sdnq config={kwargs.get("quantization_config", None)}')
        return kwargs
    kwargs = create_bnb_config(kwargs, allow=allow, module=module, modules_to_not_convert=modules_to_not_convert)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=bnb config={kwargs.get("quantization_config", None)}')
        return kwargs
    kwargs = create_quanto_config(kwargs, allow=allow, module=module, modules_to_not_convert=modules_to_not_convert)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=quanto config={kwargs.get("quantization_config", None)}')
        return kwargs
    kwargs = create_ao_config(kwargs, allow=allow, module=module, modules_to_not_convert=modules_to_not_convert)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=torchao config={kwargs.get("quantization_config", None)}')
        return kwargs
    kwargs = create_trt_config(kwargs, allow=allow, module=module, modules_to_not_convert=modules_to_not_convert)
    if kwargs is not None and 'quantization_config' in kwargs:
        if debug:
            log.trace(f'Quantization: type=tensorrt config={kwargs.get("quantization_config", None)}')
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
            install('bitsandbytes==0.47.0', quiet=True)
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


def load_trt(msg='', silent=False):
    global trt # pylint: disable=global-statement
    if trt is not None:
        return trt
    try:
        install('nvidia-modelopt')
        import pydantic
        if pydantic.__version__.startswith('1'):
            log.error('Quantization: type=tensorrt pydantic==2 required')
            return None
        import modelopt
        trt = modelopt
        fn = f'{sys._getframe(3).f_code.co_name}:{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=tensorrt version={trt.__version__} fn={fn}') # pylint: disable=protected-access
        return trt
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import tensorrt: {e}")
        trt = None
        if not silent:
            raise
    return None


def upcast_non_layerwise_modules(model, dtype): # pylint: disable=unused-argument
    from diffusers.hooks.layerwise_casting import _GO_LC_SUPPORTED_PYTORCH_LAYERS
    model_children = list(model.children())
    if not model_children:
        if not isinstance(model, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
            model = model.to(dtype)
        return model
    for module in model_children:
        has_children = list(module.children())
        if not has_children:
            if not isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
                module = module.to(dtype)
        else:
            module = upcast_non_layerwise_modules(module, dtype)
    return model


def load_fp8_model_layerwise(checkpoint_info, load_model_func, diffusers_load_config):
    model = None
    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path
    try:
        import torch
        from modules import devices, shared
        from diffusers.quantizers import quantization_config
        if not hasattr(quantization_config.QuantizationMethod, 'LAYERWISE'):
            setattr(quantization_config.QuantizationMethod, 'LAYERWISE', 'layerwise') # noqa: B010
        if "e5m2" in repo_path.lower():
            storage_dtype = torch.float8_e5m2
        else:
            storage_dtype= torch.float8_e4m3fn
        load_args = diffusers_load_config.copy()
        load_args["torch_dtype"] = storage_dtype
        model = load_model_func(repo_path, **load_args)
        model = upcast_non_layerwise_modules(model, devices.dtype)
        model._skip_layerwise_casting_patterns = None # pylint: disable=protected-access
        model.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=shared.opts.diffusers_offload_nonblocking, skip_modules_pattern=[])
        model.layerwise_storage_dtype = storage_dtype
        model.quantization_method = 'LayerWise'
    except Exception as e:
        log.error(f"Load model: Failed to load FP8 model: {e}")
        model = None
    return model


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
    if not hasattr(quantization_config.QuantizationMethod, 'LAYERWISE'):
        setattr(quantization_config.QuantizationMethod, 'LAYERWISE', 'layerwise') # noqa: B010
    for module in sd_models.get_signature(sd_model).keys():
        if not hasattr(sd_model, module):
            continue
        try:
            cls = getattr(sd_model, module).__class__.__name__
            m = getattr(sd_model, module)
            if getattr(m, "quantization_method", None) in {'LayerWise', quantization_config.QuantizationMethod.LAYERWISE}: # pylint: disable=no-member
                storage_dtype = getattr(m, "layerwise_storage_dtype", storage_dtype)
                m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=shared.opts.diffusers_offload_nonblocking)
            elif module.startswith('unet') and ('Model' in shared.opts.layerwise_quantization):
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=shared.opts.diffusers_offload_nonblocking)
                    m.layerwise_storage_dtype = storage_dtype
                    m.quantization_method = 'LayerWise'
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not shared.opts.diffusers_offload_nonblocking}')
            elif module.startswith('transformer') and ('Model' in shared.opts.layerwise_quantization):
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=shared.opts.diffusers_offload_nonblocking)
                    m.layerwise_storage_dtype = storage_dtype
                    m.quantization_method = 'LayerWise'
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not shared.opts.diffusers_offload_nonblocking}')
            elif module.startswith('text_encoder') and ('TE' in shared.opts.layerwise_quantization) and ('clip' not in cls.lower()):
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=shared.opts.diffusers_offload_nonblocking)
                    m.layerwise_storage_dtype = storage_dtype
                    m.quantization_method = quantization_config.QuantizationMethod.LAYERWISE # pylint: disable=no-member
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not shared.opts.diffusers_offload_nonblocking}')
        except Exception as e:
            if 'Hook with name' not in str(e):
                log.error(f'Quantization: type=layerwise {e}')


def sdnq_quantize_model(model, op=None, sd_model=None, do_gc: bool = True, weights_dtype: str = None, quantized_matmul_dtype: str = None, modules_to_not_convert: list = None, modules_dtype_dict: dict = None):
    global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
    from modules import devices, shared, timer
    from modules.sdnq import sdnq_post_load_quant
    from modules.sdnq.common import use_torch_compile as sdnq_use_torch_compile

    if shared.opts.sdnq_use_quantized_matmul and not sdnq_use_torch_compile:
        shared.log.warning('SDNQ Quantized MatMul requires a working Triton install. Disabling Quantized MatMul.')
        shared.opts.sdnq_use_quantized_matmul = False

    if weights_dtype is None:
        if (op is not None) and ("text_encoder" in op or op in {"TE", "LLM"}) and (shared.opts.sdnq_quantize_weights_mode_te not in {"Same as model", "default"}):
            weights_dtype = shared.opts.sdnq_quantize_weights_mode_te
        else:
            weights_dtype = shared.opts.sdnq_quantize_weights_mode
    if weights_dtype is None or weights_dtype == 'none':
        return model

    if quantized_matmul_dtype is None:
        if (op is not None) and ("text_encoder" in op or op in {"TE", "LLM"}) and (shared.opts.sdnq_quantize_matmul_mode_te not in {"Same as model", "default"}):
            quantized_matmul_dtype = shared.opts.sdnq_quantize_matmul_mode_te
        else:
            quantized_matmul_dtype = shared.opts.sdnq_quantize_matmul_mode
    if quantized_matmul_dtype == "auto":
        quantized_matmul_dtype = None

    quantization_device, return_device = get_sdnq_devices(mode="post")

    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}

    sdnq_modules_to_not_convert = [m.strip() for m in re.split(';|,| ', shared.opts.sdnq_modules_to_not_convert) if len(m.strip()) > 1]
    if len(sdnq_modules_to_not_convert) > 0:
        modules_to_not_convert.extend(sdnq_modules_to_not_convert)

    try:
        if len(shared.opts.sdnq_modules_dtype_dict) > 2:
            sdnq_modules_dtype_dict = shared.opts.sdnq_modules_dtype_dict
            if "{" not in sdnq_modules_dtype_dict:
                sdnq_modules_dtype_dict = "{" + sdnq_modules_dtype_dict + "}"
            sdnq_modules_dtype_dict = json.loads(bytes(sdnq_modules_dtype_dict, 'utf-8'))
            for key, value in sdnq_modules_dtype_dict.items():
                if isinstance(value, str):
                    value = [m.strip() for m in re.split(';|,| ', value) if len(m.strip()) > 1]
                if key not in modules_dtype_dict.keys():
                    modules_dtype_dict[key] = value
                else:
                    modules_dtype_dict[key].extend(value)
    except Exception as e:
        log.warning(f'Quantization: SDNQ failed to parse sdnq_modules_dtype_dict: {e}')

    t0 = time.time()

    model = sdnq_post_load_quant(
        model,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        torch_dtype=devices.dtype,
        group_size=shared.opts.sdnq_quantize_weights_group_size,
        svd_rank=shared.opts.sdnq_svd_rank,
        svd_steps=shared.opts.sdnq_svd_steps,
        dynamic_loss_threshold=shared.opts.sdnq_dynamic_loss_threshold,
        use_svd=shared.opts.sdnq_use_svd,
        quant_conv=shared.opts.sdnq_quantize_conv_layers,
        use_quantized_matmul=shared.opts.sdnq_use_quantized_matmul,
        use_quantized_matmul_conv=shared.opts.sdnq_use_quantized_matmul_conv,
        use_dynamic_quantization=shared.opts.sdnq_use_dynamic_quantization,
        dequantize_fp32=shared.opts.sdnq_dequantize_fp32,
        non_blocking=shared.opts.diffusers_offload_nonblocking,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict.copy(),
    )

    t1 = time.time()
    timer.load.add('sdnq', t1 - t0)

    if op is not None and shared.opts.sdnq_quantize_shuffle_weights:
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            if do_gc:
                devices.torch_gc(force=True, reason='sdnq')
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram or shared.opts.diffusers_offload_mode != "none":
            quant_last_model_name = op
            quant_last_model_device = model.device
        else:
            quant_last_model_name = None
            quant_last_model_device = None
        model.to(devices.device)
    elif (shared.opts.diffusers_offload_mode != "none") and (not shared.opts.diffusers_to_gpu):
        model = model.to(devices.cpu)
    if do_gc:
        devices.torch_gc(force=True, reason='sdnq')

    if quantized_matmul_dtype is None:
        quantized_matmul_dtype = "auto" # set for logging
    log.debug(f'Quantization: module="{op if op is not None else model.__class__}" type=sdnq mode=post dtype={weights_dtype} matmul_dtype={quantized_matmul_dtype} matmul={shared.opts.sdnq_use_quantized_matmul} svd={shared.opts.sdnq_use_svd}dynamic={shared.opts.sdnq_use_dynamic_quantization}:group={shared.opts.sdnq_quantize_weights_group_size}:rank={shared.opts.sdnq_svd_rank}:steps={shared.opts.sdnq_svd_steps}:loss={shared.opts.sdnq_dynamic_loss_threshold} quant_conv={shared.opts.sdnq_quantize_conv_layers} matmul_conv={shared.opts.sdnq_use_quantized_matmul_conv} fp32={shared.opts.sdnq_dequantize_fp32} gpu={shared.opts.sdnq_quantize_with_gpu} device={quantization_device} return={return_device} map={shared.opts.device_map} non_blocking={shared.opts.diffusers_offload_nonblocking} modules_skip={modules_to_not_convert} modules_dtype={modules_dtype_dict}')
    return model


def sdnq_quantize_weights(sd_model):
    try:
        t0 = time.time()
        from modules import shared, devices, sd_models
        log.debug(f"Quantization: type=SDNQ modules={shared.opts.sdnq_quantize_weights} dtype={shared.opts.sdnq_quantize_weights_mode} dtype_te={shared.opts.sdnq_quantize_weights_mode_te} offload={shared.opts.diffusers_offload_mode} pre_forward={shared.opts.diffusers_offload_pre}")
        global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement

        sd_model = sd_models.apply_function_to_model(sd_model, sdnq_quantize_model, shared.opts.sdnq_quantize_weights, op="sdnq")
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True, reason='sdnq')
        quant_last_model_name = None
        quant_last_model_device = None

        t1 = time.time()
        log.info(f"Quantization: type=SDNQ time={t1-t0:.2f}")
    except Exception as e:
        log.warning(f"Quantization: type=SDNQ {e}")
        from modules import errors
        errors.display(e, 'Quantization')
    return sd_model


def optimum_quanto_model(model, op=None, sd_model=None, weights=None, activations=None):
    from modules import devices, shared
    quanto = load_quanto('Quantize model: type=Optimum Quanto')
    global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
    if model.__class__.__name__ in {"FluxTransformer2DModel", "ChromaTransformer2DModel"}: # LayerNorm is not supported
        exclude_list = ["transformer_blocks.*.norm1.norm", "transformer_blocks.*.norm2", "transformer_blocks.*.norm1_context.norm", "transformer_blocks.*.norm2_context", "single_transformer_blocks.*.norm.norm", "norm_out.norm"]
        if model.__class__.__name__ == "ChromaTransformer2DModel":
            # we ignore the distilled guidance layer because it degrades quality too much
            # see: https://github.com/huggingface/diffusers/pull/11698#issuecomment-2969717180 for more details
            exclude_list.append("distilled_guidance_layer.*")
    elif model.__class__.__name__ == "QwenImageTransformer2DModel":
        exclude_list = ["transformer_blocks.0.img_mod.1.weight", "time_text_embed", "img_in", "txt_in", "proj_out", "norm_out", "pos_embed"]
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
            devices.torch_gc(force=True, reason='quanto')
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram or shared.opts.diffusers_offload_mode != "none":
            quant_last_model_name = op
            quant_last_model_device = model.device
        else:
            quant_last_model_name = None
            quant_last_model_device = None
        model.to(devices.device)
    devices.torch_gc(force=True, reason='quanto')
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
            devices.torch_gc(force=True, reason='quanto')
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
            devices.torch_gc(force=True, reason='quanto')

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


def get_dit_args(load_config:dict=None, module:str=None, device_map:bool=False, allow_quant:bool=True, modules_to_not_convert: list = None, modules_dtype_dict: dict = None):
    from modules import shared, devices
    config = {} if load_config is None else load_config.copy()
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
    # if 'variant' in config:
    #     del config['variant']
    if device_map:
        if devices.backend == 'ipex' and os.environ.get('UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS', '0') != '1' and module in {'TE', 'LLM'}:
            config['device_map'] = 'cpu' # alchemist gpus hits the 4GB allocation limit with transformers, UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS emulates above 4GB allocations
        elif shared.opts.device_map == 'cpu':
            config['device_map'] = 'cpu'
        elif shared.opts.device_map == 'gpu':
            config['device_map'] = devices.device
    if allow_quant:
        quant_args = create_config(module=module, modules_to_not_convert=modules_to_not_convert, modules_dtype_dict=modules_dtype_dict)
    else:
        quant_args = {}
    return config, quant_args


def do_post_load_quant(sd_model, allow=True):
    from modules import shared
    if dont_quant():
        return sd_model
    if shared.opts.sdnq_quantize_weights and (shared.opts.sdnq_quantize_mode == 'post' or (allow and shared.opts.sdnq_quantize_mode == 'auto')):
        shared.log.debug('Load model: post_quant=sdnq')
        sd_model = sdnq_quantize_weights(sd_model)
    if len(shared.opts.optimum_quanto_weights) > 0:
        shared.log.debug('Load model: post_quant=quanto')
        sd_model = optimum_quanto_weights(sd_model)
    if shared.opts.torchao_quantization and (shared.opts.torchao_quantization_mode == 'post' or (allow and shared.opts.torchao_quantization_mode == 'auto')):
        shared.log.debug('Load model: post_quant=torchao')
        sd_model = torchao_quantization(sd_model)
    if shared.opts.layerwise_quantization:
        shared.log.debug('Load model: post_quant=layerwise')
        apply_layerwise(sd_model)
    return sd_model
