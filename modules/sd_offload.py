import os
import re
import sys
import time
import math
import inspect
import itertools
import torch
import accelerate.hooks
import accelerate.utils.modeling
from modules.logger import log
from modules import shared, devices, errors, model_quant, sd_models, sd_offload_aux
from modules.timer import process as process_timer


debug = os.environ.get('SD_MOVE_DEBUG', None) is not None
verbose = os.environ.get('SD_MOVE_VERBOSE', None) is not None
debug_move = log.trace if debug else lambda *args, **kwargs: None
offload_allow_none = ['sd', 'sdxl']
offload_post = ['h1']
offload_hook_instance = None
group_offload_vae_limit = 1.0 # GB; vae-class components above this rest on cpu and onload whole at encode/decode
balanced_offload_exclude = ['CogView4Pipeline', 'MeissonicPipeline']
no_split_module_classes = [
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Embedding",
    "SDNQLinear", "SDNQConv1d", "SDNQConv2d", "SDNQConv3d", "SDNQConvTranspose1d", "SDNQConvTranspose2d", "SDNQConvTranspose3d", "SDNQEmbedding",
    "WanTransformerBlock",
    "MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock",
]
accelerate_dtype_byte_size = None
move_stream = None


def dtype_byte_size(dtype: torch.dtype):
    try:
        if dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]:
            dtype = accelerate.utils.modeling.CustomDtype.FP8
    except Exception: # catch since older torch many not have defined dtypes
        pass
    return accelerate_dtype_byte_size(dtype)


def get_signature(cls):
    signature = inspect.signature(cls.__init__, follow_wrapped=True)
    return signature.parameters


def disable_offload(sd_model):
    remove_group_offload(sd_model) # group hooks block the meta move at unload, keeping component weights alive for as long as any reference to the pipe survives
    if not getattr(sd_model, 'has_accelerate', False):
        return
    for module_name in get_module_names(sd_model):
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module):
            network_layer_name = getattr(module, "network_layer_name", None)
            try:
                module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            except Exception as e:
                log.warning(f'Offload: remove hook module={module_name} {e}')
            if network_layer_name:
                module.network_layer_name = network_layer_name
    sd_model.has_accelerate = False


def set_accelerate(sd_model):
    def set_accelerate_to_module(model):
        if hasattr(model, "pipe"):
            set_accelerate_to_module(model.pipe)
        for module_name in get_module_names(model):
            component = getattr(model, module_name, None)
            if isinstance(component, torch.nn.Module):
                component.has_accelerate = True

    sd_model.has_accelerate = True
    set_accelerate_to_module(sd_model)
    if hasattr(sd_model, "prior_pipe"):
        set_accelerate_to_module(sd_model.prior_pipe)
    if hasattr(sd_model, "decoder_pipe"):
        set_accelerate_to_module(sd_model.decoder_pipe)


def group_offload_config(main: bool) -> dict:
    """Effective group offload settings for one component. Components that run once per
    generation take the leaf no-stream policy regardless of the main settings, so their
    weights are never held in pinned host memory."""
    stream = shared.opts.group_offload_stream if main else False
    blocks = max(1, int(shared.opts.group_offload_blocks))
    if stream and blocks != 1:
        blocks = 1 # streamed prefetch supports one block per group; upstream clamps with a warning otherwise
    return {
        'offload_type': shared.opts.group_offload_type if main else 'leaf_level',
        'num_blocks_per_group': blocks,
        'non_blocking': shared.opts.diffusers_offload_nonblocking,
        'use_stream': stream,
        'record_stream': shared.opts.group_offload_record and stream, # record without streams is rejected upstream
        'low_cpu_mem_usage': stream and not shared.opts.group_offload_pin,
    }


def remove_group_offload_component(module) -> bool:
    if getattr(module, 'sdnext_group_offload_sig', None) is None:
        module = getattr(module, 'model', None) # wrapper components carry the hooks on the inner model
        if module is None or getattr(module, 'sdnext_group_offload_sig', None) is None:
            return False
    from diffusers.hooks.group_offloading import _GROUP_OFFLOADING, _LAYER_EXECUTION_TRACKER, _LAZY_PREFETCH_GROUP_OFFLOADING
    from diffusers.hooks.hooks import HookRegistry
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.remove_hook(_GROUP_OFFLOADING, recurse=True)
    registry.remove_hook(_LAYER_EXECUTION_TRACKER, recurse=True)
    registry.remove_hook(_LAZY_PREFETCH_GROUP_OFFLOADING, recurse=True)
    module.sdnext_group_offload_sig = None
    return True


def remove_group_offload(sd_model):
    removed = []
    for module_name in get_module_names(sd_model):
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module) and remove_group_offload_component(module):
            removed.append(module_name)
    for module_name in getattr(sd_model, 'sdnext_ondemand_modules', None) or []:
        module = getattr(sd_model, module_name, None)
        if module is not None:
            module.sdnext_ondemand = False
            if hasattr(module, '_hf_hook'):
                module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            removed.append(f'{module_name}:ondemand')
    if getattr(sd_model, 'sdnext_ondemand_modules', None):
        sd_model.sdnext_ondemand_modules = []
    if removed:
        log.debug(f'Offload: type=group op=remove modules={removed}')


def apply_group_offload_component(module, module_name: str, main: bool) -> bool:
    """Apply group offload to one component. Re-application with unchanged settings is a no-op:
    the hooks silently keep their original config when re-applied and raise before the first
    forward, so a changed config must remove the old hooks first."""
    from diffusers.hooks import apply_group_offloading
    cfg = group_offload_config(main)
    if cfg['use_stream'] and not cfg['low_cpu_mem_usage']:
        size_gb, _params = get_module_size(module)
        pin_ok = getattr(module, 'sdnext_group_offload_pin', None)
        if pin_ok is None: # decide once per module: a granted pin moves the weights into locked memory, so re-reading available on the next apply would see it lower by the pinned size and revoke its own grant
            from modules import memstats
            avail_gb = memstats.ram_stats().get('avail', 0)
            reserve_gb = max(8.0, 0.25 * shared.cpu_memory) # pinned pages cannot be reclaimed or swapped, so a quarter of the machine, floored at 8 GB, stays pageable for the process and page cache
            limit_gb = (avail_gb - reserve_gb) if avail_gb > 0 else (0.5 * shared.cpu_memory) # budget from memory free right now; total-derived ceiling only when psutil cannot say
            pin_ok = size_gb <= limit_gb
            module.sdnext_group_offload_pin = pin_ok
            module.sdnext_group_offload_pin_limit = limit_gb
        if not pin_ok:
            # unpinned streaming degrades to per-transfer staging and leaf groups make that a per-module cost,
            # so the whole leaf+stream shape goes with the pin: few large synchronous groups instead
            cfg['low_cpu_mem_usage'] = True
            cfg['use_stream'] = False
            cfg['record_stream'] = False
            cfg['offload_type'] = 'block_level'
            cfg['num_blocks_per_group'] = max(4, int(shared.opts.group_offload_blocks))
            log.warning(f'Offload: type=group module={module_name} size={size_gb:.3f} limit={getattr(module, "sdnext_group_offload_pin_limit", 0):.3f} pin=denied type=block_level blocks={cfg["num_blocks_per_group"]} expect ~{size_gb:.0f} GB transferred per step')
    sig = f'{devices.device}:{main}:' + ':'.join(str(v) for v in cfg.values())
    if getattr(module, 'sdnext_group_offload_sig', None) == sig:
        return False
    if hasattr(module, '_hf_hook'): # leftover accelerate hooks from a previous offload mode abort the group apply upstream
        module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
    remove_group_offload_component(module)
    module.requires_grad_(False)
    log.debug(f'Offload: type=group op=apply module={module_name} pin={cfg["use_stream"] and not cfg["low_cpu_mem_usage"]}') # before the apply: pinning large components takes a while and would otherwise run silently
    apply_group_offloading(module, onload_device=devices.device, offload_device=devices.cpu, **cfg)
    module.sdnext_group_offload_sig = sig
    return True


def set_group_resident(module):
    """VAE-class components never take group hooks: the hooks are forward-scoped, while
    pipelines enter through encode/decode, and tiled calls re-enter per tile."""
    if hasattr(module, '_hf_hook'):
        module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
    remove_group_offload_component(module)
    module.requires_grad_(False)
    module.to(devices.device)


def group_offload_role(module_name: str, module) -> str:
    cls = module.__class__.__name__
    if 'vae' in module_name.lower() or cls.startswith(('Autoencoder', 'VQModel', 'AsymmetricAutoencoder', 'ConsistencyDecoder')):
        return 'resident'
    if module_name.startswith(('text_encoder', 'image_encoder', 'safety_checker')):
        return 'aux'
    return 'main'


def has_entry_bridge(module) -> bool:
    """Entry points decorated with diffusers' apply_forward_hook fire _hf_hook.pre_forward,
    which is what carries the on-demand onload for encode and decode calls that bypass forward."""
    for name in ('decode', 'encode'):
        fn = getattr(module, name, None)
        if fn is not None and getattr(fn, '__qualname__', '').startswith('apply_forward_hook'):
            return True
    return False


class OnDemandHook(accelerate.hooks.ModelHook):
    """Whole-module onload for components entered through decode or encode rather than forward.
    Tiled calls re-enter inside one entry point, so the module is on device before the first
    tile; the return to cpu happens at the processing seams once outputs are materialized."""
    def pre_forward(self, module, *args, **kwargs):
        param = next(module.parameters(), None)
        if param is not None and not devices.same_device(param.device, devices.device):
            t0 = time.time()
            module.to(devices.device)
            dt = time.time() - t0
            process_timer.add('onload', dt)
            log.debug(f'Offload: type=ondemand op=onload module={module.__class__.__name__} time={dt:.3f}')
        return args, kwargs


def set_group_vae(sd_model, module, module_name: str) -> str:
    """Placement policy for vae-class components, which never take group hooks. Small
    components stay resident; components above group_offload_vae_limit rest on cpu and
    onload whole when their decode or encode entry point fires."""
    size_gb, _params = get_module_size(module)
    if size_gb < group_offload_vae_limit or not has_entry_bridge(module):
        set_group_resident(module)
        module.sdnext_ondemand = False # a lingering stamp would let the seams offload a component with no onload hook
        names = getattr(sd_model, 'sdnext_ondemand_modules', None) or []
        if module_name in names:
            sd_model.sdnext_ondemand_modules = [n for n in names if n != module_name]
        return 'resident'
    if not getattr(module, 'sdnext_ondemand', False) or not hasattr(module, '_hf_hook'):
        if hasattr(module, '_hf_hook'):
            module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
        remove_group_offload_component(module)
        module.requires_grad_(False)
        accelerate.hooks.add_hook_to_module(module, OnDemandHook(), append=False)
        module.sdnext_ondemand = True
        module.to(devices.cpu)
    names = getattr(sd_model, 'sdnext_ondemand_modules', None) or []
    if module_name not in names:
        sd_model.sdnext_ondemand_modules = names + [module_name]
    return 'ondemand'


def offload_ondemand(sd_model):
    """Return on-demand components to cpu once their outputs are materialized."""
    if sd_model is None:
        return
    names = getattr(sd_model, 'sdnext_ondemand_modules', None)
    if not names and hasattr(sd_model, 'pipe'):
        sd_model = sd_model.pipe
        names = getattr(sd_model, 'sdnext_ondemand_modules', None)
    for module_name in names or []:
        module = getattr(sd_model, module_name, None)
        param = next(module.parameters(), None) if module is not None else None
        if param is not None and not devices.same_device(param.device, devices.cpu):
            t0 = time.time()
            module.to(devices.cpu)
            dt = time.time() - t0
            process_timer.add('offload', dt)
            log.debug(f'Offload: type=ondemand op=offload module={module_name} time={dt:.3f}')


def report_group_stats(sd_model, module_names):
    """Per-component stats block once per loaded model; balanced mode prints its own from the hook map."""
    if getattr(sd_model, 'sdnext_group_stats_reported', False):
        return
    sd_model.sdnext_group_stats_reported = True
    total = 0.0
    counted = []
    for module_name in module_names:
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module):
            total += get_module_size(module)[0]
            counted.append(module_name)
            report_model_stats(module_name, module)
    log.info(f'Model class={sd_model.__class__.__name__} modules={len(counted)} size={total:.3f}')


def apply_modular_group_offload(sd_model):
    """Per-component group offload for modular pipelines, which lack the pipeline-level
    enable_*_offload entry points. The model and sequential modes also route here."""
    if shared.opts.diffusers_offload_mode != 'group' and not getattr(sd_model, 'sdnext_modular_offload_warned', False):
        sd_model.sdnext_modular_offload_warned = True
        log.warning(f'Offload: desired={shared.opts.diffusers_offload_mode} override=group reason="modular pipeline"')
    applied = []
    for name in ('transformer', 'transformer_ref'):
        transformer = getattr(sd_model, name, None)
        if transformer is not None and apply_group_offload_component(transformer, name, main=True):
            applied.append(name)
    text_encoder = getattr(sd_model, 'text_encoder', None)
    if text_encoder is not None:
        # offload targets the inner model when present: conditioning may call it directly,
        # and hooks on the wrapper forward would never fire
        if apply_group_offload_component(getattr(text_encoder, 'model', text_encoder), 'text_encoder', main=False):
            applied.append('text_encoder')
    for name in ('vae', 'audio_vae'):
        component = getattr(sd_model, name, None)
        if component is not None:
            placement = set_group_vae(sd_model, component, name)
            applied.append(f'{name}:{placement}')
    # has_accelerate stays unset: group hooks are not accelerate hooks, and the modular
    # pipeline's own to() skips group-offloaded components when move_model runs
    if any(':' not in name for name in applied):
        log.info(f'Offload: type=group type={shared.opts.group_offload_type} modules={applied}')
    report_group_stats(sd_model, ('transformer', 'transformer_ref', 'text_encoder', 'vae', 'audio_vae'))


def apply_group_offload(sd_model):
    applied, resident, ondemand = [], [], []
    for module_name in get_module_names(sd_model):
        module = getattr(sd_model, module_name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            role = group_offload_role(module_name, module)
            if role == 'resident':
                if set_group_vae(sd_model, module, module_name) == 'ondemand':
                    ondemand.append(module_name)
                else:
                    resident.append(module_name)
            elif apply_group_offload_component(module, module_name, main=role == 'main'):
                applied.append(module_name)
        except Exception as e:
            log.error(f'Offload: type=group module={module_name} {e}')
    set_accelerate(sd_model)
    if applied:
        log.info(f'Offload: type=group type={shared.opts.group_offload_type} modules={applied} resident={resident} ondemand={ondemand}')
    report_group_stats(sd_model, get_module_names(sd_model))
    return sd_model


def apply_model_offload(sd_model, quiet:bool=False):
    try:
        remove_group_offload(sd_model)
        log.quiet(quiet, f'Offload: type={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
        if not hasattr(sd_model, "_all_hooks") or len(sd_model._all_hooks) == 0: # pylint: disable=protected-access
            sd_model.enable_model_cpu_offload(device=devices.device)
        else:
            sd_model.maybe_free_model_hooks()
        set_accelerate(sd_model)
    except Exception as e:
        log.error(f'Offload: type={shared.opts.diffusers_offload_mode} {e}')


def apply_sequential_offload(sd_model, op:str='model', quiet:bool=False):
    try:
        remove_group_offload(sd_model)
        log.quiet(quiet, f'Offload: type={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
        if sd_model.has_accelerate:
            if op == "vae": # reapply sequential offload to vae
                from accelerate import cpu_offload
                sd_model.vae.to(devices.cpu)
                cpu_offload(sd_model.vae, devices.device, offload_buffers=len(sd_model.vae._parameters) > 0) # pylint: disable=protected-access
            else:
                pass # do nothing if offload is already applied
        else:
            sd_model.enable_sequential_cpu_offload(device=devices.device)
        set_accelerate(sd_model)
    except Exception as e:
        log.error(f'Offload: type={shared.opts.diffusers_offload_mode} {e}')


def apply_none_offload(sd_model, quiet:bool=False):
    if shared.sd_model_type not in offload_allow_none:
        log.warning(f'Offload: type={shared.opts.diffusers_offload_mode} cls={shared.sd_model.__class__.__name__} large model')
    else:
        log.quiet(quiet, f'Offload: type={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
    try:
        sd_model.has_accelerate = False
        remove_group_offload(sd_model)
        if hasattr(sd_model, 'maybe_free_model_hooks'):
            sd_model.maybe_free_model_hooks()
        sd_model = accelerate.hooks.remove_hook_from_module(sd_model, recurse=True)
    except Exception:
        pass
    sd_models.move_model(sd_model, devices.device, force=True)


def set_diffuser_offload(sd_model, op:str='model', quiet:bool=False, force:bool=False):
    global accelerate_dtype_byte_size # pylint: disable=global-statement
    t0 = time.time()
    if sd_model is None:
        log.warning(f'{op} is not loaded')
        return
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False
    if accelerate_dtype_byte_size is None:
        accelerate_dtype_byte_size = accelerate.utils.modeling.dtype_byte_size
        accelerate.utils.modeling.dtype_byte_size = dtype_byte_size

    if sd_models.get_diffusers_task(sd_model) == sd_models.DiffusersTaskType.MODULAR and shared.opts.diffusers_offload_mode in {'model', 'sequential', 'group'}:
        apply_modular_group_offload(sd_model)
        process_timer.add('offload', time.time() - t0)
        return

    if shared.opts.diffusers_offload_mode == "none":
        log.warning('Offload: type=none "use balanced offload with model type set not to offload"')
        apply_none_offload(sd_model, quiet=quiet)

    if shared.opts.diffusers_offload_mode == "model" and hasattr(sd_model, "enable_model_cpu_offload"):
        log.warning('Offload: type=model "use balanced offload instead"')
        apply_model_offload(sd_model, quiet=quiet)

    if shared.opts.diffusers_offload_mode == "sequential" and hasattr(sd_model, "enable_sequential_cpu_offload"):
        apply_sequential_offload(sd_model, op=op, quiet=quiet)

    if shared.opts.diffusers_offload_mode == "group":
        sd_model = apply_group_offload(sd_model)

    if shared.opts.diffusers_offload_mode == "balanced":
        sd_model = apply_balanced_offload(sd_model, force=force)

    process_timer.add('offload', time.time() - t0)


class OffloadHook(accelerate.hooks.ModelHook):
    def __init__(self, checkpoint_name):
        if shared.opts.diffusers_offload_max_gpu_memory > 1:
            shared.opts.diffusers_offload_max_gpu_memory = 0.75
        if shared.opts.diffusers_offload_max_cpu_memory > 1:
            shared.opts.diffusers_offload_max_cpu_memory = 0.75
        self.checkpoint_name = checkpoint_name
        self.min_watermark = shared.opts.diffusers_offload_min_gpu_memory
        self.max_watermark = shared.opts.diffusers_offload_max_gpu_memory
        self.cpu_watermark = shared.opts.diffusers_offload_max_cpu_memory
        self.offload_always = [m.strip() for m in re.split(';|,| ', shared.opts.diffusers_offload_always) if len(m.strip()) > 2]
        self.offload_never = [m.strip() for m in re.split(';|,| ', shared.opts.diffusers_offload_never) if len(m.strip()) > 2]
        self.gpu = int(shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory * 1024*1024*1024)
        self.cpu = int(shared.cpu_memory * shared.opts.diffusers_offload_max_cpu_memory * 1024*1024*1024)
        self.offload_map = {}
        self.param_map = {}
        self.last_pre = None
        self.last_post = None
        self.last_cls = None
        gpu = f'{(shared.gpu_memory * shared.opts.diffusers_offload_min_gpu_memory):.2f}-{(shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory):.2f}:{shared.gpu_memory:.2f}'
        log.info(f'Offload: type=balanced op=init watermark={self.min_watermark}-{self.max_watermark} gpu={gpu} cpu={shared.cpu_memory:.3f} limit={shared.opts.cuda_mem_fraction:.2f} always={self.offload_always} never={self.offload_never} pre={shared.opts.diffusers_offload_pre} streams={shared.opts.diffusers_offload_streams}')
        self.validate()
        super().__init__()

    def validate(self):
        if shared.opts.diffusers_offload_mode != 'balanced':
            return
        if shared.opts.diffusers_offload_min_gpu_memory < 0 or shared.opts.diffusers_offload_min_gpu_memory > 1:
            shared.opts.diffusers_offload_min_gpu_memory = 0.2
            log.warning(f'Offload: type=balanced op=validate: watermark low={shared.opts.diffusers_offload_min_gpu_memory} invalid value')
        if shared.opts.diffusers_offload_max_gpu_memory < 0.1 or shared.opts.diffusers_offload_max_gpu_memory > 1:
            shared.opts.diffusers_offload_max_gpu_memory = 0.7
            log.warning(f'Offload: type=balanced op=validate: watermark high={shared.opts.diffusers_offload_max_gpu_memory} invalid value')
        if shared.opts.diffusers_offload_min_gpu_memory > shared.opts.diffusers_offload_max_gpu_memory:
            shared.opts.diffusers_offload_min_gpu_memory = shared.opts.diffusers_offload_max_gpu_memory
            log.warning(f'Offload: type=balanced op=validate: watermark low={shared.opts.diffusers_offload_min_gpu_memory} reset')
        if shared.opts.diffusers_offload_max_gpu_memory * shared.gpu_memory < 3:
            log.warning(f'Offload: type=balanced op=validate: watermark high={shared.opts.diffusers_offload_max_gpu_memory} low memory')

    def model_size(self):
        return sum(self.offload_map.values())

    def matches(self, module, names: list, module_name: str | None = None) -> bool:
        """Match against an always/never list by class name or by pipeline component name.
        Component entries such as `text_encoder` cover every architecture without listing each encoder class."""
        if module.__class__.__name__ in names:
            return True
        module_name = module_name or getattr(module, 'module_name', None)
        return module_name is not None and module_name in names

    def init_hook(self, module):
        return module

    def offload_allowed(self, module):
        if hasattr(module, "offload_never"):
            return False
        if hasattr(module, 'nets') and any(hasattr(n, "offload_never") for n in module.nets):
            return False
        if shared.sd_model_type.lower() in [m.lower().strip() for m in re.split(r'[ ,]+', shared.opts.models_not_to_offload)]:
            return False
        return True

    def pre_forward(self, module, *args, **kwargs):
        _id = id(module)

        do_offload = (self.last_pre != _id) or (module.__class__.__name__ != self.last_cls)

        if do_offload and self.offload_allowed(module): # offload every other module first time when new module starts pre-forward
            if shared.opts.diffusers_offload_pre:
                t0 = time.time()
                debug_move(f'Offload: type=balanced op=pre module={module.__class__.__name__}')
                sd_offload_aux.evict_aux(reason=f'pre:{module.__class__.__name__}')
                for pipe in get_pipe_variants():
                    for module_name in get_module_names(pipe):
                        module_instance = getattr(pipe, module_name, None)
                        if (module_instance is not None) and (_id != id(module_instance)) and (not self.matches(module_instance, self.offload_never, module_name)) and (not devices.same_device(getattr(module_instance, "device", devices.cpu), devices.cpu)):
                            apply_balanced_offload_to_module(module_instance, op='pre')
                self.last_cls = module.__class__.__name__
                process_timer.add('offload', time.time() - t0)

        if not devices.same_device(getattr(module, "device", devices.cpu), devices.device): # move-to-device
            t0 = time.time()
            device_index = torch.device(devices.device).index
            if device_index is None:
                device_index = 0
            max_memory = { device_index: self.gpu, "cpu": self.cpu }
            device_map = getattr(module, "balanced_offload_device_map", None)
            if (device_map is None) or (max_memory != getattr(module, "balanced_offload_max_memory", None)):
                device_map = accelerate.infer_auto_device_map(module,
                                                              max_memory=max_memory,
                                                              no_split_module_classes=no_split_module_classes,
                                                              verbose=verbose,
                                                              clean_result=False,
                                                             )
            offload_dir = getattr(module, "offload_dir", os.path.join(shared.opts.accelerate_offload_path, module.__class__.__name__))
            if debug:
                log.trace(f'Offload: type=balanced op=dispatch map={device_map}')
            if device_map is not None:
                skip_keys = getattr(module, "_skip_keys", None)
                try:
                    module = accelerate.dispatch_model(module,
                                                       main_device=torch.device(devices.device),
                                                       device_map=device_map,
                                                       offload_dir=offload_dir,
                                                       skip_keys=skip_keys,
                                                       force_hooks=True,
                                                      )
                except Exception as e: # reapply hook
                    log.warning(f'Offload: type=balanced op=dispatch module={module.__class__.__name__} {e}')
                    module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
                    module.balanced_offload_device_map = None
                    sd_models.move_model(module, devices.device, force=True)
                    module = accelerate.hooks.add_hook_to_module(module, self, append=True)
            module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
            module.balanced_offload_device_map = device_map
            module.balanced_offload_max_memory = max_memory
            process_timer.add('onload', time.time() - t0)

        if debug:
            for _i, pipe in enumerate(get_pipe_variants()):
                for module_name in get_module_names(pipe):
                    module_instance = getattr(pipe, module_name, None)
                    log.trace(f'Offload: type=balanced op=pre:status forward={module.__class__.__name__} module={module_name} class={module_instance.__class__.__name__} pipe={_i} device={getattr(module_instance, "device", devices.cpu)} dtype={module_instance.dtype}')

        self.last_pre = _id
        return args, kwargs

    def post_forward(self, module, output):
        if self.last_post != id(module):
            self.last_post = id(module)
        if getattr(module, "offload_post", False) and (module.device != devices.cpu):
            apply_balanced_offload_to_module(module, op='post')
        return output

    def detach_hook(self, module):
        return module


def get_pipe_variants(pipe=None):
    if pipe is None:
        if shared.sd_loaded:
            pipe = shared.sd_model
        else:
            return [pipe]
    variants = [pipe]
    if hasattr(pipe, "pipe"):
        variants.append(pipe.pipe)
    if hasattr(pipe, "prior_pipe"):
        variants.append(pipe.prior_pipe)
    if hasattr(pipe, "decoder_pipe"):
        variants.append(pipe.decoder_pipe)
    return variants


def get_module_names(pipe=None, exclude=None):
    def is_valid(module):
        if isinstance(getattr(pipe, module, None), torch.nn.ModuleDict):
            return True
        if isinstance(getattr(pipe, module, None), torch.nn.ModuleList):
            return True
        if isinstance(getattr(pipe, module, None), torch.nn.Module):
            return True
        return False

    if exclude is None:
        exclude = []
    if pipe is None:
        if shared.sd_loaded:
            pipe = shared.sd_model
        else:
            return []
    modules_names = []
    try:
        dict_keys = pipe._internal_dict.keys() # pylint: disable=protected-access
        modules_names.extend(dict_keys)
    except Exception:
        pass
    try:
        dict_keys = get_signature(pipe).keys()
        modules_names.extend(dict_keys)
    except Exception:
        pass
    modules_names = [m for m in modules_names if m not in exclude and not m.startswith('_')]
    modules_names = [m for m in modules_names if is_valid(m)]
    modules_names = sorted(set(modules_names))
    return modules_names


def get_module_memory(module: torch.nn.Module) -> dict[str, float]:
    tensors = list(itertools.chain(module.parameters(), module.buffers()))
    logical_gib = sum(tensor.numel() * tensor.element_size() for tensor in tensors) / 1024**3
    storages = {}
    for tensor in tensors:
        try:
            storage = tensor.untyped_storage()
        except (AttributeError, RuntimeError):
            continue
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    storage_gib = sum(storages.values()) / 1024**3
    return {
        "logical": round(logical_gib, 3),
        "storage": round(storage_gib, 3),
        "overhead": round(storage_gib - logical_gib, 3),
        "tensors": len(tensors),
        "storages": len(storages),
    }


def get_module_size(module: torch.nn.Module) -> tuple[float, float]:
    module_size = 0
    param_num = 0
    if not isinstance(module, torch.nn.Module):
        return 0, 0
    try:
        # module_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
        tensors = set(itertools.chain(module.parameters(recurse=True), module.buffers(recurse=True)))
        module_size = sum(t.numel() * t.element_size() for t in tensors) / 1024**3
        param_num = sum(p.numel() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
    except Exception as e:
        log.error(f'Offload: type=balanced op=calc module={module.__class__.__name__} {e}')
        module_size = 0
        param_num = 0
    return module_size, param_num


def get_module_sizes(pipe=None, exclude=None):
    if exclude is None:
        exclude = []
    modules = {}
    for module_name in get_module_names(pipe, exclude):
        module_size = offload_hook_instance.offload_map.get(module_name, None)
        if module_size is None:
            module = getattr(pipe, module_name, None)
            module_size, param_num = get_module_size(module)
            offload_hook_instance.offload_map[module_name] = module_size
            offload_hook_instance.param_map[module_name] = param_num
        modules[module_name] = module_size
    modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
    return modules


def move_module_to_cpu(module, op='unk', force:bool=False):
    def do_move(module):
        if shared.opts.diffusers_offload_streams:
            global move_stream # pylint: disable=global-statement
            if move_stream is None:
                move_stream = torch.cuda.Stream(device=devices.device)
            with torch.cuda.stream(move_stream):
                module = module.to(devices.cpu)
        else:
            module = module.to(devices.cpu)
        return module

    try:
        module_name = getattr(module, "module_name", module.__class__.__name__)
        module_size = offload_hook_instance.offload_map.get(module_name, offload_hook_instance.model_size())
        used_gpu, used_ram = devices.torch_gc(fast=True)
        perc_gpu = used_gpu / shared.gpu_memory
        prev_gpu = used_gpu
        module_cls = module.__class__.__name__
        op = f'{op}:skip'
        if force:
            op = f'{op}:force'
            module = do_move(module)
            used_gpu -= module_size
        elif offload_hook_instance.matches(module, offload_hook_instance.offload_never, module_name):
            op = f'{op}:never'
        elif offload_hook_instance.matches(module, offload_hook_instance.offload_always, module_name):
            op = f'{op}:always'
            module = do_move(module)
            used_gpu -= module_size
        elif perc_gpu > shared.opts.diffusers_offload_min_gpu_memory:
            op = f'{op}:mem'
            module = do_move(module)
            used_gpu -= module_size
        if debug:
            quant = getattr(module, "quantization_method", None)
            debug_move(f'Offload: type=balanced op={op} gpu={prev_gpu:.3f}:{used_gpu:.3f} perc={perc_gpu:.2f}:{shared.opts.diffusers_offload_min_gpu_memory} ram={used_ram:.3f} current={module.device} dtype={module.dtype} quant={quant} module={module_cls} size={module_size:.3f}')
    except Exception as e:
        if 'out of memory' in str(e):
            devices.torch_gc(fast=True, force=True, reason='oom')
        elif 'bitsandbytes' in str(e):
            pass
        else:
            log.error(f'Offload: type=balanced op=apply module={getattr(module, "__name__", None)} cls={module.__class__ if inspect.isclass(module) else None} {e}')
        if os.environ.get('SD_MOVE_DEBUG', None):
            errors.display(e, f'Offload: type=balanced op=apply module={getattr(module, "__name__", None)}')


def apply_balanced_offload_to_module(module, op="apply", force:bool=False):
    module_name = getattr(module, "module_name", module.__class__.__name__)
    network_layer_name = getattr(module, "network_layer_name", None)
    device_map = getattr(module, "balanced_offload_device_map", None)
    max_memory = getattr(module, "balanced_offload_max_memory", None)
    try:
        module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
    except Exception as e:
        log.warning(f'Offload remove hook: module={module_name} {e}')
    move_module_to_cpu(module, op=op, force=force)
    try:
        module = accelerate.hooks.add_hook_to_module(module, offload_hook_instance, append=True)
    except Exception as e:
        log.warning(f'Offload add hook: module={module_name} {e}')
    module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
    if network_layer_name:
        module.network_layer_name = network_layer_name
    if device_map and max_memory:
        module.balanced_offload_device_map = device_map
        module.balanced_offload_max_memory = max_memory
    module.offload_post = shared.sd_model_type in offload_post and module_name.startswith("text_encoder")
    if shared.opts.layerwise_quantization or getattr(module, 'quantization_method', None) == 'LayerWise':
        model_quant.apply_layerwise(module, quiet=True) # need to reapply since hooks were removed/re-added
    devices.torch_gc(fast=True, force=True, reason='offload')


def get_logical_param_count(module: torch.nn.Module) -> int:
    if hasattr(module, "sdnq_dequantizer"):
        original_shape = module.sdnq_dequantizer.original_shape
        count = math.prod(original_shape)
        if getattr(module, "bias", None) is not None:
            count += module.bias.numel()
        return int(count)
    count = sum(p.numel() for p in module.parameters(recurse=False))
    for child in module.children():
        count += get_logical_param_count(child)
    return count


def report_model_stats(module_name, module):
    try:
        size = offload_hook_instance.offload_map.get(module_name, 0) if offload_hook_instance is not None else 0
        if size == 0:
            size, _params = get_module_size(module)
        quant = getattr(module, "quantization_method", None)
        params = sum(p.numel() for p in module.parameters(recurse=True))
        logical = get_logical_param_count(module)
        log.debug(f'Module: name={module_name} cls={module.__class__.__name__} size={size:.3f} params={params} logical={logical} quant={quant}')
    except Exception as e:
        log.error(f'Module stats: name={module_name} {e}')


def apply_balanced_offload(sd_model=None, exclude: list[str] | None = None, force: bool = False, silent: bool = False):
    global offload_hook_instance # pylint: disable=global-statement
    if shared.opts.diffusers_offload_mode != "balanced":
        return sd_model
    if sd_model is None:
        if not shared.sd_loaded:
            return sd_model
        sd_model = shared.sd_model
    if sd_model is None:
        return sd_model
    if exclude is None:
        exclude = []
    if sd_model.__class__.__name__ in balanced_offload_exclude:
        return sd_model
    remove_group_offload(sd_model)

    t0 = time.time()
    cached = True
    checkpoint_name = sd_model.sd_checkpoint_info.name if getattr(sd_model, "sd_checkpoint_info", None) is not None else sd_model.__class__.__name__
    if force or (offload_hook_instance is None) or (offload_hook_instance.min_watermark != shared.opts.diffusers_offload_min_gpu_memory) or (offload_hook_instance.max_watermark != shared.opts.diffusers_offload_max_gpu_memory) or (checkpoint_name != offload_hook_instance.checkpoint_name):
        cached = False
        offload_hook_instance = OffloadHook(checkpoint_name)

    if cached and shared.opts.diffusers_offload_pre:
        debug_move('Offload: type=balanced op=apply skip')
        return sd_model

    for pipe in get_pipe_variants(sd_model):
        for module_name, _module_size in get_module_sizes(pipe, exclude):
            module = getattr(pipe, module_name, None)
            if module is None:
                continue
            module.module_name = module_name
            module.offload_dir = os.path.join(shared.opts.accelerate_offload_path, checkpoint_name, module_name)
            apply_balanced_offload_to_module(module, op='apply', force=force)
            if not silent:
                report_model_stats(module_name, module)

    set_accelerate(sd_model)
    t = time.time() - t0
    process_timer.add('offload', t)
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    debug_move(f'Apply offload: time={t:.2f} type=balanced fn={fn}')
    if not cached:
        log.info(f'Model class={sd_model.__class__.__name__} modules={len(offload_hook_instance.offload_map)} size={offload_hook_instance.model_size():.3f}')
    return sd_model
