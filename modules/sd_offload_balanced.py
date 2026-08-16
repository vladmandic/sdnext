import os
import sys
import time
import inspect
import torch
import accelerate.hooks
import accelerate.utils.modeling
from modules.logger import log
from modules import shared, devices, errors, model_quant, sd_models, sd_offload_aux
from modules.timer import process as process_timer
from modules.sd_offload_utils import offload_list, offload_matches, offload_model_types, get_pipe_variants, get_module_names, get_module_size, set_accelerate, report_model_stats
from modules.sd_offload_group import remove_group_offload
import modules.sd_offload_state as s


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
        self.offload_always = offload_list(shared.opts.diffusers_offload_always)
        self.offload_never = offload_list(shared.opts.diffusers_offload_never)
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
        return offload_matches(module, module_name, names)

    def init_hook(self, module):
        return module

    def offload_allowed(self, module):
        if hasattr(module, "offload_never"):
            return False
        if hasattr(module, 'nets') and any(hasattr(n, "offload_never") for n in module.nets):
            return False
        if shared.sd_model_type.lower() in offload_model_types():
            return False
        return True

    def pre_forward(self, module, *args, **kwargs):
        _id = id(module)

        do_offload = (self.last_pre != _id) or (module.__class__.__name__ != self.last_cls)

        if do_offload and self.offload_allowed(module): # offload every other module first time when new module starts pre-forward
            if shared.opts.diffusers_offload_pre:
                t0 = time.time()
                s.debug_move(f'Offload: type=balanced op=pre module={module.__class__.__name__}')
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
                                                              no_split_module_classes=s.no_split_module_classes,
                                                              verbose=s.verbose,
                                                              clean_result=False,
                                                             )
            offload_dir = getattr(module, "offload_dir", os.path.join(shared.opts.accelerate_offload_path, module.__class__.__name__))
            if s.debug:
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

        if s.debug:
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


def get_module_sizes(pipe=None, exclude=None):
    if exclude is None:
        exclude = []
    modules = {}
    for module_name in get_module_names(pipe, exclude):
        module_size = s.offload_hook_instance.offload_map.get(module_name, None)
        if module_size is None:
            module = getattr(pipe, module_name, None)
            module_size, param_num = get_module_size(module)
            s.offload_hook_instance.offload_map[module_name] = module_size
            s.offload_hook_instance.param_map[module_name] = param_num
        modules[module_name] = module_size
    modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
    return modules


def move_module_to_cpu(module, op='unk', force:bool=False):
    def do_move(module):
        if shared.opts.diffusers_offload_streams:
            if s.move_stream is None:
                s.move_stream = torch.cuda.Stream(device=devices.device)
            with torch.cuda.stream(s.move_stream):
                module = module.to(devices.cpu)
        else:
            module = module.to(devices.cpu)
        return module

    try:
        module_name = getattr(module, "module_name", module.__class__.__name__)
        module_size = s.offload_hook_instance.offload_map.get(module_name, s.offload_hook_instance.model_size())
        used_gpu, used_ram = devices.torch_gc(fast=True)
        perc_gpu = used_gpu / shared.gpu_memory
        prev_gpu = used_gpu
        module_cls = module.__class__.__name__
        op = f'{op}:skip'
        if force:
            op = f'{op}:force'
            module = do_move(module)
            used_gpu -= module_size
        elif s.offload_hook_instance.matches(module, s.offload_hook_instance.offload_never, module_name):
            op = f'{op}:never'
        elif s.offload_hook_instance.matches(module, s.offload_hook_instance.offload_always, module_name):
            op = f'{op}:always'
            module = do_move(module)
            used_gpu -= module_size
        elif perc_gpu > shared.opts.diffusers_offload_min_gpu_memory:
            op = f'{op}:mem'
            module = do_move(module)
            used_gpu -= module_size
        if s.debug:
            quant = getattr(module, "quantization_method", None)
            s.debug_move(f'Offload: type=balanced op={op} gpu={prev_gpu:.3f}:{used_gpu:.3f} perc={perc_gpu:.2f}:{shared.opts.diffusers_offload_min_gpu_memory} ram={used_ram:.3f} current={module.device} dtype={module.dtype} quant={quant} module={module_cls} size={module_size:.3f}')
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
        module = accelerate.hooks.add_hook_to_module(module, s.offload_hook_instance, append=True)
    except Exception as e:
        log.warning(f'Offload add hook: module={module_name} {e}')
    module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
    if network_layer_name:
        module.network_layer_name = network_layer_name
    if device_map and max_memory:
        module.balanced_offload_device_map = device_map
        module.balanced_offload_max_memory = max_memory
    module.offload_post = shared.sd_model_type in s.offload_post and module_name.startswith("text_encoder")
    if shared.opts.layerwise_quantization or getattr(module, 'quantization_method', None) == 'LayerWise':
        model_quant.apply_layerwise(module, quiet=True) # need to reapply since hooks were removed/re-added
    devices.torch_gc(fast=True, force=True, reason='offload')


def apply_balanced_offload(sd_model=None, exclude: list[str] | None = None, force: bool = False, silent: bool = False):
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
    if sd_model.__class__.__name__ in s.balanced_offload_exclude:
        return sd_model
    remove_group_offload(sd_model)

    t0 = time.time()
    cached = True
    checkpoint_name = sd_model.sd_checkpoint_info.name if getattr(sd_model, "sd_checkpoint_info", None) is not None else sd_model.__class__.__name__
    if force or (s.offload_hook_instance is None) or (s.offload_hook_instance.min_watermark != shared.opts.diffusers_offload_min_gpu_memory) or (s.offload_hook_instance.max_watermark != shared.opts.diffusers_offload_max_gpu_memory) or (checkpoint_name != s.offload_hook_instance.checkpoint_name):
        cached = False
        s.offload_hook_instance = OffloadHook(checkpoint_name)

    if cached and shared.opts.diffusers_offload_pre:
        s.debug_move('Offload: type=balanced op=apply skip')
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
    t1 = time.time()
    process_timer.add('offload', t1 - t0)
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    s.debug_move(f'Apply offload: time={t1 - t0:.2f} type=balanced fn={fn}')
    if not cached:
        log.info(f'Model class={sd_model.__class__.__name__} modules={len(s.offload_hook_instance.offload_map)} size={s.offload_hook_instance.model_size():.3f}')
    return sd_model
