import os
import sys
import time
import inspect
import torch
import diffusers
import accelerate.hooks
from modules import shared, devices, errors, model_quant
from modules.timer import process as process_timer


debug_move = shared.log.trace if os.environ.get('SD_MOVE_DEBUG', None) is not None else lambda *args, **kwargs: None
should_offload = ['sc', 'sd3', 'f1', 'hunyuandit', 'auraflow', 'omnigen', 'hunyuanvideo', 'cogvideox', 'mochi', 'cogview4']
offload_hook_instance = None
balanced_offload_exclude = ['OmniGenPipeline', 'CogView4Pipeline']


def get_signature(cls):
    signature = inspect.signature(cls.__init__, follow_wrapped=True)
    return signature.parameters


def disable_offload(sd_model):
    if not getattr(sd_model, 'has_accelerate', False):
        return
    if hasattr(sd_model, "_internal_dict"):
        keys = sd_model._internal_dict.keys() # pylint: disable=protected-access
    else:
        keys = get_signature(sd_model).keys()
    for module_name in keys: # pylint: disable=protected-access
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module):
            network_layer_name = getattr(module, "network_layer_name", None)
            module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            if network_layer_name:
                module.network_layer_name = network_layer_name
    sd_model.has_accelerate = False


def set_accelerate(sd_model):
    def set_accelerate_to_module(model):
        if hasattr(model, "pipe"):
            set_accelerate_to_module(model.pipe)
        if hasattr(model, "_internal_dict"):
            for k in model._internal_dict.keys(): # pylint: disable=protected-access
                component = getattr(model, k, None)
                if isinstance(component, torch.nn.Module):
                    component.has_accelerate = True

    sd_model.has_accelerate = True
    set_accelerate_to_module(sd_model)
    if hasattr(sd_model, "prior_pipe"):
        set_accelerate_to_module(sd_model.prior_pipe)
    if hasattr(sd_model, "decoder_pipe"):
        set_accelerate_to_module(sd_model.decoder_pipe)


def set_diffuser_offload(sd_model, op:str='model', quiet:bool=False):
    t0 = time.time()
    if not shared.native:
        shared.log.warning('Attempting to use offload with backend=original')
        return
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False
    if shared.opts.diffusers_offload_mode == "none":
        if shared.sd_model_type in should_offload:
            shared.log.warning(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} type={shared.sd_model.__class__.__name__} large model')
        else:
            shared.log.quiet(quiet, f'Setting {op}: offload={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
        if hasattr(sd_model, 'maybe_free_model_hooks'):
            sd_model.maybe_free_model_hooks()
            sd_model.has_accelerate = False
    if shared.opts.diffusers_offload_mode == "model" and hasattr(sd_model, "enable_model_cpu_offload"):
        try:
            shared.log.quiet(quiet, f'Setting {op}: offload={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Model CPU offload" is enabled')
            if not hasattr(sd_model, "_all_hooks") or len(sd_model._all_hooks) == 0: # pylint: disable=protected-access
                sd_model.enable_model_cpu_offload(device=devices.device)
            else:
                sd_model.maybe_free_model_hooks()
            set_accelerate(sd_model)
        except Exception as e:
            shared.log.error(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} {e}')
    if shared.opts.diffusers_offload_mode == "sequential" and hasattr(sd_model, "enable_sequential_cpu_offload"):
        try:
            shared.log.debug(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Sequential CPU offload" is enabled')
            if sd_model.has_accelerate:
                if op == "vae": # reapply sequential offload to vae
                    from accelerate import cpu_offload
                    sd_model.vae.to("cpu")
                    cpu_offload(sd_model.vae, devices.device, offload_buffers=len(sd_model.vae._parameters) > 0) # pylint: disable=protected-access
                else:
                    pass # do nothing if offload is already applied
            else:
                sd_model.enable_sequential_cpu_offload(device=devices.device)
            set_accelerate(sd_model)
        except Exception as e:
            shared.log.error(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} {e}')
    if shared.opts.diffusers_offload_mode == "balanced":
        sd_model = apply_balanced_offload(sd_model)
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
        self.gpu = int(shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory * 1024*1024*1024)
        self.cpu = int(shared.cpu_memory * shared.opts.diffusers_offload_max_cpu_memory * 1024*1024*1024)
        self.offload_map = {}
        self.param_map = {}
        gpu = f'{shared.gpu_memory * shared.opts.diffusers_offload_min_gpu_memory:.3f}-{shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory}:{shared.gpu_memory}'
        shared.log.info(f'Offload: type=balanced op=init watermark={self.min_watermark}-{self.max_watermark} gpu={gpu} cpu={shared.cpu_memory:.3f} limit={shared.opts.cuda_mem_fraction:.2f}')
        self.validate()
        super().__init__()

    def validate(self):
        if shared.opts.diffusers_offload_mode != 'balanced':
            return
        if shared.opts.diffusers_offload_min_gpu_memory < 0 or shared.opts.diffusers_offload_min_gpu_memory > 1:
            shared.opts.diffusers_offload_min_gpu_memory = 0.25
            shared.log.warning(f'Offload: type=balanced op=validate: watermark low={shared.opts.diffusers_offload_min_gpu_memory} invalid value')
        if shared.opts.diffusers_offload_max_gpu_memory < 0.1 or shared.opts.diffusers_offload_max_gpu_memory > 1:
            shared.opts.diffusers_offload_max_gpu_memory = 0.75
            shared.log.warning(f'Offload: type=balanced op=validate: watermark high={shared.opts.diffusers_offload_max_gpu_memory} invalid value')
        if shared.opts.diffusers_offload_min_gpu_memory > shared.opts.diffusers_offload_max_gpu_memory:
            shared.opts.diffusers_offload_min_gpu_memory = shared.opts.diffusers_offload_max_gpu_memory
            shared.log.warning(f'Offload: type=balanced op=validate: watermark low={shared.opts.diffusers_offload_min_gpu_memory} reset')
        if shared.opts.diffusers_offload_max_gpu_memory * shared.gpu_memory < 4:
            shared.log.warning(f'Offload: type=balanced op=validate: watermark high={shared.opts.diffusers_offload_max_gpu_memory} low memory')

    def model_size(self):
        return sum(self.offload_map.values())

    def init_hook(self, module):
        return module

    def pre_forward(self, module, *args, **kwargs):
        if devices.normalize_device(module.device) != devices.normalize_device(devices.device):
            device_index = torch.device(devices.device).index
            if device_index is None:
                device_index = 0
            max_memory = { device_index: self.gpu, "cpu": self.cpu }
            device_map = getattr(module, "balanced_offload_device_map", None)
            if device_map is None or max_memory != getattr(module, "balanced_offload_max_memory", None):
                device_map = accelerate.infer_auto_device_map(module, max_memory=max_memory)
            offload_dir = getattr(module, "offload_dir", os.path.join(shared.opts.accelerate_offload_path, module.__class__.__name__))
            if devices.backend == "directml":
                keys = device_map.keys()
                for v in keys:
                    if isinstance(device_map[v], int):
                        # int implies CUDA or XPU device, but it will break DirectML backend.
                        # Therefore, the type of device should be added.
                        device_map[v] = f"{devices.device.type}:{device_map[v]}"
            module = accelerate.dispatch_model(module, device_map=device_map, offload_dir=offload_dir)
            module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
            module.balanced_offload_device_map = device_map
            module.balanced_offload_max_memory = max_memory
        return args, kwargs

    def post_forward(self, module, output):
        return output

    def detach_hook(self, module):
        return module


def apply_balanced_offload(sd_model, exclude=[]):
    global offload_hook_instance # pylint: disable=global-statement
    if shared.opts.diffusers_offload_mode != "balanced":
        return sd_model
    if sd_model is None:
        if not shared.sd_loaded:
            return sd_model
        sd_model = shared.sd_model
    if sd_model is None:
        return sd_model
    t0 = time.time()
    if sd_model.__class__.__name__ in balanced_offload_exclude:
        return sd_model
    cached = True
    checkpoint_name = sd_model.sd_checkpoint_info.name if getattr(sd_model, "sd_checkpoint_info", None) is not None else None
    if checkpoint_name is None:
        checkpoint_name = sd_model.__class__.__name__
    if (offload_hook_instance is None) or (offload_hook_instance.min_watermark != shared.opts.diffusers_offload_min_gpu_memory) or (offload_hook_instance.max_watermark != shared.opts.diffusers_offload_max_gpu_memory) or (checkpoint_name != offload_hook_instance.checkpoint_name):
        cached = False
        offload_hook_instance = OffloadHook(checkpoint_name)

    def get_pipe_modules(pipe):
        if hasattr(pipe, "_internal_dict"):
            modules_names = pipe._internal_dict.keys() # pylint: disable=protected-access
        else:
            modules_names = get_signature(pipe).keys()
        modules_names = [m for m in modules_names if m not in exclude and not m.startswith('_')]
        modules = {}
        for module_name in modules_names:
            module_size = offload_hook_instance.offload_map.get(module_name, None)
            if module_size is None:
                module = getattr(pipe, module_name, None)
                if not isinstance(module, torch.nn.Module):
                    continue
                try:
                    module_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
                    param_num = sum(p.numel() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
                except Exception as e:
                    shared.log.error(f'Offload: type=balanced op=calc module={module_name} {e}')
                    module_size = 0
                offload_hook_instance.offload_map[module_name] = module_size
                offload_hook_instance.param_map[module_name] = param_num
            modules[module_name] = module_size
        modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
        return modules

    def apply_balanced_offload_to_module(pipe):
        used_gpu, used_ram = devices.torch_gc(fast=True)
        if hasattr(pipe, "pipe"):
            apply_balanced_offload_to_module(pipe.pipe)
        if hasattr(pipe, "_internal_dict"):
            keys = pipe._internal_dict.keys() # pylint: disable=protected-access
        else:
            keys = get_signature(pipe).keys()
        keys = [k for k in keys if k not in exclude and not k.startswith('_')]
        for module_name, module_size in get_pipe_modules(pipe): # pylint: disable=protected-access
            module = getattr(pipe, module_name, None)
            if module is None:
                continue
            network_layer_name = getattr(module, "network_layer_name", None)
            device_map = getattr(module, "balanced_offload_device_map", None)
            max_memory = getattr(module, "balanced_offload_max_memory", None)
            module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            perc_gpu = used_gpu / shared.gpu_memory
            try:
                prev_gpu = used_gpu
                do_offload = (perc_gpu > shared.opts.diffusers_offload_min_gpu_memory) and (module.device != devices.cpu)
                if do_offload:
                    non_blocking = devices.backend != "ipex" # non_blocking on ipex causes 2x slowdown
                    module = module.to(devices.cpu, non_blocking=non_blocking)
                    used_gpu -= module_size
                cls = module.__class__.__name__
                quant = getattr(module, "quantization_method", None)
                if not cached:
                    shared.log.debug(f'Model module={module_name} type={cls} dtype={module.dtype} quant={quant} params={offload_hook_instance.param_map[module_name]:.3f} size={offload_hook_instance.offload_map[module_name]:.3f}')
                debug_move(f'Offload: type=balanced op={"move" if do_offload else "skip"} gpu={prev_gpu:.3f}:{used_gpu:.3f} perc={perc_gpu:.2f} ram={used_ram:.3f} current={module.device} dtype={module.dtype} quant={quant} module={cls} size={module_size:.3f}')
            except Exception as e:
                if 'out of memory' in str(e):
                    devices.torch_gc(fast=True, force=True, reason='oom')
                elif 'bitsandbytes' in str(e):
                    pass
                else:
                    shared.log.error(f'Offload: type=balanced op=apply module={module_name} {e}')
                if os.environ.get('SD_MOVE_DEBUG', None):
                    errors.display(e, f'Offload: type=balanced op=apply module={module_name}')
            module.offload_dir = os.path.join(shared.opts.accelerate_offload_path, checkpoint_name, module_name)
            module = accelerate.hooks.add_hook_to_module(module, offload_hook_instance, append=True)
            module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
            if network_layer_name:
                module.network_layer_name = network_layer_name
            if device_map and max_memory:
                module.balanced_offload_device_map = device_map
                module.balanced_offload_max_memory = max_memory
        devices.torch_gc(fast=True, force=True, reason='offload')

    apply_balanced_offload_to_module(sd_model)
    if hasattr(sd_model, "pipe"):
        apply_balanced_offload_to_module(sd_model.pipe)
    if hasattr(sd_model, "prior_pipe"):
        apply_balanced_offload_to_module(sd_model.prior_pipe)
    if hasattr(sd_model, "decoder_pipe"):
        apply_balanced_offload_to_module(sd_model.decoder_pipe)

    if shared.opts.layerwise_quantization:
        model_quant.apply_layerwise(sd_model, quiet=True) # need to reapply since hooks were removed/readded
    if shared.opts.pab_enabled and hasattr(sd_model, 'transformer'):
        pab_config = diffusers.PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=shared.opts.pab_block_skip_range,
            spatial_attention_timestep_skip_range=(int(100 * shared.opts.pab_timestep_skip_start), int(100 * shared.opts.pab_timestep_skip_end)),
            current_timestep_callback=lambda: sd_model.current_timestep, # pylint: disable=protected-access
        )
        try:
            diffusers.apply_pyramid_attention_broadcast(sd_model.transformer, pab_config)
        except Exception: # hook may already exist
            pass
        if not cached:
            shared.log.info(f'Applying PAB: cls={sd_model.transformer.__class__.__name__} block={shared.opts.pab_block_skip_range} start={shared.opts.pab_timestep_skip_start} end={shared.opts.pab_timestep_skip_end}')

    set_accelerate(sd_model)
    t = time.time() - t0
    process_timer.add('offload', t)
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    debug_move(f'Apply offload: time={t:.2f} type=balanced fn={fn}')
    if not cached:
        shared.log.info(f'Model class={sd_model.__class__.__name__} modules={len(offload_hook_instance.offload_map)} size={offload_hook_instance.model_size():.3f}')
    return sd_model
