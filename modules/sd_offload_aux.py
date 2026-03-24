import os
import dataclasses
import torch
from modules.logger import log
from modules import shared, devices


move_stream = None
debug = os.environ.get('SD_MOVE_DEBUG', None) is not None
verbose = os.environ.get('SD_MOVE_VERBOSE', None) is not None
debug_move = log.trace if debug else lambda *args, **kwargs: None


@dataclasses.dataclass
class AuxModel:
    model: torch.nn.Module
    name: str
    size: float  # GB


aux_models: dict[str, AuxModel] = {}


def register_aux(name: str, model: torch.nn.Module) -> None:
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    aux_models[name] = AuxModel(model=model, name=name, size=size)
    debug_move(f'Offload: type=aux op=register name={name} size={size:.3f}')


def deregister_aux(name: str) -> None:
    entry = aux_models.pop(name, None)
    if entry:
        debug_move(f'Offload: type=aux op=deregister name={name}')


def evict_aux(exclude: str | None = None, reason: str = 'evict') -> None:
    for name, entry in aux_models.items():
        if name == exclude:
            continue
        if entry.model is not None and hasattr(entry.model, 'device') and not devices.same_device(entry.model.device, devices.cpu):
            _do_move_to_cpu(entry.model, f'aux:{reason}:{name}', entry.size)


def _do_move_to_cpu(model, op_label, size):
    if shared.opts.diffusers_offload_streams:
        global move_stream  # pylint: disable=global-statement
        if move_stream is None:
            move_stream = torch.cuda.Stream(device=devices.device)
        with torch.cuda.stream(move_stream):
            model.to(devices.cpu)
    else:
        model.to(devices.cpu)
    debug_move(f'Offload: type=aux op={op_label} size={size:.3f}')


def move_aux_to_gpu(name: str) -> None:
    entry = aux_models.get(name)
    if entry is None or entry.model is None:
        return
    if hasattr(entry.model, 'device') and devices.same_device(entry.model.device, devices.device):
        return
    # 1. Evict other auxiliary models first
    evict_aux(exclude=name, reason='pre')
    # 2. If balanced offload active, evict diffusers pipeline modules if memory is tight
    if shared.sd_loaded:
        from modules.sd_offload import apply_balanced_offload
        shared.sd_model = apply_balanced_offload(shared.sd_model)
    # 3. Move to GPU (stream + sync)
    if shared.opts.diffusers_offload_streams:
        global move_stream  # pylint: disable=global-statement
        if move_stream is None:
            move_stream = torch.cuda.Stream(device=devices.device)
        with torch.cuda.stream(move_stream):
            entry.model.to(devices.device)
        move_stream.synchronize()
    else:
        entry.model.to(devices.device)
    debug_move(f'Offload: type=aux op=to_gpu name={name} size={entry.size:.3f}')


def offload_aux(name: str) -> None:
    if not shared.opts.caption_offload:
        return
    entry = aux_models.get(name)
    if entry is None or entry.model is None:
        return
    if hasattr(entry.model, 'device') and devices.same_device(entry.model.device, devices.cpu):
        return
    _do_move_to_cpu(entry.model, f'post:{name}', entry.size)
