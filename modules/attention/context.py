"""Per-generation state for attention consumers: the component running, the denoiser forward about to run, and the model."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch


@dataclass
class GenerationContext:
    active: bool = False
    role: str | None = None # 'transformer', 'te' or 'vae' while a generation runs, None outside one
    step: int = 0 # index of the denoiser forward about to run
    steps: int = 0 # forwards in the current pass
    forwards: int = 0
    model_key: tuple[str, str | None] | None = None # pipeline class and denoiser class, telemetry only
    step_buffer: torch.Tensor | None = None # the step as a device scalar updated in place, so compiled readers keep their graph
    layout: object | None = None # TokenLayout published by whoever knows the packing, None until something does


current = GenerationContext()


def denoiser_name(pipe) -> str | None:
    for name in ('transformer', 'unet'):
        module = getattr(pipe, name, None)
        if module is not None:
            return module.__class__.__name__
    return None


# every slot a pipeline can enter once per denoising step, from sd_offload_state.group_offload_main. The aux
# components on that list (decoder, controlnet, prior) are left alone: they pack no attention sequence, and a
# publication from one would clear the layout the denoiser just set
DENOISER_SLOTS = ('transformer', 'unet', 'transformer_2', 'transformer_ref', 'unconditional_transformer')


def install_layout_hook(pipe) -> None:
    """Let a classic pipeline's denoiser publish its own packing: the modular path has its own hook, this is the rest."""
    from modules import shared
    if pipe is None or not getattr(shared.opts, 'sparse_attention_enabled', False):
        return
    from modules.attention.sparse import layout as sparse_layout

    def publish(denoiser, args, kwargs): # pylint: disable=unused-argument
        set_layout(sparse_layout.layout_from_kwargs(kwargs, denoiser.__class__.__name__))

    for name in DENOISER_SLOTS:
        module = getattr(pipe, name, None)
        if module is None or getattr(module, 'sdnext_layout_hook', None) is not None or getattr(module, 'sdnext_state_hook', None) is not None:
            continue
        module.sdnext_layout_hook = module.register_forward_pre_hook(publish, with_kwargs=True)


def begin(pipe, steps: int = 0) -> None:
    from modules import devices
    current.active = True
    current.role = 'transformer'
    current.layout = None
    current.model_key = (pipe.__class__.__name__, denoiser_name(pipe)) if pipe is not None else None
    install_layout_hook(pipe)
    device = devices.device if devices.device is not None else torch.device('cpu')
    if current.step_buffer is None or current.step_buffer.device != device:
        current.step_buffer = torch.zeros((), dtype=torch.int64, device=device)
    new_pass(steps)


def new_pass(steps: int = 0) -> None:
    """Restart the step count for a denoising pass: base, hires or refiner."""
    current.steps = int(steps or 0)
    current.forwards = 0
    set_step(0)


def set_step(step: int) -> None:
    current.step = int(step)
    if current.step_buffer is not None:
        current.step_buffer.fill_(current.step)


def tick(step: int | None = None) -> None:
    """Advance to the next forward: the classic callback passes the completed step plus one, the modular pre-hook passes nothing and counts forwards."""
    set_step(current.forwards if step is None else step)
    current.forwards = current.step + 1


def set_layout(layout) -> None:
    """Publish what the packed sequence holds; callers that know the packing set this per forward."""
    current.layout = layout


def end() -> None:
    from modules.attention import debug
    current.active = False
    current.role = None
    current.model_key = None
    current.layout = None
    new_pass(0)
    debug.end_generation()


def set_role(name: str | None) -> None:
    current.role = name


@contextmanager
def role(name: str):
    previous = current.role
    current.role = name
    try:
        yield
    finally:
        current.role = previous
