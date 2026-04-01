# source <https://github.com/Trojaner/RAS_Simplified>
# original: <https://github.com/microsoft/RAS>

from modules import shared, processing
from modules.logger import log


def apply(pipe, p: processing.StableDiffusionProcessing):
    if shared.sd_model_type != "sd3" or not shared.opts.ras_enable:
        return
    from .ras_manager import MANAGER
    from .ras_scheduler import RASFlowMatchEulerDiscreteScheduler
    from .ras_attention import RASJointAttnProcessor2_0
    from .ras_forward import ras_forward
    scheduler = RASFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    MANAGER.num_steps = p.steps
    MANAGER.scheduler_end_step = p.steps
    MANAGER.width = p.width
    MANAGER.height = p.height
    MANAGER.error_reset_steps = [int(1*p.steps/3), int(2*p.steps/3)]
    log.info(f'RAS: scheduler={pipe.scheduler.__class__.__name__} {str(MANAGER)}')
    MANAGER.reset_cache()
    MANAGER.generate_skip_token_list()
    pipe.transformer.old_forward = pipe.transformer.forward
    pipe.transformer.forward = ras_forward.__get__(pipe.transformer, pipe.transformer.__class__) # pylint: disable=no-value-for-parameter
    for block in pipe.transformer.transformer_blocks:
        block.attn.set_processor(RASJointAttnProcessor2_0())


def unapply(pipe):
    if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, "old_forward"):
        from diffusers.models.attention_processor import JointAttnProcessor2_0
        pipe.transformer.forward = pipe.transformer.old_forward
        del pipe.transformer.old_forward
        for block in pipe.transformer.transformer_blocks:
            block.attn.set_processor(JointAttnProcessor2_0())
