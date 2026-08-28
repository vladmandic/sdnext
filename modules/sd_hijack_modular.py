import os
import time
import logging
import torch
import diffusers
from modules.logger import log
from modules import shared, sd_offload, timer
from modules.attention import context as attention_context


debug = os.environ.get('SD_MODULAR_DEBUG', None) is not None
intercepted = set()


def modular_intercept(self, components, state: diffusers.modular_pipelines.modular_pipeline.BlockState, *args, **kwargs):
    t0 = time.time()
    block = type(self).__name__
    keys = state if isinstance(state, list) else list(state.__dict__.keys())
    # run code before block call
    result = self.__orig_call__(components, state, *args, **kwargs)
    t1 = time.time()
    timer.blocks.add(block, t1 - t0)
    # run code after block call
    # TODO modular: intercept latents and set current latents for preview
    """
        if 'latents' in keys:
            ...
        t2 = time.time()
        timer.blocks.add('callback', t2 - t1)
    """
    if debug:
        log.trace(f'Modular intercept: block={block} state={keys} time={t1 - t0:.4f}')
    return result


def patch_blocks(blocks: diffusers.ModularPipelineBlocks):
    """recursively walks the block tree and patches the CLS __call__ method"""
    def _patch_recursive(current_block):
        block_cls = type(current_block)
        if (block_cls not in intercepted) and (block_cls != diffusers.ModularPipelineBlocks):
            if callable(block_cls) and not getattr(block_cls, "_is_patched", False):
                block_cls.__orig_call__ = block_cls.__call__ # store original call for reference
                block_cls.__call__ = modular_intercept
                block_cls._is_patched = True # pylint: disable=protected-access
                intercepted.add(block_cls)
                if debug:
                    log.trace(f'Modular hijack: {block_cls.__name__}')
        for attr in ("sub_blocks", "blocks"): # recurse into child blocks if containers exist
            sub = getattr(current_block, attr, None)
            if isinstance(sub, dict):
                for child in sub.values():
                    if isinstance(child, diffusers.ModularPipelineBlocks):
                        _patch_recursive(child)
            elif isinstance(sub, (list, tuple)):
                for child in sub:
                    if isinstance(child, diffusers.ModularPipelineBlocks):
                        _patch_recursive(child)

    _patch_recursive(blocks)


def register_callbacks(pipe: diffusers.ModularPipeline):
    intercepted.clear()
    if not isinstance(pipe, diffusers.ModularPipeline):
        return
    try:
        patch_blocks(pipe._blocks) # pylint: disable=protected-access
    except Exception as e:
        log.error(f'Modular intercept: {e}')


class InterruptLogFilter(logging.Filter):
    """Drops the per-block error dumps the modular runner logs when an interrupt raises through it."""
    def filter(self, record):
        return 'Interrupted...' not in record.msg


def install_state_hook(pipe):
    runner_log = logging.getLogger('diffusers.modular_pipelines.modular_pipeline')
    if not any(isinstance(f, InterruptLogFilter) for f in runner_log.filters):
        runner_log.addFilter(InterruptLogFilter())

    def set_phase(phase: str, module: torch.nn.Module | None = None):
        # every stage runs inside one pipeline call, so the forward hooks are the only place the current stage is visible
        if getattr(pipe, 'sdnext_phase', None) != phase:
            pipe.sdnext_phase = phase
            jobid = getattr(pipe, 'sdnext_phaseid', None) # previous jobid if any
            shared.state.end(jobid) # clear the previous job if exists
            pipe.sdnext_phaseid = shared.state.begin(phase) # start a new job for the current phase
            log.debug(f'Pipeline: phase={phase.replace(" ", "")} cls={pipe.__class__.__name__} module={module.__class__.__name__ if module is not None else None}')
            return True
        return False

    def _pre_transformer_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Generate', module)
        attention_context.set_role('transformer')
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['transformer', 'transformer_ref'], reason='generate', force=hasattr(pipe, 'sdnext_force_offload'))
        if shared.state.sampling_steps == 0 and getattr(pipe, 'num_timesteps', 0) > 0:
            shared.state.sampling_steps = pipe.num_timesteps
        if shared.state.paused:
            log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)
        shared.state.step()
        attention_context.tick()
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    def _pre_text_encode_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Text Encode', module)
        attention_context.set_role('te')
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['text_encoder'], reason='text encode', force=hasattr(pipe, 'sdnext_force_offload'))
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    def _pre_vae_decode_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Decode', module)
        attention_context.set_role('vae')
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['vae', 'audio_vae'], reason='vae decode', force=hasattr(pipe, 'sdnext_force_offload'))
        if shared.state.interrupted or shared.state.skipped: # fires per tile, so tiled decodes abort promptly
            raise AssertionError('Interrupted...')

    def _pre_vae_encode_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Encode', module)
        attention_context.set_role('vae')
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['vae', 'audio_vae'], reason='vae encode', force=hasattr(pipe, 'sdnext_force_offload'))
        if shared.state.interrupted or shared.state.skipped: # fires per tile, so tiled encodes abort promptly
            raise AssertionError('Interrupted...')

    for name in ('transformer', 'transformer_ref'):
        module = getattr(pipe, name, None)
        if module is not None:
            target = getattr(module, 'model', module) # conditioning calls the inner model directly
            if isinstance(target, torch.nn.Module) and getattr(target, 'sdnext_state_hook', None) is None:
                target.sdnext_state_hook = target.register_forward_pre_hook(_pre_transformer_hook)

    for name in ('text_encoder', 'text_encoder_2'):
        module = getattr(pipe, name, None)
        if module is not None:
            target = getattr(module, 'model', module) # conditioning calls the inner model directly
            if isinstance(target, torch.nn.Module) and getattr(target, 'sdnext_state_hook', None) is None:
                target.sdnext_state_hook = target.register_forward_pre_hook(_pre_text_encode_hook)

    for name in ('vae', 'audio_vae'):
        decoder = getattr(getattr(pipe, name, None), 'decoder', None) # decode entry points bypass forward, the inner decoder does not
        if isinstance(decoder, torch.nn.Module) and getattr(decoder, 'sdnext_state_hook', None) is None:
            decoder.sdnext_state_hook = decoder.register_forward_pre_hook(_pre_vae_decode_hook)
        encoder = getattr(getattr(pipe, name, None), 'encoder', None) # decode entry points bypass forward, the inner encoder does not
        if isinstance(encoder, torch.nn.Module) and getattr(encoder, 'sdnext_state_hook', None) is None:
            encoder.sdnext_state_hook = encoder.register_forward_pre_hook(_pre_vae_encode_hook)
