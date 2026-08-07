import time
import logging
import torch
from modules import shared, errors, devices, model_quant
from modules.logger import log


MIN_LATENT_FRAMES = 7 # decoder floor: fewer latent frames leave the chunked decode with nothing to emit


def is_modular(obj) -> bool:
    if obj is None:
        return False
    cls = obj if isinstance(obj, type) else obj.__class__
    try:
        import diffusers
        modular_cls = getattr(diffusers, 'ModularPipeline', None)
        if isinstance(modular_cls, type) and issubclass(cls, modular_cls):
            return True
    except Exception:
        pass
    return 'Modular' in cls.__name__


def load_modular_pipe(repo_cls, repo: str, workflow: str | None = None, revision: str | None = None, offline_args: dict | None = None, base: bool = False):
    if repo_cls is None or isinstance(repo_cls, str):
        log.error(f'Load modular: repo="{repo}" cls="{repo_cls}" pipeline class not found: diffusers too old')
        return None
    offline_args = offline_args or {}
    cache_dir = shared.opts.diffusers_dir if base else shared.opts.hfcache_dir # base models live in the diffusers folder so the model scan lists them; video-only models stay out of the dropdown
    try:
        t0 = time.time()
        log.debug(f'Load modular: repo="{repo}" cls={repo_cls.__name__} workflow={workflow} base={base}')
        pipe = repo_cls.from_pretrained(
            repo,
            revision=revision,
            cache_dir=cache_dir,
            **offline_args,
        )
        # workflow selection stays out of from_pretrained: pruning the blocks tree to one task
        # would disable runtime auto-dispatch between them; only the component fetch is restricted
        load_kwargs = {}
        quant_config = {}
        quant_args = model_quant.create_config(module='Model')
        if 'quantization_config' in quant_args:
            quant_config['transformer'] = quant_args['quantization_config']
            quant_config['transformer_ref'] = quant_args['quantization_config']
        te_args = model_quant.create_config(module='TE')
        if 'quantization_config' in te_args:
            quant_config['text_encoder'] = te_args['quantization_config']
        if quant_config:
            # per-component dict without a default entry: only the listed components quantize while
            # loading, everything else loads unquantized
            load_kwargs['quantization_config'] = quant_config
            log.debug(f'Load modular: quant={next(iter(quant_config.values())).__class__.__name__} modules={list(quant_config)}')
        pipe.load_components(
            workflow=workflow,
            dtype=devices.dtype,
            cache_dir=cache_dir,
            **load_kwargs,
            **offline_args,
        )
        loaded = [name for name, component in pipe.components.items() if component is not None]
        if hasattr(pipe, 'min_duration') and hasattr(pipe, 'fps'):
            pipe.sdnext_supported_min_frames = int(pipe.min_duration * pipe.fps) # fresh pipes report the true floor; still mode gates per instance
        log.debug(f'Load modular: cls={pipe.__class__.__name__} workflow={workflow} components={loaded} time={time.time()-t0:.2f}')
        return pipe
    except Exception as e:
        log.error(f'Load modular: repo="{repo}" workflow={workflow} {e}')
        errors.display(e, 'video')
        return None


def load_modular(selected, offline_args: dict):
    return load_modular_pipe(selected.repo_cls, selected.repo, workflow=selected.workflow, revision=selected.repo_revision, offline_args=offline_args, base=selected.base)


def apply_minimax_overrides(p, pipe, still: bool = False, audio: bool = True):
    """Per-generation constraints shared by the video tab and the image path: canvas and frame
    alignment, the bespoke scheduler guard, tiling, and the audio/still toggles."""
    if still:
        audio = False # a sub-second soundtrack is pure waste on a kept single frame
    multiple = pipe.canvas_multiple
    p.task_args['width'] = multiple * (p.width // multiple)
    p.task_args['height'] = multiple * (p.height // multiple)
    set_still(pipe, still)
    if still:
        frames = 5 # two latent frames; decode pads to the decoder floor and only the first frame is kept
        log.info(f'Video modular: cls={pipe.__class__.__name__} mode=still experimental')
    else:
        frames = max(getattr(p, 'frames', 1), getattr(pipe, 'sdnext_supported_min_frames', 120))
        while frames % pipe.vae_frames_per_chunk != pipe.vae_latents_per_chunk: # frame counts align to 17n+5
            frames += 1
        max_frames = int(pipe.max_duration * pipe.fps)
        while frames > max_frames:
            frames -= pipe.vae_frames_per_chunk
    if frames != getattr(p, 'frames', None):
        log.debug(f'Video modular: cls={pipe.__class__.__name__} frames={getattr(p, "frames", None)} aligned={frames}')
    p.frames = frames
    p.task_args['num_frames'] = frames
    p.steps = max(2, p.steps)
    p.task_args['num_inference_steps'] = p.steps
    pipe.num_timesteps = p.steps - 1 # sigma grid includes the terminal point; feeds the progress total
    if p.sampler_name not in ('None', 'Default'):
        log.warning(f'Video modular: cls={pipe.__class__.__name__} sampler={p.sampler_name} unsupported: using model scheduler')
    p.sampler_name = 'None' # bespoke scheduler pair must not be replaced
    pipe.vae.enable_tiling() # model always tiles; the shared vae params path may have disabled it
    set_audio(pipe, audio)
    p.task_args['output'] = ['videos', 'audio', 'sampling_rate'] if audio else ['videos']
    p.task_args['output_type'] = 'pil' # the image path otherwise requests latent output, which the decode block rejects
    p.video_still = still


def set_still(pipe, enabled: bool = True):
    """Toggle sub-floor generation for single-frame output. The duration floor is lifted only
    while the instance flag is set, so other pipes of the class and later normal runs keep the
    supported floor; decoded latents below the decoder floor are padded by duplicating the
    trailing latent. The causal VAE keeps padding out of frame 0."""
    cls = type(pipe)
    if getattr(cls, 'sdnext_min_duration_orig', None) is None:
        orig = cls.min_duration
        cls.sdnext_min_duration_orig = orig
        cls.min_duration = property(lambda self: 0.0 if getattr(self, 'sdnext_still_mode', False) else orig.fget(self))
    pipe.sdnext_still_mode = enabled
    if not enabled:
        return
    vae = getattr(pipe, 'vae', None)
    if vae is not None and getattr(vae, 'sdnext_orig_decode', None) is None:
        vae.sdnext_orig_decode = vae.decode
        def padded_decode(z, *args, **kwargs):
            if z.ndim == 5 and z.shape[2] < MIN_LATENT_FRAMES:
                pad = z[:, :, -1:].repeat(1, 1, MIN_LATENT_FRAMES - z.shape[2], 1, 1)
                z = torch.cat([z, pad], dim=2)
            return vae.sdnext_orig_decode(z, *args, **kwargs)
        vae.decode = padded_decode


def set_audio(pipe, enabled: bool):
    """Pop or restore the audio decode block. The joint denoise still carries the audio rows
    (a few percent of the sequence), but without the block the audio VAE never runs.
    Operates on the backing block tree: the public blocks property deep-copies per access."""
    blocks = getattr(pipe, '_blocks', None) # pylint: disable=protected-access
    decode = blocks.sub_blocks.get('decode', None) if blocks is not None and hasattr(blocks, 'sub_blocks') else None
    sub = getattr(decode, 'sub_blocks', None)
    if sub is None:
        return
    if enabled and 'audio' not in sub:
        stashed = getattr(pipe, 'sdnext_audio_decode_block', None)
        if stashed is not None:
            sub.insert('audio', stashed, len(sub))
            log.debug(f'Video modular: cls={pipe.__class__.__name__} audio=enabled')
    elif not enabled and 'audio' in sub:
        pipe.sdnext_audio_decode_block = sub.pop('audio')
        log.debug(f'Video modular: cls={pipe.__class__.__name__} audio=disabled')


class InterruptLogFilter(logging.Filter):
    """Drops the per-block error dumps the modular runner logs when an interrupt raises through it."""
    def filter(self, record):
        return 'Interrupted...' not in record.getMessage()


def install_state_hook(pipe):
    runner_log = logging.getLogger('diffusers.modular_pipelines.modular_pipeline')
    if not any(isinstance(f, InterruptLogFilter) for f in runner_log.filters):
        runner_log.addFilter(InterruptLogFilter())

    def state_hook(module, args): # pylint: disable=unused-argument
        if shared.state.sampling_steps == 0 and getattr(pipe, 'num_timesteps', 0) > 0:
            shared.state.sampling_steps = pipe.num_timesteps
        if shared.state.paused:
            log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)
        shared.state.step()
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    for name in ('transformer', 'transformer_ref'):
        module = getattr(pipe, name, None)
        if module is None or getattr(module, 'sdnext_state_hook', None) is not None:
            continue
        module.sdnext_state_hook = module.register_forward_pre_hook(state_hook)
