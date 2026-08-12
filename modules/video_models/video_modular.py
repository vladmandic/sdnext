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


def component_quant_config(pipe) -> dict:
    """Per-component quantization config, read off the pipeline's own component specs.

    Component names differ per architecture, so denoisers and text encoders are
    recognized by the class each spec declares rather than listed here. The result
    carries no default entry, so anything unrecognized loads unquantized.
    """
    model_args = model_quant.create_config(module='Model')
    config = {}
    for name, spec in getattr(pipe, '_component_specs', {}).items(): # pylint: disable=protected-access
        if getattr(spec, 'default_creation_method', None) != 'from_pretrained':
            continue
        cls = getattr(spec, 'type_hint', None)
        origin = getattr(cls, '__module__', '') or ''
        cls_name = getattr(cls, '__name__', '') or ''
        if origin.startswith('transformers') and 'text_encoder' in name:
            # MiniMax-H3's conditioner keeps its vision tower unquantized, which other arches on the same
            # class do not do; a second modular arch would want this passed in rather than assumed here
            te_args = model_quant.create_config(module='TE', modules_to_not_convert=['.model.visual'])
            if 'quantization_config' in te_args:
                config[name] = te_args['quantization_config']
        elif origin.startswith('diffusers') and ('Transformer' in cls_name or 'UNet' in cls_name) and 'quantization_config' in model_args:
            config[name] = model_args['quantization_config']
    return config


def missing_components(pipe, workflow: str | None) -> list:
    """Components the loaded workflow declares that did not materialize.

    A partition the workflow does not use is absent by design, so the comparison is
    against the workflow's own expected components rather than every declared spec.
    """
    blocks = getattr(pipe, '_blocks', None) # pylint: disable=protected-access
    if blocks is None:
        return []
    try:
        expected = blocks.get_workflow(workflow) if workflow else blocks
        names = [spec.name for spec in expected.expected_components]
    except Exception:
        names = list(getattr(pipe, '_component_specs', {})) # pylint: disable=protected-access
    return [name for name in names if getattr(pipe, name, None) is None]


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
        # the workflow restricts the component fetch only: passing it to from_pretrained instead would prune the blocks tree to one task and disable runtime dispatch between them
        load_kwargs = {}
        quant_config = component_quant_config(pipe)
        if quant_config:
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
        empty = [name for name, component in pipe.components.items() if component is None]
        missing = missing_components(pipe, workflow)
        pipe.sdnext_missing_components = missing # a caller that can recover a component clears its own entry
        pipe.sdnext_video_workflow = workflow # the workflow this pipe was loaded for, which is what the reference-workflow guard reads; the executed task is chosen per request
        if hasattr(pipe, 'min_duration') and hasattr(pipe, 'fps'):
            pipe.sdnext_supported_min_frames = int(pipe.min_duration * pipe.fps) # fresh pipes report the true floor; still mode gates per instance
        log.info(f'Load modular: cls={pipe.__class__.__name__} workflow={workflow} components={loaded} empty={empty} time={time.time()-t0:.2f}')
        if missing:
            # load_components builds each component in its own try/except and reports a failure as a warning on the
            # diffusers logger, so the reason is in the log above this line rather than in the exception path
            log.error(f'Load modular: cls={pipe.__class__.__name__} workflow={workflow} missing={missing} components the workflow requires did not load')
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
        log.info(f'Pipeline: cls={pipe.__class__.__name__} mode=still')
    else:
        frames = max(getattr(p, 'frames', 1), getattr(pipe, 'sdnext_supported_min_frames', 120))
        while frames % pipe.vae_frames_per_chunk != pipe.vae_latents_per_chunk: # frame counts align to 17n+5
            frames += 1
        max_frames = int(pipe.max_duration * pipe.fps)
        while frames > max_frames:
            frames -= pipe.vae_frames_per_chunk
    if frames != getattr(p, 'frames', None):
        log.debug(f'Pipeline: cls={pipe.__class__.__name__} frames={getattr(p, "frames", None)} aligned={frames}')
    p.frames = frames
    p.task_args['num_frames'] = frames
    p.steps = max(2, p.steps)
    p.task_args['num_inference_steps'] = p.steps
    pipe.num_timesteps = p.steps - 1 # sigma grid includes the terminal point; feeds the progress total
    if p.sampler_name not in ('None', 'Default'):
        log.warning(f'Pipeline: cls={pipe.__class__.__name__} sampler={p.sampler_name} unsupported: using model default')
    p.sampler_name = 'Default' # the model default is the bespoke scheduler pair, which discrete samplers must not replace
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
            log.debug(f'Pipeline: cls={pipe.__class__.__name__} audio=enabled')
    elif not enabled and 'audio' in sub:
        pipe.sdnext_audio_decode_block = sub.pop('audio')
        log.debug(f'Pipeline: cls={pipe.__class__.__name__} audio=disabled')


class InterruptLogFilter(logging.Filter):
    """Drops the per-block error dumps the modular runner logs when an interrupt raises through it."""
    def filter(self, record):
        return 'Interrupted...' not in record.getMessage()


def install_state_hook(pipe):
    runner_log = logging.getLogger('diffusers.modular_pipelines.modular_pipeline')
    if not any(isinstance(f, InterruptLogFilter) for f in runner_log.filters):
        runner_log.addFilter(InterruptLogFilter())

    def set_phase(phase: str):
        # every stage runs inside one pipeline call, so the forward hooks are the only
        # place the current stage is visible; state.begin clears the label per job
        if getattr(pipe, 'sdnext_phase', None) != phase:
            pipe.sdnext_phase = phase
            shared.state.textinfo = phase
            log.debug(f'Pipeline: cls={pipe.__class__.__name__} phase={phase}')

    def state_hook(module, args): # pylint: disable=unused-argument
        set_phase('Generate')
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

    def encode_hook(module, args): # pylint: disable=unused-argument
        set_phase('TextEncode')
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    def decode_hook(module, args): # pylint: disable=unused-argument
        set_phase('Decode')
        if shared.state.interrupted or shared.state.skipped: # fires per tile, so tiled decodes abort promptly
            raise AssertionError('Interrupted...')

    for name in ('transformer', 'transformer_ref'):
        module = getattr(pipe, name, None)
        if module is None or getattr(module, 'sdnext_state_hook', None) is not None:
            continue
        module.sdnext_state_hook = module.register_forward_pre_hook(state_hook)
    text_encoder = getattr(pipe, 'text_encoder', None)
    if text_encoder is not None:
        target = getattr(text_encoder, 'model', text_encoder) # conditioning calls the inner model directly
        if isinstance(target, torch.nn.Module) and getattr(target, 'sdnext_state_hook', None) is None:
            target.sdnext_state_hook = target.register_forward_pre_hook(encode_hook)
    for name in ('vae', 'audio_vae'):
        decoder = getattr(getattr(pipe, name, None), 'decoder', None) # decode entry points bypass forward, the inner decoder does not
        if isinstance(decoder, torch.nn.Module) and getattr(decoder, 'sdnext_state_hook', None) is None:
            decoder.sdnext_state_hook = decoder.register_forward_pre_hook(decode_hook)
