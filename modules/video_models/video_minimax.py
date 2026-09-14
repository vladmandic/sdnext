import torch
from modules.logger import log


MIN_LATENT_FRAMES = 7 # decoder floor: fewer latent frames leave the chunked decode with nothing to emit


def apply_overrides(p, pipe, still: bool = False, audio: bool = True, preview: bool = False):
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
        log.debug(f'Pipeline: cls={pipe.__class__.__name__} frames requested={getattr(p, "frames", None)} aligned={frames}')
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
    p.task_args['output_type'] = 'pil' if still else 'np'
    p.video_still = still

    if preview:
        from pipelines.minimax.minimax_latents import unpack_latents
        pipe.custom_unpack_latents = unpack_latents # add a helper to unpack the video latents from the block state
    else:
        if hasattr(pipe, 'custom_unpack_latents'):
            del pipe.custom_unpack_latents


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


def calculate_video_shift(steps: int, value: float = 0.40, max_shift: float = 16.0) -> float:
    value = max(0.05, min(0.95, value))
    return min(max_shift, round((value * steps) + 0.5))


def calculate_audio_shift(steps: int, value: float = 0.15, max_shift: float = 6.0) -> float:
    value = max(0.05, min(0.95, value))
    return min(max_shift, round((value * steps) + 0.5))


def set_sampler_shift(pipe, steps: int, video_shift: float = 12.0, audio_shift: float = 3.0):
    scheduler = getattr(pipe, 'scheduler', None)
    audio_scheduler = getattr(pipe, 'audio_scheduler', None)
    if not hasattr(scheduler, 'set_shift') or not hasattr(audio_scheduler, 'set_shift'):
        log.warning(f'Pipeline: cls={pipe.__class__.__name__} scheduler={scheduler.__class__.__name__} audio={audio_scheduler.__class__.__name__} shift unsupported')
        return
    video_calc_shift = calculate_video_shift(steps=steps, value=video_shift)
    audio_calc_shift = calculate_audio_shift(steps=steps, value=audio_shift)
    dct_video = { 'value': video_shift, 'shift': video_calc_shift }
    dct_audio = { 'value': audio_shift, 'shift': audio_calc_shift }
    # set_shift keeps the shipped value in config.shift; the default sampler restores scheduler from default_scheduler every generation, so that copy carries the shift too
    for target in (scheduler, getattr(pipe, 'default_scheduler', None)):
        if hasattr(target, 'set_shift'):
            target.set_shift(video_calc_shift)
    audio_scheduler.set_shift(audio_calc_shift)
    log.debug(f'Pipeline: scheduler={scheduler.__class__.__name__} video={dct_video} audio={dct_audio}')
