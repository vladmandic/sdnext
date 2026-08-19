import os
import copy
import time
from dataclasses import dataclass
from modules import shared, errors, sd_models, processing, devices, images, ui_common, scripts_manager, modular_load
from modules.logger import log
from modules.video_models import models_def, video_utils, video_load, video_vae, video_overrides, video_save
from modules.paths import resolve_output_path


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
REFERENCE_WORKFLOWS = ('ref2va',) # workflows that condition on references; the resolver and the limits it enforces live with the architecture


class VideoError(Exception):
    """Video generation failure; code follows HTTP semantics so API callers can map it directly."""
    def __init__(self, msg: str, code: int = 500):
        super().__init__(msg)
        self.code = code


@dataclass
class VideoResult:
    images: list # PIL frames as produced; callers decide what to surface
    video_path: str | None
    thumb_path: str | None
    num_frames: int
    fps: float # effective save fps after interpolation
    has_audio: bool
    still: bool
    processed: processing.Processed
    width: int = 0 # what was generated, which is not what was requested whenever a runner rounds or a model picks
    height: int = 0


def resolve_model(engine: str | None, model: str | None) -> tuple[models_def.Model, bool]:
    """Return (selected, needs_load): a registry row when both names are given, or a synthesized
    row describing the already-loaded pipeline when both are omitted."""
    engine_given = engine not in (None, '', 'None')
    model_given = model not in (None, '', 'None')
    if engine_given != model_given:
        raise VideoError('video model selection requires both engine and model', 400)
    if engine_given:
        selected = models_def.find(engine, model)
        if selected is None:
            available = models_def.model_names(engine) or models_def.engines()
            raise VideoError(f'video model not found: engine="{engine}" model="{model}" available={available}', 404)
        return selected, True
    cls = shared.sd_model.__class__.__name__ if shared.sd_loaded else None
    if not shared.sd_loaded or cls not in models_def.pipeline_classes():
        raise VideoError(f'no video model loaded: cls={cls} select engine and model or load a video-capable checkpoint first', 400)
    pipe = shared.sd_model
    workflow = getattr(pipe, 'sdnext_video_workflow', None)
    if workflow is None and modular_load.is_modular(pipe):
        workflow = models_def.workflow_for_class(cls) or 'auto' # modular pipes dispatch on inputs, so any workflow marker selects the modular branch
    ckpt = getattr(pipe, 'sd_checkpoint_info', None)
    selected = models_def.Model(
        name=getattr(ckpt, 'title', None) or cls,
        repo=getattr(ckpt, 'name', None),
        repo_cls=type(pipe),
        workflow=workflow,
        base=True,
    )
    return selected, False


def reference_caps(workflow: str | None):
    """Reference limits of a workflow, None when it conditions on none. The seam a client reads
    instead of mirroring the numbers."""
    if workflow not in REFERENCE_WORKFLOWS:
        return None
    from modules.minimax import minimax_references
    return minimax_references.get_reference_caps(workflow)


def validate_references(selected: models_def.Model, references: list | None, init_image) -> list | None:
    """Return the ordered references a reference workflow conditions on, None for every other model.
    Reference conditioning is exclusive to ref2va: its partition holds no keyframe transformer, and
    a mismatched request would only fail once the pipeline reached a component it never loaded.
    Checks run before the model load so a rejected request costs nothing."""
    workflow = getattr(selected, 'workflow', None)
    if workflow not in REFERENCE_WORKFLOWS:
        if references:
            raise VideoError(f'reference media requires a reference workflow: model="{selected.name}" workflow={workflow} supported={list(REFERENCE_WORKFLOWS)}', 400)
        return None
    from modules.minimax import minimax_references
    return minimax_references.resolve(workflow, references, init_image)


def run(selected: models_def.Model, *,
        prompt: str,
        negative: str = '',
        styles: list | None = None,
        width: int = 832,
        height: int = 480,
        frames: int = 17,
        steps: int = 50,
        sampler_name: str = 'Default',
        sampler_shift: float = -1.0,
        dynamic_shift: bool = False,
        seed: int = -1,
        guidance_scale: float = -1.0,
        guidance_true: float = -1.0,
        init_image=None,
        init_strength: float = 0.8,
        last_image=None,
        references: list | None = None,
        vae_type: str = 'Default',
        vae_tile_frames: int = 16,
        audio: bool = True,
        mp4_fps: int = 24,
        mp4_interpolate: int = 0,
        mp4_codec: str = 'libx264',
        mp4_ext: str = 'mp4',
        mp4_opt: str = 'crf=16',
        mp4_video: bool = True,
        mp4_frames: bool = False,
        mp4_sf: bool = False,
        mp4_thumb: bool = True,
        override_settings=None,
        engine: str | None = None,
        ui_state=None,
        scripts=None,
        script_args=(),
        per_script_args: dict | None = None,
        extra_p: dict | None = None,
        needs_load: bool = True,
       ) -> VideoResult:

    refs = validate_references(selected, references, init_image)

    if needs_load:
        if not shared.sd_loaded:
            debug('Video: model not yet loaded')
            video_load.load_model(selected)
        if selected.name != video_load.loaded_model:
            debug('Video: force reload')
            video_load.load_model(selected)
        if not shared.sd_loaded:
            debug('Video: model still not loaded')
            raise VideoError('model not loaded', 500)

    if isinstance(override_settings, (list, tuple)): # the ui override control emits "setting: value" pairs; always empty on the video tab since the control stays hidden
        from modules.generation_parameters_copypaste import create_override_settings_dict
        override_settings = create_override_settings_dict(override_settings)

    p = processing.StableDiffusionProcessingVideo(
        sd_model=shared.sd_model,
        video_engine=engine or 'Loaded',
        video_model=selected.name,
        prompt=prompt,
        negative_prompt=negative,
        styles=styles or [],
        seed=int(seed),
        sampler_name=sampler_name,
        sampler_shift=float(sampler_shift),
        steps=int(steps),
        width=16 * int(width // 16),
        height=16 * int(height // 16),
        frames=int(frames),
        denoising_strength=float(init_strength),
        init_image=init_image,
        cfg_scale=float(guidance_scale),
        cfg_true=float(guidance_true),
        vae_type=vae_type,
        vae_tile_frames=int(vae_tile_frames),
        video_audio=bool(audio),
        override_settings=override_settings,
    )
    if p.vae_type == 'Remote' and not selected.vae_remote:
        log.warning(f'Video: model={selected.name} remote vae not supported')
        p.vae_type = 'Default'

    p.state = ui_state
    if per_script_args:
        p.per_script_args.update(per_script_args)
    for k, v in (extra_p or {}).items():
        setattr(p, k, v)
    p.scripts = scripts if scripts is not None else scripts_manager.scripts_video
    p.script_args = tuple(script_args)
    p.scripts.run(p, *script_args)

    p.do_not_save_grid = True
    p.do_not_save_samples = not mp4_frames
    p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)
    mode = models_def.dispatch_mode(selected)
    if mode == 'workflow':
        # modular workflows dispatch on which inputs are present; keyframes pass through
        # unresized since the pipeline defines its own canvas placement per anchor
        p.video_still = int(frames) <= 1
        if refs is not None:
            # references outrank the keyframe inputs in every block, so those stay unset
            p.task_args['references'] = refs
            if last_image is not None:
                log.warning(f'Video: op=reference model="{selected.name}" last frame not supported, ignoring')
        else:
            if init_image is not None:
                p.task_args['image'] = init_image
            if last_image is not None:
                p.task_args['last_image'] = last_image
        if p.video_still:
            p.do_not_save_samples = False # the still is the product; save it like an image result
        elif int(mp4_fps) != 24:
            log.warning(f'Video: model="{selected.name}" fps={mp4_fps} model output is fixed at 24')
        log.debug(f'Video: op=modular workflow={selected.workflow} still={p.video_still} init={init_image} last={last_image} references={len(refs) if refs else 0}')
    elif mode == 't2v':
        if init_image is not None:
            log.warning('Video: op=T2V init image not supported')
    elif mode == 'i2v':
        if init_image is None:
            raise VideoError('No input image provided. Please upload or select an image.', 400)
        p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        if last_image is not None and video_utils.supports_last_frame(shared.sd_model):
            p.task_args['last_image'] = images.resize_image(resize_mode=2, im=last_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
            log.debug(f'Video: op=FLF2V init={init_image} last={last_image} resized={p.task_args["image"]}')
        elif last_image is not None:
            log.warning(f'Video: op=I2V model="{selected.name}" last frame not supported, ignoring')
        else:
            log.debug(f'Video: op=I2V init={init_image} resized={p.task_args["image"]}')
    elif mode == 'flf2v':
        if init_image is None:
            raise VideoError('No input image provided. Please upload or select an image.', 400)
        if last_image is None:
            raise VideoError('No last frame image provided. Please upload or select an image.', 400)
        p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        p.task_args['last_image'] = images.resize_image(resize_mode=2, im=last_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        log.debug(f'Video: op=FLF2V init={init_image} last={last_image} resized={p.task_args["image"]}')
    elif mode == 'vace':
        if init_image is not None:
            p.task_args['reference_images'] = [images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')]
            log.debug(f'Video: op=VACE reference={init_image} resized={p.task_args["reference_images"]}')
    elif mode == 'animate':
        if init_image is None:
            raise VideoError('No input image provided. Please upload or select an image.', 400)
        p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        p.task_args['mode'] = 'animate'
        p.task_args['pose_video'] = [] # input pose video to condition the generation on. must be a list of PIL images.
        p.task_args['face_video'] = [] # input face video to condition the generation on. must be a list of PIL images.
        log.debug(f'Video: op=Animate init={p.task_args["image"]} pose={p.task_args["pose_video"]} face={p.task_args["face_video"]}')
    elif mode == 'condition':
        # the conditioning inputs these models accept are wired on the ltx tab, not here
        log.warning(f'Video: op=condition model="{selected.name}" conditioning not supported here, running text to video')
        if init_image is not None:
            log.warning(f'Video: op=condition model="{selected.name}" init image not supported, ignoring')
    else:
        log.warning(f'Video: unknown model type "{selected.name}"')

    # cleanup memory
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    devices.torch_gc(force=True, reason='video')

    # set args
    processing.fix_seed(p)
    video_vae.set_vae_params(p)
    p.task_args['num_inference_steps'] = p.steps
    p.task_args['width'] = p.width
    p.task_args['height'] = p.height
    p.task_args['output_type'] = 'latent' if (p.vae_type == 'Remote') else 'pil'
    p.ops.append('video')

    # set scheduler params
    orig_dynamic_shift = shared.opts.schedulers_dynamic_shift
    orig_sampler_shift = shared.opts.schedulers_shift
    shared.opts.data['schedulers_dynamic_shift'] = dynamic_shift
    shared.opts.data['schedulers_shift'] = sampler_shift
    if hasattr(shared.sd_model, 'scheduler') and hasattr(shared.sd_model.scheduler, 'config') and hasattr(shared.sd_model.scheduler, 'register_to_config'):
        if hasattr(shared.sd_model.scheduler.config, 'use_dynamic_shifting'):
            shared.sd_model.scheduler.config.use_dynamic_shifting = dynamic_shift
            shared.sd_model.scheduler.register_to_config(use_dynamic_shifting = dynamic_shift)
        if hasattr(shared.sd_model.scheduler.config, 'flow_shift') and sampler_shift >= 0:
            shared.sd_model.scheduler.config.flow_shift = sampler_shift
            shared.sd_model.scheduler.register_to_config(flow_shift = sampler_shift)
        shared.sd_model.default_scheduler = copy.deepcopy(shared.sd_model.scheduler)

    video_overrides.set_overrides(p, selected)
    debug(f'Video: task_args={p.task_args}')

    if p.vae_type == 'Upscale':
        video_load.load_upscale_vae()
    elif hasattr(shared.sd_model, 'orig_vae'):
        shared.sd_model.vae = shared.sd_model.orig_vae

    # run processing
    log.debug(f'Video: cls={shared.sd_model.__class__.__name__} width={p.width} height={p.height} frames={p.frames} steps={p.steps}')
    err = None
    t0 = time.time()
    processed = None
    try:
        processed = processing.process_images(p)
    except Exception as e:
        err = str(e)
        errors.display(e, 'video')
    t1 = time.time()
    shared.opts.data['schedulers_dynamic_shift'] = orig_dynamic_shift
    shared.opts.data['schedulers_shift'] = orig_sampler_shift
    p.close()

    # done
    if err:
        raise VideoError(err, 500)
    if processed is None or (len(processed.images) == 0 and processed.bytes is None):
        raise VideoError('processing failed', 500)
    log.info(f'Video: name="{selected.name}" cls={shared.sd_model.__class__.__name__} frames={len(processed.images)} time={t1-t0:.2f}')

    if getattr(p, 'video_still', False):
        stills = processed.images[:1] # already trimmed in process_decode; defensive
        still_w, still_h = video_utils.pixel_size(stills, fallback=(p.width, p.height))
        return VideoResult(images=stills, video_path=None, thumb_path=None, num_frames=len(stills), fps=0.0, has_audio=False, still=True, processed=processed, width=still_w, height=still_h)

    if hasattr(processed, 'images') and processed.images is not None:
        pixels = video_save.images_to_tensor(processed.images)
    else:
        pixels = None
    if hasattr(processed, 'audio') and processed.audio is not None:
        waveform = processed.audio[0].float().cpu()
    else:
        waveform = None

    if mp4_interpolate > 0 and pixels is not None:
        p.video_interpolate = mp4_interpolate
        from modules.processing_video import apply_video_interpolation
        # pixels is 5-D (N,C,T,H,W) in [-1,1]; RIFE needs 4-D (T,C,H,W) in [0,1]
        x = pixels.squeeze(0).permute(1, 0, 2, 3)
        x = (x.clamp(-1., 1.) + 1.0) * 0.5
        x = apply_video_interpolation(p, x, count=mp4_interpolate)
        x = x * 2.0 - 1.0
        pixels = x.permute(1, 0, 2, 3).unsqueeze(0)
    from modules.processing_video import interpolation_factor
    save_fps = mp4_fps * interpolation_factor(p)
    num_frames, video_file, thumb_file = video_save.save_video(
        p=p,
        pixels=pixels,
        audio=waveform,
        aac_sample_rate=video_save.get_audio_rate(p),
        binary=processed.bytes,
        mp4_fps=save_fps,
        mp4_codec=mp4_codec,
        mp4_opt=mp4_opt,
        mp4_ext=mp4_ext,
        mp4_sf=mp4_sf,
        mp4_video=mp4_video,
        mp4_frames=mp4_frames,
        mp4_thumb=mp4_thumb,
        mp4_interpolate=mp4_interpolate,
        metadata={},
    )
    out_w, out_h = video_utils.pixel_size(processed.images, fallback=(p.width, p.height))
    del pixels
    return VideoResult(images=processed.images, video_path=video_file, thumb_path=thumb_file, num_frames=num_frames, fps=float(save_fps), has_audio=waveform is not None, still=False, processed=processed, width=out_w, height=out_h)


def generate(task_id, ui_state,
             engine, model,
             prompt, negative, styles,
             width, height, frames, steps,
             sampler_index, sampler_shift, dynamic_shift,
             seed, guidance_scale, guidance_true,
             init_image, init_strength, last_image,
             vae_type, vae_tile_frames, audio,
             mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb,
             override_settings,
             *args, **kwargs
            ):
    # gradio adapter around run(): the positional signature is frozen since external callers bind to it
    if engine is None or model is None or engine == 'None' or model == 'None':
        return video_utils.queue_err('model not selected')
    selected = models_def.find(engine, model)
    if selected is None:
        return video_utils.queue_err(f'model not found: engine="{engine}" model="{model}"')
    debug(f'Video generate: task={task_id} args={args} kwargs={kwargs}')
    try:
        res = run(selected,
                  prompt=prompt,
                  negative=negative,
                  styles=styles,
                  width=width,
                  height=height,
                  frames=frames,
                  steps=steps,
                  sampler_name=processing.get_sampler_name(sampler_index),
                  sampler_shift=sampler_shift,
                  dynamic_shift=dynamic_shift,
                  seed=seed,
                  guidance_scale=guidance_scale,
                  guidance_true=guidance_true,
                  init_image=init_image,
                  init_strength=init_strength,
                  last_image=last_image,
                  vae_type=vae_type,
                  vae_tile_frames=vae_tile_frames,
                  audio=audio,
                  mp4_fps=mp4_fps,
                  mp4_interpolate=mp4_interpolate,
                  mp4_codec=mp4_codec,
                  mp4_ext=mp4_ext,
                  mp4_opt=mp4_opt,
                  mp4_video=mp4_video,
                  mp4_frames=mp4_frames,
                  mp4_sf=mp4_sf,
                  mp4_thumb=mp4_thumb,
                  override_settings=override_settings,
                  engine=engine,
                  ui_state=ui_state,
                  script_args=args,
                 )
    except VideoError as e:
        return video_utils.queue_err(str(e))
    generation_info_js = res.processed.js()
    html_log = ui_common.plaintext_to_html(res.processed.comments)
    if res.still:
        return res.images, None, generation_info_js, res.processed.info, html_log
    result_images = res.images if mp4_frames else []
    return result_images, res.video_path, generation_info_js, res.processed.info, html_log
