import os
import time
from modules import shared, errors, sd_models, processing, devices, images, ui_common
from modules.video_models import models_def, video_utils, video_load, video_vae, video_overrides


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def generate(*args, **kwargs):
    task_id, ui_state, engine, model, prompt, negative, styles, width, height, frames, steps, sampler_index, sampler_shift, dynamic_shift, seed, guidance_scale, guidance_true, init_image, init_strength, last_image, vae_type, vae_tile_frames, save_frames, video_type, video_duration, video_loop, video_pad, video_interpolate, override_settings = args
    if engine is None or model is None or engine == 'None' or model == 'None':
        return video_utils.queue_err('model not selected')
    found = [model.name for model in models_def.models.get(engine, [])]
    selected: models_def.Model = [m for m in models_def.models[engine] if m.name == model][0] if len(found) > 0 else None
    if not shared.sd_loaded:
        debug('Video: model not yet loaded')
        video_load.load_model(selected)
    if selected.name != video_load.loaded_model:
        debug('Video: force reload')
        video_load.load_model(selected)
    if not shared.sd_loaded:
        debug('Video: model still not loaded')
        return video_utils.queue_err('model not loaded')
    debug(f'Video generate: task={task_id} args={args} kwargs={kwargs}')

    p = processing.StableDiffusionProcessingVideo(
        sd_model=shared.sd_model,
        prompt=prompt,
        negative_prompt=negative,
        styles=styles,
        seed=int(seed),
        sampler_name = processing.get_sampler_name(sampler_index),
        sampler_shift=float(sampler_shift),
        steps=int(steps),
        width=16 * int(width // 16),
        height=16 * int(height // 16),
        frames=int(frames),
        denoising_strength=float(init_strength),
        init_image=init_image,
        cfg_scale=float(guidance_scale),
        pag_scale=float(guidance_true),
        vae_type=vae_type,
        vae_tile_frames=int(vae_tile_frames),
        override_settings=override_settings,
    )
    if p.vae_type == 'Remote' and not selected.vae_remote:
        shared.log.warning(f'Video: model={selected.name} remote vae not supported')
        p.vae_type = 'Default'
    p.scripts = None
    p.script_args = None
    p.state = ui_state
    p.do_not_save_grid = True
    p.do_not_save_samples = not save_frames
    p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_video
    if 'I2V' in model:
        if init_image is None:
            return video_utils.queue_err('init image not set')
        p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        shared.log.debug(f'Video: op=I2V init={init_image} resized={p.task_args["image"]}')
    elif 'FLF2V' in model:
        if init_image is None:
            return video_utils.queue_err('init image not set')
        if last_image is None:
            return video_utils.queue_err('last image not set')
        p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        p.task_args['last_image'] = images.resize_image(resize_mode=2, im=last_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        shared.log.debug(f'Video: op=FLF2V init={init_image} last={last_image} resized={p.task_args["image"]}')
    elif 'T2V' in model:
        if init_image is not None:
            shared.log.warning('Video: op=T2V init image not supported')

    # cleanup memory
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    devices.torch_gc(force=True, reason='video')

    # set args
    processing.fix_seed(p)
    video_vae.set_vae_params(p)
    video_utils.set_prompt(p)
    p.task_args['num_inference_steps'] = p.steps
    p.task_args['width'] = p.width
    p.task_args['height'] = p.height
    p.task_args['output_type'] = 'latent' if (p.vae_type == 'Remote') else 'pil'
    p.ops.append('video')
    orig_dynamic_shift = shared.opts.schedulers_dynamic_shift
    orig_sampler_shift = shared.opts.schedulers_shift
    shared.opts.data['schedulers_dynamic_shift'] = dynamic_shift
    shared.opts.data['schedulers_shift'] = sampler_shift
    video_overrides.set_overrides(p, selected)
    debug(f'Video: task_args={p.task_args}')

    # run processing
    shared.state.disable_preview = True
    shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} width={p.width} height={p.height} frames={p.frames} steps={p.steps}')
    err = None
    t0 = time.time()
    processed = None
    try:
        processed = processing.process_images(p)
    except Exception as e:
        err = str(e)
        errors.display(e, 'video')
    t1 = time.time()
    shared.state.disable_preview = False
    shared.opts.data['schedulers_dynamic_shift'] = orig_dynamic_shift
    shared.opts.data['schedulers_shift'] = orig_sampler_shift
    p.close()

    # done
    if err:
        return video_utils.queue_err(err)
    if processed is None or len(processed.images) == 0:
        return video_utils.queue_err('processing failed')
    shared.log.info(f'Video: name="{selected.name}" cls={shared.sd_model.__class__.__name__} frames={len(processed.images)} time={t1-t0:.2f}')
    video_file = images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=video_duration, loop=video_loop, pad=video_pad, interpolate=video_interpolate)
    generation_info_js = processed.js() if processed is not None else ''
    return processed.images, video_file, generation_info_js, processed.info, ui_common.plaintext_to_html(processed.comments)
