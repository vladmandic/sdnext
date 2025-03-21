import os
import time
from modules import shared, errors, sd_models, processing, devices, images, ui_common
from modules.video_models import models_def, video_utils, video_load, video_vae, video_cache


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def generate(keyword, *args, **kwargs):
    task_id, ui_state, engine, model, prompt, negative, styles, width, height, frames, steps, sampler_index, sampler_shift, dynamic_shift, seed, guidance_scale, guidance_true, init_image, vae_type, vae_tile_frames, save_frames, video_type, video_duration, video_loop, video_pad, video_interpolate, faster_cache, pyramid_attention, override_settings = args
    if engine is None or model is None or engine == 'None' or model == 'None':
        return video_utils.queue_err('model not selected')
    found = [model.name for model in models_def.models.get(engine, [])]
    selected: models_def.Model = [m for m in models_def.models[engine] if m.name == model][0] if len(found) > 0 else None
    if not shared.sd_loaded or keyword not in shared.sd_model.__class__.__name__:
        video_load.load_model(selected)
    if not shared.sd_loaded or keyword not in shared.sd_model.__class__.__name__:
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
        init_image=init_image,
        cfg_scale=float(guidance_scale),
        diffusers_guidance_rescale=float(guidance_true),
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
    if 'I2V' in model:
        if init_image is None:
            return video_utils.queue_err('init image not set')
        p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')

    # cleanup memory
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    devices.torch_gc(force=True)

    # set args
    processing.fix_seed(p)
    video_vae.set_vae_params(p)
    video_cache.set_cache(faster_cache=faster_cache, pyramid_attention_broadcast=pyramid_attention)
    video_utils.set_prompt(p)
    p.task_args['output_type'] = 'latent' if (p.vae_type == 'Remote') else 'pil'
    p.ops.append('video')
    orig_dynamic_shift = shared.opts.schedulers_dynamic_shift
    orig_sampler_shift = shared.opts.schedulers_shift
    shared.opts.data['schedulers_dynamic_shift'] = dynamic_shift
    shared.opts.data['schedulers_shift'] = sampler_shift
    debug(f'Video: task_args={p.task_args}')

    # run processing
    shared.state.disable_preview = True
    shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} width={p.width} height={p.height} frames={p.frames} steps={p.steps}')
    err = None
    t0 = time.time()
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
