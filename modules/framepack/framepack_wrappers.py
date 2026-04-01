import os
import re
import random
import numpy as np
import torch
import gradio as gr
from PIL import Image
from modules import shared, processing, timer, paths, extra_networks, progress, ui_video_vlm, call_queue
from modules.logger import log
from modules.video_models.video_utils import check_av
from modules.framepack import framepack_install # pylint: disable=wrong-import-order
from modules.framepack import framepack_load # pylint: disable=wrong-import-order
from modules.framepack import framepack_worker # pylint: disable=wrong-import-order
from modules.framepack import framepack_hijack # pylint: disable=wrong-import-order


tmp_dir = os.path.join(paths.data_path, 'tmp', 'framepack')
git_dir = os.path.join(os.path.dirname(__file__), 'framepack')
git_repo = 'https://github.com/lllyasviel/framepack'
git_commit = 'c5d375661a2557383f0b8da9d11d14c23b0c4eaf'
loaded_variant = None


def prepare_image(image, resolution):
    from modules.framepack.pipeline.utils import resize_and_center_crop
    buckets = [
        (416, 960), (448, 864), (480, 832), (512, 768), (544, 704), (576, 672), (608, 640),
        (640, 608), (672, 576), (704, 544), (768, 512), (832, 480), (864, 448), (960, 416),
    ]
    if isinstance(image, Image.Image):
        image = np.array(image)
    h, w, _c = image.shape
    min_metric = float('inf')
    scale_factor = resolution / 640.0
    scaled_h, scaled_w = h, w
    for (bucket_h, bucket_w) in buckets:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            scaled_h = round(bucket_h * scale_factor / 16) * 16
            scaled_w = round(bucket_w * scale_factor / 16) * 16

    image = resize_and_center_crop(image, target_height=scaled_h, target_width=scaled_w)
    h0, w0, _c = image.shape
    log.debug(f'FramePack prepare: input="{w}x{h}" resized="{w0}x{h0}" resolution={resolution} scale={scale_factor}')
    return image


def interpolate_prompts(prompts, steps):
    interpolated_prompts = [''] * steps
    if prompts is None:
        return interpolated_prompts
    if isinstance(prompts, str):
        prompts = re.split(r'[,\n]', prompts)
        prompts = [p.strip() for p in prompts]
    if len(prompts) == 0:
        return interpolated_prompts
    if len(prompts) == steps:
        return prompts
    factor = steps / len(prompts)
    for i in range(steps):
        prompt_index = int(i / factor)
        interpolated_prompts[i] = prompts[prompt_index]
        # log.trace(f'FramePack interpolate: section={i} prompt="{interpolated_prompts[i]}"')
    return interpolated_prompts


def prepare_prompts(p, init_image, prompt:str, section_prompt:str, num_sections:int, vlm_enhance:bool, vlm_model:str, vlm_system_prompt:str):
    section_prompts = interpolate_prompts(section_prompt, num_sections)
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    shared.prompt_styles.apply_styles_to_extra(p)
    p.prompts, p.network_data = extra_networks.parse_prompts([p.prompt])
    extra_networks.activate(p)
    prompt = p.prompts[0]
    generated_prompts = [''] * num_sections
    previous_prompt = None
    for i in range(num_sections):
        current_prompt = (prompt + ' ' + section_prompts[i]).strip()
        if current_prompt == previous_prompt:
            generated_prompts[i] = generated_prompts[i - 1]
        else:
            generated_prompts[i] = ui_video_vlm.enhance_prompt(
                enable=vlm_enhance,
                model=vlm_model,
                image=init_image,
                prompt=current_prompt,
                system_prompt=vlm_system_prompt,
            )
            previous_prompt = current_prompt
    return generated_prompts


def load_model(variant, attention):
    global loaded_variant # pylint: disable=global-statement
    if (shared.sd_model_type != 'hunyuanvideo') or (loaded_variant != variant):
        yield gr.update(), gr.update(), 'Verifying FramePack'
        framepack_install.install_requirements(attention)
        # framepack_install.git_clone(git_repo=git_repo, git_dir=git_dir, tmp_dir=tmp_dir)
        # framepack_install.git_update(git_dir=git_dir, git_commit=git_commit)
        # sys.path.append(git_dir)
        framepack_hijack.set_progress_bar_config()
        yield gr.update(), gr.update(), 'Model loading...', ''
        loaded_variant = framepack_load.load_model(variant)
        if loaded_variant is not None:
            yield gr.update(), gr.update(), 'Model loaded'
        else:
            yield gr.update(), gr.update(), 'Model load failed'


def unload_model():
    log.debug('FramePack unload')
    framepack_load.unload_model()
    yield gr.update(), gr.update(), 'Model unloaded'


def run_framepack(task_id, _ui_state, init_image, end_image, start_weight, end_weight, vision_weight, prompt, system_prompt, optimized_prompt, section_prompt, negative_prompt, styles, seed, resolution, duration, latent_ws, steps, cfg_scale, cfg_distilled, cfg_rescale, shift, use_teacache, use_cfgzero, use_preview, mp4_fps, mp4_codec, mp4_sf, mp4_video, mp4_frames, mp4_opt, mp4_ext, mp4_interpolate, attention, vae_type, variant, vlm_enhance, vlm_model, vlm_system_prompt):
    variant = variant or 'bi-directional'
    if init_image is None:
        init_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        mode = 't2v'
    elif end_image is not None:
        mode = 'flf2v'
    else:
        mode = 'i2v'

    av = check_av()
    if av is None:
        yield gr.update(), gr.update(), 'AV package not installed'
        return

    progress.add_task_to_queue(task_id)
    with call_queue.get_lock():
        progress.start_task(task_id)

        yield from load_model(variant, attention)
        if shared.sd_model_type != 'hunyuanvideo':
            progress.finish_task(task_id)
            yield gr.update(), gr.update(), 'Model load failed'
            return

        yield gr.update(), gr.update(), 'Generate starting...'
        from modules.framepack.pipeline.thread_utils import AsyncStream, async_run
        framepack_worker.stream = AsyncStream()

        if seed is None or seed == '' or seed == -1:
            random.seed()
            seed = random.randrange(4294967294)
        seed = int(seed)
        torch.manual_seed(seed)
        num_sections = len(framepack_worker.get_latent_paddings(mp4_fps, mp4_interpolate, latent_ws, duration, variant))
        num_frames = (latent_ws * 4 - 3) * num_sections + 1
        log.info(f'FramePack start: mode={mode} variant="{variant}" frames={num_frames} sections={num_sections} resolution={resolution} seed={seed} duration={duration} teacache={use_teacache} thres={shared.opts.teacache_thresh} cfgzero={use_cfgzero}')
        log.info(f'FramePack params: steps={steps} start={start_weight} end={end_weight} vision={vision_weight} scale={cfg_scale} distilled={cfg_distilled} rescale={cfg_rescale} shift={shift}')
        init_image = prepare_image(init_image, resolution)
        if end_image is not None:
            end_image = prepare_image(end_image, resolution)
        w, h, _c = init_image.shape
        p = processing.StableDiffusionProcessingVideo(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=styles,
            steps=steps,
            seed=seed,
            width=w,
            height=h,
        )
        p.ops.append('video')
        prompts = prepare_prompts(p, init_image, prompt, section_prompt, num_sections, vlm_enhance, vlm_model, vlm_system_prompt)

        async_run(
            framepack_worker.worker,
            init_image, end_image,
            start_weight, end_weight, vision_weight,
            prompts, p.negative_prompt, system_prompt, optimized_prompt, vlm_enhance,
            seed,
            duration,
            latent_ws,
            p.steps,
            cfg_scale, cfg_distilled, cfg_rescale,
            shift,
            use_teacache, use_cfgzero, use_preview,
            mp4_fps, mp4_codec, mp4_sf, mp4_video, mp4_frames, mp4_opt, mp4_ext, mp4_interpolate,
            vae_type, variant,
        )

        output_filename = None
        while True:
            flag, data = framepack_worker.stream.output_queue.next()
            if flag == 'file':
                output_filename = data
                yield output_filename, gr.update(), gr.update()
            if flag == 'progress':
                preview, text = data
                summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
                memory = shared.mem_mon.summary()
                stats = f"<div class='performance'><p>{summary} {memory}</p></div>"
                yield gr.update(), gr.update(value=preview), f'{text} {stats}'
            if flag == 'end':
                yield output_filename, gr.update(value=None), gr.update()
                break

        progress.finish_task(task_id)
    yield gr.update(), gr.update(), 'Generate finished'
    return
