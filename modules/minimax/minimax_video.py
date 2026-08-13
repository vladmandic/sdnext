import os
import time
from PIL import Image
import numpy as np
from modules.logger import log
from modules import shared, processing, timer, progress, paths, sd_models, scripts_manager, call_queue, memstats, processing_video
from modules.video_models import models_def, video_save, video_utils


engine = 'MiniMax'
loaded = ''


def load_model(model: str):
    global loaded # pylint: disable=global-statement
    if model is None or model == '' or model == 'None':
        shared.sd_model = None
        return False
    t0 = time.time()
    from pipelines.model_minimax import load_minimax
    selected: models_def.Model = [m for m in models_def.models[engine] if m.name == model][0]
    requested = f'repo={selected.repo} workflow={selected.workflow}'
    if (loaded == requested) and (shared.sd_model is not None) and (shared.sd_model.__class__.__name__ == 'MiniMaxH3ModularPipeline'):
        log.info(f'Load video: engine="{engine}" selected="{model}" cached')
        return selected.workflow

    log.info(f'Load video: engine="{engine}" selected="{model}"')
    t0 = time.time()
    ckpt = sd_models.CheckpointInfo(filename=selected.repo)
    shared.sd_model = load_minimax(ckpt, workflow=selected.workflow)
    sd_models.set_diffuser_options(shared.sd_model) # apply attention, offload, etc.
    loaded = f'repo={selected.repo} workflow={selected.workflow}'
    t1 = time.time()
    timer.process.add('load', t1 - t0)
    if shared.sd_model is not None:
        return selected.workflow
    return None


def prepare_inputs(workflow: str, p: processing.StableDiffusionProcessingVideo, init_image, last_image, reference_media):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference
    if workflow == 'fl2va':
        if init_image is not None:
            p.task_args['image'] = init_image
        if last_image is not None:
            p.task_args['last_image'] = last_image
        log.debug(f'Prepare inputs: workflow={workflow} first={init_image} last={last_image}')
    if workflow == 'ref2va':
        files = []
        references = []
        for fn in reference_media:
            try:
                if hasattr(fn, 'name'): # gradio tempfile wrapper as files end up uploaded and not embedded
                    fn = fn.name
                if not os.path.exists(fn):
                    log.warning(f'Prepare inputs: workflow={workflow} file="{fn}" not found')
                    continue
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    files.append(fn)
                    references.append(MiniMaxH3ImageReference.from_file(fn))
                elif fn.lower().endswith((".mp4", ".mov", ".avi")):
                    files.append(fn)
                    references.append(MiniMaxH3VideoReference.from_file(fn))
                elif fn.lower().endswith((".wav", ".mp3", ".flac", ".aac")):
                    files.append(fn)
                    references.append(MiniMaxH3AudioReference.from_file(fn))
            except Exception as e:
                log.error(f'Prepare inputs: workflow={workflow} file="{fn}" {e}')
        if len(references) > 0:
            p.task_args['references'] = references
        log.debug(f'Prepare inputs: workflow={workflow} files={files}')


def generate(task_id, _ui_state,
             model, workflow,
             prompt, styles,
             width, height, frames,
             steps, seed,
             init_image, last_image, reference_media,
             mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb,
             audio_enable,
            _overrides,
            *args,
            **_kwargs,
           ):
    video_utils.check_av()
    from modules.video_models import video_minimax
    progress.add_task_to_queue(task_id)

    with call_queue.get_lock():
        t0 = time.time()
        progress.start_task(task_id)
        memstats.reset_stats()
        timer.process.reset()
        workflow = load_model(model) # override workflow based on loaded model
        if not workflow:
            progress.finish_task(task_id)
            log.error('Video: model not loaded')
            return None, 'Model not loaded'
        p = processing.StableDiffusionProcessingVideo(
            sd_model=shared.sd_model,
            video_engine=engine,
            video_model=model,
            prompt=prompt,
            styles=styles,
            seed=int(seed) if seed is not None else -1,
            steps=int(steps),
            width=width,
            height=height,
            frames=frames,
            do_not_save_grid=True,
            do_not_save_samples=not mp4_frames,
            outpath_samples=paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video),
            ops=['video'],
        )
        video_minimax.apply_overrides(p, shared.sd_model, still=False, audio=audio_enable)
        log.debug(f'Video: engine="{engine}" model="{model}" workflow={workflow} cls={shared.sd_model.__class__.__name__} kwargs={p.task_args}')
        processing.fix_seed(p)
        p.ops.append('video')
        p.scripts = scripts_manager.scripts_video
        p.script_args = args

        prepare_inputs(workflow, p, init_image, last_image, reference_media)

        _processed: processing.Processed = scripts_manager.scripts_video.run(p, *args)
        processed = processing.process_images(p)

        # init vars
        pixels = None
        num_frames = 0
        video_file = None
        aac_sample_rate = 32000

        audio = getattr(processed, 'audio', None) if audio_enable else None
        if audio is not None:
            audio = audio[0].float().cpu() if audio.ndim == 3 else audio.float().cpu()
            aac_sample_rate = getattr(shared.sd_model, 'audio_sampling_rate', 32000)

        images = getattr(processed, 'images', [])
        if isinstance(images, list):
            pixels = video_save.images_to_tensor(images)
        elif isinstance(images, np.ndarray):
            pixels = video_save.numpy_to_tensor(images)
        else:
            log.error(f'Video: images={images} type={type(images)} unsupported')

        if pixels is None:
            return None, "MiniMax: No frames generated"

        save_fps = mp4_fps * processing_video.interpolation_factor(p)
        num_frames, video_file, _thumb = video_save.save_video(
            p=p,
            pixels=pixels,
            audio=audio,
            mp4_fps=save_fps,
            mp4_codec=mp4_codec,
            mp4_opt=mp4_opt,
            mp4_ext=mp4_ext,
            mp4_sf=mp4_sf,
            mp4_video=mp4_video,
            mp4_frames=mp4_frames,
            mp4_thumb=mp4_thumb,
            mp4_interpolate=mp4_interpolate,
            aac_sample_rate=aac_sample_rate,
            metadata={},
        )
        _n, _c, _t, h, w = pixels.shape

        t1 = time.time()
        progress.finish_task(task_id)
        p.close()

        resolution = f'{w}x{h}' if num_frames > 0 else None
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        total_time = max(t1 - t0, 1e-6)
        fps = f'{num_frames/total_time:.2f}'
        its = f'{(steps)/total_time:.3f}'
        log.info(f'Processed: fn="{video_file}" frames={num_frames} fps={fps} its={its} resolution={resolution} time={total_time:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')

        ui_text = f'MiniMax: Generation completed | File {video_file} | Frames {num_frames} | Resolution {resolution} | f/s {fps} | it/s {its} ' + f"<div class='performance'><p>{summary} {memory}</p></div>"
        return video_file, ui_text
