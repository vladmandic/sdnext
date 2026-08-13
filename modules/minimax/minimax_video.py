import time
from modules.logger import log
from modules import shared, processing, timer, progress, paths, sd_models, scripts_manager, call_queue, memstats
from modules.video_models import models_def


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
    loaded = f'repo={selected.repo} workflow={selected.workflow}'
    t1 = time.time()
    timer.process.add('load', t1 - t0)
    if shared.sd_model is not None:
        return selected.workflow
    return None


def generate(task_id, _ui_state,
             model, workflow,
             prompt, styles,
             width, height, frames,
             steps, seed,
             init_image, last_image, reference_images,
             mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb,
             audio_enable,
            _overrides,
            *args,
            **_kwargs,
           ):
    from modules.video_models import video_minimax
    progress.add_task_to_queue(task_id)

    with call_queue.get_lock():
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
        processed: processing.Processed = scripts_manager.scripts_video.run(p, *args)

        if workflow == 'fl2va':
            # init images
        if workflow == 'ref2va':
            # init reference

        return None, 'Whatever'
