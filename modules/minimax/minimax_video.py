import os
import time
from PIL import Image
import numpy as np
from modules.logger import log
from modules import shared, devices, errors, processing, timer, progress, paths, sd_models, scripts_manager, call_queue, memstats, processing_video
from modules.video_models import models_def, video_save, video_utils, video_upscale


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
    if shared.sd_model is None:
        log.error(f'Load video: engine="{engine}" selected="{model}" failed')
        return None
    sd_models.set_diffuser_options(shared.sd_model) # apply attention, offload, etc.
    loaded = f'repo={selected.repo} workflow={selected.workflow}'
    t1 = time.time()
    timer.video.add('load', t1 - t0)
    if shared.sd_model is not None:
        return selected.workflow
    return None


def unwrap_file(entry):
    """The path behind a gradio file entry: an upload arrives as a tempfile wrapper or a dict, not a path."""
    if hasattr(entry, 'name'):
        return entry.name
    if isinstance(entry, dict) and 'name' in entry:
        return entry['name']
    return entry


def prepare_inputs(workflow: str | None, init_image: Image.Image | None, last_image: Image.Image | None, reference_media: list | None) -> dict:
    """The task args a workflow conditions on, resolved before the model load so a rejected request costs nothing."""
    t_inputs = time.time()
    from modules.minimax import minimax_references
    if minimax_references.get_reference_caps(workflow) is not None:
        entries = [unwrap_file(entry) for entry in (reference_media or [])]
        references = minimax_references.resolve(workflow, entries, init_image)
        log.debug(f'Prepare inputs: workflow={workflow} references={len(references)}')
        return {'references': references}
    task_args = {}
    if init_image is not None:
        task_args['image'] = init_image
    if last_image is not None:
        task_args['last_image'] = last_image
    if reference_media:
        log.warning(f'Video: op=reference workflow={workflow} references not supported, ignoring: count={len(reference_media)}')
    log.debug(f'Prepare inputs: workflow={workflow} first={init_image} last={last_image}')
    timer.video.ts('inputs', t_inputs)
    return task_args


def generate(task_id, _ui_state,
             model,
             workflow,
             prompt, styles,
             width, height,
             frames,
             steps,
             seed,
             init_image, last_image, reference_media,
             video_shift, audio_shift,
             mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt,
             mp4_video, mp4_frames, mp4_sf, mp4_thumb,
             mp4_scale, mp4_upscaler,
             audio_enable,
            _overrides,
            *args,
            **_kwargs,
           ):
    video_utils.check_av()
    from modules.video_models import video_minimax, video_run
    progress.add_task_to_queue(task_id)

    with call_queue.get_lock():
        t0 = time.time()
        progress.start_task(task_id)
        memstats.reset_stats()
        timer.process.reset()
        timer.video.reset()

        # init vars
        p = None
        workflow = None # the incoming argument is the ui's display label, so the row and then the load supply the real one
        pixels = None
        num_frames = 0
        video_file = None
        aac_sample_rate = 32000

        try:
            # resolved off the registry row so a bad reference is rejected before the load, the same as on the api path
            selected = models_def.find(engine, model)
            workflow = getattr(selected, 'workflow', None)
            task_args = prepare_inputs(workflow, init_image, last_image, reference_media)
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
            video_minimax.set_sampler_shift(shared.sd_model, video_shift=video_shift, audio_shift=audio_shift)
            log.debug(f'Video: engine="{engine}" model="{model}" workflow={workflow} cls={shared.sd_model.__class__.__name__} shift={video_shift}:{audio_shift} kwargs={p.task_args}')
            processing.fix_seed(p)
            p.ops.append('video')
            p.scripts = scripts_manager.scripts_video
            p.script_args = args

            p.task_args.update(task_args)

            _processed: processing.Processed = scripts_manager.scripts_video.run(p, *args)

            if os.environ.get("SD_MINIMAX_CHUNK", None) is not None:
                from modules.minimax.minimax_chunking import minimax_attention
                chunk_size = int(os.environ.get("SD_MINIMAX_CHUNK", 0))
                log.debug(f'Video: engine="{engine}" model="{model}" chunking=True size={chunk_size}')
                with minimax_attention(chunk_size=chunk_size):
                    processed = processing.process_images(p)
            else:
                processed = processing.process_images(p)

            sd_models.offload_ondemand(shared.sd_model, reason='finish', force=True) # force offload all loaded modules to cpu
            devices.torch_gc(force=True) # free gpu memory before saving video

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

            if mp4_interpolate > 0:
                t_interpolate = time.time()
                p.video_interpolate = mp4_interpolate
                from modules.processing_video import apply_video_interpolation
                # pixels is 5-D (N,C,T,H,W) in [-1,1]; RIFE needs 4-D (T,C,H,W) in [0,1]
                x = pixels.squeeze(0).permute(1, 0, 2, 3)
                x = (x.clamp(-1., 1.) + 1.0) * 0.5
                x = apply_video_interpolation(p, x, count=mp4_interpolate)
                x = x * 2.0 - 1.0
                pixels = x.permute(1, 0, 2, 3).unsqueeze(0)
                timer.video.ts('interpolate', t_interpolate)
                p.video_interpolated = True # notice so main save_video does not do it again

            if mp4_upscaler is not None and len(mp4_upscaler) > 0:
                t_upscale = time.time()
                pixels = video_upscale.upscale_video(pixels, scale=mp4_scale, upscaler_name=mp4_upscaler)
                timer.video.ts('upscale', t_upscale)
                p.video_upscaled = True # notice so main save_video does not do it again

            save_fps = mp4_fps * processing_video.interpolation_factor(p)
            t_save = time.time()
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
                upscale_scale=mp4_scale,
                upscale_upscaler=mp4_upscaler,
                metadata={},
            )
            timer.video.ts('save', t_save)
            _n, _c, _t, h, w = pixels.shape
            del pixels
            if audio is not None:
                del audio

        except video_run.VideoError as e: # a rejected input, so the reason belongs in the output box and not only in the log
            log.error(f'Video: engine="{engine}" model="{model}" workflow={workflow} {e}')
            return None, f'Error: {e}'
        except Exception as e:
            log.error(f'Video: engine="{engine}" model="{model}" workflow={workflow} {e}')
            errors.display(e, 'Video')
        finally:
            jobid = getattr(shared.sd_model, 'sdnext_phaseid', None) if shared.sd_loaded else None # sd_model loads on access, and a request rejected before the load must not trigger one
            shared.state.end(jobid) # clear the previous job if exists
            progress.finish_task(task_id)
            if p is not None: # a request rejected before the processing object exists has nothing to close
                p.close()

        t1 = time.time()
        resolution = f'{w}x{h}' if num_frames > 0 else None
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        total_time = max(t1 - t0, 1e-6)
        timer.video.merge(timer.process)
        timer.video.set('wall', total_time)
        log.debug(f'Video: timers={timer.video.dct(no_total=True)}')
        fps = f'{num_frames/total_time:.2f}'
        its = f'{(steps)/total_time:.3f}'
        log.info(f'Processed: fn="{video_file}" frames={num_frames} fps={fps} its={its} resolution={resolution} time={total_time:.2f}')

        ui_text = f'Video | File {video_file} | Frames {num_frames} | Resolution {resolution} | f/s {fps} | it/s {its} ' + f"<div class='performance'><p>{summary} {memory}</p></div>"
        return video_file, ui_text
