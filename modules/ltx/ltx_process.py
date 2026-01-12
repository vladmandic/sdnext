"""
- modernui
- teacache and others
"""
import os
import time
import threading
from modules import shared, errors, timer, memstats, progress, processing, sd_models, sd_samplers, extra_networks
from modules.video_models.video_save import save_video
from modules.video_models.video_utils import check_av
from modules.processing_callbacks import diffusers_callback
from modules.ltx.ltx_util import get_bucket, get_frames, load_model, load_upsample, get_conditions, get_generator, get_prompts, vae_decode


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
# engine, model = 'LTX Video', 'LTXVideo 0.9.7 13B'
upsample_repo_id = "a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers"
upsample_pipe = None
queue_lock = threading.Lock()


def run_ltx(task_id,
            _ui_state,
            model:str,
            prompt:str,
            negative:str,
            styles:list[str],
            width:int,
            height:int,
            frames:int,
            steps:int,
            sampler_index:int,
            seed:int,
            upsample_enable:bool,
            upsample_ratio:float,
            refine_enable:bool,
            refine_strength:float,
            condition_strength: float,
            condition_image,
            condition_last,
            condition_files,
            condition_video,
            condition_video_frames:int,
            condition_video_skip:int,
            decode_timestep:float,
            image_cond_noise_scale:float,
            mp4_fps:int,
            mp4_interpolate:int,
            mp4_codec:str,
            mp4_ext:str,
            mp4_opt:str,
            mp4_video:bool,
            mp4_frames:bool,
            mp4_sf:bool,
            _overrides,
           ):

    def abort(e, ok:bool=False, p=None):
        if ok:
            shared.log.info(e)
        else:
            shared.log.error(f'Video: cls={shared.sd_model.__class__.__name__} op=base {e}')
            errors.display(e, 'LTX')
        if p is not None:
            extra_networks.deactivate(p)
        shared.state.end()
        progress.finish_task(task_id)
        yield None, f'LTX Error: {str(e)}'

    if model is None or len(model) == 0:
        yield from abort('Video: no model selected', ok=True)
        return
    # from diffusers import LTXConditionPipeline # pylint: disable=unused-import
    check_av()
    progress.add_task_to_queue(task_id)
    with queue_lock:
        progress.start_task(task_id)
        memstats.reset_stats()
        timer.process.reset()
        yield None, 'LTX: Loading...'
        engine = 'LTX Video'
        load_model(engine, model)
        debug(f'Video: cls={shared.sd_model.__class__.__name__} op=init model="{model}"')
        if not shared.sd_model.__class__.__name__.startswith("LTX"):
            yield from abort(f'Video: cls={shared.sd_model.__class__.__name__} selected model is not LTX model', ok=True)
            return

        videojob = shared.state.begin('Video', task_id=task_id)
        shared.state.job_count = 1

        p = processing.StableDiffusionProcessingVideo(
            video_engine=engine,
            video_model=model,
            prompt=prompt,
            negative_prompt=negative,
            styles=styles,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            sampler_index=sampler_index,
            seed=seed,
        )
        p.ops.append('video')

        condition_images = []
        if condition_image is not None:
            condition_images.append(condition_image)
        if condition_last is not None:
            condition_images.append(condition_last)
        conditions = get_conditions(
            width,
            height,
            condition_strength,
            condition_images,
            condition_files,
            condition_video,
            condition_video_frames,
            condition_video_skip,
        )

        prompt, negative, networks = get_prompts(prompt, negative, styles)
        sampler_name = processing.get_sampler_name(sampler_index)
        sd_samplers.create_sampler(sampler_name, shared.sd_model)
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=init styles={styles} networks={networks} sampler={shared.sd_model.scheduler.__class__.__name__}')
        extra_networks.activate(p, networks)

        t0 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        t1 = time.time()
        base_args = {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": get_bucket(width),
            "height": get_bucket(height),
            "num_frames": get_frames(frames),
            "num_inference_steps": steps,
            "generator": get_generator(seed),
            "callback_on_step_end": diffusers_callback,
        }
        if 'LTX2' in shared.sd_model.__class__.__name__:
            base_args["output_type"] = "np"
        else:
            base_args["output_type"] = "latent"
        if 'Condition' in shared.sd_model.__class__.__name__:
            base_args["image_cond_noise_scale"] = image_cond_noise_scale
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=base {base_args}')
        if len(conditions) > 0:
            base_args["conditions"] = conditions
        yield None, 'LTX: Generate in progress...'
        samplejob = shared.state.begin('Sample')
        try:
            latents = shared.sd_model(**base_args).frames[0]
        except AssertionError as e:
            yield from abort(e, ok=True, p=p)
            return
        except Exception as e:
            yield from abort(e, ok=False, p=p)
            return
        t2 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        t3 = time.time()
        timer.process.add('offload', t1 - t0)
        timer.process.add('base', t2 - t1)
        timer.process.add('offload', t3 - t2)
        shared.state.end(samplejob)

        if upsample_enable:
            t4 = time.time()
            upsamplejob = shared.state.begin('Upsample')
            global upsample_pipe # pylint: disable=global-statement
            upsample_pipe = load_upsample(upsample_pipe, upsample_repo_id)
            upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe)
            upscale_args = {
                "width": get_bucket(upsample_ratio * width),
                "height": get_bucket(upsample_ratio * height),
                "generator": get_generator(seed),
                "output_type": "latent",
            }
            if latents.ndim == 4:
                latents = latents.unsqueeze(0) # add batch dimension
            shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=upsample latents={latents.shape} {upscale_args}')
            yield None, 'LTX: Upsample in progress...'
            try:
                upsampled_latents = upsample_pipe(latents=latents, **upscale_args).frames[0]
            except AssertionError as e:
                yield from abort(e, ok=True, p=p)
                return
            except Exception as e:
                yield from abort(e, ok=False, p=p)
                return
            latents = upsampled_latents
            t5 = time.time()
            upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe)
            t6 = time.time()
            timer.process.add('upsample', t5 - t4)
            timer.process.add('offload', t6 - t5)
            shared.state.end(upsamplejob)

        if refine_enable:
            t7 = time.time()
            refinejob = shared.state.begin('Refine')
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            refine_args = {
                "prompt": prompt,
                "negative_prompt": negative,
                "width": get_bucket(upsample_ratio * width),
                "height": get_bucket(upsample_ratio * height),
                "num_frames": get_frames(frames),
                "denoise_strength": refine_strength,
                "num_inference_steps": steps,
                "image_cond_noise_scale": image_cond_noise_scale,
                "generator": get_generator(seed),
                "callback_on_step_end": diffusers_callback,
                "output_type": "latent",
            }
            if latents.ndim == 4:
                latents = latents.unsqueeze(0) # add batch dimension
            shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=refine latents={latents.shape} {refine_args}')
            if len(conditions) > 0:
                refine_args["conditions"] = conditions
            yield None, 'LTX: Refine in progress...'
            try:
                refined_latents = shared.sd_model(latents=latents, **refine_args).frames[0]
            except AssertionError as e:
                yield from abort(e, ok=True, p=p)
                return
            except Exception as e:
                yield from abort(e, ok=False, p=p)
                return
            latents = refined_latents
            t8 = time.time()
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            t9 = time.time()
            timer.process.add('refine', t8 - t7)
            timer.process.add('offload', t9 - t8)
            shared.state.end(refinejob)

        extra_networks.deactivate(p)

        yield None, 'LTX: VAE decode in progress...'
        try:
            frames = vae_decode(latents, decode_timestep, seed)
        except AssertionError as e:
            yield from abort(e, ok=True, p=p)
            return
        except Exception as e:
            yield from abort(e, ok=False, p=p)
            return
        t10 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        t11 = time.time()
        timer.process.add('offload', t11 - t10)

        num_frames, video_file = save_video(
            p=p,
            pixels=frames,
            mp4_fps=mp4_fps,
            mp4_codec=mp4_codec,
            mp4_opt=mp4_opt,
            mp4_ext=mp4_ext,
            mp4_sf=mp4_sf,
            mp4_video=mp4_video,
            mp4_frames=mp4_frames,
            mp4_interpolate=mp4_interpolate,
            metadata={},
        )

        t_end = time.time()
        _n, _c, _t, h, w = frames.shape
        resolution = f'{w}x{h}' if num_frames > 0 else None
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        fps = f'{num_frames/(t_end-t0):.2f}'
        its = f'{(steps)/(t_end-t0):.2f}'

        shared.state.end(videojob)
        progress.finish_task(task_id)

        shared.log.info(f'Processed: fn="{video_file}" frames={num_frames} fps={fps} its={its} resolution={resolution} time={t_end-t0:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')
        yield video_file, f'LTX: Generation completed | File {video_file} | Frames {len(frames)} | Resolution {resolution} | f/s {fps} | it/s {its} '+ f"<div class='performance'><p>{summary} {memory}</p></div>"
