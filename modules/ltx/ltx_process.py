"""
- condition upload image
- condition upload video
- condition video get frames
- custom sampler
- new way of generate video
- modernui
- lora loader
"""
# import diffusers.LTXConditionPipeline
import os
import time
import threading
from modules import shared, sd_models, errors, timer, memstats, progress
from modules.processing_callbacks import diffusers_callback
from modules.ltx.ltx_util import get_bucket, get_frames, load_model, load_upsample, get_conditions, get_generator, get_prompts, vae_decode


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
engine, model = 'LTX Video', 'LTXVideo 0.9.7 13B'
upsample_repo_id = "a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers"
upsample_pipe = None
queue_lock = threading.Lock()


def run_ltx(task_id,
            _ui_state,
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
            condition_image_strength:float,
            condition_video_strength:float,
            condition_video_frames:int,
            condition_image,
            condition_video,
            decode_timestep:float,
            image_cond_noise_scale:float,
            _overrides,
           ):

    def abort(e, ok:bool=False):
        if ok:
            shared.log.info(e)
        else:
            shared.log.error(f'Video: cls={shared.sd_model.__class__.__name__} op=base {e}')
            errors.display(e, 'LTX')
        shared.state.end()
        progress.finish_task(task_id)
        yield None, f'LTX Error: {str(e)}'

    progress.add_task_to_queue(task_id)
    with queue_lock:
        progress.start_task(task_id)
        memstats.reset_stats()
        timer.process.reset()
        yield None, 'LTX: Loading...'
        load_model(engine, model)

        shared.state.begin('Video', task_id=task_id)
        shared.state.job_count = 1

        conditions = get_conditions(
            condition_image,
            condition_image_strength,
            condition_video,
            condition_video_strength,
            condition_video_frames,
        )

        prompt, negative, networks = get_prompts(prompt, negative, styles)
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=init prompt="{prompt}" negative="{negative}" styles={styles} networks={networks}')

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
            "image_cond_noise_scale": image_cond_noise_scale,
            "generator": get_generator(seed),
            "callback_on_step_end": diffusers_callback,
            "output_type": "latent",
        }
        if len(conditions) > 0:
            base_args["conditions"] = conditions
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=base {base_args}')
        yield None, 'LTX: Generate in progress...'
        try:
            latents = shared.sd_model(**base_args).frames[0]
        except AssertionError as e:
            yield from abort(e, ok=True)
            return
        except Exception as e:
            yield from abort(e, ok=False)
            return
        t2 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        t3 = time.time()
        timer.process.add('offload', t1 - t0)
        timer.process.add('base', t2 - t1)
        timer.process.add('offload', t3 - t2)

        if upsample_enable:
            t4 = time.time()
            shared.state.begin('Upsample')
            global upsample_pipe # python-lint: disable=global-statement
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
                yield from abort(e, ok=True)
                return
            except Exception as e:
                yield from abort(e, ok=False)
                return
            latents = upsampled_latents
            t5 = time.time()
            upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe)
            t6 = time.time()
            timer.process.add('upsample', t5 - t4)
            timer.process.add('offload', t6 - t5)
            shared.state.end()

        if refine_enable:
            t7 = time.time()
            shared.state.begin('Refine')
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
            if len(conditions) > 0:
                refine_args["conditions"] = conditions
            if latents.ndim == 4:
                latents = latents.unsqueeze(0) # add batch dimension
            shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=refine latents={latents.shape} {refine_args}')
            yield None, 'LTX: Refine in progress...'
            try:
                refined_latents = shared.sd_model(latents=latents, **refine_args).frames[0]
            except AssertionError as e:
                yield from abort(e, ok=True)
                return
            except Exception as e:
                yield from abort(e, ok=False)
                return
            latents = refined_latents
            t8 = time.time()
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            t9 = time.time()
            timer.process.add('refine', t8 - t7)
            timer.process.add('offload', t9 - t8)
            shared.state.end()

        yield None, 'LTX: VAE decode in progress...'
        try:
            frames = vae_decode(latents, decode_timestep, seed)
        except AssertionError as e:
            yield from abort(e, ok=True)
            return
        except Exception as e:
            yield from abort(e, ok=False)
            return
        t10 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        t11 = time.time()
        timer.process.add('offload', t11 - t10)
        shared.state.end()
        progress.finish_task(task_id)

        t_end = time.time()
        num_frames = len(frames)
        resolution = f'{frames[0].width}x{frames[0].height}' if num_frames > 0 else None
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        fps = f'{num_frames/(t_end-t0):.2f}'
        its = f'{(steps)/(t_end-t0):.2f}'

        shared.log.info(f'Processed: frames={num_frames} fps={fps} its={its} resolution={resolution} time={t_end-t0:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')
        yield frames, f'LTX: Generation completed | Frames {len(frames)} | Resolution {resolution} | f/s {fps} | it/s {its} '+ f"<div class='performance'><p>{summary} {memory}</p></div>"
