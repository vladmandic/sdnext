import os
import copy
import time
import numpy as np
import torch
from PIL import Image

from modules import shared, errors, timer, memstats, progress, processing, sd_models, sd_samplers, devices, extra_networks, call_queue
from modules.logger import log
from modules.ltx import ltx_capabilities
from modules.ltx.ltx_util import get_bucket, get_frames, load_model, load_upsample, get_conditions, get_generator, get_prompts, vae_decode
from modules.processing_callbacks import diffusers_callback
from modules.video_models.video_vae import set_vae_params
from modules.video_models.video_save import save_video
from modules.video_models.video_utils import check_av


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
upsample_repo_id_09 = 'a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers'
upsample_pipe = None


def _latent_pass(caps, prompt, negative, width, height, frames, steps, guidance_scale, mp4_fps, conditions, image_cond_noise_scale, seed, image=None):
    base_args = {
        'prompt': prompt,
        'negative_prompt': negative,
        'width': get_bucket(width),
        'height': get_bucket(height),
        'num_frames': get_frames(frames),
        'num_inference_steps': steps,
        'generator': get_generator(seed),
        'callback_on_step_end': diffusers_callback,
        'output_type': 'latent',
    }
    if guidance_scale is not None and guidance_scale > 0:
        base_args['guidance_scale'] = guidance_scale
    if caps.supports_frame_rate_kwarg:
        base_args['frame_rate'] = float(mp4_fps)
    if caps.supports_image_cond_noise_scale and image_cond_noise_scale is not None:
        base_args['image_cond_noise_scale'] = image_cond_noise_scale
    if caps.supports_multi_condition and conditions:
        base_args['conditions'] = conditions
    if caps.is_i2v and caps.repo_cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') and image is not None:
        base_args['image'] = image
    if caps.family == '2.x' and caps.is_distilled:
        # distilled 2.x was trained with a fixed sigma schedule; override diffusers' linspace default
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        base_args['sigmas'] = list(DISTILLED_SIGMA_VALUES)
        base_args.pop('num_inference_steps', None)
    log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=latent_pass args_keys={list(base_args.keys())}')
    result = shared.sd_model(**base_args)
    latents = result.frames[0] if hasattr(result, 'frames') else None
    audio = None
    if hasattr(result, 'audio') and result.audio is not None:
        audio = result.audio[0].float().cpu()
    return latents, audio


def run_ltx(task_id,
            _ui_state,
            model: str,
            prompt: str,
            negative: str,
            styles: list,
            width: int,
            height: int,
            frames: int,
            steps: int,
            sampler_index: int,
            guidance_scale: float,
            sampler_shift: float,
            dynamic_shift: bool,
            seed: int,
            upsample_enable: bool,
            upsample_ratio: float,
            refine_enable: bool,
            refine_strength: float,
            condition_strength: float,
            ltx_init_image,
            condition_image,
            condition_last,
            condition_files,
            condition_video,
            condition_video_frames: int,
            condition_video_skip: int,
            decode_timestep: float,
            image_cond_noise_scale: float,
            mp4_fps: int,
            mp4_interpolate: int,
            mp4_codec: str,
            mp4_ext: str,
            mp4_opt: str,
            mp4_video: bool,
            mp4_frames: bool,
            mp4_sf: bool,
            audio_enable: bool,
            _overrides,
           ):

    def abort(e, ok: bool = False, p=None):
        if ok:
            log.info(e)
        else:
            log.error(f'Video: cls={shared.sd_model.__class__.__name__} op=base {e}')
            errors.display(e, 'LTX')
        if p is not None:
            extra_networks.deactivate(p)
        shared.state.end()
        progress.finish_task(task_id)
        yield None, f'LTX Error: {str(e)}'

    if model is None or len(model) == 0 or model == 'None':
        yield from abort('Video: no model selected', ok=True)
        return
    check_av()
    progress.add_task_to_queue(task_id)

    with call_queue.get_lock():
        progress.start_task(task_id)
        memstats.reset_stats()
        timer.process.reset()
        yield None, 'LTX: Loading...'

        engine = 'LTX Video'
        load_model(engine, model)
        caps = ltx_capabilities.get_caps(model)
        if caps is None or not shared.sd_model.__class__.__name__.startswith('LTX'):
            yield from abort(f'Video: cls={shared.sd_model.__class__.__name__} selected model is not LTX', ok=True)
            return

        videojob = shared.state.begin('Video', task_id=task_id)
        shared.state.job_count = 1

        from modules.video_models import models_def, video_overrides
        selected = next((m for m in models_def.models.get(engine, []) if m.name == model), None)

        effective_init_image = ltx_init_image if ltx_init_image is not None else condition_image

        if caps.is_i2v and caps.repo_cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') and effective_init_image is None:
            yield from abort('No input image provided. Please upload or select an image.', ok=True)
            return

        condition_images = []
        if effective_init_image is not None:
            condition_images.append(effective_init_image)
        if condition_last is not None:
            condition_images.append(condition_last)
        conditions = []
        if caps.supports_multi_condition:
            conditions = get_conditions(
                width, height, condition_strength,
                condition_images, condition_files, condition_video,
                condition_video_frames, condition_video_skip,
                family=caps.family,
            )

        sampler_name = processing.get_sampler_name(sampler_index)
        sd_samplers.create_sampler(sampler_name, shared.sd_model)
        log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=init caps={caps.family} styles={styles} sampler={shared.sd_model.scheduler.__class__.__name__}')

        from modules.paths import resolve_output_path
        p = processing.StableDiffusionProcessingVideo(
            sd_model=shared.sd_model,
            video_engine=engine,
            video_model=model,
            prompt=prompt,
            negative_prompt=negative,
            styles=styles,
            seed=int(seed) if seed is not None else -1,
            sampler_name=sampler_name,
            sampler_shift=float(sampler_shift),
            steps=int(steps),
            width=get_bucket(width),
            height=get_bucket(height),
            frames=get_frames(frames),
            cfg_scale=float(guidance_scale) if guidance_scale is not None and guidance_scale > 0 else caps.default_cfg,
            denoising_strength=float(condition_strength) if condition_strength is not None else 1.0,
            init_image=effective_init_image,
            vae_type='Default',
            vae_tile_frames=16,
        )
        p.scripts = None
        p.script_args = None
        p.do_not_save_grid = True
        p.do_not_save_samples = not mp4_frames
        p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)
        p.ops.append('video')

        p.task_args['num_inference_steps'] = p.steps
        p.task_args['width'] = p.width
        p.task_args['height'] = p.height
        # force pil: 'latent' output triggers frame collapse in process_samples
        p.task_args['output_type'] = 'pil'
        if caps.supports_frame_rate_kwarg:
            p.task_args['frame_rate'] = float(mp4_fps)
        if caps.supports_image_cond_noise_scale and image_cond_noise_scale is not None:
            p.task_args['image_cond_noise_scale'] = image_cond_noise_scale
        if caps.supports_decode_timestep and decode_timestep is not None:
            p.task_args['decode_timestep'] = decode_timestep
        if caps.supports_multi_condition and conditions:
            p.task_args['conditions'] = conditions

        if caps.is_i2v and caps.repo_cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') and effective_init_image is not None:
            from modules import images
            p.task_args['image'] = images.resize_image(resize_mode=2, im=effective_init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')

        if caps.family == '2.x' and caps.is_distilled:
            # distilled 2.x was trained with a fixed sigma schedule; override diffusers' linspace default
            from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
            p.task_args['sigmas'] = list(DISTILLED_SIGMA_VALUES)
            p.task_args.pop('num_inference_steps', None)

        framewise = caps.family == '0.9'
        set_vae_params(p, framewise=framewise)

        orig_dynamic_shift = shared.opts.schedulers_dynamic_shift
        orig_sampler_shift = shared.opts.schedulers_shift
        shared.opts.data['schedulers_dynamic_shift'] = dynamic_shift
        shared.opts.data['schedulers_shift'] = sampler_shift
        if hasattr(shared.sd_model, 'scheduler') and hasattr(shared.sd_model.scheduler, 'config') and hasattr(shared.sd_model.scheduler, 'register_to_config'):
            if hasattr(shared.sd_model.scheduler.config, 'use_dynamic_shifting'):
                shared.sd_model.scheduler.config.use_dynamic_shifting = dynamic_shift
                shared.sd_model.scheduler.register_to_config(use_dynamic_shifting=dynamic_shift)
            if hasattr(shared.sd_model.scheduler.config, 'flow_shift') and sampler_shift is not None and sampler_shift >= 0:
                shared.sd_model.scheduler.config.flow_shift = sampler_shift
                shared.sd_model.scheduler.register_to_config(flow_shift=sampler_shift)
            shared.sd_model.default_scheduler = copy.deepcopy(shared.sd_model.scheduler)

        if selected is not None:
            video_overrides.set_overrides(p, selected)

        t0 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        t1 = time.time()

        samplejob = shared.state.begin('Sample')
        yield None, 'LTX: Generate in progress...'

        audio = None
        pixels = None
        frames_out = None
        needs_latent_path = upsample_enable or refine_enable

        try:
            if needs_latent_path:
                prompt_final, negative_final, networks = get_prompts(prompt, negative, styles)
                extra_networks.activate(p, networks)
                latents, audio = _latent_pass(
                    caps=caps,
                    prompt=prompt_final,
                    negative=negative_final,
                    width=width,
                    height=height,
                    frames=frames,
                    steps=steps,
                    guidance_scale=p.cfg_scale,
                    mp4_fps=mp4_fps,
                    conditions=conditions,
                    image_cond_noise_scale=image_cond_noise_scale if caps.supports_image_cond_noise_scale else None,
                    seed=int(seed) if seed is not None else -1,
                    image=p.task_args.get('image'),
                )
            else:
                processed = processing.process_images(p)
                if processed is None or processed.images is None or len(processed.images) == 0:
                    yield from abort('Video: process_images returned no frames', ok=True, p=p)
                    return
                pixels = processed.images
                if getattr(processed, 'audio', None) is not None:
                    audio = processed.audio
                latents = None
        except AssertionError as e:
            yield from abort(e, ok=True, p=p)
            return
        except Exception as e:
            yield from abort(e, ok=False, p=p)
            return

        t2 = time.time()
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        devices.torch_gc(force=True, reason='ltx:base')
        t3 = time.time()
        timer.process.add('offload', t1 - t0)
        timer.process.add('base', t2 - t1)
        timer.process.add('offload', t3 - t2)
        shared.state.end(samplejob)

        if upsample_enable and latents is not None:
            t4 = time.time()
            upsamplejob = shared.state.begin('Upsample')
            try:
                if caps.family == '0.9':
                    global upsample_pipe # pylint: disable=global-statement
                    upsample_pipe = load_upsample(upsample_pipe, upsample_repo_id_09)
                    upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe)
                    up_args = {
                        'width': get_bucket(upsample_ratio * width),
                        'height': get_bucket(upsample_ratio * height),
                        'generator': get_generator(int(seed) if seed is not None else -1),
                        'output_type': 'latent',
                    }
                    if latents.ndim == 4:
                        latents = latents.unsqueeze(0)
                    log.debug(f'Video: op=upsample family=0.9 latents={latents.shape} {up_args}')
                    yield None, 'LTX: Upsample in progress...'
                    latents = upsample_pipe(latents=latents, **up_args).frames[0]
                    upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe)
                else:
                    from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline
                    log.info(f'Video load: cls={LTX2LatentUpsamplePipeline.__name__} family=2.x')
                    up_pipe = LTX2LatentUpsamplePipeline.from_pretrained(
                        'Lightricks/LTX-2-Latent-Upsampler',
                        vae=shared.sd_model.vae,
                        cache_dir=shared.opts.hfcache_dir,
                        torch_dtype=devices.dtype,
                    )
                    up_pipe = sd_models.apply_balanced_offload(up_pipe)
                    up_args = {
                        'width': get_bucket(upsample_ratio * width),
                        'height': get_bucket(upsample_ratio * height),
                        'num_frames': get_frames(frames),
                        'latents_normalized': True,
                        'generator': get_generator(int(seed) if seed is not None else -1),
                        'output_type': 'latent',
                    }
                    if latents.ndim == 4:
                        latents = latents.unsqueeze(0)
                    log.debug(f'Video: op=upsample family=2.x latents={latents.shape} {up_args}')
                    yield None, 'LTX: Upsample in progress...'
                    latents = up_pipe(latents=latents, **up_args).frames[0]
                    up_pipe = sd_models.apply_balanced_offload(up_pipe)
            except AssertionError as e:
                yield from abort(e, ok=True, p=p)
                return
            except Exception as e:
                yield from abort(e, ok=False, p=p)
                return
            t5 = time.time()
            timer.process.add('upsample', t5 - t4)
            shared.state.end(upsamplejob)

        if refine_enable and latents is not None:
            t7 = time.time()
            refinejob = shared.state.begin('Refine')
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            devices.torch_gc(force=True, reason='ltx:refine')
            # refine is the terminal stage when enabled: let the pipeline decode internally so the final vae pass runs
            # inside the same offload/cudnn context as a normal generation, matching the Generic Video tab
            refine_args = {
                'prompt': prompt_final,
                'negative_prompt': negative_final,
                'width': get_bucket((upsample_ratio if upsample_enable else 1.0) * width),
                'height': get_bucket((upsample_ratio if upsample_enable else 1.0) * height),
                'num_frames': get_frames(frames),
                'num_inference_steps': steps,
                'generator': get_generator(int(seed) if seed is not None else -1),
                'callback_on_step_end': diffusers_callback,
                'output_type': 'pil',
            }
            if p.cfg_scale is not None and p.cfg_scale > 0:
                refine_args['guidance_scale'] = p.cfg_scale
            if caps.supports_frame_rate_kwarg:
                refine_args['frame_rate'] = float(mp4_fps)
            if caps.supports_image_cond_noise_scale and image_cond_noise_scale is not None:
                refine_args['image_cond_noise_scale'] = image_cond_noise_scale
            if caps.supports_multi_condition and conditions:
                refine_args['conditions'] = conditions
            if caps.family == '2.x':
                if caps.is_distilled:
                    # distilled variants have a canonical Stage-2 refine schedule they were trained on;
                    # see diffusers.pipelines.ltx2.utils and Lightricks/LTX-2 ti2vid_two_stages pipeline
                    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
                    refine_args['sigmas'] = list(STAGE_2_DISTILLED_SIGMA_VALUES)
                else:
                    # non-distilled: truncate the default linspace schedule to match user-controlled refine_strength
                    default_sigmas = np.linspace(1.0, 1.0 / steps, steps)
                    num_skip = max(steps - max(int(steps * refine_strength), 1), 0)
                    refine_args['sigmas'] = default_sigmas[num_skip:].tolist()
                refine_args.pop('num_inference_steps', None)
            elif caps.repo_cls_name == 'LTXConditionPipeline':
                refine_args['denoise_strength'] = refine_strength
            if latents.ndim == 4:
                latents = latents.unsqueeze(0)
            log.debug(f'Video: op=refine cls={caps.repo_cls_name} latents={latents.shape}')
            yield None, 'LTX: Refine in progress...'
            try:
                result = shared.sd_model(latents=latents, **refine_args)
                pixels = result.frames[0] if hasattr(result, 'frames') else None
                if hasattr(result, 'audio') and result.audio is not None:
                    audio = result.audio[0].float().cpu()
                latents = None
            except AssertionError as e:
                yield from abort(e, ok=True, p=p)
                return
            except Exception as e:
                yield from abort(e, ok=False, p=p)
                return
            t8 = time.time()
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            t9 = time.time()
            timer.process.add('refine', t8 - t7)
            timer.process.add('offload', t9 - t8)
            shared.state.end(refinejob)

        shared.opts.data['schedulers_dynamic_shift'] = orig_dynamic_shift
        shared.opts.data['schedulers_shift'] = orig_sampler_shift

        if needs_latent_path:
            extra_networks.deactivate(p)

        if needs_latent_path and latents is not None:
            # only reached when upsample ran without refine; refine decodes through the pipeline and sets latents=None
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'], force=True)
            devices.torch_gc(force=True, reason='ltx:vae')
            yield None, 'LTX: VAE decode in progress...'
            try:
                if torch.is_tensor(latents):
                    # 0.9.x returns raw latents with output_type='latent'; 2.x pre-denormalizes them
                    frames_out = vae_decode(latents, decode_timestep if caps.supports_decode_timestep else 0.0, int(seed) if seed is not None else -1, denormalize=caps.family == '0.9')
                else:
                    frames_out = latents
            except AssertionError as e:
                yield from abort(e, ok=True, p=p)
                return
            except Exception as e:
                yield from abort(e, ok=False, p=p)
                return
            pixels = frames_out
            t10 = time.time()
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            t11 = time.time()
            timer.process.add('offload', t11 - t10)

        if not audio_enable:
            audio = None

        try:
            aac_sample_rate = shared.sd_model.vocoder.config.output_sampling_rate
        except Exception:
            aac_sample_rate = 24000

        num_frames, video_file, _thumb = save_video(
            p=p,
            pixels=pixels,
            audio=audio,
            mp4_fps=mp4_fps,
            mp4_codec=mp4_codec,
            mp4_opt=mp4_opt,
            mp4_ext=mp4_ext,
            mp4_sf=mp4_sf,
            mp4_video=mp4_video,
            mp4_frames=mp4_frames,
            mp4_interpolate=mp4_interpolate,
            aac_sample_rate=aac_sample_rate,
            metadata={},
        )

        t_end = time.time()
        if isinstance(pixels, list) and len(pixels) > 0 and isinstance(pixels[0], Image.Image):
            w, h = pixels[0].size
        elif hasattr(pixels, 'ndim') and pixels.ndim == 5:
            _n, _c, _t, h, w = pixels.shape
        elif hasattr(pixels, 'ndim') and pixels.ndim == 4:
            _n, h, w, _c = pixels.shape
        elif hasattr(pixels, 'shape'):
            h, w = pixels.shape[-2], pixels.shape[-1]
        else:
            w, h = p.width, p.height
        resolution = f'{w}x{h}' if num_frames > 0 else None
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        total_time = max(t_end - t0, 1e-6)
        fps = f'{num_frames/total_time:.2f}'
        its = f'{(steps)/total_time:.2f}'

        shared.state.end(videojob)
        progress.finish_task(task_id)
        p.close()

        log.info(f'Processed: fn="{video_file}" frames={num_frames} fps={fps} its={its} resolution={resolution} time={t_end-t0:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')
        yield video_file, f'LTX: Generation completed | File {video_file} | Frames {num_frames} | Resolution {resolution} | f/s {fps} | it/s {its} ' + f"<div class='performance'><p>{summary} {memory}</p></div>"
