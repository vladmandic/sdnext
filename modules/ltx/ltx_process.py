import os
import time
import torch
from PIL import Image

from modules import shared, errors, timer, memstats, progress, processing, sd_models, sd_samplers, devices, extra_networks, call_queue
from modules.logger import log
from modules.ltx import ltx_capabilities
from modules.ltx.ltx_util import get_bucket, get_frames, load_model, load_upsample, load_upsample_2x, get_conditions, get_generator, get_prompts, temp_scheduler_opts, vae_decode
from modules.processing_callbacks import diffusers_callback
from modules.video_models.video_vae import set_vae_params
from modules.video_models.video_save import save_video
from modules.video_models.video_utils import check_av


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
upsample_repo_id_09 = 'a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers'
# Upsampler weights are tied to the family VAE; using the wrong one preserves structure
# but drifts per-channel latent statistics (decodes desaturated / crushed contrast).
upsample_repo_id_20 = 'Lightricks/LTX-2'
upsample_repo_id_23 = 'CalamitousFelicitousness/LTX-2.3-Spatial-Upsampler-x2-1.1-Diffusers'
upsample_pipe = None
upsample_pipe_2x = None

STAGE2_DEV_LORA_ADAPTER = 'ltx2_stage2_distilled'


def _canonical_ltx2_guidance(caps) -> dict:
    # Four-way composition (cfg + stg + modality + rescale) from huggingface/diffusers#13217.
    # Distilled bakes these into its sigma schedule; skip or we double-apply.
    if caps.family != '2.x' or caps.is_distilled:
        return {}
    return {
        'stg_scale': caps.stg_default_scale,
        'modality_scale': caps.modality_default_scale,
        'guidance_rescale': caps.guidance_rescale_default,
        'spatio_temporal_guidance_blocks': list(caps.stg_default_blocks),
        'audio_guidance_scale': 7.0,
        'audio_stg_scale': 1.0,
        'audio_modality_scale': 3.0,
        'audio_guidance_rescale': 0.7,
    }


def _canonical_stage2_dev_kwargs() -> dict:
    # Stage 2 identity guidance from huggingface/diffusers#13217. The distilled LoRA makes Dev
    # behave like Distilled, which was trained at identity; Stage 1's four-way composition on
    # top double-dips and produces striping/flicker.
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    return {
        'sigmas': list(STAGE_2_DISTILLED_SIGMA_VALUES),
        'noise_scale': float(STAGE_2_DISTILLED_SIGMA_VALUES[0]),
        'guidance_scale': 1.0,
        'stg_scale': 0.0,
        'modality_scale': 1.0,
        'guidance_rescale': 0.0,
        'audio_guidance_scale': 1.0,
        'audio_stg_scale': 0.0,
        'audio_modality_scale': 1.0,
        'audio_guidance_rescale': 0.0,
        'spatio_temporal_guidance_blocks': None,
    }


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
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        base_args['sigmas'] = list(DISTILLED_SIGMA_VALUES)
        base_args.pop('num_inference_steps', None)
    base_args.update(_canonical_ltx2_guidance(caps))
    if caps.use_cross_timestep:
        base_args['use_cross_timestep'] = True
    log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=latent_pass args_keys={list(base_args.keys())}')
    result = shared.sd_model(**base_args)
    # video latents strip the batch dim; audio latents keep it so LTX2Pipeline.prepare_audio_latents
    # can rewrap them when re-entered as ndim==4 at Stage 2.
    latents = result.frames[0] if hasattr(result, 'frames') else None
    audio_latents = None
    if hasattr(result, 'audio') and result.audio is not None:
        audio_latents = result.audio
    return latents, audio_latents


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

        # Lightricks TI2VidTwoStagesPipeline: Stage 1 at half-res, 2x upsample, Stage 2 refine at target.
        # Auto-couple when the user picks Refine but not Upsample. Condition variants still need per-stage
        # conditioning rebuild, so keep them on the same-resolution path.
        auto_refine_upsample = (
            refine_enable
            and caps.supports_canonical_stage2
            and not upsample_enable
            and not caps.supports_multi_condition
        )
        effective_upsample_enable = upsample_enable or auto_refine_upsample
        effective_upsample_ratio = upsample_ratio if upsample_enable else 2.0
        target_w = get_bucket(width)
        target_h = get_bucket(height)
        if auto_refine_upsample:
            # Stage 1 at target/2 needs multiple-of-32; 2x upsample then forces final divisible by 64.
            # Derive final from base, otherwise Stage 2 silently falls to base*2 != target.
            base_w = get_bucket(target_w // 2)
            base_h = get_bucket(target_h // 2)
            final_w = base_w * 2
            final_h = base_h * 2
            if (final_w, final_h) != (target_w, target_h):
                log.warning(f'LTX: two-stage refine needs resolution divisible by 64; adjusting {target_w}x{target_h} -> {final_w}x{final_h}')
        elif effective_upsample_enable:
            base_w = target_w
            base_h = target_h
            final_w = get_bucket(effective_upsample_ratio * target_w)
            final_h = get_bucket(effective_upsample_ratio * target_h)
        else:
            base_w = target_w
            base_h = target_h
            final_w = target_w
            final_h = target_h
        log.debug(f'LTX: resolution planning target={target_w}x{target_h} base={base_w}x{base_h} final={final_w}x{final_h} auto_refine_upsample={auto_refine_upsample}')

        videojob = shared.state.begin('Video', task_id=task_id)
        shared.state.job_count = 1

        from modules.video_models import models_def, video_overrides
        selected = next((m for m in models_def.models.get(engine, []) if m.name == model), None)

        if caps.is_i2v and caps.repo_cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') and ltx_init_image is None:
            yield from abort('No input image provided. Please upload or select an image.', ok=True)
            return

        condition_images = []
        if ltx_init_image is not None:
            condition_images.append(ltx_init_image)
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
            width=base_w,
            height=base_h,
            frames=get_frames(frames),
            cfg_scale=float(guidance_scale) if guidance_scale is not None and guidance_scale > 0 else caps.default_cfg,
            denoising_strength=float(condition_strength) if condition_strength is not None else 1.0,
            init_image=ltx_init_image,
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

        if caps.is_i2v and caps.repo_cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') and ltx_init_image is not None:
            from modules import images
            p.task_args['image'] = images.resize_image(resize_mode=2, im=ltx_init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')

        if caps.family == '2.x' and caps.is_distilled:
            from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
            p.task_args['sigmas'] = list(DISTILLED_SIGMA_VALUES)
            p.task_args.pop('num_inference_steps', None)
        p.task_args.update(_canonical_ltx2_guidance(caps))

        framewise = caps.family == '0.9'
        set_vae_params(p, framewise=framewise)

        # Scheduler + shared.opts mutation is wrapped in temp_scheduler_opts so restore runs on
        # every exit path (normal return, abort, interrupt, Stage 2 scheduler swap). See the
        # helper's docstring for the five pieces of state it snapshots.
        with temp_scheduler_opts(shared.sd_model, dynamic_shift=dynamic_shift, sampler_shift=sampler_shift):
            if selected is not None:
                video_overrides.set_overrides(p, selected)

            t0 = time.time()
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
            t1 = time.time()

            samplejob = shared.state.begin('Sample')
            yield None, 'LTX: Generate in progress...'

            audio = None
            stage1_audio_latents = None
            pixels = None
            frames_out = None
            needs_latent_path = upsample_enable or refine_enable

            try:
                if needs_latent_path:
                    prompt_final, negative_final, networks = get_prompts(prompt, negative, styles)
                    extra_networks.activate(p, networks)
                    latents, stage1_audio_latents = _latent_pass(
                        caps=caps,
                        prompt=prompt_final,
                        negative=negative_final,
                        width=base_w,
                        height=base_h,
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

            if effective_upsample_enable and latents is not None:
                t4 = time.time()
                upsamplejob = shared.state.begin('Upsample')
                try:
                    if caps.family == '0.9':
                        global upsample_pipe # pylint: disable=global-statement
                        upsample_pipe = load_upsample(upsample_pipe, upsample_repo_id_09)
                        upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe)
                        up_args = {
                            'width': final_w,
                            'height': final_h,
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
                        global upsample_pipe_2x # pylint: disable=global-statement
                        upsample_repo = upsample_repo_id_23 if caps.variant == '2.3' else upsample_repo_id_20
                        upsample_pipe_2x = load_upsample_2x(upsample_pipe_2x, upsample_repo)
                        upsample_pipe_2x = sd_models.apply_balanced_offload(upsample_pipe_2x)
                        # 2.x base pass returns denormalized latents; latents_normalized=False tells the
                        # upsampler "already raw, do not denormalize again".
                        up_args = {
                            'width': final_w,
                            'height': final_h,
                            'num_frames': get_frames(frames),
                            'latents_normalized': False,
                            'generator': get_generator(int(seed) if seed is not None else -1),
                            'output_type': 'latent',
                        }
                        if latents.ndim == 4:
                            latents = latents.unsqueeze(0)
                        log.debug(f'Video: op=upsample family=2.x latents={latents.shape} auto={auto_refine_upsample} {up_args}')
                        yield None, 'LTX: Upsample in progress...'
                        latents = upsample_pipe_2x(latents=latents, **up_args).frames[0]
                        upsample_pipe_2x = sd_models.apply_balanced_offload(upsample_pipe_2x)
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
                # Refine is terminal: let the pipe decode internally so the final VAE pass runs inside
                # the same offload/cudnn context as a normal generation (matches Generic Video tab).
                refine_args = {
                    'prompt': prompt_final,
                    'negative_prompt': negative_final,
                    'width': final_w,
                    'height': final_h,
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
                # Thread Stage-1 I2V init image through Stage 2 so first-frame identity survives refine.
                if caps.is_i2v and caps.repo_cls_name in ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') and p.task_args.get('image') is not None:
                    refine_args['image'] = p.task_args['image']
                # Thread Stage-1 audio latents into Stage 2 on 2.x. The video branch cross-attends
                # audio every layer; letting prepare_audio_latents fall back to fresh noise biases
                # the video branch off-distribution (desaturated output on distilled 2.x).
                if caps.family == '2.x':
                    if stage1_audio_latents is not None:
                        refine_args['audio_latents'] = stage1_audio_latents.to(device=devices.device)
                    if caps.use_cross_timestep:
                        refine_args['use_cross_timestep'] = True

                saved_scheduler_stage2 = None
                try:
                    if caps.supports_canonical_stage2:
                        # Dev 2.x Stage 2: swap scheduler, fuse distilled LoRA, 3 steps on the distilled
                        # sigma schedule at identity guidance (huggingface/diffusers#13217).
                        log.info(f'LTX: canonical Stage 2 via distilled LoRA repo={caps.stage2_dev_lora_repo}')
                        from diffusers import FlowMatchEulerDiscreteScheduler
                        offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
                        saved_scheduler_stage2 = shared.sd_model.scheduler
                        shared.sd_model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                            saved_scheduler_stage2.config,
                            use_dynamic_shifting=False,
                            shift_terminal=None,
                        )
                        shared.sd_model.load_lora_weights(
                            caps.stage2_dev_lora_repo,
                            adapter_name=STAGE2_DEV_LORA_ADAPTER,
                            cache_dir=shared.opts.hfcache_dir,
                            **offline_args,
                        )
                        shared.sd_model.set_adapters([STAGE2_DEV_LORA_ADAPTER], [1.0])
                        # Do NOT apply _canonical_ltx2_guidance on this path; its audio-branch kwargs
                        # would clobber the identity set.
                        refine_args.update(_canonical_stage2_dev_kwargs())
                        refine_args.pop('num_inference_steps', None)
                    elif caps.family == '2.x':
                        # Distilled 2.x. Dev 2.x with a LoRA hit the branch above.
                        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
                        refine_args['sigmas'] = list(STAGE_2_DISTILLED_SIGMA_VALUES)
                        refine_args.pop('num_inference_steps', None)
                        # LTX2Pipeline/LTX2ImageToVideoPipeline default noise_scale=0.0 when not passed;
                        # sigma=0 user latents mismatched against sigmas[0] scheduler collapses output.
                        # LTX2ConditionPipeline auto-infers this; do the same explicitly for T2V/I2V.
                        refine_args['noise_scale'] = float(refine_args['sigmas'][0])
                        refine_args.update(_canonical_ltx2_guidance(caps))
                    elif caps.repo_cls_name == 'LTXConditionPipeline':
                        refine_args['denoise_strength'] = refine_strength
                    if latents.ndim == 4:
                        latents = latents.unsqueeze(0)
                    log.debug(f'Video: op=refine cls={caps.repo_cls_name} latents={latents.shape} canonical_stage2={caps.supports_canonical_stage2}')
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
                finally:
                    if saved_scheduler_stage2 is not None:
                        try:
                            from modules.lora.extra_networks_lora import unload_diffusers
                            unload_diffusers()
                        except Exception as e:
                            log.warning(f'LTX: canonical Stage 2 LoRA unload failed: {e}')
                        shared.sd_model.scheduler = saved_scheduler_stage2
                        log.debug('LTX: canonical Stage 2 cleanup done (LoRA unloaded, scheduler restored)')
                t8 = time.time()
                shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
                t9 = time.time()
                timer.process.add('refine', t8 - t7)
                timer.process.add('offload', t9 - t8)
                shared.state.end(refinejob)

            if needs_latent_path:
                extra_networks.deactivate(p)

            if needs_latent_path and latents is not None:
                # Only reached on upsample-without-refine; refine decodes through the pipe and nulls latents.
                shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'], force=True)
                devices.torch_gc(force=True, reason='ltx:vae')
                yield None, 'LTX: VAE decode in progress...'
                try:
                    if torch.is_tensor(latents):
                        # 0.9.x returns raw latents with output_type='latent'; 2.x pre-denormalizes.
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
