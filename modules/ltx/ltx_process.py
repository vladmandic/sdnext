import os
import time
import torch
from PIL import Image

from modules import shared, errors, timer, memstats, progress, processing, sd_models, sd_samplers, devices, extra_networks, call_queue, scripts_manager
from modules.logger import log
from modules.ltx import ltx_capabilities
from modules.ltx.ltx_diffusers_patch import apply_patch as apply_ltx_diffusers_patch
from modules.ltx.ltx_util import get_bucket, get_frames, load_model, load_upsample, load_upsample_2x, get_conditions, get_generator, get_prompts, ltx_scheduler_opts, vae_decode

apply_ltx_diffusers_patch()
from modules.processing_callbacks import diffusers_callback
from modules.video_models import video_run, video_utils
from modules.video_models.video_vae import set_vae_params
from modules.video_models.video_save import save_video, get_audio_rate


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
upsample_repo_id_09 = 'a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers'
upsample_pipe = None
upsample_pipe_2x = None

STAGE2_DEV_LORA_ADAPTER = 'ltx2_stage2_distilled'
I2V_IMAGE_CLASSES = ('LTXImageToVideoPipeline', 'LTX2ImageToVideoPipeline') # the i2v pipes that take the init image as a plain kwarg rather than as a condition


def _prompt_tensors_to_device(*tensors):
    return tuple(t.to(device=devices.device) if torch.is_tensor(t) else t for t in tensors)


def identity_ltx2_guidance() -> dict:
    # Named rather than omitted: pipeline defaults track the current upstream model, so a missing
    # term guides a schedule that already bakes it in.
    return {
        'stg_scale': 0.0,
        'modality_scale': 1.0,
        'guidance_rescale': 0.0,
        'spatio_temporal_guidance_blocks': None,
        'audio_guidance_scale': 1.0,
        'audio_stg_scale': 0.0,
        'audio_modality_scale': 1.0,
        'audio_guidance_rescale': 0.0,
    }


def _canonical_ltx2_guidance(caps) -> dict:
    # Four-way composition (cfg + stg + modality + rescale) from huggingface/diffusers#13217.
    # Distilled bakes these into its sigma schedule and runs at identity.
    if caps.family != '2.x':
        return {}
    if caps.is_distilled:
        return identity_ltx2_guidance()
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


def _canonical_stage2_kwargs() -> dict:
    # Stage 2 identity guidance from huggingface/diffusers#13217. Applied to both Dev (with
    # distilled LoRA on top) and Distilled. Distilled was trained at identity; Dev's four-way
    # composition on top of the LoRA double-dips and produces striping/flicker.
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    return {
        'sigmas': list(STAGE_2_DISTILLED_SIGMA_VALUES),
        'noise_scale': float(STAGE_2_DISTILLED_SIGMA_VALUES[0]),
        'guidance_scale': 1.0,
        **identity_ltx2_guidance(),
    }


def _latent_pass(caps, prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, width, height, frames, steps, guidance_scale, mp4_fps, conditions, image_cond_noise_scale, seed, image=None):
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = _prompt_tensors_to_device(
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    )
    base_args = {
        'prompt_embeds': prompt_embeds,
        'prompt_attention_mask': prompt_attention_mask,
        'negative_prompt_embeds': negative_prompt_embeds,
        'negative_prompt_attention_mask': negative_prompt_attention_mask,
        'width': get_bucket(width),
        'height': get_bucket(height),
        'num_frames': get_frames(frames) if frames is not None else None, # None defers to the duration head
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
    if caps.is_i2v and caps.repo_cls_name in I2V_IMAGE_CLASSES and image is not None:
        base_args['image'] = image
    if caps.family == '2.x' and caps.is_distilled:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        base_args['sigmas'] = list(DISTILLED_SIGMA_VALUES)
        base_args.pop('num_inference_steps', None)
    base_args.update(_canonical_ltx2_guidance(caps))
    if caps.family == '2.x':
        base_args['use_cross_timestep'] = caps.use_cross_timestep
    log.debug(f'Video: cls={shared.sd_model.__class__.__name__} op=latent_pass args_keys={list(base_args.keys())}')
    result = shared.sd_model(**base_args)
    latents = result.frames[0] if hasattr(result, 'frames') else None
    return latents


def reject(msg: str, code: int):
    """Refuse the run with a logged reason. The message otherwise only travels in the raised error,
    and a caller that turns it into a string leaves no trace of why the job did nothing."""
    if code >= 500:
        log.error(f'Video: op=ltx code={code} {msg}')
    else:
        log.info(f'Video: op=ltx code={code} {msg}')
    raise video_run.VideoError(msg, code)


def run(model: str, *,
        prompt: str,
        negative: str = '',
        styles: list | None = None,
        width: int = 768,
        height: int = 512,
        frames: int = 121,
        auto_duration: bool = False,
        steps: int = 0, # <=0 takes the model default
        sampler_name: str = 'Default',
        sampler_shift: float = -1.0, # <0 keeps the model default
        dynamic_shift: bool = False,
        seed: int = -1,
        guidance_scale: float = -1.0, # <=0 takes the model default
        upsample_enable: bool = False,
        upsample_ratio: float = 2.0,
        refine_enable: bool = False,
        refine_strength: float = 0.4,
        condition_strength: float = 1.0,
        init_image=None,
        condition_last=None,
        condition_files: list | None = None,
        condition_video: str | None = None,
        condition_video_frames: int = -1,
        condition_video_skip: int = 0,
        decode_timestep: float = 0.05,
        image_cond_noise_scale: float = 0.025,
        audio: bool = True,
        mp4_fps: int = 24,
        mp4_interpolate: int = 0,
        mp4_codec: str = 'libx264',
        mp4_ext: str = 'mp4',
        mp4_opt: str = 'crf=16',
        mp4_video: bool = True,
        mp4_frames: bool = False,
        mp4_sf: bool = False,
        mp4_thumb: bool = True,
        mp4_scale: float = 1.0,
        mp4_upscaler: str = '',
        override_settings=None,
        ui_state=None,
        scripts=None,
        script_args=(),
        per_script_args: dict | None = None,
        extra_p: dict | None = None,
       ) -> video_run.VideoResult:
    """Generate one LTX video and save it.

    Every failure leaves as a VideoError whose code follows HTTP semantics, with 499 reserved for
    an interrupt so a cancel is distinguishable from a crash. LTX decodes through its own VAE path,
    so it takes no vae_type: the decode is always full.
    """
    if model is None or len(model) == 0 or model == 'None':
        reject('no model selected', 400)
    if model.startswith('─'):
        reject('dropdown separator selected, pick an actual model below', 400)
    if mp4_video and video_utils.check_av() is None:
        reject('video encoding is unavailable: the av package failed to load', 500)

    engine = 'LTX Video'
    load_model(engine, model)
    caps = ltx_capabilities.get_caps(model)
    cls = shared.sd_model.__class__.__name__ if shared.sd_loaded else None
    if caps is None or cls is None or not cls.startswith('LTX'):
        reject(f'selected model is not LTX: model="{model}" cls={cls}', 400)
    takes_init_image = caps.is_i2v and caps.repo_cls_name in I2V_IMAGE_CLASSES
    if takes_init_image and init_image is None:
        reject('No input image provided. Please upload or select an image.', 400)

    steps = int(steps) if steps is not None and int(steps) > 0 else caps.default_steps
    cfg_scale = float(guidance_scale) if guidance_scale is not None and guidance_scale > 0 else caps.default_cfg
    auto_frames = bool(auto_duration) and caps.supports_auto_duration
    if auto_duration and not auto_frames:
        log.warning(f'LTX: model="{model}" auto duration unsupported, using frames={get_frames(frames)}')

    p = None
    t0 = time.time()
    try:
        # Lightricks TI2VidTwoStagesPipeline: Stage 1 at half-res, 2x upsample, Stage 2 refine at target.
        # Auto-couple when the user picks Refine but not Upsample. Both Dev and Distilled refine paths
        # expect upsampled latents; same-res refine on Distilled produces oversaturation. Condition
        # variants still need per-stage conditioning rebuild and are excluded by supports_two_stage_refine.
        auto_refine_upsample = (
            refine_enable
            and caps.supports_two_stage_refine
            and not upsample_enable
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
                log.warning(f'LTX: resolution={target_w}x{target_h} adjusted={final_w}x{final_h} two-stage refine needs resolution divisible by 64')
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
        log.debug(f'LTX: resolution planning target={target_w}x{target_h} base={base_w}x{base_h} final={final_w}x{final_h} upsample={auto_refine_upsample}')

        from modules.video_models import models_def, video_overrides
        selected = models_def.find(engine, model)

        condition_images = [init_image] if init_image is not None else []
        conditions = []
        conditions_stage2 = []
        if caps.supports_multi_condition:
            # Stage 1 conditions match base latent dims; Stage 2 rebuilds at final dims so frame
            # indices and spatial sizes survive the 2x upsample. Same source PIL/file refs feed
            # both calls; get_conditions handles the resize.
            conditions = get_conditions(
                base_w, base_h, condition_strength,
                condition_images, condition_files, condition_video,
                condition_video_frames, condition_video_skip,
                family=caps.family, num_frames=get_frames(frames), condition_last=condition_last,
            )
            if (final_w, final_h) != (base_w, base_h):
                conditions_stage2 = get_conditions(
                    final_w, final_h, condition_strength,
                    condition_images, condition_files, condition_video,
                    condition_video_frames, condition_video_skip,
                    family=caps.family, num_frames=get_frames(frames), condition_last=condition_last,
                )
            else:
                conditions_stage2 = conditions

        sd_samplers.create_sampler(sampler_name, shared.sd_model)
        log.debug(f'Video: cls={cls} op=init caps={caps.family} styles={styles} sampler={shared.sd_model.scheduler.__class__.__name__}')

        from modules.paths import resolve_output_path
        p = processing.StableDiffusionProcessingVideo(
            sd_model=shared.sd_model,
            video_engine=engine,
            video_model=model,
            prompt=prompt,
            negative_prompt=negative,
            styles=styles or [],
            seed=int(seed) if seed is not None else -1,
            sampler_name=sampler_name,
            sampler_shift=float(sampler_shift),
            steps=steps,
            width=base_w,
            height=base_h,
            frames=get_frames(frames),
            cfg_scale=cfg_scale,
            denoising_strength=float(condition_strength) if condition_strength is not None else 1.0,
            init_image=init_image,
            vae_type='Default',
            vae_tile_frames=16,
            override_settings=video_run.normalize_override_settings(override_settings),
        )
        processing.fix_seed(p)
        p.state = ui_state
        p.do_not_save_grid = True
        p.do_not_save_samples = not mp4_frames
        p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)
        p.ops.append('video')
        if per_script_args:
            p.per_script_args.update(per_script_args)
        for k, v in (extra_p or {}).items():
            setattr(p, k, v)

        p.scripts = scripts if scripts is not None else scripts_manager.scripts_video
        p.script_args = tuple(script_args)
        p.scripts.run(p, *p.script_args)

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

        if takes_init_image:
            from modules import images
            p.task_args['image'] = images.resize_image(resize_mode=2, im=init_image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')

        if caps.family == '2.x' and caps.is_distilled:
            from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
            p.task_args['sigmas'] = list(DISTILLED_SIGMA_VALUES)
            p.task_args.pop('num_inference_steps', None)
        p.task_args.update(_canonical_ltx2_guidance(caps))
        if caps.family == '2.x':
            p.task_args['use_cross_timestep'] = caps.use_cross_timestep
        if auto_frames:
            p.task_args['num_frames'] = None

        framewise = caps.family == '0.9'
        set_vae_params(p, framewise=framewise)

        # Scheduler + shared.opts mutation is wrapped in ltx_scheduler_opts so restore runs on
        # every exit path (normal return, abort, interrupt, Stage 2 scheduler swap).
        with ltx_scheduler_opts(shared.sd_model, dynamic_shift=dynamic_shift, sampler_shift=sampler_shift, shift_terminal=caps.scheduler_shift_terminal):
            if selected is not None:
                video_overrides.set_overrides(p, selected)

            t_offload = time.time()
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, silent=True)
            t1 = time.time()
            timer.process.add('offload', t1 - t_offload)

            audio_out = None
            pixels = None
            latents = None
            prompt_embeds = prompt_attention_mask = negative_prompt_embeds = negative_prompt_attention_mask = None
            needs_latent_path = upsample_enable or refine_enable

            with video_utils.phase('Sample'):
                if needs_latent_path:
                    if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                        p.scripts.before_process(p)
                    prompt_final, negative_final, networks = get_prompts(p)
                    extra_networks.activate(p, networks)
                    # Encode once and reuse across stages; encode_prompt short-circuits when
                    # embeds are passed to __call__. CPU park keeps them off GPU between stages.
                    with devices.inference_context():
                        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = shared.sd_model.encode_prompt(
                            prompt=prompt_final,
                            negative_prompt=negative_final,
                            do_classifier_free_guidance=True,
                            device=devices.device,
                        )
                    prompt_embeds = prompt_embeds.cpu()
                    prompt_attention_mask = prompt_attention_mask.cpu() if prompt_attention_mask is not None else None
                    negative_prompt_embeds = negative_prompt_embeds.cpu() if negative_prompt_embeds is not None else None
                    negative_prompt_attention_mask = negative_prompt_attention_mask.cpu() if negative_prompt_attention_mask is not None else None
                    # encode_prompt outside pipe.__call__ bypasses the post-forward offload hook;
                    # re-anchor so the text encoder doesn't stay pinned through Stage 1 forward.
                    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, force=True, silent=True)
                    devices.torch_gc(force=True, reason='ltx:encode')
                    latents = _latent_pass(
                        caps=caps,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_prompt_attention_mask=negative_prompt_attention_mask,
                        width=base_w,
                        height=base_h,
                        frames=None if auto_frames else frames,
                        steps=steps,
                        guidance_scale=p.cfg_scale,
                        mp4_fps=mp4_fps,
                        conditions=conditions,
                        image_cond_noise_scale=image_cond_noise_scale if caps.supports_image_cond_noise_scale else None,
                        seed=p.seed,
                        image=p.task_args.get('image'),
                    )
                    if auto_frames and torch.is_tensor(latents):
                        # upsample and refine take the realized length; re-predicting would drift
                        frames = (latents.shape[-3] - 1) * getattr(shared.sd_model, 'vae_temporal_compression_ratio', 8) + 1
                        p.frames = frames
                        log.debug(f'LTX: auto duration frames={frames}')
                else:
                    processed = processing.process_images(p)
                    if processed is None or processed.images is None or len(processed.images) == 0:
                        # process_images swallows the interrupt assertion, so an empty result is the
                        # only place a cancel and a genuine failure are still distinguishable
                        if shared.state.interrupted or shared.state.skipped:
                            reject('interrupted', 499)
                        reject('process_images returned no frames', 500)
                    pixels = processed.images
                    raw_audio = getattr(processed, 'audio', None)
                    if raw_audio is not None:
                        # Strip batch dim from (B, 2, N); write_audio expects (2, N) for the
                        # transpose-to-interleaved path used by AAC s16.
                        audio_out = raw_audio[0].float().cpu() if raw_audio.ndim == 3 else raw_audio.float().cpu()

                t2 = time.time()
                # silent=True everywhere: per-module stats were already dumped during the load-time
                # balanced_offload pass. Upsample/refine boundaries force a rebuild because the global
                # offload_hook_instance is keyed on checkpoint_name (sd_offload_balanced), but re-logging
                # the same inventory adds noise without information.
                shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, silent=True)
                devices.torch_gc(force=True, reason='ltx:base')
                t3 = time.time()
                timer.process.add('base', t2 - t1)
                timer.process.add('offload', t3 - t2)

            if effective_upsample_enable and latents is not None:
                with video_utils.phase('Upsample'):
                    t4 = time.time()
                    # Shared-VAE exclude: both upsample pipes receive shared.sd_model.vae as a
                    # constructor formality (pure latent -> latent forward). The main pipe already
                    # owns the VAE's hook lifecycle, so walking it again here hits meta tensors
                    # from the prior offload pass. Excluding also shortens the walk to the one
                    # module that actually belongs to this pipe: latent_upsampler.
                    upsample_exclude = ['vae']
                    if latents.ndim == 4:
                        latents = latents.unsqueeze(0)
                    if caps.family == '0.9':
                        global upsample_pipe # pylint: disable=global-statement
                        upsample_pipe = load_upsample(upsample_pipe, upsample_repo_id_09)
                        upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe, exclude=upsample_exclude, silent=True)
                        up_args = {
                            'width': final_w,
                            'height': final_h,
                            'generator': get_generator(p.seed),
                            'output_type': 'latent',
                        }
                        log.debug(f'Video: op=upsample family=0.9 latents={latents.shape} {up_args}')
                        latents = upsample_pipe(latents=latents, **up_args).frames[0]
                        upsample_pipe = sd_models.apply_balanced_offload(upsample_pipe, exclude=upsample_exclude, silent=True)
                    else:
                        global upsample_pipe_2x # pylint: disable=global-statement
                        upsample_pipe_2x = load_upsample_2x(upsample_pipe_2x, caps.upsample_repo, caps.variant)
                        upsample_pipe_2x = sd_models.apply_balanced_offload(upsample_pipe_2x, exclude=upsample_exclude, silent=True)
                        # 2.x base pass returns denormalized latents; latents_normalized=False tells the
                        # upsampler "already raw, do not denormalize again".
                        up_args = {
                            'width': final_w,
                            'height': final_h,
                            'num_frames': get_frames(frames),
                            'latents_normalized': False,
                            'generator': get_generator(p.seed),
                            'output_type': 'latent',
                        }
                        log.debug(f'Video: op=upsample family=2.x latents={latents.shape} auto={auto_refine_upsample} {up_args}')
                        latents = upsample_pipe_2x(latents=latents, **up_args).frames[0]
                        upsample_pipe_2x = sd_models.apply_balanced_offload(upsample_pipe_2x, exclude=upsample_exclude, silent=True)
                    t5 = time.time()
                    timer.process.add('upsample', t5 - t4)

            if refine_enable and latents is not None:
                with video_utils.phase('Refine'):
                    t7 = time.time()
                    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, silent=True)
                    devices.torch_gc(force=True, reason='ltx:refine')
                    # Refine is terminal: let the pipe decode internally so the final VAE pass runs inside
                    # the same offload/cudnn context as a normal generation (matches Generic Video tab).
                    refine_args = {
                        'prompt_embeds': prompt_embeds,
                        'prompt_attention_mask': prompt_attention_mask,
                        'negative_prompt_embeds': negative_prompt_embeds,
                        'negative_prompt_attention_mask': negative_prompt_attention_mask,
                        'width': final_w,
                        'height': final_h,
                        'num_frames': get_frames(frames),
                        'num_inference_steps': steps,
                        'generator': get_generator(p.seed),
                        'callback_on_step_end': diffusers_callback,
                        'output_type': 'pil',
                    }
                    if p.cfg_scale is not None and p.cfg_scale > -1:
                        refine_args['guidance_scale'] = p.cfg_scale
                    if caps.supports_frame_rate_kwarg:
                        refine_args['frame_rate'] = float(mp4_fps)
                    if caps.supports_image_cond_noise_scale and image_cond_noise_scale is not None:
                        refine_args['image_cond_noise_scale'] = image_cond_noise_scale
                    if caps.supports_multi_condition and conditions_stage2:
                        refine_args['conditions'] = conditions_stage2
                    # Thread Stage-1 I2V init image through Stage 2 so first-frame identity survives refine.
                    if takes_init_image and p.task_args.get('image') is not None:
                        refine_args['image'] = p.task_args['image']
                    if caps.family == '2.x':
                        refine_args['use_cross_timestep'] = caps.use_cross_timestep
                    # output_type='latent' skips the post-loop audio_vae + vocoder pass when audio
                    # is unwanted; per-step audio cross-attention still runs for video conditioning.
                    # Internal video decode is also skipped; vae_decode below picks it up.
                    want_audio = caps.supports_audio and audio
                    if not want_audio:
                        refine_args['output_type'] = 'latent'

                    saved_scheduler_stage2 = None
                    try:
                        if caps.family == '2.x':
                            # Stage 2 recipe (huggingface/diffusers#13217): fresh scheduler with shifting
                            # disabled, 3 steps on STAGE_2_DISTILLED_SIGMA_VALUES, identity guidance.
                            # Dev runs Distilled-on-Dev via the LoRA; Distilled is already at identity.
                            from diffusers import FlowMatchEulerDiscreteScheduler
                            saved_scheduler_stage2 = shared.sd_model.scheduler
                            shared.sd_model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                                saved_scheduler_stage2.config,
                                use_dynamic_shifting=False,
                                shift_terminal=None,
                            )
                            if caps.supports_canonical_stage2:
                                log.debug(f'LTX: stage=2 distilled=LoRA repo={caps.stage2_dev_lora_repo} weight={caps.stage2_dev_lora_weight}')
                                offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
                                # 2.5 keeps the LoRA in the model repo, so the file has to be named
                                lora_args ={'weight_name': caps.stage2_dev_lora_weight} if caps.stage2_dev_lora_weight is not None else {}
                                shared.sd_model.load_lora_weights(
                                    caps.stage2_dev_lora_repo,
                                    adapter_name=STAGE2_DEV_LORA_ADAPTER,
                                    cache_dir=shared.opts.hfcache_dir,
                                    **lora_args,
                                    **offline_args,
                                )
                                shared.sd_model.set_adapters([STAGE2_DEV_LORA_ADAPTER], [1.0])
                            else:
                                log.debug('LTX: stage=2 distilled=native')
                            # Identity kwargs override any guidance left from earlier in refine_args.
                            refine_args.update(_canonical_stage2_kwargs())
                            refine_args.pop('num_inference_steps', None)
                        elif caps.repo_cls_name == 'LTXConditionPipeline':
                            refine_args['denoise_strength'] = refine_strength
                        if latents.ndim == 4:
                            latents = latents.unsqueeze(0)
                        (
                            refine_args['prompt_embeds'],
                            refine_args['prompt_attention_mask'],
                            refine_args['negative_prompt_embeds'],
                            refine_args['negative_prompt_attention_mask'],
                        ) = _prompt_tensors_to_device(
                            refine_args['prompt_embeds'],
                            refine_args['prompt_attention_mask'],
                            refine_args['negative_prompt_embeds'],
                            refine_args['negative_prompt_attention_mask'],
                        )
                        log.debug(f'Video: op=refine cls={caps.repo_cls_name} latents={latents.shape} canonical_stage2={caps.supports_canonical_stage2}')
                        result = shared.sd_model(latents=latents, **refine_args)
                        out = result.frames[0] if hasattr(result, 'frames') else None
                        if want_audio:
                            pixels = out
                            if hasattr(result, 'audio') and result.audio is not None:
                                audio_out = result.audio[0].float().cpu()
                            latents = None
                        else:
                            latents = out
                    finally:
                        if saved_scheduler_stage2 is not None:
                            if caps.supports_canonical_stage2:
                                try:
                                    from modules.lora.extra_networks_lora import unload_diffusers
                                    unload_diffusers()
                                except Exception as e:
                                    log.warning(f'LTX: stage=2 distilled=LoRA unload failed: {e}')
                            shared.sd_model.scheduler = saved_scheduler_stage2
                    t8 = time.time()
                    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, silent=True)
                    t9 = time.time()
                    timer.process.add('refine', t8 - t7)
                    timer.process.add('offload', t9 - t8)

            if needs_latent_path and latents is not None:
                # Decode any path that leaves latents intact: upsample-without-refine, or
                # refine with output_type='latent' (audio disabled).
                with video_utils.phase('VAE Decode'):
                    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'], force=True, silent=True)
                    devices.torch_gc(force=True, reason='ltx:vae')
                    if torch.is_tensor(latents):
                        # 0.9.x returns raw latents with output_type='latent'; 2.x pre-denormalizes.
                        pixels = vae_decode(latents, decode_timestep if caps.supports_decode_timestep else 0.0, p.seed, denormalize=caps.family == '0.9')
                    else:
                        pixels = latents
                    t10 = time.time()
                    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, silent=True)
                    t11 = time.time()
                    timer.process.add('offload', t11 - t10)

            if not audio:
                audio_out = None

            if mp4_interpolate > 0 and pixels is not None:
                p.video_interpolate = mp4_interpolate
                from modules.processing_video import apply_video_interpolation
                # refine path returns PIL list (output_type='pil'); decode path returns 5-D tensor
                if isinstance(pixels, list) and len(pixels) > 0 and isinstance(pixels[0], Image.Image):
                    from modules.video_models.video_save import images_to_tensor
                    pixels = images_to_tensor(pixels)
                # pixels is 5-D (N,C,T,H,W) in [-1,1]; RIFE needs 4-D (T,C,H,W) in [0,1]
                x = pixels.squeeze(0).permute(1, 0, 2, 3)
                x = (x.clamp(-1., 1.) + 1.0) * 0.5
                x = apply_video_interpolation(p, x, count=mp4_interpolate)
                x = x * 2.0 - 1.0
                pixels = x.permute(1, 0, 2, 3).unsqueeze(0)

            # LTX is conditioned on mp4_fps as the source rate; scale saved fps to keep duration constant
            from modules.processing_video import interpolation_factor
            save_fps = mp4_fps * interpolation_factor(p)
            num_frames, video_file, thumb_file = save_video(
                p=p,
                pixels=pixels,
                audio=audio_out,
                mp4_fps=save_fps,
                mp4_codec=mp4_codec,
                mp4_opt=mp4_opt,
                mp4_ext=mp4_ext,
                mp4_sf=mp4_sf,
                mp4_video=mp4_video,
                mp4_frames=mp4_frames,
                mp4_thumb=mp4_thumb,
                mp4_interpolate=mp4_interpolate,
                aac_sample_rate=get_audio_rate(p),
                upscale_scale=mp4_scale,
                upscale_upscaler=mp4_upscaler,
                metadata={},
            )

            out_w, out_h = video_utils.pixel_size(pixels, fallback=(p.width, p.height))
            total_time = max(time.time() - t0, 1e-6)
            log.info(f'Processed: fn="{video_file}" frames={num_frames} fps={num_frames/total_time:.2f} its={p.steps/total_time:.3f} resolution={out_w}x{out_h} time={total_time:.2f}')
            # the decode paths never materialize PIL, so frames come back through the saved file
            images_out = pixels if isinstance(pixels, list) else []
            processed_out = processing.Processed(p, images_out, seed=p.seed, audio=audio_out)
            del pixels
            return video_run.VideoResult(
                images=images_out,
                video_path=video_file,
                thumb_path=thumb_file,
                num_frames=num_frames,
                fps=float(save_fps),
                has_audio=audio_out is not None,
                still=False,
                processed=processed_out,
                width=out_w,
                height=out_h,
            )
    except video_run.VideoError:
        raise
    except AssertionError as e:
        # diffusers_callback raises this to unwind the denoise loop on interrupt
        log.info(f'Video: op=ltx {e}')
        raise video_run.VideoError('interrupted', 499) from e
    except Exception as e:
        log.error(f'Video: cls={shared.sd_model.__class__.__name__} op=ltx {e}')
        errors.display(e, 'LTX')
        raise video_run.VideoError(str(e), 500) from e
    finally:
        if p is not None:
            extra_networks.deactivate(p)
            p.close()


def run_ltx(task_id,
            _ui_state,
            model: str,
            prompt: str,
            negative: str,
            styles: list,
            width: int,
            height: int,
            frames: int,
            auto_duration: bool,
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
            mp4_thumb: bool,
            mp4_scale: float,
            mp4_upscaler: str,
            audio_enable: bool,
            _overrides,
            *args,
            **_kwargs,
           ):
    # gradio adapter around run(): the signature is frozen since external callers bind to it by keyword
    progress.add_task_to_queue(task_id)
    with call_queue.get_lock():
        progress.start_task(task_id)
        memstats.reset_stats()
        timer.process.reset()
        yield None, 'LTX: Loading...'
        videojob = shared.state.begin('Video', task_id=task_id)
        shared.state.job_count = 1
        err = None
        res = None
        t0 = time.time()
        try:
            res = run(model,
                      prompt=prompt,
                      negative=negative,
                      styles=styles,
                      width=width,
                      height=height,
                      frames=frames,
                      auto_duration=auto_duration,
                      steps=steps,
                      sampler_name=processing.get_sampler_name(sampler_index),
                      sampler_shift=sampler_shift,
                      dynamic_shift=dynamic_shift,
                      seed=seed,
                      guidance_scale=guidance_scale,
                      upsample_enable=upsample_enable,
                      upsample_ratio=upsample_ratio,
                      refine_enable=refine_enable,
                      refine_strength=refine_strength,
                      condition_strength=condition_strength,
                      init_image=ltx_init_image,
                      condition_last=condition_last,
                      condition_files=condition_files,
                      condition_video=condition_video,
                      condition_video_frames=condition_video_frames,
                      condition_video_skip=condition_video_skip,
                      decode_timestep=decode_timestep,
                      image_cond_noise_scale=image_cond_noise_scale,
                      audio=audio_enable,
                      mp4_fps=mp4_fps,
                      mp4_interpolate=mp4_interpolate,
                      mp4_codec=mp4_codec,
                      mp4_ext=mp4_ext,
                      mp4_opt=mp4_opt,
                      mp4_video=mp4_video,
                      mp4_frames=mp4_frames,
                      mp4_sf=mp4_sf,
                      mp4_thumb=mp4_thumb,
                      mp4_scale=mp4_scale,
                      mp4_upscaler=mp4_upscaler,
                      override_settings=_overrides,
                      ui_state=_ui_state,
                      script_args=args,
                     )
        except video_run.VideoError as e:
            err = str(e)
        finally:
            shared.state.end(videojob)
            progress.finish_task(task_id)
        if res is None:
            yield None, f'LTX Error: {err}'
            return
        total_time = max(time.time() - t0, 1e-6)
        resolution = f'{res.width}x{res.height}' if res.num_frames > 0 else None
        fps = f'{res.num_frames/total_time:.2f}'
        its = f'{res.processed.steps/total_time:.3f}'
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        yield res.video_path, f'Video | File {res.video_path} | Frames {res.num_frames} | Resolution {resolution} | f/s {fps} | it/s {its} ' + f"<div class='performance'><p>{summary} {memory}</p></div>"
