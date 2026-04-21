import time
from contextlib import contextmanager
import torch
from PIL import Image
from modules import devices, shared, sd_models, timer, extra_networks
from modules.logger import log


def get_bucket(size: int):
    # LTX pipes validate width/height divisible by 32 across all families.
    ratio = getattr(shared.sd_model, 'vae_spatial_compression_ratio', None)
    if not isinstance(ratio, int) or ratio < 32:
        ratio = 32
    size = int(size)
    return size - (size % ratio)


def get_frames(frames: int):
    return int(8 * (int(frames) // 8)) + 1


def load_model(engine: str, model: str):
    if model is None or model == '' or model == 'None':
        shared.sd_model = None
        return
    t0 = time.time()
    from modules.video_models import models_def, video_load
    selected: models_def.Model = [m for m in models_def.models[engine] if m.name == model][0]
    # video_load owns the cache; pipe-class mismatch inside it invalidates the name-based hit
    # when Unload Models (or any external swap) silently replaced shared.sd_model.
    log.info(f'Video load: engine="{engine}" selected="{model}" {selected}')
    video_load.load_model(selected)
    t1 = time.time()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    t2 = time.time()
    timer.process.add('load', t1 - t0)
    timer.process.add('offload', t2 - t1)


def load_upsample(upsample_pipe, upsample_repo_id):
    if upsample_pipe is None:
        t0 = time.time()
        from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline
        log.info(f'Video load: cls={LTXLatentUpsamplePipeline.__name__} repo="{upsample_repo_id}"')
        upsample_pipe = LTXLatentUpsamplePipeline.from_pretrained(
            upsample_repo_id,
            vae=shared.sd_model.vae,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
        t1 = time.time()
        timer.process.add('load', t1 - t0)
    return upsample_pipe


def load_upsample_2x(upsample_pipe, upsample_repo_id):
    # 2.x ships the upsampler as a bare nn.Module in a subfolder; no from_pretrained on the
    # pipeline wrapper, so we load the model + construct the pipeline manually.
    if upsample_pipe is None:
        t0 = time.time()
        from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from modules import sd_checkpoint
        log.info(f'Video load: cls={LTX2LatentUpsamplePipeline.__name__} repo="{upsample_repo_id}"')
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            upsample_repo_id,
            subfolder='latent_upsampler',
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        ).to(devices.device)
        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=shared.sd_model.vae,
            latent_upsampler=latent_upsampler,
        )
        # Synthetic checkpoint_info gives this pipe its own OffloadHook cache slot, so routing
        # it through apply_balanced_offload does not invalidate the main pipe's module map
        # (sd_offload.py:488 keys on sd_checkpoint_info.name).
        upsample_pipe.sd_checkpoint_info = sd_checkpoint.CheckpointInfo('ltx-upsampler-2.x')
        t1 = time.time()
        timer.process.add('load', t1 - t0)
    return upsample_pipe


@contextmanager
def ltx_scheduler_opts(sd_model, *, dynamic_shift=None, sampler_shift=None):
    # Run-scoped override of shared.opts scheduler settings and scheduler.config. Snapshots
    # five pieces of state (shared.opts dynamic_shift + shift, scheduler object, default_scheduler
    # snapshot, and scheduler.config use_dynamic_shifting + flow_shift) and restores every one on
    # exit. Keeps run-specific sampler settings out of config.json and prevents default_scheduler
    # from getting clobbered by a deepcopy of the mutated scheduler at video_load.py:171. The
    # scheduler-object restore matters for Stage 2 refine, which swaps the scheduler entirely.
    orig_dynamic_shift = shared.opts.schedulers_dynamic_shift
    orig_sampler_shift = shared.opts.schedulers_shift
    orig_scheduler = sd_model.scheduler
    orig_default_scheduler = getattr(sd_model, 'default_scheduler', None)
    orig_use_dynamic_shifting = getattr(orig_scheduler.config, 'use_dynamic_shifting', None) if hasattr(orig_scheduler, 'config') else None
    orig_flow_shift = getattr(orig_scheduler.config, 'flow_shift', None) if hasattr(orig_scheduler, 'config') else None

    try:
        if dynamic_shift is not None:
            shared.opts.data['schedulers_dynamic_shift'] = dynamic_shift
        if sampler_shift is not None:
            shared.opts.data['schedulers_shift'] = sampler_shift
        if hasattr(sd_model, 'scheduler') and hasattr(sd_model.scheduler, 'config') and hasattr(sd_model.scheduler, 'register_to_config'):
            if dynamic_shift is not None and hasattr(sd_model.scheduler.config, 'use_dynamic_shifting'):
                sd_model.scheduler.config.use_dynamic_shifting = dynamic_shift
                sd_model.scheduler.register_to_config(use_dynamic_shifting=dynamic_shift)
            if sampler_shift is not None and sampler_shift >= 0 and hasattr(sd_model.scheduler.config, 'flow_shift'):
                sd_model.scheduler.config.flow_shift = sampler_shift
                sd_model.scheduler.register_to_config(flow_shift=sampler_shift)
        yield
    finally:
        shared.opts.data['schedulers_dynamic_shift'] = orig_dynamic_shift
        shared.opts.data['schedulers_shift'] = orig_sampler_shift
        if sd_model.scheduler is not orig_scheduler:
            sd_model.scheduler = orig_scheduler
        if orig_default_scheduler is not None and sd_model.default_scheduler is not orig_default_scheduler:
            sd_model.default_scheduler = orig_default_scheduler
        if hasattr(sd_model.scheduler, 'config') and hasattr(sd_model.scheduler, 'register_to_config'):
            if orig_use_dynamic_shifting is not None and hasattr(sd_model.scheduler.config, 'use_dynamic_shifting'):
                sd_model.scheduler.config.use_dynamic_shifting = orig_use_dynamic_shifting
                sd_model.scheduler.register_to_config(use_dynamic_shifting=orig_use_dynamic_shifting)
            if orig_flow_shift is not None and hasattr(sd_model.scheduler.config, 'flow_shift'):
                sd_model.scheduler.config.flow_shift = orig_flow_shift
                sd_model.scheduler.register_to_config(flow_shift=orig_flow_shift)
        log.debug(f'LTX: scheduler/opts restored dynamic_shift={orig_dynamic_shift} sampler_shift={orig_sampler_shift}')


def _condition_cls(family: str):
    if family == '2.x':
        try:
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
            return LTX2VideoCondition
        except ImportError:
            log.warning('LTX conditions: LTX2VideoCondition not available in installed diffusers')
            return None
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    return LTXVideoCondition


def make_condition(condition_cls, family: str, frames, strength: float, is_video: bool):
    if family == '2.x':
        return condition_cls(frames=frames, index=0, strength=strength)
    if is_video:
        return condition_cls(video=frames, frame_index=0, strength=strength)
    return condition_cls(image=frames, frame_index=0, strength=strength)


def get_conditions(width, height, condition_strength, condition_images, condition_files, condition_video, condition_video_frames, condition_video_skip, family: str = '0.9'):
    condition_cls = _condition_cls(family)
    if condition_cls is None:
        return []
    conditions = []
    if condition_images is not None:
        for condition_image in condition_images:
            try:
                if isinstance(condition_image, str):
                    from modules.api.api import decode_base64_to_image
                    condition_image = decode_base64_to_image(condition_image)
                condition_image = condition_image.convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS)
                conditions.append(make_condition(condition_cls, family, condition_image, condition_strength, is_video=False))
                log.debug(f'Video condition: family={family} image={condition_image.size} strength={condition_strength}')
            except Exception as e:
                log.error(f'LTX condition image: {e}')
    if condition_files is not None:
        batch_images = []
        for fn in condition_files:
            try:
                if hasattr(fn, 'name'):
                    condition_image = Image.open(fn.name).convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS)
                else:
                    condition_image = fn.convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS)
                batch_images.append(condition_image)
            except Exception as e:
                log.error(f'LTX condition files: {e}')
        if len(batch_images) > 0:
            conditions.append(make_condition(condition_cls, family, batch_images, condition_strength, is_video=True))
            log.debug(f'Video condition: family={family} files={len(batch_images)} size={batch_images[0].size} strength={condition_strength}')
    if condition_video is not None:
        from modules.video_models.video_utils import get_video_frames
        try:
            condition_frames = get_video_frames(condition_video, num_frames=condition_video_frames, skip_frames=condition_video_skip)
            condition_frames = [f.convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS) for f in condition_frames]
            if len(condition_frames) > 0:
                conditions.append(make_condition(condition_cls, family, condition_frames, condition_strength, is_video=True))
                log.debug(f'Video condition: family={family} frames={len(condition_frames)} size={condition_frames[0].size} strength={condition_strength}')
        except Exception as e:
            log.error(f'LTX condition video: {e}')
    return conditions


def get_prompts(prompt, negative, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    negative = shared.prompt_styles.apply_negative_styles_to_prompt(negative, styles)
    prompts, networks = extra_networks.parse_prompts([prompt])
    prompt = prompts[0] if len(prompts) > 0 else prompt
    return prompt, negative, networks


def get_generator(seed):
    import random
    if seed is None or seed < 0:
        random.seed()
        seed = int(random.randrange(4294967294))
    return torch.Generator().manual_seed(seed)


def vae_decode(latents, decode_timestep, seed, denormalize: bool = True):
    t0 = time.time()
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    log.debug(f'Video: cls={shared.sd_model.vae.__class__.__name__} op=vae latents={latents.shape} timestep={decode_timestep} denormalize={denormalize}')
    from diffusers.utils.torch_utils import randn_tensor
    if denormalize:
        latents = shared.sd_model._denormalize_latents( # pylint: disable=protected-access
            latents,
            shared.sd_model.vae.latents_mean,
            shared.sd_model.vae.latents_std,
            shared.sd_model.vae.config.scaling_factor
        )
    latents = latents.to(device=devices.device, dtype=devices.dtype)
    if not shared.sd_model.vae.config.timestep_conditioning:
        timestep = None
    else:
        noise = randn_tensor(latents.shape, generator=get_generator(seed), device=devices.device, dtype=devices.dtype)
        timestep = torch.tensor([decode_timestep], device=devices.device, dtype=latents.dtype)
        noise_scale = torch.tensor([decode_timestep], device=devices.device, dtype=devices.dtype)[:, None, None, None, None]
        latents = (1 - noise_scale) * latents + noise_scale * noise
    frames = shared.sd_model.vae.decode(latents, timestep, return_dict=False)[0]
    t1 = time.time()
    timer.process.add('vae', t1 - t0)
    return frames
