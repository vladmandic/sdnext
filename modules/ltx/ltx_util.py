import time
import torch
from PIL import Image
from modules import devices, shared, sd_models, timer, extra_networks
from modules.logger import log


loaded_model: str = None


def get_bucket(size: int):
    if not hasattr(shared.sd_model, 'vae_temporal_compression_ratio'):
        return int(size) - (int(size) % 32)
    return int(size) - (int(size) % shared.sd_model.vae_temporal_compression_ratio)


def get_frames(frames: int):
    return int(8 * (int(frames) // 8)) + 1


def load_model(engine: str, model: str):
    global loaded_model # pylint: disable=global-statement
    if loaded_model == model:
        return
    if model is None or model == '' or model=='None':
        loaded_model = None
        shared.sd_model = None
        return
    t0 = time.time()
    from modules.video_models import models_def, video_load
    selected: models_def.Model = [m for m in models_def.models[engine] if m.name == model][0]
    log.info(f'Video load: engine="{engine}" selected="{model}" {selected}')
    video_load.load_model(selected)
    loaded_model = model
    t1 = time.time()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    t2 = time.time()
    timer.process.add('load', t1 - t0)
    timer.process.add('offload', t2 - t1)


def load_upsample(upsample_pipe, upsample_repo_id):
    if upsample_pipe is None:
        t0 = time.time()
        from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline
        log.info(f'Video load: cls={LTXLatentUpsamplePipeline.__class__.__name__} repo="{upsample_repo_id}"')
        upsample_pipe = LTXLatentUpsamplePipeline.from_pretrained(
            upsample_repo_id,
            vae=shared.sd_model.vae,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
        t1 = time.time()
        timer.process.add('load', t1 - t0)
    return upsample_pipe


def get_conditions(width, height, condition_strength, condition_images, condition_files, condition_video, condition_video_frames, condition_video_skip):
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    conditions = []
    if condition_images is not None:
        for condition_image in condition_images:
            try:
                if isinstance(condition_image, str):
                    from modules.api.api import decode_base64_to_image
                    condition_image = decode_base64_to_image(condition_image)
                condition_image = condition_image.convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS)
                conditions.append(LTXVideoCondition(image=condition_image, frame_index=0, strength=condition_strength))
                log.debug(f'Video condition: image={condition_image.size} strength={condition_strength}')
            except Exception as e:
                log.error(f'LTX condition image: {e}')
    if condition_files is not None:
        condition_images = []
        for fn in condition_files:
            try:
                if hasattr(fn, 'name'):
                    condition_image = Image.open(fn.name).convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS)
                else:
                    condition_image = fn.convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS)
                condition_images.append(condition_image)
            except Exception as e:
                log.error(f'LTX condition files: {e}')
        if len(condition_images) > 0:
            conditions.append(LTXVideoCondition(video=condition_images, frame_index=0, strength=condition_strength))
            log.debug(f'Video condition: files={len(condition_images)} size={condition_images[0].size} strength={condition_strength}')
    if condition_video is not None:
        from modules.video_models.video_utils import get_video_frames
        try:
            condition_frames = get_video_frames(condition_video, num_frames=condition_video_frames, skip_frames=condition_video_skip)
            condition_frames = [f.convert('RGB').resize((width, height), resample=Image.Resampling.LANCZOS) for f in condition_frames]
            if len(condition_frames) > 0:
                conditions.append(LTXVideoCondition(video=condition_frames, frame_index=0, strength=condition_strength))
                log.debug(f'Video condition: frames={len(condition_frames)} size={condition_frames[0].size} strength={condition_strength}')
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


def vae_decode(latents, decode_timestep, seed):
    t0 = time.time()
    log.debug(f'Video: cls={shared.sd_model.vae.__class__.__name__} op=vae latents={latents.shape} timestep={decode_timestep}')
    from diffusers.utils.torch_utils import randn_tensor
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
    frames = shared.sd_model.vae.decode(latents, timestep, return_dict=False)[0] # n, c, f, h, w
    # frames = frames.squeeze(0) if frames.ndim == 5 else frames
    # frames = frames.permute(1, 2, 3, 0)
    # frames = shared.sd_model.video_processor.postprocess_video(frames, output_type='pil')
    t1 = time.time()
    timer.process.add('vae', t1 - t0)
    return frames
