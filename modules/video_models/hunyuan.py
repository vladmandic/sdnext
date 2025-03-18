from dataclasses import dataclass
import os
import time
import torch
import transformers
import diffusers
from modules import shared, sd_models, sd_checkpoint, sd_samplers, processing, model_quant, devices, images, timer, ui_common


@dataclass
class Model():
    name: str
    repo: str
    dit: str
    subfolder: str

models = {
    'None': [],
    'Hunyuan Video': [
        Model(name='None', repo=None, dit=None, subfolder=None),
        Model(name='Hunyuan Video T2V', repo='hunyuanvideo-community/HunyuanVideo', dit='hunyuanvideo-community/HunyuanVideo', subfolder='transformer'),
        Model(name='Hunyuan Video I2V', repo='hunyuanvideo-community/HunyuanVideo-I2V', dit='hunyuanvideo-community/HunyuanVideo-I2V', subfolder='transformer'), # https://github.com/huggingface/diffusers/pull/10983
        Model(name='SkyReels Hunyuan T2V', repo='hunyuanvideo-community/HunyuanVideo', dit='Skywork/SkyReels-V1-Hunyuan-T2V', subfolder=None), # https://github.com/huggingface/diffusers/pull/10837
        Model(name='SkyReels Hunyuan I2V', repo='hunyuanvideo-community/HunyuanVideo', dit='Skywork/SkyReels-V1-Hunyuan-I2V', subfolder=None),
        Model(name='Fast Hunyuan T2V', repo='hunyuanvideo-community/HunyuanVideo', dit='FastVideo/FastHunyuan-diffusers', subfolder='transformer'), # https://github.com/hao-ai-lab/FastVideo/blob/8a77cf22c9b9e7f931f42bc4b35d21fd91d24e45/fastvideo/models/hunyuan/inference.py#L213
    ]
}
debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
loaded_model = None
prompt_template = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>"
        "\nDescribe the video by detailing the following aspects: \n"
        "1. The main content and theme of the video.\n"
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n"
        "3. Actions, events, behaviors, temporal relationships, and physical movement changes of the objects.\n"
        "4. Background environment, light, style and atmosphere.\n"
        "5. Camera angles, movements, and transitions used in the video.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


def hijack_decode(*args, **kwargs):
    t0 = time.time()
    vae: diffusers.AutoencoderKLHunyuanVideo = shared.sd_model.vae
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.log.debug(f'Video: vae={vae.__class__.__name__} tile={vae.tile_sample_min_width}:{vae.tile_sample_min_height}:{vae.tile_sample_min_num_frames} stride={vae.tile_sample_stride_width}:{vae.tile_sample_stride_height}:{vae.tile_sample_stride_num_frames} time={t1-t0:.2f}')
    return res


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    shared.log.debug(f'Video: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


def load(selected):
    if selected is None:
        return
    global loaded_model # pylint: disable=global-statement
    if loaded_model == selected.name:
        return
    sd_models.unload_model_weights()
    t0 = time.time()

    quant_args = model_quant.create_config(module='Model')
    cls = diffusers.HunyuanVideoTransformer3DModel
    try:
        debug(f'Video load: module=transofrmer repo="{selected.dit}" subfolder="{selected.subfolder}" cls={cls.__name__} quant={quant_args is not None}')
        transformer = cls.from_pretrained(
            pretrained_model_name_or_path=selected.dit,
            subfolder=selected.subfolder,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
            **quant_args
        )
    except Exception as e:
        shared.log.error(f'video load: module=transformer repo="{selected.dit}" subfolder="{selected.subfolder}" cls={cls.__name__} {e}')

    quant_args = model_quant.create_config(module='Text Encoder')
    if 'I2V' in selected.repo:
        cls = transformers.LlavaForConditionalGeneration
    else:
        cls = transformers.LlamaModel
    try:
        debug(f'Video load: module=te repo="{selected.repo}" cls={cls.__name__} quant={quant_args is not None}')
        text_encoder = cls.from_pretrained(
            pretrained_model_name_or_path=selected.repo,
            subfolder="text_encoder",
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
            **quant_args
        )
    except Exception as e:
        shared.log.error(f'video load: module=te repo="{selected.repo}" cls={cls.__name__} {e}')

    cls = transformers.CLIPTextModel
    try:
        debug(f'Video load: module=clip repo="{selected.repo}" cls={cls.__name__} quant=False')
        text_encoder_2 = transformers.CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=selected.repo,
            subfolder="text_encoder_2",
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
    except Exception as e:
        shared.log.error(f'video load: module=clip repo="{selected.repo}" cls={cls.__name__} {e}')

    cls = diffusers.AutoencoderKLHunyuanVideo
    try:
        debug(f'Video load: module=vae repo="{selected.repo}" cls={cls.__name__} quant=False')
        vae = diffusers.AutoencoderKLHunyuanVideo.from_pretrained(
            pretrained_model_name_or_path=selected.repo,
            subfolder="vae",
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
    except Exception as e:
        shared.log.error(f'video load: module=vae repo="{selected.repo}" cls={cls.__name__} {e}')

    if selected.name == 'Hunyuan Video I2V':
        cls = diffusers.HunyuanVideoImageToVideoPipeline
    elif selected.name == 'SkyReels Hunyuan I2V':
        cls = diffusers.HunyuanSkyreelsImageToVideoPipeline
    else:
        cls = diffusers.HunyuanVideoPipeline
    try:
        debug(f'Video load: module=pipe repo="{selected.repo}" cls={cls.__name__} quant=False')
        shared.sd_model = cls.from_pretrained(
            pretrained_model_name_or_path=selected.repo,
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
    except Exception as e:
        shared.log.error(f'video load: module=pipe repo="{selected.repo}" cls={cls.__name__} {e}')

    t1 = time.time()
    sd_models.set_diffuser_options(shared.sd_model)
    shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(selected.repo)
    shared.sd_model.sd_model_hash = None
    shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
    shared.sd_model.vae.decode = hijack_decode
    shared.sd_model.orig_encode_prompt = shared.sd_model.encode_prompt
    shared.sd_model.encode_prompt = hijack_encode_prompt
    shared.sd_model.vae.enable_slicing()
    loaded_model = selected.name
    msg = f'Video load: cls={shared.sd_model.__class__.__name__} model="{selected.name}" time={t1-t0:.2f}'
    shared.log.info(msg)
    return msg


def generate(*args, **kwargs):
    task_id, ui_state, engine, model, prompt, negative, styles, width, height, frames, steps, sampler_index, sampler_shift, seed, guidance_scale, guidance_true, init_image, vae_type, vae_tile_frames, save_frames, video_type, video_duration, video_loop, video_pad, video_interpolate, override_settings = args
    if engine is None or model is None or engine == 'None' or model == 'None':
        shared.log.error('Video: model not selected')
        return [], '', '', 'Video model not selected'
    if not shared.sd_loaded or 'Hunyuan' not in shared.sd_model.__class__.__name__:
        found = [model.name for model in models.get(engine, [])]
        selected = [m for m in models[engine] if m.name == model][0] if len(found) > 0 else None
        load(selected)
    if not shared.sd_loaded or 'Hunyuan' not in shared.sd_model.__class__.__name__:
        shared.log.error('Video: model not loaded')
        return [], '', '', 'Video model not loaded'
    debug(f'Video generate: task={task_id} args={args} kwargs={kwargs}')

    p = processing.StableDiffusionProcessingVideo(
        sd_model=shared.sd_model,
        styles=styles,
        seed=int(seed),
        sampler_name = processing.get_sampler_name(sampler_index),
        sampler_shift=float(sampler_shift),
        steps=int(steps),
        width=16 * int(width // 16),
        height=16 * int(height // 16),
        frames=int(frames),
        init_image=init_image,
        cfg_scale=float(guidance_scale),
        diffusers_guidance_rescale=float(guidance_true),
        vae_type=vae_type,
        vae_tile_frames=int(vae_tile_frames),
        override_settings=override_settings,
    )
    p.scripts = None
    p.script_args = args
    p.state = ui_state
    p.do_not_save_grid = True
    p.do_not_save_samples = not save_frames
    if 'I2V' in model:
        if init_image is None:
            shared.log.error('Video: init image not set')
            return [], '', '', 'Error: init image not set'
        p.task_args['image'] = init_image

    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    devices.torch_gc(force=True)

    # handle sampler and seed
    if p.sampler_name != 'Default':
        shared.sd_model.scheduler = sd_samplers.create_sampler(p.sampler_name, shared.sd_model)
        p.sampler_name = 'Default' # avoid double creation
    if hasattr(shared.sd_model.scheduler, '_shift') and sampler_shift > 0:
        shared.sd_model.scheduler._shift = sampler_shift # pylint: disable=protected-access

    # handle vae
    if vae_tile_frames > p.frames:
        shared.sd_model.vae.tile_sample_min_num_frames = vae_tile_frames
        shared.sd_model.vae.use_framewise_decoding = True
        shared.sd_model.vae.enable_tiling()
    else:
        shared.sd_model.vae.use_framewise_decoding = False
        shared.sd_model.vae.disable_tiling()

    # set args
    processing.fix_seed(p)
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(negative, p.styles)
    p.task_args['width'] = p.width
    p.task_args['height'] = p.height
    p.task_args['num_inference_steps'] = p.steps
    p.task_args['num_frames'] = p.frames
    p.task_args['generator'] = torch.manual_seed(p.seed)
    p.task_args['guidance_scale'] = p.cfg_scale
    p.task_args['true_cfg_scale'] = p.diffusers_guidance_rescale
    p.task_args['prompt_template'] = prompt_template
    p.task_args['output_type'] = 'pil'
    p.task_args['prompt'] = p.prompt
    p.task_args['negative_prompt'] = p.negative_prompt
    p.ops.append('video')
    debug(f'Video: task_args={p.task_args}')

    # run processing
    shared.state.disable_preview = True
    shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} width={p.width} height={p.height} frames={p.frames} steps={p.steps}')
    t0 = time.time()
    processed = processing.process_images(p)
    t1 = time.time()
    shared.state.disable_preview = False

    p.close()
    if processed is None or len(processed.images) == 0:
        return [], '', '', 'Error: processing failed'
    shared.log.info(f'Video: frames={len(processed.images)} time={t1-t0:.2f}')
    if video_type != 'None':
        images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=video_duration, loop=video_loop, pad=video_pad, interpolate=video_interpolate)

    generation_info_js = processed.js() if processed is not None else ''
    return processed.images, generation_info_js, processed.info, ui_common.plaintext_to_html(processed.comments)
