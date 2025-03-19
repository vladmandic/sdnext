import os
import time
from modules import shared, errors, timer, sd_models, sd_checkpoint, model_quant, devices
from modules.video_models import models_def


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def queue_err(msg):
    shared.log.error(f'Video: {msg}')
    return [], None, '', '', f'Error: {msg}'


def get_quant(args):
    if args is not None and "quantization_config" in args:
        return args['quantization_config'].__class__.__name__
    return None


def get_url(url):
    return f'&nbsp <a href="{url}" target="_blank" rel="noopener noreferrer" style="color: var(--button-primary-background-fill); font-weight: normal">{url}</a><br>' if url else ''


def set_prompt(p):
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    p.task_args['prompt'] = p.prompt
    p.task_args['negative_prompt'] = p.negative_prompt


def set_vae_params(frames, tile_frames):
    if tile_frames > frames:
        if hasattr(shared.sd_model.vae, 'tile_sample_min_num_frames'):
            shared.sd_model.vae.tile_sample_min_num_frames = tile_frames
        if hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
            shared.sd_model.vae.use_framewise_decoding = True
        if hasattr(shared.sd_model.vae, 'enable_tiling'):
            shared.sd_model.vae.enable_tiling()
    else:
        if hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
            shared.sd_model.vae.use_framewise_decoding = False
        if hasattr(shared.sd_model.vae, 'disable_tiling'):
            shared.sd_model.vae.disable_tiling()


def hijack_vae_decode(*args, **kwargs):
    t0 = time.time()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    debug(f'Video decode: vae={shared.sd_model.vae.__class__.__name__} time={t1-t0:.2f}')
    return res


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    debug(f'Video encode: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res

loaded_model = None


def load_model(selected: models_def.Model):
    if selected is None:
        return
    global loaded_model # pylint: disable=global-statement
    if loaded_model == selected.name:
        return
    sd_models.unload_model_weights()
    t0 = time.time()

    # text encoder
    try:
        quant_args = model_quant.create_config(module='Text Encoder')
        debug(f'Video load: module=te repo="{selected.te or selected.repo}" folder="{selected.te_folder}" cls={selected.te_cls.__name__} quant={get_quant(quant_args)}')
        text_encoder = selected.te_cls.from_pretrained(
            pretrained_model_name_or_path=selected.te or selected.repo,
            subfolder=selected.te_folder,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
            **quant_args
        )
    except Exception as e:
        shared.log.error(f'video load: module=te cls={selected.te_cls.__name__} {e}')
        errors.display(e, 'video')
        text_encoder = None

    # transformer
    try:
        quant_args = model_quant.create_config(module='Model')
        debug(f'Video load: module=transformer repo="{selected.dit or selected.repo}" folder="{selected.dit_folder}" cls={selected.dit_cls.__name__} quant={get_quant(quant_args)}')
        transformer = selected.dit_cls.from_pretrained(
            pretrained_model_name_or_path=selected.dit or selected.repo,
            subfolder=selected.dit_folder,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
            **quant_args
        )
    except Exception as e:
        shared.log.error(f'video load: module=transformer cls={selected.dit_cls.__name__} {e}')
        errors.display(e, 'video')
        transformer = None

    # model
    try:
        debug(f'Video load: module=pipe repo="{selected.repo}" cls={selected.repo_cls.__name__}')
        shared.sd_model = selected.repo_cls.from_pretrained(
            pretrained_model_name_or_path=selected.repo,
            transformer=transformer,
            text_encoder=text_encoder,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
        )
    except Exception as e:
        shared.log.error(f'video load: module=pipe repo="{selected.repo}" cls={selected.repo_cls.__name__} {e}')
        errors.display(e, 'video')

    t1 = time.time()
    sd_models.set_diffuser_options(shared.sd_model)
    shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(selected.repo)
    shared.sd_model.sd_model_hash = None
    if selected.vae_hijack:
        shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
        shared.sd_model.vae.decode = hijack_vae_decode
    if selected.te_hijack:
        shared.sd_model.orig_encode_prompt = shared.sd_model.encode_prompt
        shared.sd_model.encode_prompt = hijack_encode_prompt
    shared.sd_model.vae.enable_slicing()
    loaded_model = selected.name
    msg = f'Video load: cls={shared.sd_model.__class__.__name__} model="{selected.name}" time={t1-t0:.2f}'
    shared.log.info(msg)
    return msg
