import os
import sys
import copy
import time
import diffusers
from modules import shared, errors, sd_models, sd_checkpoint, model_quant, devices, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from modules.video_models import models_def, video_utils, video_overrides, video_cache


def _loader(component):
    """Return loader type for log messages."""
    if sys.platform != 'linux':
        return 'default'
    if component == 'diffusers':
        return 'runai' if shared.opts.runai_streamer_diffusers else 'default'
    return 'runai' if shared.opts.runai_streamer_transformers else 'default'


loaded_model = None


def load_custom(model_name: str):
    log.debug(f'Video load: module=pipe repo="{model_name}" cls=Custom')
    if 'veo-3.1' in model_name:
        from modules.video_models.google_veo import load_veo
        pipe = load_veo(model_name)
        return pipe
    return None


def load_model(selected: models_def.Model):
    from modules import sdnq # pylint: disable=unused-import
    if selected is None or selected.repo is None:
        return ''
    global loaded_model # pylint: disable=global-statement
    if not shared.sd_loaded:
        loaded_model = None
    if loaded_model == selected.name:
        return ''
    if shared.sd_loaded:
        sd_models.unload_model_weights()

    t0 = time.time()
    jobid = shared.state.begin('Load model')

    video_cache.apply_teacache_patch(selected.dit_cls)

    # overrides
    offline_args = {}
    if shared.opts.offline_mode:
        offline_args["local_files_only"] = True
        os.environ['HF_HUB_OFFLINE'] = '1'
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.unsetenv('HF_HUB_OFFLINE')

    kwargs = video_overrides.load_override(selected, **offline_args)

    # text encoder
    if selected.te_cls is not None:
        try:
            load_args, quant_args = model_quant.get_dit_args({}, module='TE', device_map=True)

            # loader deduplication of text-encoder models
            if selected.te_cls.__name__ == 'T5EncoderModel' and shared.opts.te_shared_t5:
                selected.te = 'Disty0/t5-xxl'
                selected.te_folder = ''
                selected.te_revision = None
            if selected.te_cls.__name__ == 'UMT5EncoderModel' and shared.opts.te_shared_t5:
                if 'SDNQ' in selected.name:
                    selected.te = 'Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32'
                else:
                    selected.te = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
                selected.te_folder = 'text_encoder'
                selected.te_revision = None
            if selected.te_cls.__name__ == 'LlamaModel' and shared.opts.te_shared_t5:
                selected.te = 'hunyuanvideo-community/HunyuanVideo'
                selected.te_folder = 'text_encoder'
                selected.te_revision = None
            if selected.te_cls.__name__ == 'Qwen2_5_VLForConditionalGeneration' and shared.opts.te_shared_t5:
                selected.te = 'ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers'
                selected.te_folder = 'text_encoder'
                selected.te_revision = None

            log.debug(f'Video load: module=te repo="{selected.te or selected.repo}" folder="{selected.te_folder}" cls={selected.te_cls.__name__} quant={model_quant.get_quant_type(quant_args)} loader={_loader("transformers")}')
            kwargs["text_encoder"] = selected.te_cls.from_pretrained(
                pretrained_model_name_or_path=selected.te or selected.repo,
                subfolder=selected.te_folder,
                revision=selected.te_revision or selected.repo_revision,
                cache_dir=shared.opts.hfcache_dir,
                **load_args,
                **quant_args,
                **offline_args,
            )
        except Exception as e:
            log.error(f'video load: module=te cls={selected.te_cls.__name__} {e}')
            errors.display(e, 'video')

    # transformer
    if selected.dit_cls is not None:
        try:
            def load_dit_folder(dit_folder):
                if dit_folder is not None and dit_folder not in kwargs:
                    # get a new quant arg on every loop to prevent the quant config classes getting entangled
                    load_args, quant_args = model_quant.get_dit_args({}, module='Model', device_map=True)
                    log.debug(f'Video load: module=transformer repo="{selected.dit or selected.repo}" module="{dit_folder}" folder="{dit_folder}" cls={selected.dit_cls.__name__} quant={model_quant.get_quant_type(quant_args)} loader={_loader("diffusers")}')
                    kwargs[dit_folder] = selected.dit_cls.from_pretrained(
                        pretrained_model_name_or_path=selected.dit or selected.repo,
                        subfolder=dit_folder,
                        revision=selected.dit_revision or selected.repo_revision,
                        cache_dir=shared.opts.hfcache_dir,
                        **load_args,
                        **quant_args,
                        **offline_args,
                    )
                else:
                    log.debug(f'Video load: module=transformer repo="{selected.dit or selected.repo}" module="{dit_folder}" folder="{dit_folder}" cls={selected.dit_cls.__name__} loader={_loader("diffusers")} skip')

            if selected.dit_folder is None:
                selected.dit_folder = ['transformer']
            if isinstance(selected.dit_folder, list) or isinstance(selected.dit_folder, tuple):
                for dit_folder in selected.dit_folder: # wan a14b has transformer and transformer_2
                    load_dit_folder(dit_folder)
            else:
                load_dit_folder(selected.dit_folder)
        except Exception as e:
            log.error(f'video load: module=transformer cls={selected.dit_cls.__name__} {e}')
            errors.display(e, 'video')

    # model
    try:
        if selected.repo_cls is None:
            shared.sd_model = load_custom(selected.repo)
        else:
            log.debug(f'Video load: module=pipe repo="{selected.repo}" cls={selected.repo_cls.__name__}')
            shared.sd_model = selected.repo_cls.from_pretrained(
                pretrained_model_name_or_path=selected.repo,
                revision=selected.repo_revision,
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **kwargs,
                **offline_args,
            )
    except Exception as e:
        log.error(f'video load: module=pipe repo="{selected.repo}" cls={selected.repo_cls.__name__} {e}')
        errors.display(e, 'video')

    if shared.sd_model is None:
        msg = f'Video load: model="{selected.name}" failed'
        log.error(msg)
        return msg

    t1 = time.time()
    if shared.sd_model.__class__.__name__.startswith("LTX"):
        shared.sd_model.scheduler.config.use_dynamic_shifting = False
    shared.sd_model.default_scheduler = copy.deepcopy(shared.sd_model.scheduler) if hasattr(shared.sd_model, "scheduler") else None
    shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(selected.repo)
    shared.sd_model.sd_model_hash = None
    sd_models.set_diffuser_options(shared.sd_model, offload=False)

    decode, text, image, slicing, tiling, framewise = False, False, False, False, False, False
    if selected.vae_hijack and hasattr(shared.sd_model, 'vae') and hasattr(shared.sd_model.vae, 'decode'):
        sd_hijack_vae.init_hijack(shared.sd_model)
        decode = True
    if selected.te_hijack and hasattr(shared.sd_model, 'encode_prompt'):
        sd_hijack_te.init_hijack(shared.sd_model)
        text = True
    if selected.image_hijack and hasattr(shared.sd_model, 'encode_image'):
        shared.sd_model.orig_encode_image = shared.sd_model.encode_image
        shared.sd_model.encode_image = video_utils.hijack_encode_image
        image = True
    if hasattr(shared.sd_model, 'vae') and hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
        shared.sd_model.vae.use_framewise_decoding = True
        framewise = True
    if hasattr(shared.sd_model, 'vae') and hasattr(shared.sd_model.vae, 'enable_slicing'):
        shared.sd_model.vae.enable_slicing()
        slicing = True
    if hasattr(shared.sd_model, 'vae') and hasattr(shared.sd_model.vae, 'enable_tiling'):
        shared.sd_model.vae.enable_tiling()
        tiling = True
    if hasattr(shared.sd_model, "set_progress_bar_config"):
        shared.sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m', ncols=80, colour='#327fba')

    shared.sd_model = model_quant.do_post_load_quant(shared.sd_model, allow=False)
    sd_models.set_diffuser_offload(shared.sd_model)

    loaded_model = selected.name
    msg = f'Video load: cls={shared.sd_model.__class__.__name__} model="{selected.name}" time={t1-t0:.2f}'
    log.info(msg)
    log.debug(f'Video hijacks: decode={decode} text={text} image={image} slicing={slicing} tiling={tiling} framewise={framewise}')
    shared.state.end(jobid)
    return msg


def load_upscale_vae():
    if not hasattr(shared.sd_model, 'vae'):
        return
    if hasattr(shared.sd_model.vae, '_asymmetric_upscale_vae'):
        return # already loaded
    if shared.sd_model.vae.__class__.__name__ != 'AutoencoderKLWan':
        log.warning('Video decode: upscale VAE unsupported')
        return

    repo_id = 'spacepxl/Wan2.1-VAE-upscale2x'
    subfolder = "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1"
    vae_decode = diffusers.AutoencoderKLWan.from_pretrained(repo_id, subfolder=subfolder, cache_dir=shared.opts.hfcache_dir)
    vae_decode.requires_grad_(False)
    vae_decode = vae_decode.to(device=devices.device, dtype=devices.dtype)
    vae_decode.eval()
    log.debug(f'Decode: load="{repo_id}"')
    shared.sd_model.orig_vae = shared.sd_model.vae
    shared.sd_model.vae = vae_decode
    shared.sd_model.vae._asymmetric_upscale_vae = True # pylint: disable=protected-access
    sd_hijack_vae.init_hijack(shared.sd_model)
    sd_models.apply_balanced_offload(shared.sd_model, force=True) # reapply offload
