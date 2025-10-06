import copy
import time
from modules import shared, errors, sd_models, sd_checkpoint, model_quant, devices, sd_hijack_te, sd_hijack_vae
from modules.video_models import models_def, video_utils, video_overrides, video_cache


loaded_model = None


def load_model(selected: models_def.Model):
    if selected is None:
        return ''
    global loaded_model # pylint: disable=global-statement
    if loaded_model == selected.name:
        return ''
    sd_models.unload_model_weights()
    t0 = time.time()
    jobid = shared.state.begin('Load model')

    video_cache.apply_teacache_patch(selected.dit_cls)

    # overrides
    kwargs = video_overrides.load_override(selected)

    # text encoder
    try:
        load_args, quant_args = model_quant.get_dit_args({}, module='TE', device_map=True)
        if selected.te_cls.__name__ == 'T5EncoderModel' and shared.opts.te_shared_t5:
            selected.te = 'Disty0/t5-xxl'
            selected.te_folder = ''
            selected.te_revision = None
        shared.log.debug(f'Video load: module=te repo="{selected.te or selected.repo}" folder="{selected.te_folder}" cls={selected.te_cls.__name__} quant={model_quant.get_quant_type(quant_args)}')
        kwargs["text_encoder"] = selected.te_cls.from_pretrained(
            pretrained_model_name_or_path=selected.te or selected.repo,
            subfolder=selected.te_folder,
            revision=selected.te_revision or selected.repo_revision,
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args
        )
    except Exception as e:
        shared.log.error(f'video load: module=te cls={selected.te_cls.__name__} {e}')
        errors.display(e, 'video')

    # transformer
    try:
        def load_dit_folder(dit_folder):
            if dit_folder is not None and dit_folder not in kwargs:
                # get a new quant arg on every loop to prevent the quant config classes getting entangled
                load_args, quant_args = model_quant.get_dit_args({}, module='Model', device_map=True)
                shared.log.debug(f'Video load: module=transformer repo="{selected.dit or selected.repo}" module="{dit_folder}" folder="{dit_folder}" cls={selected.dit_cls.__name__} quant={model_quant.get_quant_type(quant_args)}')
                kwargs[dit_folder] = selected.dit_cls.from_pretrained(
                    pretrained_model_name_or_path=selected.dit or selected.repo,
                    subfolder=dit_folder,
                    revision=selected.dit_revision or selected.repo_revision,
                    cache_dir=shared.opts.hfcache_dir,
                    **load_args,
                    **quant_args
                )
            else:
                shared.log.debug(f'Video load: module=transformer repo="{selected.dit or selected.repo}" module="{dit_folder}" folder="{dit_folder}" cls={selected.dit_cls.__name__} skip')

        if selected.dit_folder is None:
            selected.dit_folder = ['transformer']
        if isinstance(selected.dit_folder, list) or isinstance(selected.dit_folder, tuple):
            for dit_folder in selected.dit_folder: # wan a14b has transformer and transformer_2
                load_dit_folder(dit_folder)
        else:
            load_dit_folder(selected.dit_folder)
    except Exception as e:
        shared.log.error(f'video load: module=transformer cls={selected.dit_cls.__name__} {e}')
        errors.display(e, 'video')

    # model
    try:
        shared.log.debug(f'Video load: module=pipe repo="{selected.repo}" cls={selected.repo_cls.__name__}')
        shared.sd_model = selected.repo_cls.from_pretrained(
            pretrained_model_name_or_path=selected.repo,
            revision=selected.repo_revision,
            cache_dir=shared.opts.hfcache_dir,
            torch_dtype=devices.dtype,
            **kwargs,
        )
    except Exception as e:
        shared.log.error(f'video load: module=pipe repo="{selected.repo}" cls={selected.repo_cls.__name__} {e}')
        errors.display(e, 'video')

    t1 = time.time()
    if shared.sd_model.__class__.__name__.startswith("LTX"):
        shared.sd_model.scheduler.config.use_dynamic_shifting = False
    shared.sd_model.default_scheduler = copy.deepcopy(shared.sd_model.scheduler) if hasattr(shared.sd_model, "scheduler") else None
    shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(selected.repo)
    shared.sd_model.sd_model_hash = None
    sd_models.set_diffuser_options(shared.sd_model, offload=False)

    decode, text, image, slicing, tiling = False, False, False, False, False
    if selected.vae_hijack and hasattr(shared.sd_model.vae, 'decode'):
        sd_hijack_vae.init_hijack(shared.sd_model)
        decode = True
    if selected.te_hijack and hasattr(shared.sd_model, 'encode_prompt'):
        sd_hijack_te.init_hijack(shared.sd_model)
        text = True
    if selected.image_hijack and hasattr(shared.sd_model, 'encode_image'):
        shared.sd_model.orig_encode_image = shared.sd_model.encode_image
        shared.sd_model.encode_image = video_utils.hijack_encode_image
        image = True
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
    shared.log.info(msg)
    shared.log.debug(f'Video hijacks: decode={decode} text={text} image={image} slicing={slicing} tiling={tiling}')
    shared.state.end(jobid)
    return msg
