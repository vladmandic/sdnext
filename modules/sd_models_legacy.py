import io
import os
import sys
import contextlib

from modules import shared


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'


def get_checkpoint_state_dict(checkpoint_info, timer):
    from modules.sd_models_utils import read_state_dict
    if not os.path.isfile(checkpoint_info.filename):
        return None
    """
    if checkpoint_info in checkpoints_loaded:
        shared.log.info("Load model: cache")
        checkpoints_loaded.move_to_end(checkpoint_info, last=True)  # FIFO -> LRU cache
        return checkpoints_loaded[checkpoint_info]
    """
    res = read_state_dict(checkpoint_info.filename, what='model')
    """
    if shared.opts.sd_checkpoint_cache > 0 and not shared.native:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = res
        # clean up cache if limit is reached
        while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
            checkpoints_loaded.popitem(last=False)
    """
    timer.record("load")
    return res


def repair_config(sd_config):
    from modules import paths
    if "use_ema" not in sd_config.model.params:
        sd_config.model.params.use_ema = False
    if shared.opts.no_half:
        sd_config.model.params.unet_config.params.use_fp16 = False
    elif shared.opts.upcast_sampling:
        sd_config.model.params.unet_config.params.use_fp16 = True if sys.platform != 'darwin' else False
    if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
        sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"
    # For UnCLIP-L, override the hardcoded karlo directory
    if "noise_aug_config" in sd_config.model.params and "clip_stats_path" in sd_config.model.params.noise_aug_config.params:
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)


def load_model_weights(model, checkpoint_info, state_dict, timer):
    from modules.modeldata import model_data
    from modules.memstats import memory_stats
    from modules import devices, sd_vae
    shared.log.debug(f'Load model: memory={memory_stats()}')
    timer.record("hash")
    if model_data.sd_dict == 'None':
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        shared.log.error(f'Load model: path="{checkpoint_info.filename}"')
        shared.log.error(' '.join(str(e).splitlines()[:2]))
        return False
    del state_dict
    timer.record("apply")
    if shared.opts.opt_channelslast:
        import torch
        model.to(memory_format=torch.channels_last)
        timer.record("channels")
    if not shared.opts.no_half:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)
        if shared.opts.no_half_vae: # remove VAE from model when doing half() to prevent its weights from being converted to float16
            model.first_stage_model = None
        if shared.opts.upcast_sampling and depth_model: # with don't convert the depth model weights to float16
            model.depth_model = None
        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model
    if shared.opts.cuda_cast_unet:
        devices.dtype_unet = model.model.diffusion_model.dtype
    else:
        model.model.diffusion_model.to(devices.dtype_unet)
    model.first_stage_model.to(devices.dtype_vae)
    model.sd_model_hash = checkpoint_info.calculate_shorthash()
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    model.is_sdxl = False # a1111 compatibility item
    model.is_sd2 = hasattr(model.cond_stage_model, 'model') # a1111 compatibility item
    model.is_sd1 = not hasattr(model.cond_stage_model, 'model') # a1111 compatibility item
    model.logvar = model.logvar.to(devices.device) if hasattr(model, 'logvar') else None # fix for training
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("vae")
    return True


def load_model(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'):
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf
    from modules import devices, lowvram, sd_hijack, sd_models_config, script_callbacks
    from modules.timer import Timer
    from modules.memstats import memory_stats
    from modules.modeldata import model_data
    from modules.sd_models import unload_model_weights, move_model
    from modules.sd_checkpoint import select_checkpoint
    checkpoint_info = checkpoint_info or select_checkpoint(op=op)
    if checkpoint_info is None:
        return
    if op == 'model' or op == 'dict':
        if (model_data.sd_model is not None) and (getattr(model_data.sd_model, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (getattr(model_data.sd_refiner, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return
    shared.log.debug(f'Load {op}: name={checkpoint_info.filename} dict={already_loaded_state_dict is not None}')
    if timer is None:
        timer = Timer()
    current_checkpoint_info = None
    if op == 'model' or op == 'dict':
        if model_data.sd_model is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            current_checkpoint_info = getattr(model_data.sd_model, 'sd_checkpoint_info', None)
            unload_model_weights(op=op)
    else:
        if model_data.sd_refiner is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            current_checkpoint_info = getattr(model_data.sd_refiner, 'sd_checkpoint_info', None)
            unload_model_weights(op=op)

    if not shared.native:
        from modules import sd_hijack_inpainting
        sd_hijack_inpainting.do_inpainting_hijack()

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    if state_dict is None or checkpoint_config is None:
        shared.log.error(f'Load {op}: path="{checkpoint_info.filename}"')
        if current_checkpoint_info is not None:
            shared.log.info(f'Load {op}: previous="{current_checkpoint_info.filename}" restore')
            load_model(current_checkpoint_info, None)
        return
    shared.log.debug(f'Model dict loaded: {memory_stats()}')
    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config)
    timer.record("config")
    shared.log.debug(f'Model config loaded: {memory_stats()}')
    sd_model = None
    stdout = io.StringIO()
    if os.environ.get('SD_LDM_DEBUG', None) is not None:
        sd_model = instantiate_from_config(sd_config.model)
    else:
        with contextlib.redirect_stdout(stdout):
            sd_model = instantiate_from_config(sd_config.model)
        for line in stdout.getvalue().splitlines():
            if len(line) > 0:
                shared.log.info(f'LDM: {line.strip()}')
    shared.log.debug(f"Model created from config: {checkpoint_config}")
    sd_model.used_config = checkpoint_config
    sd_model.has_accelerate = False
    timer.record("create")
    ok = load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    if not ok:
        model_data.sd_model = sd_model
        current_checkpoint_info = None
        unload_model_weights(op=op)
        shared.log.debug(f'Model weights unloaded: {memory_stats()} op={op}')
        if op == 'refiner':
            # shared.opts.data['sd_model_refiner'] = 'None'
            shared.opts.sd_model_refiner = 'None'
        return
    else:
        shared.log.debug(f'Model weights loaded: {memory_stats()}')
    timer.record("load")
    if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        move_model(sd_model, devices.device)
    timer.record("move")
    shared.log.debug(f'Model weights moved: {memory_stats()}')
    sd_hijack.model_hijack.hijack(sd_model)
    timer.record("hijack")
    sd_model.eval()
    if op == 'refiner':
        model_data.sd_refiner = sd_model
    else:
        model_data.sd_model = sd_model
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model
    timer.record("embeddings")
    script_callbacks.model_loaded_callback(sd_model)
    timer.record("callbacks")
    shared.log.info(f"Model loaded in {timer.summary()}")
    current_checkpoint_info = None
    devices.torch_gc(force=True)
    shared.log.info(f'Model load finished: {memory_stats()}')
