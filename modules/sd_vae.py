import os
import glob
import torch
from modules import shared, errors, paths, devices, sd_models, sd_detect


vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}
base_vae = None
loaded_vae_file = None
checkpoint_info = None
vae_path = os.path.abspath(os.path.join(paths.models_path, 'VAE'))
debug = os.environ.get('SD_VAE_DEBUG', None) is not None
unspecified = object()
vae_scale_override = {
    'WanPipeline': 16,
    'ChronoEditPipeline': 16,
}


def get_vae_scale_factor(model=None):
    if not shared.sd_loaded:
        vae_scale_factor = 8
        return vae_scale_factor
    patch_size = 1
    if model is None:
        model = shared.sd_model
    if model is None:
        vae_scale_factor = 8
    elif model.__class__.__name__ in vae_scale_override:
        vae_scale_factor = vae_scale_override[model.__class__.__name__]
    elif hasattr(model, 'vae_scale_factor_spatial'):
        vae_scale_factor = model.vae_scale_factor_spatial
    elif hasattr(model, 'vae_scale_factor'):
        vae_scale_factor = model.vae_scale_factor
    elif hasattr(model, 'pipe') and hasattr(model.pipe, 'vae_scale_factor'):
        vae_scale_factor = model.pipe.vae_scale_factor
    elif hasattr(model, 'config') and hasattr(model.config, 'vae_scale_factor'):
        vae_scale_factor = model.config.vae_scale_factor
    else:
        # shared.log.warning(f'VAE: cls={model.__class__.__name__ if model else "None"} scale=unknown')
        vae_scale_factor = 8
    if hasattr(model, 'patch_size'):
        patch_size = model.patch_size
    if debug:
        shared.log.trace(f'VAE: cls={model.__class__.__name__ if model else "None"} scale={vae_scale_factor} patch={patch_size}')
    return vae_scale_factor * patch_size


def load_vae_dict(filename):
    vae_ckpt = sd_models.read_state_dict(filename, what='vae')
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


def get_filename(filepath):
    if filepath.endswith(".json"):
        return os.path.basename(os.path.dirname(filepath))
    else:
        return os.path.basename(filepath)


def refresh_vae_list():
    global vae_path # pylint: disable=global-statement
    vae_path = shared.opts.vae_dir
    vae_dict.clear()
    vae_paths = []
    if sd_models.model_path is not None and os.path.isdir(sd_models.model_path):
        vae_paths += [os.path.join(sd_models.model_path, 'VAE', '**/*.vae.safetensors')]
    if shared.opts.ckpt_dir is not None and os.path.isdir(shared.opts.ckpt_dir):
        vae_paths += [os.path.join(shared.opts.ckpt_dir, '**/*.vae.safetensors')]
    if shared.opts.vae_dir is not None and os.path.isdir(shared.opts.vae_dir):
        vae_paths += [os.path.join(shared.opts.vae_dir, '**/*.safetensors')]
    vae_paths += [
        os.path.join(sd_models.model_path, 'VAE', '**/*.json'),
        os.path.join(shared.opts.vae_dir, '**/*.json'),
    ]
    candidates = []
    for path in vae_paths:
        candidates += glob.iglob(path, recursive=True)
    candidates = [os.path.abspath(path) for path in candidates]
    for filepath in candidates:
        name = get_filename(filepath)
        if name == 'VAE':
            continue
        if filepath.endswith(".json"):
            vae_dict[name] = os.path.dirname(filepath)
        else:
            vae_dict[name] = filepath
    shared.log.info(f'Available VAEs: path="{vae_path}" items={len(vae_dict)}')
    return vae_dict


def find_vae_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.splitext(checkpoint_file)[0]
    for vae_location in [f"{checkpoint_path}.vae.pt", f"{checkpoint_path}.vae.ckpt", f"{checkpoint_path}.vae.safetensors"]:
        if os.path.isfile(vae_location):
            return vae_location
    return None


def resolve_vae(checkpoint_file):
    if shared.opts.sd_vae == 'TAESD':
        return None, None
    if shared.cmd_opts.vae is not None: # 1st
        return shared.cmd_opts.vae, 'forced'
    if shared.opts.sd_vae == "Default": # 2nd
        return None, None
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None: # 3rd
        return vae_near_checkpoint, 'near-checkpoint'
    if shared.opts.sd_vae == "Automatic": # 4th
        basename = os.path.splitext(os.path.basename(checkpoint_file))[0]
        if vae_dict.get(basename, None) is not None:
            return vae_dict[basename], 'automatic'
    else:
        vae_from_options = vae_dict.get(shared.opts.sd_vae, None) # 5th
        if vae_from_options is not None:
            return vae_from_options, 'settings'
        vae_from_options = vae_dict.get(shared.opts.sd_vae + '.safetensors', None) # 6th
        if vae_from_options is not None:
            return vae_from_options, 'settings'
        shared.log.warning(f"VAE not found: {shared.opts.sd_vae}")
    return None, None


def apply_vae_config(model_file, vae_file, sd_model):
    def get_vae_config():
        config_file = os.path.join(paths.sd_configs_path, os.path.splitext(os.path.basename(model_file))[0] + '_vae.json')
        if config_file is not None and os.path.exists(config_file):
            return shared.readfile(config_file, as_type="dict")
        config_file = os.path.join(paths.sd_configs_path, os.path.splitext(os.path.basename(vae_file))[0] + '.json') if vae_file else None
        if config_file is not None and os.path.exists(config_file):
            return shared.readfile(config_file, as_type="dict")
        config_file = os.path.join(paths.sd_configs_path, shared.sd_model_type, 'vae', 'config.json')
        if config_file is not None and os.path.exists(config_file):
            return shared.readfile(config_file, as_type="dict")
        return {}

    if hasattr(sd_model, 'vae') and hasattr(sd_model.vae, 'config'):
        config = get_vae_config()
        for k, v in config.items():
            if k in sd_model.vae.config and not k.startswith('_'):
                sd_model.vae.config[k] = v


def load_vae_diffusers(model_file, vae_file=None, vae_source="unknown-source"):
    if vae_file is None:
        return None
    if not os.path.exists(vae_file):
        shared.log.error(f'VAE not found: model{vae_file}')
        return None
    diffusers_load_config = {
        "low_cpu_mem_usage": False,
        "torch_dtype": devices.dtype_vae,
        "use_safetensors": True,
    }
    if shared.opts.diffusers_vae_load_variant == 'default':
        if devices.dtype_vae == torch.float16:
            diffusers_load_config['variant'] = 'fp16'
    elif shared.opts.diffusers_vae_load_variant == 'fp32':
        pass
    else:
        diffusers_load_config['variant'] = shared.opts.diffusers_vae_load_variant
    if shared.opts.diffusers_vae_upcast != 'default':
        diffusers_load_config['force_upcast'] = True if shared.opts.diffusers_vae_upcast == 'true' else False
    _pipeline, model_type = sd_detect.detect_pipeline(model_file, 'vae')
    vae_config = sd_detect.get_load_config(model_file, model_type, config_type='json')
    if vae_config is not None:
        diffusers_load_config['config'] = os.path.join(vae_config, 'vae')
    shared.log.info(f'Load module: type=VAE model="{vae_file}" source={vae_source} config={diffusers_load_config}')
    try:
        import diffusers
        if os.path.isfile(vae_file):
            if os.path.getsize(vae_file) > 1310944880: # 1.3GB
                vae = diffusers.ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder', **diffusers_load_config) # consistency decoder does not have from single file, so we'll just download it once more
            elif os.path.getsize(vae_file) < 10000000: # 10MB
                vae = diffusers.AutoencoderTiny.from_single_file(vae_file, **diffusers_load_config)
            else:
                vae = diffusers.AutoencoderKL.from_single_file(vae_file, **diffusers_load_config)
                if getattr(vae.config, 'scaling_factor', 0) == 0.18125 and shared.sd_model_type == 'sdxl':
                    vae.config.scaling_factor = 0.13025
                    shared.log.debug('Setting model: component=VAE fix scaling factor')
            vae = vae.to(devices.dtype_vae)
        else:
            if 'consistency-decoder' in vae_file:
                vae = diffusers.ConsistencyDecoderVAE.from_pretrained(vae_file, **diffusers_load_config)
            else:
                vae = diffusers.AutoencoderKL.from_pretrained(vae_file, **diffusers_load_config)
        global loaded_vae_file # pylint: disable=global-statement
        loaded_vae_file = os.path.basename(vae_file)
        # shared.log.debug(f'Diffusers VAE config: {vae.config}')
        if shared.opts.diffusers_offload_mode == 'none':
            sd_models.move_model(vae, devices.device)
        return vae
    except Exception as e:
        shared.log.error(f"Load VAE failed: model={vae_file} {e}")
        if debug:
            errors.display(e, 'VAE')
    return None


def reload_vae_weights(sd_model=None, vae_file=unspecified):
    if not sd_model:
        sd_model = shared.sd_model
    if sd_model is None:
        return None
    global checkpoint_info # pylint: disable=global-statement
    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename
    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file)
    else:
        vae_source = "function-argument"
    if vae_file is None or vae_file == 'None':
        if hasattr(sd_model, 'original_vae'):
            sd_models.set_diffuser_options(sd_model, vae=sd_model.original_vae, op='vae')
            shared.log.info("VAE restored")
            return None
    if loaded_vae_file == vae_file:
        return None

    if hasattr(sd_model, "vae") and getattr(sd_model, "sd_checkpoint_info", None) is not None:
        vae = load_vae_diffusers(sd_model.sd_checkpoint_info.filename, vae_file, vae_source)
        if vae is not None:
            if not hasattr(sd_model, 'original_vae'):
                sd_model.original_vae = sd_model.vae
                sd_models.move_model(sd_model.original_vae, devices.cpu)
            sd_models.set_diffuser_options(sd_model, vae=vae, op='vae')
            apply_vae_config(sd_model.sd_checkpoint_info.filename, vae_file, sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_models.move_model(sd_model, devices.device)
    return sd_model
