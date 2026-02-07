import os
from modules import shared, devices, files_cache, sd_models, model_quant


unet_dict = {}
loaded_unet = None
failed_unet = []
debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


dit_models = ['Flux', 'StableDiffusion3', 'HiDream', 'Lumina2', 'Chroma', 'Wan', 'Qwen']


def load_unet_sdxl_nunchaku(repo_id):
    try:
        from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
    except Exception:
        shared.log.error(f'Load module: quant=Nunchaku module=unet repo="{repo_id}" low nunchaku version')
        return None
    if 'turbo' in repo_id.lower():
        nunchaku_repo = 'nunchaku-ai/nunchaku-sdxl-turbo/svdq-int4_r32-sdxl-turbo.safetensors'
    else:
        nunchaku_repo = 'nunchaku-ai/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors'

    if shared.opts.nunchaku_offload:
        shared.log.warning('Load module: quant=Nunchaku module=unet offload not supported for SDXL, ignoring')
    shared.log.debug(f'Load module: quant=Nunchaku module=unet repo="{nunchaku_repo}"')
    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        nunchaku_repo,
        torch_dtype=devices.dtype,
        cache_dir=shared.opts.hfcache_dir,
    )
    unet.quantization_method = 'SVDQuant'
    return unet


def load_unet(model, repo_id:str=None):
    global loaded_unet # pylint: disable=global-statement

    if ("StableDiffusionXLPipeline" in model.__class__.__name__) and (('stable-diffusion-xl-base' in repo_id) or ('sdxl-turbo' in repo_id)):
        if model_quant.check_nunchaku('Model'):
            unet = load_unet_sdxl_nunchaku(repo_id)
            if unet is not None:
                model.unet = unet
                return

    if shared.opts.sd_unet == 'Default' or shared.opts.sd_unet == 'None':
        return

    if shared.opts.sd_unet not in list(unet_dict):
        shared.log.error(f'Load module: type=UNet not found: {shared.opts.sd_unet}')
        return

    config_file = os.path.splitext(unet_dict[shared.opts.sd_unet])[0] + '.json'
    if os.path.exists(config_file):
        config = shared.readfile(config_file, as_type="dict")
    else:
        config = None
        config_file = 'default'

    try:
        if shared.opts.sd_unet == loaded_unet or shared.opts.sd_unet in failed_unet:
            pass
        elif "StableCascade" in model.__class__.__name__:
            from pipelines.model_stablecascade import load_prior
            prior_unet, prior_text_encoder = load_prior(unet_dict[shared.opts.sd_unet], config_file=config_file)
            loaded_unet = shared.opts.sd_unet
            if prior_unet is not None:
                model.prior_pipe.prior = None # Prevent OOM
                model.prior_pipe.prior = prior_unet.to(devices.device, dtype=devices.dtype_unet)
            if prior_text_encoder is not None:
                model.prior_pipe.text_encoder = None # Prevent OOM
                model.prior_pipe.text_encoder = prior_text_encoder.to(devices.device, dtype=devices.dtype)
        elif any([m in model.__class__.__name__ for m in dit_models]) or hasattr(model, 'transformer'): # noqa: C419 # pylint: disable=use-a-generator
            loaded_unet = shared.opts.sd_unet
            sd_models.load_diffuser() # TODO model load: force-reloading entire model as loading transformers only leads to massive memory usage
        else:
            if not hasattr(model, 'unet') or model.unet is None:
                shared.log.error('Load module: type=UNET not found in current model')
                return
            shared.log.info(f'Load module: type=UNet name="{shared.opts.sd_unet}" file="{unet_dict[shared.opts.sd_unet]}" config="{config_file}"')
            from diffusers import UNet2DConditionModel
            from safetensors.torch import load_file
            unet = UNet2DConditionModel.from_config(model.unet.config if config is None else config).to(devices.device, devices.dtype)
            state_dict = load_file(unet_dict[shared.opts.sd_unet])
            unet.load_state_dict(state_dict)
            model.unet = unet.to(devices.device, devices.dtype_unet)
    except Exception as e:
        shared.log.error(f'Failed to load UNet model: {e}')
        if debug:
            from modules import errors
            errors.display(e, 'UNet load:')
        return
    devices.torch_gc()


def refresh_unet_list():
    unet_dict.clear()
    for file in files_cache.list_files(shared.opts.unet_dir, ext_filter=[".safetensors", ".gguf", ".pth"]):
        basename = os.path.basename(file)
        name = os.path.splitext(basename)[0] if ".safetensors" in basename else basename
        unet_dict[name] = file
    shared.log.info(f'Available UNets: path="{shared.opts.unet_dir}" items={len(unet_dict)}')
