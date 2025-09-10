from modules import shared, sd_models, devices, model_quant, errors # pylint: disable=unused-import


def load_hyimage(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    shared.log.error(f'Load model: type=NextStep model="{checkpoint_info.name}" repo="{repo_id}" not supported')

    """
    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    shared.log.debug(f'Load model: type=HunyuanImage repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    sys.path.append(os.path.dirname(__file__))
    from pipelines.hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline, HunyuanImagePipelineConfig
    from pipelines.hyimage.common.config import instantiate

    use_distilled = 'distilled' in repo_id.lower()
    config = HunyuanImagePipelineConfig.create_default(version='v2.1', use_distilled=use_distilled)
    config.torch_dtype = devices.dtype
    config.device = devices.device
    config.enable_dit_offloading = False
    config.enable_reprompt_model_offloading = False
    config.enable_refiner_offloading = False

    pipe = HunyuanImagePipeline(config=config)

    snapshot_folder = snapshot_download(repo_id, cache_dir=shared.opts.hfcache_dir, allow_patterns='vae/vae_2_1/*')
    pipe.config.vae_config.load_from = os.path.join(snapshot_folder, 'vae/vae_2_1')
    pipe.vae = instantiate(pipe.config.vae_config.model, vae_path=pipe.config.vae_config.load_from)

    pipe._load_dit() # pylint: disable=protected-access
    pipe._load_byt5() # pylint: disable=protected-access
    pipe._load_text_encoder() # pylint: disable=protected-access
    pipe._load_reprompt_model() # pylint: disable=protected-access

    pipe.task_args = {
        'use_reprompt': False,
        'use_refiner': False,
    }

    devices.torch_gc(force=True, reason='load')
    """
    return None
