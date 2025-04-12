def load_omnigen(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    from modules import shared, devices, sd_models, shared_items
    repo_id = sd_models.path_to_repo(checkpoint_info.name)

    # load
    from modules.omnigen import OmniGenPipeline
    shared_items.pipelines['OmniGen'] = OmniGenPipeline
    pipe = OmniGenPipeline.from_pretrained(
        model_name=repo_id,
        vae_path='madebyollin/sdxl-vae-fp16-fix',
        cache_dir=shared.opts.diffusers_dir,
    )

    # init
    pipe.device = devices.device
    pipe.dtype = devices.dtype
    pipe.model.device = devices.device
    pipe.separate_cfg_infer = True
    pipe.use_kv_cache = False
    pipe.model.to(device=devices.device, dtype=devices.dtype)
    if shared.opts.diffusers_eval:
        pipe.model.eval()
    pipe.vae.to(devices.device, dtype=devices.dtype)
    devices.torch_gc(force=True)
    return pipe
