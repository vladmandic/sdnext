import diffusers


def load_pixart(checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader, model_te
    modelloader.hf_login()
    # shared.opts.data['cuda_dtype'] = 'FP32' # override
    # shared.opts.data['diffusers_offload_mode}'] = "model" # override
    # devices.set_cuda_params()
    fn = checkpoint_info.path.replace('huggingface/', '')
    t5 = model_te.load_t5(name=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    transformer = diffusers.PixArtTransformer2DModel.from_pretrained(
        fn,
        subfolder = 'transformer',
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    transformer.to(devices.device)
    kwargs = { 'transformer': transformer }
    if t5 is not None:
        kwargs['text_encoder'] = t5
    diffusers_load_config.pop('variant', None)
    pipe = diffusers.PixArtSigmaPipeline.from_pretrained(
        'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        cache_dir = shared.opts.diffusers_dir,
        **kwargs,
        **diffusers_load_config,
    )
    devices.torch_gc(force=True)
    return pipe
