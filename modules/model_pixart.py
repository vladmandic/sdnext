import transformers
import diffusers


def load_pixart(checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader, sd_models, model_quant
    modelloader.hf_login()
    repo_id = sd_models.path_to_repo(checkpoint_info.name)

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Transformer')
    transformer = diffusers.PixArtTransformer2DModel.from_pretrained(
        repo_id,
        subfolder='transformer',
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    text_encoder = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.PixArtSigmaPipeline.from_pretrained(
        'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        **load_args,
    )
    devices.torch_gc(force=True)
    return pipe
