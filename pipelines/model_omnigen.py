import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te


def load_omnigen(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    repo_id = sd_models.path_to_repo(checkpoint_info)
    vae = None

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='Model')
    transformer = diffusers.OmniGenTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    if vae is not None:
        load_config['vae'] = vae
    pipe = diffusers.OmniGenPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
    )

    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
