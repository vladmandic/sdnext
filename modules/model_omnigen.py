import os
import diffusers
from modules import errors, shared, devices, sd_models, model_quant

debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_omnigen(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
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

    devices.torch_gc(force=True)
    return pipe
