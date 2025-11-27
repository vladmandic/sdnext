import diffusers
from modules import shared, sd_models, sd_hijack_te, devices, model_quant
from pipelines import generic


def load_z_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    shared.log.debug(f'Load model: type=Z-Image repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.ZImageTransformer2DModel, load_config=diffusers_load_config)

    load_config, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.ZImagePipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        **load_config,
    )

    del transformer
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
