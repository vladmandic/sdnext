import diffusers
from modules import shared, devices, sd_models
from modules.logger import log
from pipelines import generic


def load_flux3(checkpoint_info, diffusers_load_config=None):
    from modules.modular_load import load_modular_pipe
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    log.debug(f'Load model: type=Flux3 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Flux3Transformer2DModel, load_config=diffusers_load_config)
    if transformer is None:
        return None
    # the step-distilled checkpoints are flow maps: their pipeline passes the transformer the timestep each step jumps to
    repo_cls = diffusers.Flux3DistilledModularPipeline if getattr(transformer.config, 'flow_map', False) else diffusers.Flux3ModularPipeline
    pipe = load_modular_pipe(
        repo_cls,
        repo_id,
        offline_args=offline_args,
        base=True,
        load_config=diffusers_load_config,
        loaded={'transformer': transformer},
    )
    del transformer
    if pipe is None:
        return None

    pipe.sd_checkpoint_info = checkpoint_info
    pipe.patch_size = 2 # latents pack into 2x2 patches, so sizes are multiples of vae_scale_factor * 2
    pipe.max_condition_images = 10 # references keep their own resolution, the request sets the output size
    devices.torch_gc(force=True, reason='load')
    return pipe
