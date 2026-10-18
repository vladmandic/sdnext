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

    log.debug(f'Load model: type=Flux3 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')
    if repo_id is None or repo_id.lower() == 'none':
        return None

    from pipelines.flux3 import FLUX3_SPEC
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Flux3Transformer2DModel, load_config=diffusers_load_config, native_spec=FLUX3_SPEC)
    if transformer is None:
        return None
    # the step-distilled checkpoints are flow maps: their blocks tree passes the transformer the timestep each step jumps to, and the transformer rejects the other tree
    if getattr(transformer.config, 'flow_map', False):
        repo_cls, blocks = diffusers.Flux3DistilledModularPipeline, diffusers.Flux3DistilledAutoBlocks()
    else:
        repo_cls, blocks = diffusers.Flux3ModularPipeline, diffusers.Flux3AutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler") # do not load 9b upsampler by default
    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    pipe = load_modular_pipe(
        repo_cls,
        repo_id,
        blocks=blocks,
        offline_args=offline_args,
        base=True,
        load_config=diffusers_load_config,
        loaded={'transformer': transformer},
    )
    del transformer
    if pipe is None:
        return None

    from pipelines.flux3 import unpack_latents
    pipe.custom_unpack_latents = unpack_latents # add a helper to unpack the video latents from the block state
    pipe.sd_checkpoint_info = checkpoint_info
    pipe.patch_size = 2 # latents pack into 2x2 patches, so sizes are multiples of vae_scale_factor * 2
    pipe.max_condition_images = 10 # references keep their own resolution, the request sets the output size
    pipe.single_prompt = True # the text encoder step takes one prompt per call, the batch comes from num_images_per_prompt
    devices.torch_gc(force=True, reason='load')
    return pipe
