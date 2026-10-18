import diffusers
from modules import devices, sd_models, shared, sd_hijack_vae, sd_hijack_te
from modules.logger import log
from pipelines import generic


class MingImageModularPipeline(diffusers.ModularPipeline): # empty pipeline so it has its own class
    pass


def load_ming_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    generic.set_pipeline('MingImage', MingImageModularPipeline)

    if repo_id is None or repo_id.lower() == 'none':
        return None
    sd_models.hf_auth_check(checkpoint_info)

    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    log.debug(f'Load model: type=MingImage repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')
    sd_models.warn_group_offload(min_vram=20)

    from modules.modular_load import load_modular_pipe, missing_components
    from pipelines.ming.modular_blocks_ming_image import MingImageBlocks
    blocks = MingImageBlocks(components_repo=repo_id)
    pipe = load_modular_pipe(
        MingImageModularPipeline,
        repo_id,
        blocks=blocks,
        offline_args=offline_args,
        base=True,
        load_config=diffusers_load_config,
        trust_remote_code=False,
        load_repo_config=False,
    )
    if pipe is None:
        return None

    missing = missing_components(pipe, workflow=None)
    if missing:
        raise RuntimeError(f'Ming-Image components failed to load: {missing}')
    pipe.task_args = {'output_type': 'np'} # pylint: disable=attribute-defined-outside-init
    pipe.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    del blocks
    devices.torch_gc(force=True, reason='load')
    return pipe
