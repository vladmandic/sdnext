import diffusers
from modules import shared, devices, sd_models
from modules.logger import log


def load_minimax(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    from modules.video_models import video_modular, video_load
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    if repo_id is None or repo_id.lower() == 'none':
        return None
    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    log.debug(f'Load model: type=MiniMaxH3 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    pipe = video_modular.load_modular_pipe(
        getattr(diffusers, 'MiniMaxH3ModularPipeline', None),
        repo_id,
        workflow='fl2va',
        offline_args=offline_args,
        base=True,
    )
    if pipe is None:
        return None

    video_modular.install_state_hook(pipe)
    video_load.loaded_model = None # image-path load invalidates the video tab's name cache
    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()

    devices.torch_gc()
    return pipe
