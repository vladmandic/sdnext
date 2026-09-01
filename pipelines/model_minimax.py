import diffusers
from modules import shared, devices, sd_models
from modules.logger import log


def load_minimax(checkpoint_info, diffusers_load_config = None, workflow: str | None = None):
    from modules.video_models import video_load
    from modules.modular_load import load_modular_pipe
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    workflow = (workflow or getattr(checkpoint_info, 'subfolder', None) or 'fl2va').lower() # one repo holds both checkpoint partitions; reference entries select ref2va via the subfolder tag
    log.debug(f'Load model: type=MiniMaxH3 repo="{repo_id}" workflow={workflow} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    sd_models.warn_group_offload(min_vram=20)
    repo_cls = diffusers.MiniMaxH3ModularPipeline
    pipe = load_modular_pipe(
        repo_cls,
        repo_id,
        workflow=workflow,
        offline_args=offline_args,
        base=True,
        load_config=diffusers_load_config,
    )
    if pipe is None:
        return None

    pipe.sd_checkpoint_info = checkpoint_info
    pipe.sdnext_force_offload = True # very large model, so each stage ends with an on-demand offload sweep
    if hasattr(pipe, 'min_duration') and hasattr(pipe, 'fps'):
        pipe.sdnext_supported_min_frames = int(pipe.min_duration * pipe.fps) # fresh pipes report the true floor; still mode gates per instance

    video_load.loaded_model = None # image-path load invalidates the video tab's name cache
    # if hasattr(pipe, 'vae'):
    #    pipe.vae = pipe.vae.to(torch.float16) # minimax loads vae in float32
    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()

    from pipelines.minimax.minimax_latents import unpack_latents
    pipe.custom_unpack_latents = unpack_latents # add a helper to unpack the video latents from the block state
    devices.torch_gc()
    return pipe
