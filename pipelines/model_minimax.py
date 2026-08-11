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
    workflow = (getattr(checkpoint_info, 'subfolder', None) or 'fl2va').lower() # one repo holds both checkpoint partitions; reference entries select ref2va via the subfolder tag
    log.debug(f'Load model: type=MiniMaxH3 repo="{repo_id}" workflow={workflow} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    repo_cls = diffusers.MiniMaxH3ModularPipeline
    pipe = video_modular.load_modular_pipe(
        repo_cls,
        repo_id,
        workflow=workflow,
        offline_args=offline_args,
        base=True,
    )
    if pipe is None:
        return None
    if pipe.text_encoder is None:
        # TODO minimax missing te: we should never be here
        import transformers
        from pipelines import generic
        text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3VLForConditionalGeneration, load_config=diffusers_load_config, allow_shared=False)
        pipe.update_components(text_encoder=text_encoder)

    video_modular.install_state_hook(pipe)
    video_load.loaded_model = None # image-path load invalidates the video tab's name cache
    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()

    devices.torch_gc()
    return pipe
