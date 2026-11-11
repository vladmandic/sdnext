import diffusers
from modules.logger import log


warned = False


def unpack_latents(latents, components: diffusers.modular_pipelines.ModularPipeline, state: diffusers.modular_pipelines.BlockState):
    global warned # pylint: disable=global-statement
    if warned:
        return latents
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames, video_latent_num_frames
    from modules import processing_callbacks
    frames = getattr(processing_callbacks.p, 'frames', 1)
    width = getattr(processing_callbacks.p, 'width', 1024)
    height = getattr(processing_callbacks.p, 'height', 1024)
    if frames <= 0 or width <= 0 or height <= 0:
        return latents
    try:
        num_frames = align_num_frames(frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk)
        num_latent_frames = video_latent_num_frames(num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk)
        latent_height = height // components.vae_spatial_compression_ratio
        latent_width = width // components.vae_spatial_compression_ratio
        patch_t, patch_h, patch_w = components.patch_size
        channels = components.vae_latent_channels
        rows = state.latents[state.num_condition_video_rows :]
        rows = rows.reshape(
            -1,
            num_latent_frames // patch_t,
            latent_height // patch_h,
            latent_width // patch_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
        latents = rows.reshape(
            -1,
            channels,
            num_latent_frames,
            latent_height,
            latent_width,
        ).contiguous()
    except Exception as e:
        # fails with sliced attention due to shape mismatch as state.latents only contains a subset of the full video latents
        if not warned:
            warned = True
            log.warning(f'Video unpack latents: {e}')
    return latents


def unpack_image_latents(latents, components: diffusers.modular_pipelines.ModularPipeline, state: diffusers.modular_pipelines.BlockState):
    # based on MiniMaxH3ImageUnpackLatentsStep in diffusers/modular_pipelines/minimax_h3_image/decoders.py
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import MiniMaxH3ImagePrepareCanvasStep, get_image_canvas_size
    from modules import processing_callbacks, processing_helpers
    width, height = processing_helpers.pass_size(processing_callbacks.p)
    area = getattr(components, 'task_args', {}).get('min_generation_area', None)
    if area is None:
        area = next(param.default for param in MiniMaxH3ImagePrepareCanvasStep().inputs if param.name == 'min_generation_area')
    canvas_height, canvas_width = get_image_canvas_size(height, width, area, components.canvas_multiple) # a request below the area floor is generated on a larger canvas
    latent_height = canvas_height // components.vae_spatial_compression_ratio
    latent_width = canvas_width // components.vae_spatial_compression_ratio
    _, patch_h, patch_w = components.patch_size
    channels = components.vae_latent_channels
    rows = latents[state.num_reference_rows:] # reference rows lead the packed sequence
    rows = rows.reshape(-1, latent_height // patch_h, latent_width // patch_w, channels, patch_h, patch_w)
    return rows.permute(0, 3, 1, 4, 2, 5).reshape(-1, channels, latent_height, latent_width).contiguous()
