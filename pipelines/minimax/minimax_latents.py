import diffusers


def unpack_latents(latents, components: diffusers.modular_pipelines.ModularPipeline, state: diffusers.modular_pipelines.BlockState):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames, video_latent_num_frames
    from modules import processing_callbacks
    frames = getattr(processing_callbacks.p, 'frames', 1)
    width = getattr(processing_callbacks.p, 'width', 1024)
    height = getattr(processing_callbacks.p, 'height', 1024)
    if frames <= 0 or width <= 0 or height <= 0:
        return latents
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
    return latents
