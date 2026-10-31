import diffusers


def unpack_latents(latents, components: diffusers.modular_pipelines.ModularPipeline, state: diffusers.modular_pipelines.BlockState):
    # based on methods from diffusers/modular_pipelines/flux3/decoders.py
    from modules import processing_callbacks, processing_helpers
    width, height = processing_helpers.pass_size(processing_callbacks.p) # a hires pass runs at its own size, not the request's

    # unpack: (batch_size, latent_height * latent_width, num_channels) -> (batch_size, num_channels, latent_height, latent_width)
    latent_width = width // (components.vae_scale_factor * 2)
    latent_height = height // (components.vae_scale_factor * 2)
    batch_size, _, num_channels = state.latents.shape
    unpacked_latents = latents.view(batch_size, latent_height, latent_width, num_channels).permute(0, 3, 1, 2)

    # unpatchify: (batch_size, num_channels, latent_height, latent_width) -> (batch_size, num_channels // 4, latent_height * 2, latent_width * 2)
    batch_size, num_channels_latents, height, width = unpacked_latents.shape
    unpatched_latents = unpacked_latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    unpatched_latents = unpatched_latents.permute(0, 1, 4, 2, 5, 3)
    unpatched_latents = unpatched_latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)

    return unpatched_latents
