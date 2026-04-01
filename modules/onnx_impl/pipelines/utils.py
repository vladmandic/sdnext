import numpy as np
import torch


def extract_generator_seed(generator: torch.Generator | list[torch.Generator]) -> list[int]:
    if isinstance(generator, list):
        generator = [g.seed() for g in generator]
    else:
        generator = [generator.seed()]
    return generator


def randn_tensor(shape, dtype: np.dtype, generator: torch.Generator | list[torch.Generator] | int | list[int]):
    if hasattr(generator, "seed") or (isinstance(generator, list) and hasattr(generator[0], "seed")):
        generator = extract_generator_seed(generator)
        if len(generator) == 1:
            generator = generator[0]
    return np.random.default_rng(generator).standard_normal(shape).astype(dtype)


def prepare_latents(
    init_noise_sigma: float,
    batch_size: int,
    height: int,
    width: int,
    dtype: np.dtype,
    generator: torch.Generator | list[torch.Generator],
    latents: np.ndarray | None = None,
    num_channels_latents = 4,
    vae_scale_factor = 8,
):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)

    if latents is None:
        latents = randn_tensor(shape, dtype, generator)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * np.float64(init_noise_sigma)

    return latents
