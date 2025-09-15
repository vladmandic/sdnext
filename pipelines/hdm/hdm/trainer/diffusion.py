import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from diffusers import EulerDiscreteScheduler


def get_noise_noisy_latents_and_timesteps(
    noise_scheduler: EulerDiscreteScheduler, latents
):
    noise = torch.randn_like(latents, device=latents.device)
    b_size = latents.shape[0]
    min_timestep = 0
    max_timestep = noise_scheduler.config.num_train_timesteps

    timesteps = torch.randint(
        min_timestep, max_timestep, (b_size,), device=latents.device
    )

    sigmas = noise_scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(latents.device)
    timesteps = timesteps.to(latents.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(latents.shape):
        sigma = sigma.unsqueeze(-1)

    # Diffusion Forward process
    noisy_samples = latents + noise * sigma
    scale = 1 / (sigma**2 + 1) ** 0.5
    return noisy_samples * scale, noise, timesteps


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss


def apply_debiased_estimation(loss, timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(
        snr_t, torch.ones_like(snr_t) * 1000
    )  # if timestep is 0, snr_t is inf, so limit it to 1000
    weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)
