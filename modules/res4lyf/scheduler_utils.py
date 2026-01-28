import math
from typing import Literal

import numpy as np
import torch

try:
    import scipy.stats
    _scipy_available = True
except ImportError:
    _scipy_available = False

def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: Literal["cosine", "exp", "laplace"] = "cosine",
) -> torch.Tensor:
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "laplace":
        def alpha_bar_fn(t):
            lmb = -0.5 * math.copysign(1, 0.5 - t) * math.log(1 - 2 * math.fabs(0.5 - t) + 1e-6)
            snr = math.exp(lmb)
            return math.sqrt(snr / (1 + snr))
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    ramp = np.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n))
    return torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

def get_sigmas_beta(n, sigma_min, sigma_max, alpha=0.6, beta=0.6, device="cpu"):
    if not _scipy_available:
        raise ImportError("scipy is required for beta sigmas")
    sigmas = np.array(
        [
            sigma_min + (ppf * (sigma_max - sigma_min))
            for ppf in [
                scipy.stats.beta.ppf(timestep, alpha, beta)
                for timestep in 1 - np.linspace(0, 1, n)
            ]
        ]
    )
    return torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

def get_sigmas_flow(n, sigma_min, sigma_max, device="cpu"):
    # Linear flow sigmas
    sigmas = np.linspace(sigma_max, sigma_min, n)
    return torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

def apply_shift(sigmas, shift):
    return shift * sigmas / (1 + (shift - 1) * sigmas)

def get_dynamic_shift(mu, base_shift, max_shift, base_seq_len, max_seq_len):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return m * mu + b

def index_for_timestep(timestep, timesteps):
    index_candidates = (timesteps == timestep).nonzero()

    if len(index_candidates) == 0:
        step_index = len(timesteps) - 1
    elif len(index_candidates) > 1:
        step_index = index_candidates[0].item()
    else:
        step_index = index_candidates.item()

    return step_index

def add_noise_to_sample(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    timestep: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    step_index = index_for_timestep(timestep, timesteps)
    sigma = sigmas[step_index]

    noisy_samples = original_samples + sigma * noise
    return noisy_samples
