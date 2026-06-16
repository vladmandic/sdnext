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
    dtype: torch.dtype = torch.float32,
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
    return torch.tensor(betas, dtype=dtype)

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

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu", dtype: torch.dtype = torch.float32):
    ramp = np.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.from_numpy(sigmas).to(dtype=dtype, device=device)

def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu", dtype: torch.dtype = torch.float32):
    sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n))
    return torch.from_numpy(sigmas).to(dtype=dtype, device=device)

def get_sigmas_beta(n, sigma_min, sigma_max, alpha=0.6, beta=0.6, device="cpu", dtype: torch.dtype = torch.float32):
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
    return torch.from_numpy(sigmas).to(dtype=dtype, device=device)

def get_sigmas_flow(n, sigma_min, sigma_max, device="cpu", dtype: torch.dtype = torch.float32):
    # Linear flow sigmas
    sigmas = np.linspace(sigma_max, sigma_min, n)
    return torch.from_numpy(sigmas).to(dtype=dtype, device=device)

def apply_shift(sigmas, shift):
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def get_base_sigmas(alphas_cumprod: torch.Tensor) -> np.ndarray:
    """Return the standard diffusion sigmas derived from alphas_cumprod."""
    return np.array(((1 - alphas_cumprod.cpu().numpy()) / alphas_cumprod.cpu().numpy()) ** 0.5, dtype=np.float32)


def sigma_to_t(sigma: np.ndarray, log_sigmas: np.ndarray) -> np.ndarray:
    """Convert sigma values to corresponding discrete timesteps via interpolation."""
    sigma = np.array(sigma, dtype=np.float32)
    log_sigma = np.log(np.maximum(sigma, 1e-10))
    dists = log_sigma - log_sigmas[:, np.newaxis]
    low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1
    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]
    w = (low - log_sigma) / (low - high)
    w = np.clip(w, 0, 1)
    t = (1 - w) * low_idx + w * high_idx
    return t.reshape(sigma.shape).astype(np.float32)


def get_dynamic_shift(mu, base_shift, max_shift, base_seq_len, max_seq_len):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return m * mu + b

def index_for_timestep(timestep, timesteps):
    # Normalize inputs to numpy arrays for a robust, device-agnostic argmin
    if isinstance(timestep, torch.Tensor):
        timestep_np = timestep.detach().cpu().numpy()
    else:
        timestep_np = np.array(timestep)

    if isinstance(timesteps, torch.Tensor):
        timesteps_np = timesteps.detach().cpu().numpy()
    else:
        timesteps_np = np.array(timesteps)

    # Use numpy argmin on absolute difference for stability
    idx = np.abs(timesteps_np - timestep_np).argmin()
    return int(idx)

def add_noise_to_sample(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    timestep: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    step_index = index_for_timestep(timestep, timesteps)
    sigma = sigmas[step_index].to(original_samples.dtype)

    noisy_samples = original_samples + sigma * noise
    return noisy_samples


def validate_custom_schedule_args(timesteps, sigmas):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` should be set.")
    if timesteps is None and sigmas is None:
        raise ValueError("Must pass exactly one of `timesteps` or `sigmas`.")


def prepare_res4lyf_timesteps_and_sigmas(
    config,
    alphas_cumprod: torch.Tensor,
    num_inference_steps: int | None = None,
    timesteps=None,
    sigmas=None,
    device: str | torch.device = None,
    dtype: torch.dtype = torch.float32,
):
    validate_custom_schedule_args(timesteps, sigmas)

    base_sigmas = get_base_sigmas(alphas_cumprod)

    if timesteps is not None:
        if getattr(config, "use_karras_sigmas", False) or getattr(config, "use_exponential_sigmas", False) or getattr(config, "use_beta_sigmas", False):
            raise ValueError("Cannot set `timesteps` when karras/exponential/beta sigmas are enabled.")
        if getattr(config, "use_flow_sigmas", False):
            raise ValueError("Cannot set `timesteps` when `use_flow_sigmas` is enabled.")

        timesteps_array = np.array(timesteps, dtype=np.float32)
        sigmas_array = np.interp(timesteps_array, np.arange(len(base_sigmas), dtype=np.float32), base_sigmas)
        num_inference_steps = len(timesteps_array)
        sigmas_array = np.concatenate([sigmas_array, [0.0]])
        return num_inference_steps, timesteps_array, sigmas_array

    sigmas_array = np.array(sigmas, dtype=np.float32)
    if num_inference_steps is None:
        num_inference_steps = len(sigmas_array) - 1
    log_sigmas = np.log(base_sigmas)
    timesteps_array = np.array([sigma_to_t(sigma, log_sigmas) for sigma in sigmas_array[:-1]], dtype=np.float32)
    return num_inference_steps, timesteps_array, sigmas_array
