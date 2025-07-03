import random
from os import environ
import numpy as np
import torch


JPEG_QUALITY = 100


def seed_everything(seed):
    random.seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exists(x):
    return x is not None


def get(x, default):
    if exists(x):
        return x
    return default


def get_self_recurrence_schedule(schedule, num_inference_steps):
    self_recurrence_schedule = [0] * num_inference_steps
    for schedule_current in reversed(schedule):
        if schedule_current is None or len(schedule_current) == 0:
            continue
        [start, end, repeat] = schedule_current
        start_i = round(num_inference_steps * start)
        end_i = round(num_inference_steps * end)
        for i in range(start_i, end_i):
            self_recurrence_schedule[i] = repeat
    return self_recurrence_schedule


def batch_dict_to_tensor(batch_dict, batch_order):
    batch_tensor = []
    for batch_type in batch_order:
        batch_tensor.append(batch_dict[batch_type])
    batch_tensor = torch.cat(batch_tensor, dim=0)
    return batch_tensor


def batch_tensor_to_dict(batch_tensor, batch_order):
    batch_tensor_chunk = batch_tensor.chunk(len(batch_order))
    batch_dict = {}
    for i, batch_type in enumerate(batch_order):
        batch_dict[batch_type] = batch_tensor_chunk[i]
    return batch_dict


def noise_prev(scheduler, timestep, x_0, noise=None):
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if noise is None:
        noise = torch.randn_like(x_0).to(x_0)

    # From DDIMScheduler step function (hopefully this works)
    timestep_i = (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0][0].item()
    if timestep_i + 1 >= scheduler.timesteps.shape[0]:  # We are at t = 0 (ish)
        return x_0
    prev_timestep = scheduler.timesteps[timestep_i + 1:timestep_i + 2]  # Make sure t is not 0-dim

    x_t_prev = scheduler.add_noise(x_0, noise, prev_timestep)
    return x_t_prev


def noise_t2t(scheduler, timestep, timestep_target, x_t, noise=None):
    assert timestep_target >= timestep
    if noise is None:
        noise = torch.randn_like(x_t).to(x_t)

    alphas_cumprod = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)

    timestep = timestep.to(torch.long)
    timestep_target = timestep_target.to(torch.long)

    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_tt = alphas_cumprod[timestep_target]
    alpha_prod = alpha_prod_tt / alpha_prod_t

    sqrt_alpha_prod = (alpha_prod ** 0.5).flatten()
    while len(sqrt_alpha_prod.shape) < len(x_t.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = ((1 - alpha_prod) ** 0.5).flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    x_tt = sqrt_alpha_prod * x_t + sqrt_one_minus_alpha_prod * noise
    return x_tt
