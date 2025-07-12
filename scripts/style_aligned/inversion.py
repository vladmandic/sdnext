# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from diffusers import StableDiffusionXLPipeline
import torch
from tqdm import tqdm
if TYPE_CHECKING:
    import numpy as np


T = torch.Tensor
TN = T
InversionCallback = Callable[[StableDiffusionXLPipeline, int, T, dict[str, T]], dict[str, T]]


def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
    # Tokenize text and get embeddings
    text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    if prompt == '':
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        return negative_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds


def _encode_text_sdxl(model: StableDiffusionXLPipeline, prompt: str) -> tuple[dict[str, T], T]:
    device = model._execution_device # pylint: disable=protected-access
    prompt_embeds, pooled_prompt_embeds, = _get_text_embeddings(prompt, model.tokenizer, model.text_encoder, device) # pylint: disable=unused-variable
    prompt_embeds_2, pooled_prompt_embeds2, = _get_text_embeddings( prompt, model.tokenizer_2, model.text_encoder_2, device)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
    text_encoder_projection_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), model.text_encoder.dtype, # pylint: disable=protected-access
                                           text_encoder_projection_dim).to(device)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds2, "time_ids": add_time_ids}
    return added_cond_kwargs, prompt_embeds


def _encode_text_sdxl_with_negative(model: StableDiffusionXLPipeline, prompt: str) -> tuple[dict[str, T], T]:
    added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(model, "")
    prompt_embeds = torch.cat((prompt_embeds_uncond, prompt_embeds, ))
    added_cond_kwargs = {"text_embeds": torch.cat((added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])),
                         "time_ids": torch.cat((added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])),}
    return added_cond_kwargs, prompt_embeds


def _encode_image(model: StableDiffusionXLPipeline, image: np.ndarray) -> T:
    image = torch.from_numpy(image).float() / 255.
    image = (image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
    latent = model.vae.encode(image.to(model.vae.device, model.vae.dtype))['latent_dist'].mean * model.vae.config.scaling_factor
    return latent


def _next_step(model: StableDiffusionXLPipeline, model_output: T, timestep: int, sample: T) -> T:
    timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def _get_noise_pred(model: StableDiffusionXLPipeline, latent: T, t: T, context: T, guidance_scale: float, added_cond_kwargs: dict[str, T]):
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    # latents = next_step(model, noise_pred, t, latent)
    return noise_pred


def _ddim_loop(model: StableDiffusionXLPipeline, z0, prompt, guidance_scale) -> T:
    all_latent = [z0]
    added_cond_kwargs, text_embedding = _encode_text_sdxl_with_negative(model, prompt)
    latent = z0.clone().detach().to(model.text_encoder.dtype)
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = _get_noise_pred(model, latent, t, text_embedding, guidance_scale, added_cond_kwargs)
        latent = _next_step(model, noise_pred, t, latent)
        all_latent.append(latent)
    return torch.cat(all_latent).flip(0)


def make_inversion_callback(zts, offset: int = 0):

    def callback_on_step_end(pipeline: StableDiffusionXLPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]: # pylint: disable=unused-argument
        latents = callback_kwargs['latents']
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {'latents': latents}
    return  zts[offset], callback_on_step_end


@torch.no_grad()
def ddim_inversion(model: StableDiffusionXLPipeline, x0: np.ndarray, prompt: str, num_inference_steps: int, guidance_scale,) -> T:
    z0 = _encode_image(model, x0)
    model.scheduler.set_timesteps(num_inference_steps, device=z0.device)
    zs = _ddim_loop(model, z0, prompt, guidance_scale)
    return zs
