import torch
import torch.utils.checkpoint


class PFODESolver():
    def __init__(self, scheduler, t_initial=1, t_terminal=0,) -> None:
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps # 0+1000
        self.stepsize  = (t_terminal-t_initial) / (train_step_terminal - train_step_initial) #1/1000

    def get_timesteps(self, t_start, t_end, num_steps):
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2

        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints

        timesteps = (self.scheduler.num_train_timesteps - 1) + (timepoints - self.t_initial) / self.stepsize # correspondint to StableDiffusion indexing system, from 999 (t_init) -> 0 (dt)
        return timesteps.round().long()
        # return timesteps.floor().long()

    def solve(self,
              latents,
              unet,
              t_start,
              t_end,
              prompt_embeds,
              negative_prompt_embeds,
              guidance_scale=1.0,
              num_steps = 2,
              num_windows = 1,
    ):
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))

        do_classifier_free_guidance = True if guidance_scale > 1 else False
        bsz = latents.shape[0]

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding( # pylint: disable=no-member
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # 7. Denoising loop
        with torch.no_grad():
            # for i in tqdm(range(num_steps)):
            for i in range(num_steps):

                t = torch.cat([timesteps[:, i]]*2) if do_classifier_free_guidance else timesteps[:, i]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # STEP: compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, timesteps[:, i].cpu(), latents, return_dict=False)[0]

                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval
                # prev_timestep = batch_timesteps - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]):
                    alpha_prod_t_prev[ib] = self.scheduler.alphas_cumprod[prev_timestep[ib]] if prev_timestep[ib] >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t

                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:,None,None,None] ** (0.5) * noise_pred) / alpha_prod_t[:, None,None,None] ** (0.5)
                    pred_epsilon = noise_pred
                # elif self.scheduler.config.prediction_type == "sample":
                #     pred_original_sample = noise_pred
                #     pred_epsilon = (latents - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t[:,None,None,None]**0.5) * latents - (beta_prod_t[:,None,None,None]**0.5) * noise_pred
                    pred_epsilon = (alpha_prod_t[:,None,None,None]**0.5) * noise_pred + (beta_prod_t[:,None,None,None]**0.5) * latents
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )
                pred_sample_direction = (1 - alpha_prod_t_prev[:,None,None,None]) ** (0.5) * pred_epsilon
                latents = alpha_prod_t_prev[:,None,None,None] ** (0.5) * pred_original_sample + pred_sample_direction

        return latents


class PFODESolverSDXL():
    def __init__(self, scheduler, t_initial=1, t_terminal=0,) -> None:
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps # 0+1000

        self.stepsize  = (t_terminal-t_initial) / (train_step_terminal - train_step_initial) #1/1000

    def get_timesteps(self, t_start, t_end, num_steps):
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2

        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints

        timesteps = (self.scheduler.num_train_timesteps - 1) + (timepoints - self.t_initial) / self.stepsize # correspondint to StableDiffusion indexing system, from 999 (t_init) -> 0 (dt)
        return timesteps.round().long()
        # return timesteps.floor().long()

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def solve(self,
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            guidance_scale=1.0,
            num_steps = 10,
            num_windows = 4,
            resolution = 1024,
    ):
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))
        dtype = latents.dtype
        device = latents.device
        bsz = latents.shape[0]
        do_classifier_free_guidance = True if guidance_scale > 1 else False

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = torch.cat(
            # [self._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), dtype) for _ in range(bsz)]
            [self._get_add_time_ids((resolution, resolution), (0, 0), (resolution, resolution), dtype) for _ in range(bsz)]
        ).to(device)
        negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding( # pylint: disable=no-member
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # 7. Denoising loop
        with torch.no_grad():
            # for i in tqdm(range(num_steps)):
            for i in range(num_steps):
                # expand the latents if we are doing classifier free guidance
                t = torch.cat([timesteps[:, i]]*2) if do_classifier_free_guidance else timesteps[:, i]
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


                # STEP: compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, timesteps[:, i].cpu(), latents, return_dict=False)[0]

                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval
                # prev_timestep = batch_timesteps - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]):
                    alpha_prod_t_prev[ib] = self.scheduler.alphas_cumprod[prev_timestep[ib]] if prev_timestep[ib] >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t

                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:,None,None,None] ** (0.5) * noise_pred) / alpha_prod_t[:, None,None,None] ** (0.5)
                    pred_epsilon = noise_pred
                # elif self.scheduler.config.prediction_type == "sample":
                #     pred_original_sample = noise_pred
                #     pred_epsilon = (latents - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                # elif self.scheduler.config.prediction_type == "v_prediction":
                #     pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * noise_pred
                #     pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * latents
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )
                pred_sample_direction = (1 - alpha_prod_t_prev[:,None,None,None]) ** (0.5) * pred_epsilon
                latents = alpha_prod_t_prev[:,None,None,None] ** (0.5) * pred_original_sample + pred_sample_direction

        return latents
