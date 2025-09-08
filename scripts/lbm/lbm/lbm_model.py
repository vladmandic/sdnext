from typing import Any, Dict, List, Optional, Tuple, Union
import lpips
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper
from ..vae import AutoencoderKLDiffusers
from .lbm_config import LBMConfig


class LBMModel(BaseModel):
    """This is the LBM class which defines the model.

    Args:

        config (LBMConfig):
            Configuration for the model

        denoiser (Union[DiffusersUNet2DWrapper, DiffusersTransformer2DWrapper]):
            Denoiser to use for the diffusion model. Defaults to None

        training_noise_scheduler (EulerDiscreteScheduler):
            Noise scheduler to use for training. Defaults to None

        sampling_noise_scheduler (EulerDiscreteScheduler):
            Noise scheduler to use for sampling. Defaults to None

        vae (AutoencoderKLDiffusers):
            VAE to use for the diffusion model. Defaults to None

        conditioner (ConditionerWrapper):
            Conditioner to use for the diffusion model. Defaults to None
    """

    @classmethod
    def load_from_config(cls, config: LBMConfig):
        return cls(config=config)

    def __init__(
        self,
        config: LBMConfig,
        denoiser: Union[
            DiffusersUNet2DWrapper,
            DiffusersUNet2DCondWrapper,
        ] = None,
        training_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        sampling_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        vae: AutoencoderKLDiffusers = None,
        conditioner: ConditionerWrapper = None,
    ):
        BaseModel.__init__(self, config)

        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.sampling_noise_scheduler = sampling_noise_scheduler
        self.training_noise_scheduler = training_noise_scheduler
        self.timestep_sampling = config.timestep_sampling
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.pixel_loss_type = config.pixel_loss_type
        self.pixel_loss_max_size = config.pixel_loss_max_size
        self.pixel_loss_weight = config.pixel_loss_weight
        self.logit_mean = config.logit_mean
        self.logit_std = config.logit_std
        self.prob = config.prob
        self.selected_timesteps = config.selected_timesteps
        self.source_key = config.source_key
        self.target_key = config.target_key
        self.mask_key = config.mask_key
        self.bridge_noise_sigma = config.bridge_noise_sigma

        self.num_iterations = nn.Parameter(
            torch.tensor(0, dtype=torch.float32), requires_grad=False
        )
        if self.pixel_loss_type == "lpips" and self.pixel_loss_weight > 0:
            self.lpips_loss = lpips.LPIPS(net="vgg")

        else:
            self.lpips_loss = None

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        if self.vae is not None:
            self.vae.on_fit_start(device=device, *args, **kwargs)
        if self.conditioner is not None:
            self.conditioner.on_fit_start(device=device, *args, **kwargs)

    def forward(self, batch: Dict[str, Any], step=0, batch_idx=0, *args, **kwargs):

        self.num_iterations += 1

        # Get inputs/latents
        if self.vae is not None:
            vae_inputs = batch[self.target_key]
            z = self.vae.encode(vae_inputs)
            downsampling_factor = self.vae.downsampling_factor
        else:
            z = batch[self.target_key]
            downsampling_factor = 1

        if self.mask_key in batch:
            valid_mask = batch[self.mask_key].bool()[:, 0, :, :].unsqueeze(1)
            invalid_mask = ~valid_mask
            valid_mask_for_latent = ~torch.max_pool2d(
                invalid_mask.float(),
                downsampling_factor,
                downsampling_factor,
            ).bool()
            valid_mask_for_latent = valid_mask_for_latent.repeat((1, z.shape[1], 1, 1))

        else:
            valid_mask = torch.ones_like(batch[self.target_key]).bool()
            valid_mask_for_latent = torch.ones_like(z).bool()

        source_image = batch[self.source_key]
        source_image = torch.nn.functional.interpolate(
            source_image,
            size=batch[self.target_key].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(z.dtype)
        if self.vae is not None:
            z_source = self.vae.encode(source_image)

        else:
            z_source = source_image

        # Get conditionings
        conditioning = self._get_conditioning(batch, *args, **kwargs)

        # Sample a timestep
        timestep = self._timestep_sampling(n_samples=z.shape[0], device=z.device)
        sigmas = None

        # Create interpolant
        sigmas = self._get_sigmas(
            self.training_noise_scheduler, timestep, n_dim=4, device=z.device
        )
        noisy_sample = (
            sigmas * z_source
            + (1.0 - sigmas) * z
            + self.bridge_noise_sigma
            * (sigmas * (1.0 - sigmas)) ** 0.5
            * torch.randn_like(z)
        )

        for i, t in enumerate(timestep):
            if t.item() == self.training_noise_scheduler.timesteps[0]:
                noisy_sample[i] = z_source[i]

        # Predict noise level using denoiser
        prediction = self.denoiser(
            sample=noisy_sample,
            timestep=timestep,
            conditioning=conditioning,
            *args,
            **kwargs,
        )

        target = z_source - z
        denoised_sample = noisy_sample - prediction * sigmas
        target_pixels = batch[self.target_key]

        # Compute loss
        if self.latent_loss_weight > 0:
            loss = self.latent_loss(prediction, target.detach(), valid_mask_for_latent)
            latent_recon_loss = loss.mean()

        else:
            loss = torch.zeros(z.shape[0], device=z.device)
            latent_recon_loss = torch.zeros_like(loss)

        if self.pixel_loss_weight > 0:
            denoised_sample = self._predicted_x_0(
                model_output=prediction,
                sample=noisy_sample,
                sigmas=sigmas,
            )
            pixel_loss = self.pixel_loss(
                denoised_sample, target_pixels.detach(), valid_mask
            )
            loss += self.pixel_loss_weight * pixel_loss

        else:
            pixel_loss = torch.zeros_like(latent_recon_loss)

        return {
            "loss": loss.mean(),
            "latent_recon_loss": latent_recon_loss,
            "pixel_recon_loss": pixel_loss.mean(),
            "predicted_hr": denoised_sample,
            "noisy_sample": noisy_sample,
        }

    def latent_loss(self, prediction, model_input, valid_latent_mask):
        if self.latent_loss_type == "l2":
            return torch.mean(
                (
                    (prediction * valid_latent_mask - model_input * valid_latent_mask)
                    ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )
        elif self.latent_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    prediction * valid_latent_mask - model_input * valid_latent_mask
                ).reshape(model_input.shape[0], -1),
                1,
            )
        else:
            raise NotImplementedError(
                f"Loss type {self.latent_loss_type} not implemented"
            )

    def pixel_loss(self, prediction, model_input, valid_mask):

        latent_crop = self.pixel_loss_max_size // self.vae.downsampling_factor
        input_crop = self.pixel_loss_max_size

        crop_h = max((prediction.shape[2] - latent_crop), 0)
        crop_w = max((prediction.shape[3] - latent_crop), 0)

        input_crop_h = max((model_input.shape[2] - self.pixel_loss_max_size), 0)
        input_crop_w = max((model_input.shape[3] - self.pixel_loss_max_size), 0)

        # image random cropping
        if crop_h == 0:
            offset_h = 0
        else:
            offset_h = torch.randint(0, crop_h, (1,)).item()

        if crop_w == 0:
            offset_w = 0
        else:
            offset_w = torch.randint(0, crop_w, (1,)).item()
        input_offset_h = offset_h * self.vae.downsampling_factor
        input_offset_w = offset_w * self.vae.downsampling_factor

        prediction = prediction[
            :,
            :,
            crop_h
            - offset_h : min(crop_h - offset_h + latent_crop, prediction.shape[2]),
            crop_w
            - offset_w : min(crop_w - offset_w + latent_crop, prediction.shape[3]),
        ]

        model_input = model_input[
            :,
            :,
            input_crop_h
            - input_offset_h : min(
                input_crop_h - input_offset_h + input_crop, model_input.shape[2]
            ),
            input_crop_w
            - input_offset_w : min(
                input_crop_w - input_offset_w + input_crop, model_input.shape[3]
            ),
        ]

        valid_mask = valid_mask[
            :,
            :,
            input_crop_h
            - input_offset_h : min(
                input_crop_h - input_offset_h + input_crop, valid_mask.shape[2]
            ),
            input_crop_w
            - input_offset_w : min(
                input_crop_w - input_offset_w + input_crop, valid_mask.shape[3]
            ),
        ]

        decoded_prediction = self.vae.decode(prediction).clamp(-1, 1)

        if self.pixel_loss_type == "l2":
            return torch.mean(
                (
                    (decoded_prediction * valid_mask - model_input * valid_mask) ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )

        elif self.pixel_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    decoded_prediction * valid_mask - model_input * valid_mask
                ).reshape(model_input.shape[0], -1),
                1,
            )

        elif self.pixel_loss_type == "lpips":
            return self.lpips_loss(
                decoded_prediction * valid_mask, model_input * valid_mask
            ).mean()

    def _get_conditioning(
        self,
        batch: Dict[str, Any],
        ucg_keys: List[str] = None,
        set_ucg_rate_zero=False,
        *args,
        **kwargs,
    ):
        """
        Get the conditionings
        """
        if self.conditioner is not None:
            return self.conditioner(
                batch,
                ucg_keys=ucg_keys,
                set_ucg_rate_zero=set_ucg_rate_zero,
                vae=self.vae,
                *args,
                **kwargs,
            )
        else:
            return None

    def _timestep_sampling(self, n_samples=1, device="cpu"):
        if self.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                self.training_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device="cpu",
            )
            return self.training_noise_scheduler.timesteps[idx].to(device=device)

        elif self.timestep_sampling == "log_normal":
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(n_samples,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (
                u * self.training_noise_scheduler.config.num_train_timesteps
            ).long()
            return self.training_noise_scheduler.timesteps[indices].to(device=device)

        elif self.timestep_sampling == "custom_timesteps":
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)

            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

    def _predicted_x_0(
        self,
        model_output,
        sample,
        sigmas=None,
    ):
        """
        Predict x_0, the orinal denoised sample, using the model output and the timesteps depending on the prediction type.
        """
        pred_x_0 = sample - model_output * sigmas
        return pred_x_0

    def _get_sigmas(
        self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"
    ):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        conditioner_inputs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ):
        self.sampling_noise_scheduler.set_timesteps(
            sigmas=np.linspace(1, 1 / num_steps, num_steps)
        )

        sample = z

        # Get conditioning
        conditioning = self._get_conditioning(
            conditioner_inputs, set_ucg_rate_zero=True, device=z.device
        )

        # If max_samples parameter is provided, limit the number of samples
        if max_samples is not None:
            sample = sample[:max_samples]

        if conditioning:
            conditioning["cond"] = {
                k: v[:max_samples] for k, v in conditioning["cond"].items()
            }

        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):
            if hasattr(self.sampling_noise_scheduler, "scale_model_input"):
                denoiser_input = self.sampling_noise_scheduler.scale_model_input(
                    sample, t
                )

            else:
                denoiser_input = sample

            # Predict noise level using denoiser using conditionings
            pred = self.denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                conditioning=conditioning,
            )

            # Make one step on the reverse diffusion process
            sample = self.sampling_noise_scheduler.step(
                pred, t, sample, return_dict=False
            )[0]
            if i < len(self.sampling_noise_scheduler.timesteps) - 1:
                timestep = (
                    self.sampling_noise_scheduler.timesteps[i + 1]
                    .to(z.device)
                    .repeat(sample.shape[0])
                )
                sigmas = self._get_sigmas(
                    self.sampling_noise_scheduler, timestep, n_dim=4, device=z.device
                )
                sample = sample + self.bridge_noise_sigma * (
                    sigmas * (1.0 - sigmas)
                ) ** 0.5 * torch.randn_like(sample)
                sample = sample.to(z.dtype)

        if self.vae is not None:
            decoded_sample = self.vae.decode(sample)

        else:
            decoded_sample = sample

        return decoded_sample

    def log_samples(
        self,
        batch: Dict[str, Any],
        input_shape: Optional[Tuple[int, int, int]] = None,
        max_samples: Optional[int] = None,
        num_steps: Union[int, List[int]] = 20,
    ):
        if isinstance(num_steps, int):
            num_steps = [num_steps]

        logs = {}

        N = max_samples if max_samples is not None else len(batch[self.source_key])

        batch = {k: v[:N] for k, v in batch.items()}

        # infer input shape based on VAE configuration if not passed
        if input_shape is None:
            if self.vae is not None:
                # get input pixel size of the vae
                input_shape = batch[self.target_key].shape[2:]
                # rescale to latent size
                input_shape = (
                    self.vae.latent_channels,
                    input_shape[0] // self.vae.downsampling_factor,
                    input_shape[1] // self.vae.downsampling_factor,
                )
            else:
                raise ValueError(
                    "input_shape must be passed when no VAE is used in the model"
                )

        for num_step in num_steps:
            source_image = batch[self.source_key]
            source_image = torch.nn.functional.interpolate(
                source_image,
                size=batch[self.target_key].shape[2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=self.dtype)
            if self.vae is not None:
                z = self.vae.encode(source_image)

            else:
                z = source_image

            with torch.autocast(dtype=self.dtype, device_type="cuda"):
                logs[f"samples_{num_step}_steps"] = self.sample(
                    z,
                    num_steps=num_step,
                    conditioner_inputs=batch,
                    max_samples=N,
                )

        return logs
