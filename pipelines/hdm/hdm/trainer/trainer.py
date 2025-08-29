import os
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

from diffusers import (
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from anyschedule import AnySchedule

from ..utils import instantiate
from ..modules.text_encoders import BaseTextEncoder
from .diffusion import (
    get_noise_noisy_latents_and_timesteps,
    prepare_scheduler_for_custom_training,
)


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        *args,
        name: str = "",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_sch_configs: dict[str, Any] = {
            "lr": {
                "mode": "cosine",
                "end": 100000,
                "min_value": 0.001,
            }
        },
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate(optimizer)
        self.opt_configs = opt_configs
        self.lr = lr
        self.lr_sch_configs = lr_sch_configs
        self.use_warm_up = use_warm_up
        self.warm_up_period = warm_up_period

    def configure_optimizers(self):
        parameters = []
        assert self.train_params is not None
        for param in self.train_params:
            if param.ndim < 2:  # bias, norm, ...
                fan_in = param.numel()
            elif param.ndim > 2:  # Conv layer in patch embedding
                # For conv layers, fan_in is channels_in * kernel_size^2
                fan_ins = param.shape[1:]
                fan_in = 1
                for fan_in_i in fan_ins:
                    fan_in *= fan_in_i
            else:  # Linear layers, including attention and MLP
                fan_in = param.shape[1]
            parameters.append(
                {
                    "params": param,
                    "lr": self.lr / fan_in,
                }
            )
        optimizer = self.optimizer(parameters, lr=self.lr, **self.opt_configs)

        lr_scheduler = None
        if bool(self.lr_sch_configs):
            lr_scheduler = AnySchedule(optimizer=optimizer, config=self.lr_sch_configs)

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }


class DMTrainer(BaseTrainer):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        te: BaseTextEncoder,
        vae: AutoencoderKL | None = None,
        unet_compile: bool = False,
        te_compile: bool = False,
        vae_compile: bool = False,
        te_use_normed_ctx: bool = False,
        te_freeze: bool = True,
        vae_std: float = 7.5,
        vae_mean: float = 1.125,
        scheduler: EulerDiscreteScheduler | None = None,
        lycoris_model: nn.Module | None = None,
        *args,
        name: str = "",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_sch_configs: dict[str, Any] = {},
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        full_config: dict[str, Any] = {},
        **kwargs,
    ):
        super(DMTrainer, self).__init__(
            name=name,
            lr=lr,
            optimizer=optimizer,
            opt_configs=opt_configs,
            lr_sch_configs=lr_sch_configs,
            use_warm_up=use_warm_up,
            warm_up_period=warm_up_period,
        )
        self.save_hyperparameters(
            ignore=["unet", "scheduler", "te", "vae", "lycoris_model", "args", "kwargs"]
        )
        prepare_scheduler_for_custom_training(scheduler, self.device)

        if unet_compile:
            unet = torch.compile(unet)

        if te_compile:
            te = torch.compile(te)

        if vae_compile and vae is not None:
            vae = torch.compile(vae)

        if te_freeze:
            te.requires_grad_(False).eval()

        if vae is not None:
            vae.requires_grad_(False).eval()

        self.unet = unet
        self.te = te
        self.vae = vae
        self.scheduler = scheduler

        self.te_use_normed_ctx = te_use_normed_ctx
        self.vae_std = vae_std
        self.vae_mean = vae_mean

        self.lycoris_model = lycoris_model

        self.epoch = 0
        self.opt_step = 0
        self.ema_loss = 0
        self.ema_decay = 0.99

        if lycoris_model is not None:
            self.lycoris_model.train()
            self.train_params = self.lycoris_model.parameters()
        else:
            self.unet.requires_grad_(True).train()
            self.train_params = self.unet.parameters()

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.unet.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def training_step(self, batch, idx):
        x, captions, tokenizer_outputs = batch
        # print(type(x), type(captions), type(tokenizer_outputs), type(added_cond))

        if self.vae is not None:
            with torch.no_grad():
                latent_dist = self.vae.encode(x).latent_dist
                x = latent_dist.sample()
                x = (x - self.vae_mean) / self.vae_std

        b, c, h, w = x.shape

        noisy_latent, noise, timesteps = get_noise_noisy_latents_and_timesteps(
            self.scheduler, x
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x, noise, timesteps)
        elif self.scheduler.config.prediction_type == "sample":
            target = x
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        with torch.no_grad():
            if isinstance(self.te, BaseTextEncoder):
                normed_embedding, embedding, pooled_embedding, attn_mask = self.te(
                    tokenizer_outputs
                )
            else:
                normed_embedding, pooled_embedding, *embeddings = self.te(
                    **tokenizer_outputs[0], return_dict=False, output_hidden_states=True
                )
                embedding = embeddings[-1][-1]
            if self.te_use_normed_ctx:
                ctx = normed_embedding
            else:
                ctx = embedding

        model_output = self.unet(
            noisy_latent.to(self.dtype),
            timesteps,
            encoder_hidden_states=ctx.to(self.dtype),
            encoder_attention_mask=attn_mask,
        )[0]
        loss = F.mse_loss(model_output, target)

        ema_decay = min(self.opt_step / (10 + self.opt_step), self.ema_decay)
        self.ema_loss = ema_decay * self.ema_loss + (1 - ema_decay) * loss.item()
        self.opt_step += 1

        if self._trainer is not None:
            self.log("train/loss", loss.item(), on_step=True, logger=True)
            self.log(
                "train/ema_loss",
                self.ema_loss,
                on_step=True,
                logger=True,
                prog_bar=True,
            )
        return loss


class FlowTrainer(BaseTrainer):
    def __init__(
        self,
        unet: nn.Module,
        te: BaseTextEncoder,
        vae: AutoencoderKL | None = None,
        unet_compile: bool = False,
        te_compile: bool = False,
        vae_compile: bool = False,
        te_use_normed_ctx: bool = False,
        te_freeze: bool = True,
        vae_std: float = 7.5,
        vae_mean: float = 1.125,
        lycoris_model: nn.Module | None = None,
        *args,
        name: str = "",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_sch_configs: dict[str, Any] = {},
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        full_config: dict[str, Any] = {},
        **kwargs,
    ):
        super(FlowTrainer, self).__init__(
            name=name,
            lr=lr,
            optimizer=optimizer,
            opt_configs=opt_configs,
            lr_sch_configs=lr_sch_configs,
            use_warm_up=use_warm_up,
            warm_up_period=warm_up_period,
        )
        self.save_hyperparameters(
            ignore=[
                "unet",
                "te",
                "vae",
                "lycoris_model",
                "args",
                "kwargs",
                "full_config",
                "opt_configs",
                "lr_sch_configs",
            ]
        )

        if unet_compile:
            unet = torch.compile(unet)

        if te_compile:
            te = torch.compile(te)

        if vae_compile and vae is not None:
            vae = torch.compile(vae)

        if te_freeze:
            te.requires_grad_(False).eval()

        if vae is not None:
            vae.requires_grad_(False).eval()

        self.unet = unet
        self.te = te
        self.vae = vae

        self.te_use_normed_ctx = te_use_normed_ctx
        if self.vae is not None:
            vae_std = self.vae.config["latents_std"]
            vae_mean = self.vae.config["latents_mean"]
            self.register_buffer("vae_std", torch.tensor(vae_std).view(1, -1, 1, 1))
            self.register_buffer("vae_mean", torch.tensor(vae_mean).view(1, -1, 1, 1))
        else:
            self.vae_std = vae_std
            self.vae_mean = vae_mean

        self.lycoris_model = lycoris_model

        self.epoch = 0
        self.opt_step = 0
        self.ema_loss = 0
        self.ema_decay = 0.995

        if lycoris_model is not None:
            self.lycoris_model.train()
            self.train_params = self.lycoris_model.parameters()
        else:
            self.unet.requires_grad_(True).train()
            self.train_params = self.unet.parameters()

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.unet.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def training_step(self, batch, idx):
        x, captions, tokenizer_outputs, pos_map, *addon_info = batch

        if self.vae is not None:
            if pos_map is not None:
                pos_map = pos_map.unflatten(1, x.shape[-2:])  # (B, H, W, 2)
            with torch.no_grad():
                x = x.to(self.device)
                x = torch.concat(
                    [
                        self.vae.encode(x[i : i + 4]).latent_dist.sample()
                        for i in range(0, x.shape[0], 4)
                    ]
                )
                x = (x - self.vae_mean) / self.vae_std
            pos_map = pos_map.permute(0, 3, 1, 2)
            pos_map = (
                F.interpolate(pos_map, x.shape[-2:], mode="area")
                .permute(0, 2, 3, 1)
                .flatten(1, 2)
            )

        b, c, h, w = x.shape

        noise = torch.randn_like(x)
        t = torch.sigmoid(torch.randn(b, 1, 1, 1, device=x.device))
        noisy_latent = t * noise + (1 - t) * x
        target = noise - x

        with torch.no_grad():
            if isinstance(self.te, BaseTextEncoder):
                normed_embedding, embedding, pooled_embedding, attn_mask = self.te(
                    tokenizer_outputs
                )
            else:
                normed_embedding, pooled_embedding, *embeddings = self.te(
                    **tokenizer_outputs[0], return_dict=False, output_hidden_states=True
                )
                embedding = embeddings[-1][-1]
            if self.te_use_normed_ctx:
                ctx = normed_embedding
            else:
                ctx = embedding

        if pooled_embedding is not None:
            added_cond_kwargs = {
                "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]]).to(
                    noisy_latent
                ),
                "text_embeds": pooled_embedding.to(noisy_latent),
            }
        else:
            added_cond_kwargs = {}

        if len(addon_info) > 0:
            for addon in addon_info:
                added_cond_kwargs.update(addon)

        model_output = self.unet(
            noisy_latent.to(self.dtype),
            t,
            encoder_hidden_states=ctx.to(self.dtype),
            encoder_attention_mask=attn_mask,
            pos_map=pos_map,
            added_cond_kwargs=added_cond_kwargs,
        )[0]
        loss = F.mse_loss(model_output, target)
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        ema_decay = min(self.opt_step / (10 + self.opt_step), self.ema_decay)
        self.ema_loss = ema_decay * self.ema_loss + (1 - ema_decay) * loss.item()
        self.opt_step += 1

        if self._trainer is not None:
            self.log("train/loss", loss.item(), logger=True)
            self.log(
                "train/ema_loss",
                self.ema_loss,
                logger=True,
                prog_bar=True,
            )
        return loss
