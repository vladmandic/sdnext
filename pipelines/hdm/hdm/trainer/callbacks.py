from operator import is_
import os

import torch
import wandb

from lightning.pytorch import Callback, Trainer
from hdm.trainer import DMTrainer


class ImageGenCallback(Callback):
    def __init__(self, config, img_gen_func):
        self.config = {
            "period": 100,
            "num": 4,
            "preview_num": 4,
            "batch_size": 4,
            "steps": 24,
        }
        self.config.update(config)
        self.img_gen = img_gen_func

    @torch.no_grad()
    def on_train_batch_start(
        self, trainer: Trainer, pl_module: DMTrainer, batch, batch_idx
    ):
        if batch_idx % self.config["period"] == 0:
            is_training = pl_module.training
            pl_module.eval()
            torch.cuda.empty_cache()
            captions, images = self.img_gen(pl_module, batch, self.config)
            torch.cuda.empty_cache()

            if hasattr(trainer.logger, "id"):
                id = trainer.logger.id
            elif hasattr(trainer.logger, "experiment"):
                id = getattr(trainer.logger.experiment, "id", self.config.get("id", 0))
            else:
                id = self.config.get("id", 0)
            if not isinstance(id, (str, bytes, int, float)):
                id = self.config.get("id", 0)
            if "id" in self.config:
                id = self.config["id"]

            rank = trainer.local_rank
            base_idx = rank * self.config["num"]

            os.makedirs(f"./sample/{id}/{trainer.global_step}", exist_ok=True)
            data = []
            for idx, (caption, image) in enumerate(zip(captions, images)):
                idx = base_idx + idx
                image.save(f"./sample/{id}/{trainer.global_step}/{idx}.png")
                data.append(
                    [
                        caption,
                        wandb.Image(f"./sample/{id}/{trainer.global_step}/{idx}.png"),
                    ]
                )
            if trainer.is_global_zero:
                trainer.logger.log_table(
                    key="sample/images",
                    columns=["caption", "image"],
                    data=data[: self.config["preview_num"]],
                )
            torch.cuda.empty_cache()
            pl_module.train(is_training)
