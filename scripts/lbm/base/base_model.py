from typing import Any, Dict
import torch
import torch.nn as nn
from .model_config import ModelConfig


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.input_key = config.input_key
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts

        Args:
            device (Optional[torch.device], optional): The device to use. Usefull to set
                relevant parameters on the model and embedder to the right device only
                once at the start of the training. Defaults to None.
        """
        if device is not None:
            self.device = device
        self.to(self.device)

    def forward(self, batch: Dict[str, Any], *args, **kwargs):
        raise NotImplementedError("forward method is not implemented")

    def freeze(self):
        """Freeze the model"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        self = super().to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )

        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def compute_metrics(self, batch: Dict[str, Any], *args, **kwargs):
        """Compute the metrics"""
        return {}

    def sample(self, batch: Dict[str, Any], *args, **kwargs):
        """Sample from the model"""
        return {}

    def log_samples(self, batch: Dict[str, Any], *args, **kwargs):
        """Log the samples"""
        return None

    def on_train_batch_end(self, batch: Dict[str, Any], *args, **kwargs):
        """Update the model an optimization is perforned on a batch."""
        pass
