import torch


class Generator(torch.Generator):
    def __init__(self, device: torch.device | None = None):
        super().__init__("cpu")
