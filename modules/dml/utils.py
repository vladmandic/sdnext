from typing import Union
import torch


rDevice = Union[torch.device, int]
def get_device(device: rDevice | None=None) -> torch.device:
    if device is None:
        device = torch.dml.current_device()
    return torch.device(device)
