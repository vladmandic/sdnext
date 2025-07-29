import logging
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
from .base import BaseConditioner


KEY2CATDIM = {
    "vector": 1,
    "crossattn": 2,
    "concat": 1,
}


class ConditionerWrapper(nn.Module):
    """
    Wrapper for conditioners. This class allows to apply multiple conditioners in a single forward pass.

    Args:

        conditioners (List[BaseConditioner]): List of conditioners to apply in the forward pass.
    """

    def __init__(
        self,
        conditioners: Union[List[BaseConditioner], None] = None,
    ):
        nn.Module.__init__(self)
        self.conditioners = nn.ModuleList(conditioners)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def conditioner_sanity_check(self):
        cond_input_keys = []
        for conditioner in self.conditioners:
            cond_input_keys.append(conditioner.input_key)

        assert all([key in set(cond_input_keys) for key in self.ucg_keys])

    def on_fit_start(self, device: torch.device = None, *args, **kwargs):
        for conditioner in self.conditioners:
            conditioner.on_fit_start(device=device, *args, **kwargs)

    def forward(
        self,
        batch: Dict[str, Any],
        ucg_keys: List[str] = None,
        set_ucg_rate_zero=False,
        *args,
        **kwargs,
    ):
        """
        Forward pass through all conditioners

        Args:

            batch: batch of data
            ucg_keys: keys to use for ucg. This will force zero conditioning in all the
                conditioners that have input_keys in ucg_keys
            set_ucg_rate_zero: set the ucg rate to zero for all the conditioners except the ones in ucg_keys

        Returns:

        Dict[str, Any]: The output of the conditioner. The output of the conditioner is a dictionary with the main key "cond" and value
            is a dictionary with the keys as the type of conditioning and the value as the conditioning tensor.
        """
        if ucg_keys is None:
            ucg_keys = []
        wrapper_outputs = dict(cond={})
        for conditioner in self.conditioners:
            if conditioner.input_key in ucg_keys:
                force_zero_embedding = True
            elif conditioner.ucg_rate > 0 and not set_ucg_rate_zero:
                force_zero_embedding = bool(torch.rand(1) < conditioner.ucg_rate)
            else:
                force_zero_embedding = False

            conditioner_output = conditioner.forward(
                batch, force_zero_embedding=force_zero_embedding, *args, **kwargs
            )
            logging.debug(
                f"conditioner:{conditioner.__class__.__name__}, input_key:{conditioner.input_key}, force_ucg_zero_embedding:{force_zero_embedding}"
            )
            for key in conditioner_output:
                logging.debug(
                    f"conditioner_output:{key}:{conditioner_output[key].shape}"
                )
                if key in wrapper_outputs["cond"]:
                    wrapper_outputs["cond"][key] = torch.cat(
                        [wrapper_outputs["cond"][key], conditioner_output[key]],
                        KEY2CATDIM[key],
                    )
                else:
                    wrapper_outputs["cond"][key] = conditioner_output[key]

        return wrapper_outputs

    def to(self, *args, **kwargs):
        """
        Move all conditioners to device and dtype
        """
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        self = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        for conditioner in self.conditioners:
            conditioner.to(device=device, dtype=dtype, non_blocking=non_blocking)

        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        return self
