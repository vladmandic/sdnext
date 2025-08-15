# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight), self.bias)
