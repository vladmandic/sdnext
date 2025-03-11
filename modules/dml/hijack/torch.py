import torch
from modules.sd_hijack_utils import CondFunc


CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'privateuseone')
# https://github.com/microsoft/DirectML/issues/400
CondFunc('torch.Tensor.new', lambda orig, self, *args, **kwargs: orig(self.cpu(), *args, **kwargs).to(self.device), lambda orig, self, *args, **kwargs: torch.dml.is_directml_device(self.device))


def cuda(self: torch.Tensor):
    return self.to(torch.dml.current_device())
torch.Tensor.cuda = cuda


# https://github.com/lshqqytiger/stable-diffusion-webui-directml/issues/436
_pow_ = torch.Tensor.pow_
def pow_(self: torch.Tensor, *args, **kwargs):
    if self.dtype == torch.float64:
        return _pow_(self.cpu(), *args, **kwargs).to(self.device)
    return _pow_(self, *args, **kwargs)
torch.Tensor.pow_ = pow_


_load = torch.load
def load(f, map_location = "cpu", *args, **kwargs):
    if type(map_location) in (str, torch.device,):
        device = torch.device(map_location)
        if device.type == "privateuseone":
            data = _load(f, *args, map_location="cpu", **kwargs)
            for k in data:
                for weight in data[k]:
                    data[k][weight] = data[k][weight].to(device)
            return data
    return _load(f, *args, map_location=map_location, **kwargs)
torch.load = load
