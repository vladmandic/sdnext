import torch
import diffusers.models.lora as diffusers_lora
import modules.lora.lyco_helpers as lyco_helpers
import modules.lora.network as network
from modules import devices
from modules.errors import log


class ModuleTypeLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)
        return None


class NetworkModuleLora(network.NetworkModule):

    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        self.up_model = self.create_module(weights.w, "lora_up.weight")
        self.down_model = self.create_module(weights.w, "lora_down.weight")
        self.mid_model = self.create_module(weights.w, "lora_mid.weight", none_ok=True)
        self.dim = weights.w["lora_down.weight"].shape[0]

    def create_module(self, weights, key, none_ok=False):
        weight = weights.get(key)
        if weight is None and none_ok:
            return None
        linear_modules = [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention, diffusers_lora.LoRACompatibleLinear]
        typ = type(self.sd_module)
        is_linear = typ in linear_modules or self.sd_module.__class__.__name__ in ["NNCFLinear", "QLinear", "Linear4bit"]
        is_conv = (typ in [torch.nn.Conv2d, diffusers_lora.LoRACompatibleConv]) or (self.sd_module.__class__.__name__ in ["NNCFConv2d", "QConv2d"]) or (typ.__name__ in ['downsampler_block', 'upsampler_block'])
        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and (key == "lora_down.weight" or key == "dyn_up"):
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)
            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and (key == "lora_mid.weight"):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        elif is_conv and (key == "lora_up.weight" or key == "dyn_down"):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(f'Lora unsupported: key={key} layer={self.network_key} type={typ.__name__}')
        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)
        module.weight.requires_grad_(False)
        return module

    def calc_updown(self, target): # pylint: disable=W0237
        target_dtype = target.dtype if target.dtype != torch.uint8 else self.up_model.weight.dtype
        up = self.up_model.weight.to(target.device, dtype=target_dtype)
        down = self.down_model.weight.to(target.device, dtype=target_dtype)
        output_shape = [up.size(0), down.size(1)]
        if self.mid_model is not None:
            mid = self.mid_model.weight.to(target.device, dtype=target_dtype)
            updown = lyco_helpers.rebuild_cp_decomposition(up, down, mid) # cp-decomposition
            output_shape += mid.shape[2:]
        else:
            mid = None
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = lyco_helpers.rebuild_conventional(up, down, output_shape, self.network.dyn_dim)
        del up, down, mid
        return self.finalize_updown(updown, target, output_shape)

    def forward(self, x, y):
        self.up_model.to(device=devices.device)
        self.down_model.to(device=devices.device)
        if hasattr(y, "scale"):
            return y(scale=1) + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()
        return y + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()
