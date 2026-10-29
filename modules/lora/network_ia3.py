import modules.lora.network as network

class NetworkModuleIa3(network.NetworkModule): # pylint: disable=abstract-method
    weight_dependent = True

    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, target):
        w = self.w.to(target.device, dtype=target.dtype)
        output_shape = [w.size(0), target.size(1)]
        if self.on_input:
            output_shape.reverse()
        else:
            w = w.reshape(-1, 1)
        updown = target * w
        return self.finalize_updown(updown, target, output_shape)
