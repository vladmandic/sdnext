import modules.lora.network as network


class NetworkModuleNorm(network.NetworkModule): # pylint: disable=abstract-method
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        self.w_norm = weights.w.get("w_norm")
        self.b_norm = weights.w.get("b_norm")

    def calc_updown(self, target):
        output_shape = self.w_norm.shape
        updown = self.w_norm.to(target.device, dtype=target.dtype)
        if self.b_norm is not None:
            ex_bias = self.b_norm.to(target.device, dtype=target.dtype)
        else:
            ex_bias = None
        return self.finalize_updown(updown, target, output_shape, ex_bias)
