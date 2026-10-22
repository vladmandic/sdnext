import modules.lora.network as network


# adapted from https://github.com/KohakuBlueleaf/LyCORIS
class NetworkModuleGLora(network.NetworkModule): # pylint: disable=abstract-method
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.w1a = weights.w["a1.weight"]
        self.w1b = weights.w["b1.weight"]
        self.w2a = weights.w["a2.weight"]
        self.w2b = weights.w["b2.weight"]
        self.dim = self.w1b.shape[0]

    def calc_updown(self, target): # pylint: disable=arguments-differ
        w1a = self.w1a.to(target.device, dtype=target.dtype)
        w1b = self.w1b.to(target.device, dtype=target.dtype)
        w2a = self.w2a.to(target.device, dtype=target.dtype)
        w2b = self.w2b.to(target.device, dtype=target.dtype)
        output_shape = [w1a.size(0), w1b.size(1)]
        updown = (w2b @ w1b) + ((target @ w2a) @ w1a)
        return self.finalize_updown(updown, target, output_shape)
