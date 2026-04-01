import torch
import modules.lora.lyco_helpers as lyco_helpers
import modules.lora.network as network


class ModuleTypeLokr(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        has_1 = "lokr_w1" in weights.w or ("lokr_w1_a" in weights.w and "lokr_w1_b" in weights.w)
        has_2 = "lokr_w2" in weights.w or ("lokr_w2_a" in weights.w and "lokr_w2_b" in weights.w)
        if has_1 and has_2:
            return NetworkModuleLokr(net, weights)
        return None


def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


class NetworkModuleLokr(network.NetworkModule): # pylint: disable=abstract-method
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        self.w1 = weights.w.get("lokr_w1")
        self.w1a = weights.w.get("lokr_w1_a")
        self.w1b = weights.w.get("lokr_w1_b")
        self.dim = self.w1b.shape[0] if self.w1b is not None else self.dim
        self.w2 = weights.w.get("lokr_w2")
        self.w2a = weights.w.get("lokr_w2_a")
        self.w2b = weights.w.get("lokr_w2_b")
        self.dim = self.w2b.shape[0] if self.w2b is not None else self.dim
        self.t2 = weights.w.get("lokr_t2")

    def calc_updown(self, target):
        if self.w1 is not None:
            w1 = self.w1.to(target.device, dtype=target.dtype)
        else:
            w1a = self.w1a.to(target.device, dtype=target.dtype)
            w1b = self.w1b.to(target.device, dtype=target.dtype)
            w1 = w1a @ w1b
        if self.w2 is not None:
            w2 = self.w2.to(target.device, dtype=target.dtype)
        elif self.t2 is None:
            w2a = self.w2a.to(target.device, dtype=target.dtype)
            w2b = self.w2b.to(target.device, dtype=target.dtype)
            w2 = w2a @ w2b
        else:
            t2 = self.t2.to(target.device, dtype=target.dtype)
            w2a = self.w2a.to(target.device, dtype=target.dtype)
            w2b = self.w2b.to(target.device, dtype=target.dtype)
            w2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)
        output_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        if len(target.shape) == 4:
            output_shape = target.shape
        updown = make_kron(output_shape, w1, w2)
        return self.finalize_updown(updown, target, output_shape)


class NetworkModuleLokrChunk(NetworkModuleLokr):
    """LoKR module that returns one chunk of the Kronecker product.

    Used when a LoKR adapter targets a fused weight (e.g., QKV) but the model
    has separate modules (Q, K, V). Computes kron(w1, w2) on-the-fly and
    returns only the designated chunk, keeping memory usage minimal.
    """
    def __init__(self, net, weights, chunk_index, num_chunks):
        super().__init__(net, weights)
        self.chunk_index = chunk_index
        self.num_chunks = num_chunks

    def calc_updown(self, target):
        if self.w1 is not None:
            w1 = self.w1.to(target.device, dtype=target.dtype)
        else:
            w1a = self.w1a.to(target.device, dtype=target.dtype)
            w1b = self.w1b.to(target.device, dtype=target.dtype)
            w1 = w1a @ w1b
        if self.w2 is not None:
            w2 = self.w2.to(target.device, dtype=target.dtype)
        elif self.t2 is None:
            w2a = self.w2a.to(target.device, dtype=target.dtype)
            w2b = self.w2b.to(target.device, dtype=target.dtype)
            w2 = w2a @ w2b
        else:
            t2 = self.t2.to(target.device, dtype=target.dtype)
            w2a = self.w2a.to(target.device, dtype=target.dtype)
            w2b = self.w2b.to(target.device, dtype=target.dtype)
            w2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)
        full_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        updown = make_kron(full_shape, w1, w2)
        updown = torch.chunk(updown, self.num_chunks, dim=0)[self.chunk_index]
        output_shape = list(updown.shape)
        return self.finalize_updown(updown, target, output_shape)
