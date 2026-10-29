import torch
import modules.lora.lyco_helpers as lyco_helpers
import modules.lora.network as network


class NetworkModuleHada(network.NetworkModule): # pylint: disable=abstract-method
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape
        self.w1a = weights.w["hada_w1_a"]
        self.w1b = weights.w["hada_w1_b"]
        self.dim = self.w1b.shape[0]
        self.w2a = weights.w["hada_w2_a"]
        self.w2b = weights.w["hada_w2_b"]
        self.t1 = weights.w.get("hada_t1")
        self.t2 = weights.w.get("hada_t2")

    def calc_updown(self, target):
        w1a = self.w1a.to(target.device, dtype=target.dtype)
        w1b = self.w1b.to(target.device, dtype=target.dtype)
        w2a = self.w2a.to(target.device, dtype=target.dtype)
        w2b = self.w2b.to(target.device, dtype=target.dtype)
        output_shape = [w1a.size(0), w1b.size(1)]
        if self.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = self.t1.to(target.device, dtype=target.dtype)
            updown1 = lyco_helpers.make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = lyco_helpers.rebuild_conventional(w1a, w1b, output_shape)
        if self.t2 is not None:
            t2 = self.t2.to(target.device, dtype=target.dtype)
            updown2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = lyco_helpers.rebuild_conventional(w2a, w2b, output_shape)
        updown = updown1 * updown2
        return self.finalize_updown(updown, target, output_shape)


class NetworkModuleHadaChunk(NetworkModuleHada):
    """LoHA module that returns one row chunk of the Hadamard product.

    Used when a LoHA adapter targets a fused weight (e.g., img_attn.qkv) but the
    diffusers model exposes separate Q/K/V modules. Slices the row-side of each
    Hadamard arm (w1a, w2a) at the assigned chunk's row range and computes the
    partial product. Memory and compute scale linearly with chunk size; no full
    Hadamard temporary is materialized.

    Tucker (CP-decomposed) LoHAs are not handled here. LyCORIS only saves
    hada_t1 / hada_t2 for non-1x1 Conv layers, and fused QKV is always Linear,
    so this combination cannot arise from a conformant trainer.
    """

    def __init__(self, net, weights, chunk_index, num_chunks):
        super().__init__(net, weights)
        self.chunk_index = chunk_index
        self.num_chunks = num_chunks

    def calc_updown(self, target):
        w1a = self.w1a.to(target.device, dtype=target.dtype)
        w1b = self.w1b.to(target.device, dtype=target.dtype)
        w2a = self.w2a.to(target.device, dtype=target.dtype)
        w2b = self.w2b.to(target.device, dtype=target.dtype)
        w1a_chunk = torch.chunk(w1a, self.num_chunks, dim=0)[self.chunk_index].contiguous()
        w2a_chunk = torch.chunk(w2a, self.num_chunks, dim=0)[self.chunk_index].contiguous()
        output_shape = [w1a_chunk.size(0), w1b.size(1)]
        if len(w1b.shape) == 4:
            output_shape += w1b.shape[2:]
        updown1 = lyco_helpers.rebuild_conventional(w1a_chunk, w1b, output_shape)
        updown2 = lyco_helpers.rebuild_conventional(w2a_chunk, w2b, output_shape)
        updown = updown1 * updown2
        return self.finalize_updown(updown, target, output_shape)
