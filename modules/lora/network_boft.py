"""BOFT (Butterfly-OFT) — cascade of butterfly orthogonal factors.

Saves with the same ``oft_blocks`` key as OFT but as a 4-D tensor
``(boft_m, block_num, block_size, block_size)``. The caller in
:func:`pipelines.flux.flux2_lora.try_load_oft` discriminates BOFT from
OFT by ``oft_blocks.ndim == 4``. Math ported from
``KohakuBlueleaf/LyCORIS/lycoris/modules/boft.py``.
"""

import torch
import modules.lora.network as network


class NetworkModuleBOFT(network.NetworkModule):  # pylint: disable=abstract-method
    """Butterfly-OFT module: cascade of orthogonal factors.

    Constructor signature mirrors :class:`NetworkModuleOFT` so it slots into
    the same ``finalize_updown`` pipeline. The ``boft_m``/``block_num``/
    ``block_size`` triple is read from the saved tensor's shape rather than
    re-derived via :func:`butterfly_factor`, which keeps loading deterministic
    even if the upstream factorization heuristic changes.
    """

    def __init__(self, net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        self.org_module: list[torch.nn.Module] = [self.sd_module]
        self.scale = 1.0

        # 4-D oft_blocks: (boft_m, block_num, block_size, block_size)
        self.oft_blocks = weights.w["oft_blocks"]
        self.alpha = weights.w["alpha"]
        self.rescale = weights.w.get("rescale")
        self.boft_m = self.oft_blocks.shape[0]
        self.block_num = self.oft_blocks.shape[1]
        self.block_size = self.oft_blocks.shape[2]
        self.boft_b = self.block_size

        # Resolve out_dim from the host module — matches NetworkModuleOFT's
        # discrimination so Linear/Conv2d hosts both work.
        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]
        if is_linear:
            self.out_dim = self.sd_module.out_features
        elif is_conv:
            self.out_dim = self.sd_module.out_channels
        else:
            self.out_dim = self.block_num * self.block_size

        # constraint scales with out_dim per LyCORIS BOFT init
        self.constraint = float(self.alpha) * self.out_dim if self.alpha is not None else 0.0

    def _get_r(self, target: torch.Tensor):
        """Compute the per-stage Cayley rotations.

        Returns a tensor of shape ``(boft_m, block_num, block_size, block_size)``
        where each ``r[i]`` is a stack of ``block_num`` orthogonal matrices
        derived from the i-th butterfly factor via Cayley's parameterization
        of SO(n): ``R = (I + Q)(I - Q)^-1`` for skew-symmetric ``Q``.
        """
        eye = torch.eye(self.block_size, device=target.device, dtype=target.dtype)
        oft_blocks = self.oft_blocks.to(target.device, dtype=target.dtype)
        q = oft_blocks - oft_blocks.transpose(-1, -2)
        if self.constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                q = q * self.constraint / q_norm
        # Inverse needs fp32 to be numerically well-behaved across all dtypes;
        # cast back to target dtype after.
        r = (eye + q) @ (eye - q).float().inverse().to(target.dtype)
        return r

    def _make_weight(self, target: torch.Tensor):
        """Apply the butterfly cascade to ``target`` and return the transformed weight.

        Direct port of :meth:`ButterflyOFTModule.make_weight` (LyCORIS
        boft.py:158-191) for the merge-mode (no-bypass) path. ``target`` is the
        host weight; iteratively reshape to expose the per-stage block layout,
        einsum-multiply by the stage rotation, then reshape back. The reshape
        recipe at each stage is what makes the rotations interleave across
        butterfly partitions, giving the algorithm its O(d log d) parameter
        density.
        """
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self._get_r(target)
        inp = target

        for i in range(m):
            bi = r[i]
            g = 2
            k = 2 ** i * r_b
            inp = (
                inp.unflatten(0, (-1, g, k))
                .transpose(1, 2)
                .flatten(0, 2)
                .unflatten(0, (-1, b))
            )
            inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
            inp = (
                inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)
            )

        if self.rescale is not None:
            inp = inp * self.rescale.to(target.device, dtype=target.dtype)
        return inp

    def calc_updown(self, target: torch.Tensor):
        merged = self._make_weight(target)
        updown = merged - target
        return self.finalize_updown(updown, target, target.shape)
