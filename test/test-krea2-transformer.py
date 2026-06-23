#!/usr/bin/env python
"""Offline parity test: Krea2Transformer2DModel vs the reference SingleStreamDiT.

Builds both models from one tiny config, copies the reference state dict into the diffusers
port, runs identical inputs, and asserts the forward outputs match. No server, no checkpoint.

The reference checkpoint repo (mmdit.py) is expected at $KREA2_REF_DIR
(default /home/ohiom/database/watering-hole).
"""

import importlib.util
import os
import sys
from contextlib import nullcontext

import torch

REF_DIR = os.environ.get("KREA2_REF_DIR", "/home/ohiom/database/watering-hole")


def load_reference():
    sys.path.insert(0, REF_DIR)
    import mmdit
    # The reference pins the cuDNN SDPA backend; neutralize it so both models use the same
    # default kernel and the test can run on CPU.
    mmdit.sdpa_kernel = lambda *a, **k: nullcontext()
    return mmdit


def load_port():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "krea2", "transformer_krea2.py"))
    spec = importlib.util.spec_from_file_location("transformer_krea2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mmdit = load_reference()
    port = load_port()

    cfg = dict(
        features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
        layers=2, patch=2, channels=4, bias=False, theta=1e3,
        txtlayers=3, txtheads=2, txtkvheads=2,
    )

    torch.manual_seed(0)
    ref = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**cfg)).float().eval()
    mine = port.Krea2Transformer2DModel(**cfg).float().eval()
    missing, unexpected = mine.load_state_dict(ref.state_dict(), strict=False)
    assert not missing, f"missing keys when loading reference weights: {missing}"
    assert not unexpected, f"unexpected keys when loading reference weights: {unexpected}"

    batch, txtlen, imglen = 2, 5, 9
    cdim = cfg["channels"] * cfg["patch"] ** 2
    seq = txtlen + imglen
    gen = torch.Generator().manual_seed(1)
    img = torch.randn(batch, imglen, cdim, generator=gen)
    context = torch.randn(batch, txtlen, cfg["txtlayers"], cfg["txtdim"], generator=gen)
    timestep = torch.rand(batch, generator=gen)
    pos = torch.randint(0, 16, (batch, seq, 3), generator=gen).float()
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[0, -2:] = False  # exercise the key-padding path

    with torch.no_grad():
        out_ref = ref(img, context, timestep, pos, mask)
        out_mine = mine(
            hidden_states=img,
            encoder_hidden_states=context,
            timestep=timestep,
            position_ids=pos,
            attention_mask=mask,
            return_dict=False,
        )[0]

    assert out_ref.shape == out_mine.shape, f"shape mismatch: {out_ref.shape} vs {out_mine.shape}"
    diff = (out_ref - out_mine).abs().max().item()
    rel = diff / (out_ref.abs().max().item() + 1e-8)
    print(f"output shape: {tuple(out_mine.shape)}")
    print(f"max abs diff: {diff:.3e}   max rel diff: {rel:.3e}")
    tol = 1e-4
    assert diff < tol, f"PARITY FAILED: max abs diff {diff:.3e} >= {tol}"
    print("PARITY OK")


if __name__ == "__main__":
    main()
