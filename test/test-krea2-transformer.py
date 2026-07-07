#!/usr/bin/env python
"""Offline tests for the Krea2 transformer port and its native loader handling.

- Parity: Krea2Transformer2DModel vs the reference SingleStreamDiT. Builds both from one tiny
  config, copies the reference state dict into the diffusers port, runs identical inputs, and
  asserts the forward outputs match. Needs the reference (mmdit.py) at $KREA2_REF_DIR
  (default /home/ohiom/database/watering-hole).
- Zero-init regression: a checkpoint that omits the dormant last.up/last.down residual branch,
  after materialize_zero_init, produces the same output as the base whose up is zeroed. Port
  only, so it runs without the reference.

No server, no checkpoint.
"""

import importlib.util
import os
import sys
from contextlib import nullcontext

import torch

REF_DIR = os.environ.get("KREA2_REF_DIR", "/home/ohiom/database/watering-hole")

CFG = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, bias=False, theta=1e3,
    txtlayers=3, txtheads=2, txtkvheads=2,
)


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


def make_inputs():
    batch, txtlen, imglen = 2, 5, 9
    cdim = CFG["channels"] * CFG["patch"] ** 2
    seq = txtlen + imglen
    gen = torch.Generator().manual_seed(1)
    img = torch.randn(batch, imglen, cdim, generator=gen)
    context = torch.randn(batch, txtlen, CFG["txtlayers"], CFG["txtdim"], generator=gen)
    timestep = torch.rand(batch, generator=gen)
    pos = torch.randint(0, 16, (batch, seq, 3), generator=gen).float()
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[0, -2:] = False  # exercise the key-padding path
    return img, context, timestep, pos, mask


def run_parity(mmdit, port):
    torch.manual_seed(0)
    ref = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**CFG)).float().eval()
    mine = port.Krea2Transformer2DModel(**CFG).float().eval()
    missing, unexpected = mine.load_state_dict(ref.state_dict(), strict=False)
    assert not missing, f"missing keys when loading reference weights: {missing}"
    assert not unexpected, f"unexpected keys when loading reference weights: {unexpected}"

    img, context, timestep, pos, mask = make_inputs()
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


def load_materialize_zero_init():
    """Import the real loader helper. native_transformer pulls in modules.shared, which needs
    cmd_args parsed first, so bootstrap it the same way the native-transformer suite does."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("SD_INSTALL_QUIET", "1")
    import modules.cmd_args
    import installer
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        modules.cmd_args.parse_args()
    finally:
        sys.argv = orig_argv
    installer.add_args(modules.cmd_args.parser)
    modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])
    from pipelines.native_transformer import materialize_zero_init
    return materialize_zero_init


def run_zero_init_regression(port):
    """A finetune predating the last.up/last.down branch omits both keys. Zero-filling them
    must reproduce the base's dormant behavior: identical output to a model whose up is zeroed
    (up(down(x)) == 0), while a genuinely missing weight stays a hard mismatch."""
    materialize_zero_init = load_materialize_zero_init()
    branch_keys = ("last.up.weight", "last.down.weight")

    torch.manual_seed(0)
    full_sd = port.Krea2Transformer2DModel(**CFG).state_dict()

    # Ground truth: the branch dormant exactly as the base ships it (up zeroed, down arbitrary).
    dormant = port.Krea2Transformer2DModel(**CFG).float().eval()
    dormant_sd = dict(full_sd)
    dormant_sd["last.up.weight"] = torch.zeros_like(dormant_sd["last.up.weight"])
    dormant.load_state_dict(dormant_sd)

    # Under test: a checkpoint that omits both branch keys; the loader zero-fills them. A real
    # missing weight (blocks.0.attn.wq.weight) is dropped too, to confirm it is NOT zero-filled.
    filled = port.Krea2Transformer2DModel(**CFG).float().eval()
    hard_key = "blocks.0.attn.wq.weight"
    partial_sd = {k: v for k, v in full_sd.items() if k not in branch_keys and k != hard_key}
    missing, unexpected = filled.load_state_dict(partial_sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert set(missing) == set(branch_keys) | {hard_key}, f"unexpected missing set: {missing}"

    remaining = materialize_zero_init(filled, missing, ("last.down.", "last.up."))
    assert remaining == [hard_key], f"expected only {hard_key} to remain hard-missing, got {remaining}"
    for k in branch_keys:
        w = dict(filled.named_parameters())[k]
        assert w.abs().sum().item() == 0.0, f"{k} was not zero-filled"

    # Restore the genuinely-missing weight so the forward is well defined, then compare.
    filled.load_state_dict({hard_key: full_sd[hard_key]}, strict=False)

    img, context, timestep, pos, mask = make_inputs()

    def forward(model):
        with torch.no_grad():
            return model(
                hidden_states=img, encoder_hidden_states=context, timestep=timestep,
                position_ids=pos, attention_mask=mask, return_dict=False,
            )[0]

    diff = (forward(dormant) - forward(filled)).abs().max().item()
    print(f"zero-init regression: max abs diff vs dormant base: {diff:.3e}")
    assert diff == 0.0, f"ZERO-INIT REGRESSION FAILED: filled output differs by {diff:.3e}"
    print("ZERO-INIT OK")


def main():
    port = load_port()
    # Parity runs first in pristine torch state; the regression imports modules afterwards.
    mmdit = load_reference()
    run_parity(mmdit, port)
    run_zero_init_regression(port)


if __name__ == "__main__":
    main()
