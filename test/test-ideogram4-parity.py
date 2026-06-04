#!/usr/bin/env python
"""Numerical parity gate for the ported Ideogram 4 transformer.

Validates that ``pipelines.ideogram4.transformer_ideogram4.Ideogram4Transformer2DModel``
reproduces the upstream reference (github.com/ideogram-oss/ideogram4) exactly:
identical parameter names/shapes (checked via ``load_state_dict``) and identical
forward outputs on fixed inputs. A small config is used so it runs on CPU in
seconds rather than instantiating the real 9B model.

The upstream reference module is fetched at run time from a pinned commit into a
temp dir and imported as the parity oracle; the test SKIPS cleanly when it cannot
be fetched (offline). Run from the repo root:

    python test/test-ideogram4-parity.py
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

REF_COMMIT = "19fc3af67fd7a98b7accf0844e50eda50af9bdc9"
REF_BASE = f"https://raw.githubusercontent.com/ideogram-oss/ideogram4/{REF_COMMIT}/src/ideogram4"
REF_FILES = ("constants.py", "modeling_ideogram4.py")

# Small config (both reference and port use it) so parity runs on CPU in seconds.
SMALL = {
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "num_layers": 2,
    "intermediate_size": 128,
    "adaln_dim": 32,
    "in_channels": 16,
    "llm_features_dim": 48,
    "mrope_section": (2, 2, 2),
    "rope_theta": 5_000_000,
    "norm_eps": 1e-5,
}


def fetch_reference():
    """Download the pinned upstream modeling module into a temp package and import it."""
    tmp = tempfile.mkdtemp(prefix="ideogram4_ref_")
    pkg = os.path.join(tmp, "ideogram4_ref")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf8"):
        pass
    for fn in REF_FILES:
        with urllib.request.urlopen(f"{REF_BASE}/{fn}", timeout=30) as resp:
            data = resp.read().decode("utf8")
        # upstream imports `from ideogram4.constants import ...`; point it at the temp package
        data = data.replace("from ideogram4.constants", "from ideogram4_ref.constants")
        with open(os.path.join(pkg, fn), "w", encoding="utf8") as f:
            f.write(data)
    sys.path.insert(0, tmp)
    return importlib.import_module("ideogram4_ref.modeling_ideogram4")


def check_latent_norm() -> None:
    """Guard the latent-norm constants (offline): they are the reference values, not VAE BatchNorm stats."""
    from pipelines.ideogram4.latent_norm import get_latent_norm

    shift, scale = get_latent_norm()
    assert shift.shape == (128,) and scale.shape == (128,)
    ref_shift = torch.tensor([0.01984364, 0.10149707, 0.29689495, 0.27188619])
    ref_scale = torch.tensor([1.63933691, 1.70204478, 1.73642566, 1.90004803])
    assert torch.allclose(shift[:4], ref_shift, atol=1e-6), f"latent shift regressed: {shift[:4].tolist()}"
    assert torch.allclose(scale[:4], ref_scale, atol=1e-6), f"latent scale regressed: {scale[:4].tolist()}"
    print("PASS: latent-norm constants match reference")


def main() -> int:
    check_latent_norm()
    try:
        ref = fetch_reference()
    except Exception as e:
        print(f"SKIP: could not fetch upstream reference ({e})")
        return 0

    from pipelines.ideogram4.transformer_ideogram4 import Ideogram4Transformer2DModel

    torch.manual_seed(0)
    ref_cfg = ref.Ideogram4Config(
        emb_dim=SMALL["num_attention_heads"] * SMALL["attention_head_dim"],
        num_layers=SMALL["num_layers"],
        num_heads=SMALL["num_attention_heads"],
        intermediate_size=SMALL["intermediate_size"],
        adanln_dim=SMALL["adaln_dim"],
        in_channels=SMALL["in_channels"],
        llm_features_dim=SMALL["llm_features_dim"],
        rope_theta=SMALL["rope_theta"],
        mrope_section=SMALL["mrope_section"],
        norm_eps=SMALL["norm_eps"],
    )
    ref_model = ref.Ideogram4Transformer(ref_cfg).eval()
    mine = Ideogram4Transformer2DModel(**SMALL).eval()

    # 1. structural parity: names + shapes must match 1:1 (the shipped checkpoint
    #    was saved from the reference, so a clean load proves load compatibility).
    missing, unexpected = mine.load_state_dict(ref_model.state_dict(), strict=False)
    assert not missing, f"missing keys in port: {missing}"
    assert not unexpected, f"unexpected keys in port: {unexpected}"

    # 2. numerical parity on fixed inputs.
    n_text, n_img = 3, 4
    seq_len = n_text + n_img
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(1, seq_len, SMALL["in_channels"], generator=gen)
    t = torch.rand(1, generator=gen)
    llm = torch.randn(1, seq_len, SMALL["llm_features_dim"], generator=gen)
    position_ids = torch.randint(0, 64, (1, seq_len, 3), generator=gen)
    segment_ids = torch.ones(1, seq_len, dtype=torch.long)
    indicator = torch.tensor([[ref.LLM_TOKEN_INDICATOR] * n_text + [ref.OUTPUT_IMAGE_INDICATOR] * n_img], dtype=torch.long)

    kwargs = {"llm_features": llm, "x": x, "t": t, "position_ids": position_ids, "segment_ids": segment_ids, "indicator": indicator}
    with torch.no_grad():
        out_ref = ref_model(**kwargs)
        out_mine = mine(**kwargs)

    image_tokens = indicator[0] == ref.OUTPUT_IMAGE_INDICATOR
    diff = (out_ref[:, image_tokens] - out_mine[:, image_tokens]).abs().max().item()
    print(f"max abs diff (image tokens): {diff:.3e}")
    assert diff < 1e-4, f"parity FAILED: max diff {diff}"
    print("PASS: ported transformer matches upstream reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
