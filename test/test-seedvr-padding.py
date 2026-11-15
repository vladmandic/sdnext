"""SeedVR2 7B patch round-trip regressions; no model weights or GPU required.

Run from the repository root: python test/test-seedvr-padding.py
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.seedvr.src.common.cache import Cache
from modules.seedvr.src.models.dit import na
from modules.seedvr.src.models.dit.patch import NaPatchIn, NaPatchOut


class TestSeedVRPadding(unittest.TestCase):
    def round_trip(self, shapes, patch_size=(1, 2, 2), disable_cache=False):
        dim = patch_size[0] * patch_size[1] * patch_size[2]
        patch_in = NaPatchIn(in_channels=1, patch_size=patch_size, dim=dim)
        patch_out = NaPatchOut(out_channels=1, patch_size=patch_size, dim=dim)
        with torch.no_grad():
            for layer in (patch_in, patch_out):
                layer.proj.weight.copy_(torch.eye(dim))
                layer.proj.bias.zero_()
            samples = [torch.arange(t * h * w, dtype=torch.float32).reshape(t, h, w, 1) for t, h, w in shapes]
            original, original_shapes = na.flatten(samples)
            patched, patched_shapes = patch_in(original, original_shapes)
            restored, restored_shapes = patch_out(patched, patched_shapes, original_shapes, Cache(disable=disable_cache))
        torch.testing.assert_close(restored_shapes, original_shapes)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)

    def test_logged_failures(self):
        for shape in ((1, 578, 864), (1, 298, 292)):
            for disabled in (False, True):
                with self.subTest(shape=shape, disable_cache=disabled):
                    self.round_trip([shape], disable_cache=disabled)

    def test_spatial_padding_and_unpadded_shapes(self):
        for h in range(1, 9):
            for w in range(1, 9):
                with self.subTest(height=h, width=w):
                    self.round_trip([(1, h, w)])

    def test_mixed_batch(self):
        self.round_trip([(1, 6, 8), (1, 5, 7), (1, 8, 8)])

    def test_temporal_padding(self):
        self.round_trip([(3, 5, 7), (6, 8, 6)], patch_size=(2, 2, 2))

    def test_no_patching(self):
        self.round_trip([(1, 5, 7)], patch_size=(1, 1, 1))


if __name__ == '__main__':
    unittest.main()
