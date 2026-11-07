"""MOSS and shared VAE precision regressions; no server or model weights required.

Run: python test/test-moss-loading-precision.py
Application helpers are compiled in isolation to avoid initializing global app/device state.
The vendored MOSS pipeline itself runs with a tiny transformer and a recording VAE.
"""

import ast
import contextlib
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf8'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), 'exec'), namespace)  # pylint: disable=exec-used
    return namespace[name]


MOSS = runpy.run_path(str(ROOT / 'pipelines/moss/moss_image.py'))
BASE = ModuleType('pipelines.moss.moss_image')
BASE.__dict__.update(MOSS)
with patch.dict(sys.modules, {'pipelines.moss.moss_image': BASE}):
    I2I = runpy.run_path(str(ROOT / 'pipelines/moss/moss_image_i2i.py'))


class Config(dict):
    __getattr__ = dict.__getitem__


class RecordingVae(torch.nn.Module):
    def __init__(self, dtype, force_upcast):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        self.bn = torch.nn.BatchNorm2d(128).to(dtype=dtype)
        self.config = Config(force_upcast=force_upcast, batch_norm_eps=1e-5)
        self.seen = []
        self.fail = False

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self):
        return self.weight.device

    def encode(self, pixels):
        self.seen.append((self.dtype, pixels.dtype))
        if self.fail:
            raise RuntimeError('test VAE failure')
        latents = torch.zeros(pixels.shape[0], 32, 2, 2, dtype=self.dtype)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda **kwargs: latents))

    def decode(self, latents, **_kwargs):
        self.seen.append((self.dtype, latents.dtype))
        if self.fail:
            raise RuntimeError('test VAE failure')
        return (torch.zeros(latents.shape[0], 3, 16, 16, dtype=self.dtype),)


class TestVaePrecision(unittest.TestCase):
    def test_direct_moss_decode(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for force in (False, True):
                with self.subTest(dtype=dtype, force=force):
                    vae = RecordingVae(dtype, force)
                    transformer = MOSS['MOSSImageTransformer'](dim=16, ffn_dim=16, num_heads=2, num_layers=1, text_dim=8, freq_dim=8)
                    transformer.forward = Mock(return_value=SimpleNamespace(sample=torch.zeros(1, 128, 1, 1)))
                    pipe = MOSS['MOSSImagePipeline'](transformer, MOSS['MOSSImageFlowMatchScheduler'](), vae=vae)
                    pipe.set_progress_bar_config(disable=True)
                    result = pipe(prompt_embeds=torch.zeros(1, 2, 8), height=16, width=16, num_inference_steps=1,
                                  guidance_scale=1, uncond_skip_layers=(), output_type='pt')
                    expected = torch.float32 if force else dtype
                    self.assertEqual(vae.seen, [(expected, expected)])
                    self.assertEqual(vae.dtype, dtype)
                    self.assertTrue(torch.isfinite(result.images).all())

    def test_decode_failure_restores_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            vae = RecordingVae(dtype, True)
            vae.fail = True
            with self.assertRaisesRegex(RuntimeError, 'test VAE failure'):
                with MOSS['vae_precision'](vae):
                    vae.decode(torch.zeros(1, 32, 2, 2, dtype=vae.dtype))
            self.assertEqual(vae.seen, [(torch.float32, torch.float32)])
            self.assertEqual(vae.dtype, dtype)

    def test_img2img_encode_and_failure_restore(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for force in (False, True):
                for fail in (False, True):
                    with self.subTest(dtype=dtype, force=force, fail=fail):
                        vae = RecordingVae(dtype, force)
                        vae.fail = fail
                        pipe = SimpleNamespace(vae=vae, vae_scale_factor=8, patch_size=2,
                                               transformer=SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(in_channels=128)))
                        context = self.assertRaisesRegex(RuntimeError, 'test VAE failure') if fail else contextlib.nullcontext()
                        with context:
                            I2I['_encode_image'](pipe, torch.zeros(1, 3, 16, 16), 16, 16, torch.device('cpu'), None)
                        expected = torch.float32 if force else dtype
                        self.assertEqual(vae.seen, [(expected, expected)])
                        self.assertEqual(vae.dtype, dtype)

    def test_shared_encode_decode(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for force, no_half in ((False, False), (True, False), (False, True)):
                for operation in ('encode', 'decode'):
                    with self.subTest(dtype=dtype, force=force, no_half=no_half, operation=operation):
                        opts = SimpleNamespace(diffusers_offload_mode='none', no_half_vae=no_half, cuda_compile=[])
                        namespace = dict(torch=torch, time=SimpleNamespace(time=lambda: 0), debug=False,
                                         shared=SimpleNamespace(opts=opts), devices=SimpleNamespace(device=torch.device('cpu'), inference_context=contextlib.nullcontext),
                                         sd_models=Mock(), sd_vae=SimpleNamespace(loaded_vae_file=None, get_vae_scale_factor=lambda *a, **kw: 8),
                                         log=Mock(), log_debug=Mock(), errors=Mock())
                        fn = load_function('modules/processing_vae.py', f'full_vae_{operation}', namespace)
                        vae = RecordingVae(dtype, force)
                        fn(torch.zeros(1, 32 if operation == 'decode' else 3, 2, 2), SimpleNamespace(vae=vae))
                        expected = torch.float32 if force or no_half else dtype
                        self.assertEqual(vae.seen, [(expected, expected)])
                        self.assertEqual(vae.dtype, dtype)
                        self.assertFalse(hasattr(vae, 'orig_dtype'))


if __name__ == '__main__':
    unittest.main()
