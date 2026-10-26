"""Focused tests for MOSS-Image img2img latent and schedule handling."""

from types import SimpleNamespace

import torch

from pipelines.moss.moss_image import MOSSImageFlowMatchScheduler, MOSSImagePipeline
from pipelines.moss.moss_image_i2i import _pack_latents, _prepare_mask, _prepare_schedule


def test_pack_latents_is_inverse_of_pipeline_unpatchify():
    raw_latents = torch.randn(2, 32, 8, 12)
    packed = _pack_latents(raw_latents)
    bn = torch.nn.BatchNorm2d(128)
    bn.running_mean.copy_(torch.randn_like(bn.running_mean))
    bn.running_var.copy_(torch.rand_like(bn.running_var) + 0.5)
    epsilon = 1e-5
    standard_deviation = torch.sqrt(bn.running_var + epsilon).view(1, -1, 1, 1)
    mean = bn.running_mean.view(1, -1, 1, 1)
    normalized = (packed - mean) / standard_deviation
    pipe = SimpleNamespace(vae=SimpleNamespace(bn=bn, config=SimpleNamespace(batch_norm_eps=epsilon)))

    decoded_latents = MOSSImagePipeline.unpack_latents(pipe, normalized)

    torch.testing.assert_close(decoded_latents, raw_latents)


def test_img2img_strength_selects_fraction_of_denoising_schedule():
    for strength in (0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
        scheduler = MOSSImageFlowMatchScheduler()
        actual_sigma, original = _prepare_schedule(scheduler, 50, strength, torch.device("cpu"))
        total_steps = len(original[0])
        expected_steps = max(1, int(total_steps * strength**0.5))
        start_index = total_steps - expected_steps

        assert len(scheduler.timesteps) == expected_steps
        assert scheduler.sigmas[0].item() == actual_sigma == original[1][start_index].item()
        assert scheduler.sigmas[-1].item() == 0
        assert len(scheduler.sigmas) == len(scheduler.timesteps) + 1


def test_inpaint_mask_matches_packed_latent_grid():
    pipe = SimpleNamespace(vae_scale_factor=8, patch_size=2)
    mask = torch.ones(1, 1, 1024, 1024)

    prepared = _prepare_mask(pipe, mask, 1024, 1024, torch.device("cpu"))

    assert prepared.shape == (1, 1, 64, 64)
