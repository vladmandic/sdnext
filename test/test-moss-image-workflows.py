"""Focused tests for MOSS-Image img2img latent and schedule handling."""

from types import SimpleNamespace

import torch
from PIL import Image

from pipelines.moss.moss_image import MOSSImageFlowMatchScheduler, MOSSImagePipeline
from pipelines.moss.moss_image_i2i import _encode_image, _pack_latents, _prepare_mask, _prepare_schedule
from pipelines.moss.moss_transformer_2d import MOSSImageTransformer2DModel


def test_sdnq_text_encoder_class_matches_checkpoint_config():
    import transformers
    from pipelines.generic_text_encoder import get_shared

    cls = transformers.Qwen3_5TextModel
    assert cls.config_class.__name__ == 'Qwen3_5TextConfig'
    target_repo, args = get_shared(
        cls,
        'OzzyGT/MOSS_Image_SFT_sdnq_dynamic_4bit',
        subfolder='text_encoder',
    )
    assert target_repo == 'OzzyGT/MOSS_Image_Base_sdnq_dynamic_4bit'
    assert args.get('subfolder') == 'text_encoder'


def test_img2img_encoding_accepts_transformer_in_channels_config():
    class FakeVae:
        dtype = torch.float32
        bn = torch.nn.BatchNorm2d(128)
        config = SimpleNamespace(batch_norm_eps=1e-5)

        def encode(self, pixels):
            latents = torch.zeros(pixels.shape[0], 32, 4, 4, device=pixels.device)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: latents))

    pipe = SimpleNamespace(
        vae=FakeVae(),
        vae_scale_factor=8,
        patch_size=2,
        transformer=SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(in_channels=128)),
    )

    encoded = _encode_image(pipe, Image.new('RGB', (16, 16)), 16, 16, torch.device('cpu'), None)

    assert encoded.shape == (1, 128, 2, 2)


def test_upstream_transformer_layout_runs_with_padded_prompt_masks():
    transformer = MOSSImageTransformer2DModel(
        in_channels=4,
        num_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_dim=16,
        text_embed_dim=12,
        timestep_embed_dim=8,
    )
    expected_keys = {
        'context_refiner.0.attn.to_qkv.weight',
        'transformer_blocks.0.attn.to_q.weight',
        'transformer_blocks.0.ff_in.weight',
    }
    assert expected_keys <= set(transformer.state_dict())

    output = transformer(
        torch.randn(2, 4, 2, 2),
        torch.tensor([500, 300]),
        torch.randn(2, 3, 12),
        torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
    )

    assert output.sample.shape == (2, 4, 2, 2)


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
