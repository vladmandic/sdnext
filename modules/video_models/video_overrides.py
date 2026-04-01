import os
import torch
import diffusers
from modules import shared, processing
from modules.logger import log
from modules.video_models.models_def import Model


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_override(selected: Model, **load_args):
    kwargs = {}
    # Allegro
    if 'Allegro T2V' in selected.name:
        kwargs['vae'] = diffusers.AutoencoderKLAllegro.from_pretrained(selected.repo, subfolder="vae", torch_dtype=torch.float32, cache_dir=shared.opts.hfcache_dir, **load_args)
    # LTX
    if 'LTXVideo 0.9.5 I2V' in selected.name:
        kwargs['vae'] = diffusers.AutoencoderKLLTXVideo.from_pretrained(selected.repo, subfolder="vae", torch_dtype=torch.float32, cache_dir=shared.opts.hfcache_dir, **load_args)
    # WAN
    if 'WAN 2.1 14B' in selected.name:
        kwargs['vae'] = diffusers.AutoencoderKLWan.from_pretrained(selected.repo, subfolder="vae", torch_dtype=torch.float32, cache_dir=shared.opts.hfcache_dir, **load_args)
    if ('A14B' in selected.name) or ('14B VACE' in selected.name):
        if shared.opts.model_wan_stage == 'combined':
            kwargs['boundary_ratio'] = shared.opts.model_wan_boundary
        elif shared.opts.model_wan_stage == 'high noise':
            kwargs['transformer_2'] = None
            kwargs['boundary_ratio'] = 0.0
        elif shared.opts.model_wan_stage == 'low noise':
            kwargs['boundary_ratio'] = 1000.0
            kwargs['transformer'] = None
    debug(f'Video overrides: model="{selected.name}" kwargs={list(kwargs)}')
    return kwargs


def set_overrides(p: processing.StableDiffusionProcessingVideo, selected: Model):
    cls = shared.sd_model.__class__.__name__
    # Allegro
    if selected.name == 'Allegro T2V':
        shared.sd_model.vae.enable_tiling()
    # Latte
    if selected.name == 'Latte 1 T2V':
        p.task_args['enable_temporal_attentions'] = True
        p.task_args['video_length'] = 16 * (max(p.frames // 16, 1))
    # SkyReels
    if 'SkyReelsV2DiffusionForcing' in cls:
        p.task_args['overlap_history'] = 17
    # LTX
    if cls == 'LTXImageToVideoPipeline' or cls == 'LTXConditionPipeline':
        p.task_args['generator'] = None
    if cls == 'LTXConditionPipeline':
        p.task_args['strength'] = p.denoising_strength
    if 'LTX' in cls:
        p.task_args['width'] = 32 * (p.width // 32)
        p.task_args['height'] = 32 * (p.height // 32)
    # WAN
    if 'Wan' in cls:
        p.task_args['width'] = 16 * (p.width // 16)
        p.task_args['height'] = 16 * (p.height // 16)
        p.frames = 4 * (max(p.frames // 4, 1)) + 1
    # WAN VACE
    if 'WanVACEPipeline' in cls:
        if (getattr(p, 'init_images', None) is not None) and (len(p.init_images) > 0):
            p.task_args['reference_images'] = p.init_images
    # WAN 2.2-5B
    if 'WAN 2.2 5B' in selected.name:
        shared.sd_model.vae.disable_tiling()
    # Kandinsky 5
    if 'Kandinsky 5.0 Lite 5s' in selected.name:
        # p.task_args['time_length'] = 5
        pass
    if 'Kandinsky 5.0 Lite 10s' in selected.name:
        # p.task_args['time_length'] = 10
        shared.sd_model.transformer.set_attention_backend("flex")
