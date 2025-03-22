import os
import torch
import diffusers
from modules import shared, processing
from modules.video_models.models_def import Model


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_override(selected: Model):
    kwargs = {}
    if selected.name == 'Allegro T2V':
        kwargs['vae'] = diffusers.AutoencoderKLAllegro.from_pretrained(selected.repo,
                                                                       subfolder="vae",
                                                                       torch_dtype=torch.float32,
                                                                       cache_dir=shared.opts.hfcache_dir)
        debug(f'Video overrides: model="{selected.name}" kwargs={list(kwargs)}')
    return kwargs


def set_overrides(p: processing.StableDiffusionProcessingVideo, selected: Model):
    if selected.name == 'Latte 1 T2V':
        p.task_args['enable_temporal_attentions'] = False
        debug(f'Video overrides: model="{selected.name}" args={p.task_args}')
