import os
import torch
import diffusers
from modules import shared, processing, devices
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
    # OzzyGT LTX-2.3 mirrors pack connectors/ twice by design: sharded (*-00001-of-0000N +
    # .index.json) and unsharded diffusion_pytorch_model.safetensors of the byte-identical
    # weights. snapshot_download faithfully fetches both; diffusers' component loader picks
    # sharded when the index is present. ignore_patterns skips the ~6.3 GB unsharded copy
    # without reaching for a cleaner upstream mirror.
    ltx2_redundant_connector_repos = {
        'OzzyGT/LTX-2.3',
        'OzzyGT/LTX-2.3-sdnq-dynamic-int4',
    }
    if selected.repo in ltx2_redundant_connector_repos:
        kwargs['ignore_patterns'] = ['connectors/diffusion_pytorch_model.safetensors']
    # LTX2TextConnectors weights are byte-identical across all 2.3 variants (verified by blob
    # hash). Pre-load from a canonical repo so per-variant fetches skip connectors/ entirely.
    # FP16 variants share OzzyGT/LTX-2.3; SDNQ variants share the pre-quantized mirror.
    ltx2_connectors_cls = None
    try:
        from diffusers.pipelines.ltx2 import LTX2TextConnectors
        ltx2_connectors_cls = LTX2TextConnectors
    except ImportError as e:
        log.warning(f'Video load: LTX2TextConnectors unavailable ({e}); dedup of LTX-2.3 connectors disabled')
    if ('LTXVideo 2.3' in selected.name and shared.opts.te_shared_t5 and ltx2_connectors_cls is not None):
        conn_repo = 'OzzyGT/LTX-2.3-sdnq-dynamic-int4' if 'SDNQ' in selected.name else 'OzzyGT/LTX-2.3'
        log.debug(f'Video load: module=connectors repo="{conn_repo}" cls={ltx2_connectors_cls.__name__} shared={shared.opts.te_shared_t5}')
        kwargs['connectors'] = ltx2_connectors_cls.from_pretrained(
            conn_repo,
            subfolder='connectors',
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
            ignore_patterns=['connectors/diffusion_pytorch_model.safetensors'],
            **load_args,
        )
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
    ltx_i2v_classes = ('LTXImageToVideoPipeline', 'LTXConditionPipeline', 'LTX2ImageToVideoPipeline', 'LTX2ConditionPipeline')
    if cls in ltx_i2v_classes:
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
