import time
import torch
import diffusers
import transformers
from modules import shared, sd_models, sd_hijack_te, devices, modelloader, model_quant


def load_quants(kwargs, repo_id, cache_dir):
    kwargs_copy = kwargs.copy()
    if 'Sana_1600M' in repo_id and model_quant.check_nunchaku('Model'): # only sana-1600m
        import nunchaku
        nunchaku_precision = nunchaku.utils.get_precision()
        nunchaku_repo = f"mit-han-lab/svdq-{nunchaku_precision}-sana-1600m"
        shared.log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" precision={nunchaku_precision} attention={shared.opts.nunchaku_attention}')
        kwargs['transformer'] = nunchaku.NunchakuSanaTransformer2DModel.from_pretrained(nunchaku_repo, torch_dtype=devices.dtype)
    elif model_quant.check_quant('Model'):
        load_args, quant_args = model_quant.get_dit_args(kwargs_copy, module='Model')
        kwargs['transformer'] = diffusers.SanaTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, **load_args, **quant_args)
    if model_quant.check_quant('TE'):
        load_args, quant_args = model_quant.get_dit_args(kwargs_copy, module='TE')
        kwargs['text_encoder'] = transformers.AutoModel.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=cache_dir, **load_args, **quant_args)
    return kwargs


def load_sana(checkpoint_info, kwargs={}):
    modelloader.hf_login()
    repo_id = sd_models.path_to_repo(checkpoint_info)

    kwargs.pop('load_connected_pipeline', None)
    kwargs.pop('safety_checker', None)
    kwargs.pop('requires_safety_checker', None)
    kwargs.pop('torch_dtype', None)

    # set variant since hf repos are a mess
    if not repo_id.endswith('_diffusers'):
        repo_id = f'{repo_id}_diffusers'
    if 'Sana_1600M' in repo_id:
        if devices.dtype == torch.bfloat16 or 'BF16' in repo_id:
            if 'BF16' not in repo_id:
                repo_id = repo_id.replace('_diffusers', '_BF16_diffusers')
            kwargs['variant'] = 'bf16'
            kwargs['torch_dtype'] = devices.dtype
        else:
            kwargs['variant'] = 'fp16'
    if 'Sana_600M' in repo_id:
        kwargs['variant'] = 'fp16'

    kwargs = load_quants(kwargs, repo_id, cache_dir=shared.opts.diffusers_dir)
    shared.log.debug(f'Load model: type=Sana repo="{repo_id}" args={list(kwargs)}')
    t0 = time.time()

    if devices.dtype == torch.bfloat16 or devices.dtype == torch.float32:
        kwargs['torch_dtype'] = devices.dtype
    if 'Sprint' in repo_id:
        cls = diffusers.SanaSprintPipeline
    else:
        cls = diffusers.SanaPipeline
    pipe = cls.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        **kwargs,
    )

    # only cast if not quant-loaded
    try:
        if devices.dtype == torch.bfloat16 or devices.dtype == torch.float32:
            if 'transformer' not in kwargs:
                pipe.transformer = pipe.transformer.to(dtype=devices.dtype)
            if 'text_encoder' not in kwargs:
                pipe.text_encoder = pipe.text_encoder.to(dtype=devices.dtype)
            pipe.vae = pipe.vae.to(dtype=devices.dtype)
        if devices.dtype == torch.float16:
            if 'transformer' not in kwargs:
                pipe.transformer = pipe.transformer.to(dtype=devices.dtype)
            if 'text_encoder' not in kwargs:
                pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float32) # gemma2 does not support fp16
            pipe.vae = pipe.vae.to(dtype=torch.float32) # dc-ae often overflows in fp16
    except Exception as e:
        shared.log.error(f'Load model: type=Sana {e}')

    try:
        if shared.opts.diffusers_eval:
            pipe.text_encoder.eval()
            pipe.transformer.eval()
    except Exception:
        pass

    sd_hijack_te.init_hijack(pipe)
    t1 = time.time()
    shared.log.debug(f'Load model: type=Sana target={devices.dtype} te={pipe.text_encoder.dtype} transformer={pipe.transformer.dtype} vae={pipe.vae.dtype} time={t1-t0:.2f}')
    devices.torch_gc(force=True, reason='load')
    return pipe
