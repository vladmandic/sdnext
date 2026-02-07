import torch
import diffusers
import transformers
from modules import shared, sd_models, sd_hijack_te, devices, model_quant


def load_quants(kwargs, repo_id, cache_dir):
    kwargs_copy = kwargs.copy()
    if 'Sana_1600M_1024px' in repo_id and model_quant.check_nunchaku('Model'): # only available model
        import nunchaku
        nunchaku_precision = nunchaku.utils.get_precision()
        nunchaku_repo = "nunchaku-ai/nunchaku-sana/svdq-int4_r32-sana1.6b.safetensors"
        shared.log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" precision={nunchaku_precision} attention={shared.opts.nunchaku_attention}')
        kwargs['transformer'] = nunchaku.NunchakuSanaTransformer2DModel.from_pretrained(nunchaku_repo, torch_dtype=devices.dtype, cache_dir=cache_dir)
    elif model_quant.check_quant('Model'):
        load_args, quant_args = model_quant.get_dit_args(kwargs_copy, module='Model')
        kwargs['transformer'] = diffusers.SanaTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, **load_args, **quant_args)
    if model_quant.check_quant('TE'):
        load_args, quant_args = model_quant.get_dit_args(kwargs_copy, module='TE')
        kwargs['text_encoder'] = transformers.AutoModel.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=cache_dir, **load_args, **quant_args)
    return kwargs


def load_sana(checkpoint_info, kwargs=None):
    if kwargs is None:
        kwargs = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

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

    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
