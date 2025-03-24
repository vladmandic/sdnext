import os
import time
import torch
import diffusers
import transformers
from modules import shared, sd_models, devices, modelloader, model_quant


def load_quants(kwargs, repo_id, cache_dir):
    quant_args = {}
    quant_args = model_quant.create_config()
    if not quant_args:
        return kwargs
    load_args = kwargs.copy()
    if 'transformer' not in kwargs and ('Model' in shared.opts.bnb_quantization or 'Model' in shared.opts.torchao_quantization):
        kwargs['transformer'] = diffusers.models.SanaTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, **load_args, **quant_args)
        shared.log.debug(f'Quantization: module=transformer type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
    if 'text_encoder' not in kwargs and ('Text Encoder' in shared.opts.bnb_quantization or 'Text Encoder' in shared.opts.torchao_quantization):
        kwargs['text_encoder'] = transformers.AutoModelForCausalLM.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=cache_dir, **load_args, **quant_args)
        shared.log.debug(f'Quantization: module=t5 type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
    return kwargs


def load_sana(checkpoint_info, kwargs={}):
    modelloader.hf_login()

    fn = checkpoint_info if isinstance(checkpoint_info, str) else checkpoint_info.path
    repo_id = sd_models.path_to_repo(fn)
    kwargs.pop('load_connected_pipeline', None)
    kwargs.pop('safety_checker', None)
    kwargs.pop('requires_safety_checker', None)
    kwargs.pop('torch_dtype', None)

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
    if shared.opts.diffusers_eval:
        pipe.text_encoder.eval()
        pipe.transformer.eval()
    t1 = time.time()
    shared.log.debug(f'Load model: type=Sana target={devices.dtype} te={pipe.text_encoder.dtype} transformer={pipe.transformer.dtype} vae={pipe.vae.dtype} time={t1-t0:.2f}')

    devices.torch_gc()
    return pipe
