import transformers
import diffusers


def load_lumina(_checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader
    modelloader.hf_login()
    # {'low_cpu_mem_usage': True, 'torch_dtype': torch.float16, 'load_connected_pipeline': True, 'safety_checker': None, 'requires_safety_checker': False}
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = 'torch.float16'
    if 'low_cpu_mem_usage' in diffusers_load_config:
        del diffusers_load_config['low_cpu_mem_usage']
    if 'load_connected_pipeline' in diffusers_load_config:
        del diffusers_load_config['load_connected_pipeline']
    if 'safety_checker' in diffusers_load_config:
        del diffusers_load_config['safety_checker']
    if 'requires_safety_checker' in diffusers_load_config:
        del diffusers_load_config['requires_safety_checker']
    pipe = diffusers.LuminaText2ImgPipeline.from_pretrained(
        'Alpha-VLLM/Lumina-Next-SFT-diffusers',
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    devices.torch_gc()
    return pipe


def load_lumina2(checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, sd_models, model_quant
    quant_args = {}
    quant_args = model_quant.create_bnb_config(quant_args)
    if quant_args:
        model_quant.load_bnb(f'Load model: type=Lumina quant={quant_args}')
    if not quant_args:
        quant_args = model_quant.create_config()
    kwargs = {}
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    if (('Model' in shared.opts.bnb_quantization or 'Model' in shared.opts.torchao_quantization or 'Model' in shared.opts.quanto_quantization) or ('Transformer' in shared.opts.bnb_quantization or 'Transformer' in shared.opts.torchao_quantization or 'Transformer' in shared.opts.quanto_quantization)):
        kwargs['transformer'] = diffusers.Lumina2Transformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype, **quant_args)
    if ('TE' in shared.opts.bnb_quantization or 'TE' in shared.opts.torchao_quantization):
        kwargs['text_encoder'] = transformers.AutoModel.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype, **quant_args)
    sd_model = diffusers.Lumina2Text2ImgPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config, **quant_args, **kwargs)
    return sd_model
