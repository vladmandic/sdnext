import os
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te


def load_transformer(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    fn = None

    if shared.opts.sd_unet is not None and shared.opts.sd_unet != 'Default':
        from modules import sd_unet
        if shared.opts.sd_unet not in list(sd_unet.unet_dict):
            shared.log.error(f'Load module: type=Transformer not found: {shared.opts.sd_unet}')
            return None
        fn = sd_unet.unet_dict[shared.opts.sd_unet] if os.path.exists(sd_unet.unet_dict[shared.opts.sd_unet]) else None

    if fn is not None and 'gguf' in fn.lower():
        shared.log.error('Load model: type=HiDream format="gguf" unsupported')
        transformer = None
        from modules import ggml
        transformer = ggml.load_gguf(fn, cls=diffusers.HiDreamImageTransformer2DModel, compute_dtype=devices.dtype)
    elif fn is not None and 'safetensors' in fn.lower():
        shared.log.debug(f'Load model: type=FLEX transformer="{repo_id}" quant="{model_quant.get_quant(repo_id)}" args={load_args}')
        transformer = diffusers.FluxTransformer2DModel.from_single_file(fn, cache_dir=shared.opts.hfcache_dir, **load_args)
    else:
        shared.log.debug(f'Load model: type=FLEX transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
        transformer = diffusers.FluxTransformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
    if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
        sd_models.move_model(transformer, devices.cpu)
    return transformer


def load_text_encoders(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=FLEX t5="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder_2 = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder_2",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder_2 is not None:
        sd_models.move_model(text_encoder_2, devices.cpu)
    return text_encoder_2


def load_flex(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    transformer = load_transformer(repo_id, diffusers_load_config)
    text_encoder_2 = load_text_encoders(repo_id, diffusers_load_config)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=FLEX model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.flex2 import Flex2Pipeline
    pipe = Flex2Pipeline.from_pretrained(
        repo_id,
        # custom_pipeline=repo_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    sd_hijack_te.init_hijack(pipe)
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["flex2"] = Flex2Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["flex2"] = Flex2Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["flex2"] = Flex2Pipeline

    del text_encoder_2
    del transformer

    devices.torch_gc()
    return pipe
