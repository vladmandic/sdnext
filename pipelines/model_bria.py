import os
import sys
import transformers
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

    from pipelines.bria.transformer_bria import BriaTransformer2DModel

    if fn is not None and 'gguf' in fn.lower():
        shared.log.error('Load model: type=Bria format="gguf" unsupported')
        transformer = None
    elif fn is not None and 'safetensors' in fn.lower():
        shared.log.debug(f'Load model: type=Bria transformer="{fn}" quant="{model_quant.get_quant(repo_id)}" args={load_args}')
        transformer = BriaTransformer2DModel.from_single_file(
            fn,
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
        )
    else:
        shared.log.debug(f'Load model: type=Bria transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
        transformer = BriaTransformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
    if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
        sd_models.move_model(transformer, devices.cpu)
    return transformer


def load_text_encoder(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=Bria te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder is not None:
        sd_models.move_model(text_encoder, devices.cpu)
    return text_encoder


def load_bria(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    transformer = load_transformer(repo_id, diffusers_load_config)
    text_encoder = load_text_encoder(repo_id, diffusers_load_config)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=Bria model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.bria.bria_pipeline import BriaPipeline
    sys.path.append(os.path.join(os.path.dirname(__file__), 'bria'))

    pipe = BriaPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_args,
    )

    del text_encoder
    del transformer

    sd_hijack_te.init_hijack(pipe)
    from modules.video_models import video_vae
    pipe.vae.orig_decode = pipe.vae.decode
    pipe.vae.decode = video_vae.hijack_vae_decode

    devices.torch_gc()
    return pipe
