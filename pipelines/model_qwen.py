import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te


def load_transformer(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    shared.log.debug(f'Load model: type=Qwen transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
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
    shared.log.debug(f'Load model: type=Qwen te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder is not None:
        sd_models.move_model(text_encoder, devices.cpu)
    return text_encoder


def load_qwen(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    transformer = load_transformer(repo_id, diffusers_load_config)
    text_encoder = load_text_encoder(repo_id, diffusers_load_config)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=Qwen model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    cls = diffusers.QwenImagePipeline
    pipe = cls.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }

    del text_encoder
    del transformer

    sd_hijack_te.init_hijack(pipe)
    from modules.video_models import video_vae
    pipe.vae.orig_decode = pipe.vae.decode
    pipe.vae.decode = video_vae.hijack_vae_decode

    devices.torch_gc()
    return pipe
