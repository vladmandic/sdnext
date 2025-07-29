import sys
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te


def load_dit(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    shared.log.debug(f'Load model: type=FLite dit="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    import pipelines.f_lite
    sys.modules['f_lite'] = pipelines.f_lite
    transformer = pipelines.f_lite.DiT.from_pretrained(
        repo_id,
        subfolder="dit_model",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
        sd_models.move_model(transformer, devices.cpu)
    return transformer


def load_text_encoder(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=FLite te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
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


def load_flite(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    from pipelines.f_lite import FLitePipeline
    dit_model = load_dit(repo_id, diffusers_load_config)
    text_encoder = load_text_encoder(repo_id, diffusers_load_config)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=FLite model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')
    pipe = FLitePipeline.from_pretrained(
        repo_id,
        revision="refs/pr/8",
        dit_model=dit_model,
        text_encoder=text_encoder,
        trust_remote_code=True,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    sd_hijack_te.init_hijack(pipe)
    del text_encoder
    del dit_model

    devices.torch_gc()
    return pipe
