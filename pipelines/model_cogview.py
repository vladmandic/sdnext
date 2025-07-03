import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, modelloader


def load_cogview3(checkpoint_info, diffusers_load_config={}):
    modelloader.hf_login()
    repo_id = sd_models.path_to_repo(checkpoint_info)

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=CogView3 transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    transformer = diffusers.CogView3PlusTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
        **quant_args,
    )

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=CogView3 te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
        **quant_args,
    )

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=CogView3 model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')
    pipe = diffusers.CogView3PlusPipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    devices.torch_gc()
    return pipe


def load_cogview4(checkpoint_info, diffusers_load_config={}):
    modelloader.hf_login()
    repo_id = sd_models.path_to_repo(checkpoint_info)

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=CogView4 transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    transformer = diffusers.CogView4Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
        **quant_args,
    )

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=CogView4 te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
        **quant_args,
    )

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=CogView4 model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')
    pipe = diffusers.CogView4Pipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    if shared.opts.diffusers_eval:
        pipe.text_encoder.eval()
        pipe.transformer.eval()
    pipe.enable_model_cpu_offload() # TODO model fix: cogview4: balanced offload does not work for GlmModel
    devices.torch_gc()
    return pipe
