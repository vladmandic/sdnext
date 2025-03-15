import transformers
import diffusers
from modules import shared, devices, sd_models


def load_common(diffusers_load_config={}, module=None):
    from modules import model_quant, modelloader
    modelloader.hf_login()

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

    quant_args = {}
    if not quant_args:
        quant_args = model_quant.create_bnb_config(quant_args, module=module)
    if not quant_args:
        quant_args = model_quant.create_ao_config(quant_args, module=module)
    if quant_args:
        shared.log.debug(f'Load model: type=CogView quantization module="{module}" {quant_args}')

    return diffusers_load_config, quant_args


def load_cogview3(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    shared.log.debug(f'Load model: type=CogView3 model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    diffusers_load_config, quant_args = load_common(diffusers_load_config, module='Model')
    transformer = diffusers.CogView3PlusTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
        **quant_args,
    )

    diffusers_load_config, quant_args = load_common(diffusers_load_config, module='Text Encoder')
    text_encoder = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
        **quant_args,
    )

    pipe = diffusers.CogView3PlusPipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    devices.torch_gc()
    return pipe


def load_cogview4(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    shared.log.debug(f'Load model: type=CogView4 model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    diffusers_load_config, quant_args = load_common(diffusers_load_config, module='Model')
    transformer = diffusers.CogView4Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
        **quant_args,
    )

    diffusers_load_config, quant_args = load_common(diffusers_load_config, module='Text Encoder')
    text_encoder = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
        **quant_args,
    )

    pipe = diffusers.CogView4Pipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    if shared.opts.diffusers_eval:
        pipe.text_encoder.eval()
        pipe.transformer.eval()
    pipe.enable_model_cpu_offload() # TODO cogview4: balanced offload does not work for GlmModel
    devices.torch_gc()
    return pipe
