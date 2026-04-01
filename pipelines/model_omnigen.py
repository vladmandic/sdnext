import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log


def load_omnigen(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='Model')
    log.debug(f'Load model: type=OmniGen repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')
    transformer = diffusers.OmniGenTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.OmniGenPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
    )

    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe


def load_omnigen2(checkpoint_info, diffusers_load_config=None): # pylint: disable=unused-argument
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    from pipelines.omnigen2 import OmniGen2Pipeline, OmniGen2Transformer2DModel, Qwen2_5_VLForConditionalGeneration
    diffusers.OmniGen2Pipeline = OmniGen2Pipeline # monkey-pathch
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["omnigen2"] = diffusers.OmniGen2Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["omnigen2"] = diffusers.OmniGen2Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["omnigen2"] = diffusers.OmniGen2Pipeline

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='Model')
    log.debug(f'Load model: type=OmniGen2 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='TE')
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo_id,
        subfolder="mllm",
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_config,
        **quant_config,
    )

    pipe = OmniGen2Pipeline.from_pretrained(
        repo_id,
        # transformer=transformer,
        mllm=mllm,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_config,
    )
    pipe.transformer = transformer # for omnigen2 transformer must be loaded after pipeline

    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
