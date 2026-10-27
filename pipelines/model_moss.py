import diffusers
import transformers
from modules import shared, devices, sd_models, sd_detect, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic
from pipelines.generic_text_encoder import check_linear_attention_kernels


def load_moss_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=MOSSImage repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.moss.moss_image import MOSSImagePipeline, MOSSImageTransformer, MOSSImageFlowMatchScheduler
    from pipelines.moss.moss_transformer_2d import MOSSImageTransformer2DModel
    from pipelines.moss.moss_image_i2i import MOSSImageToImagePipeline
    from pipelines.moss.moss_image_inpaint import MOSSImageInpaintPipeline
    generic.set_pipeline('MOSSImage', MOSSImagePipeline)
    auto_pipeline = diffusers.pipelines.auto_pipeline
    auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['mossimage'] = MOSSImagePipeline
    auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['mossimage'] = MOSSImageToImagePipeline
    auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING['mossimage'] = MOSSImageInpaintPipeline

    if repo_id is None or repo_id.lower() == 'none':
        return None

    check_linear_attention_kernels('Qwen3_5Model') # before the transformer load, so an unsupported setup fails fast
    # the repo's model_index names classes from its remote-code file, so the pipeline is assembled here instead of through from_pretrained
    sd_detect.get_model_index(repo_id) # the model list only shows cached repos that hold model_index.json

    # Dual transformers implementation:
    # - MOSSImageTransformer2DModel comes from https://huggingface.co/OzzyGT/MOSS_Image_Base_sdnq_dynamic_4bit/blob/main/transformer/transformer_moss_image.py
    #   which is part of OzzyGT custom modular blocks https://huggingface.co/OzzyGT/moss_image_custom_blocks and is used by SDNQ pre-quant converted models
    # - MOSSImageTransformer comes from the vendored implementation and is used for reference model format SD-Next/MOSS-Image-SFT

    is_sdnq = 'sdnq' in repo_id.lower()
    transformer_cls = MOSSImageTransformer2DModel if is_sdnq else MOSSImageTransformer
    if is_sdnq:
        modules_to_not_convert = ['x_embedder', 'context_embedder', 'time_embedder', 'norm_out', 'proj_out']
    else:
        modules_to_not_convert = ['video_embedder', '.text_embedder.proj', 'time_embedding', 'shared_block_modulation', 'video_head_modulation', 'final_linear_video']
    from pipelines.moss import MOSS_SDNQ_SPEC, MOSS_SPEC
    transformer = generic.load_transformer(
        repo_id,
        cls_name=transformer_cls,
        load_config=diffusers_load_config,
        native_spec=MOSS_SDNQ_SPEC if is_sdnq else MOSS_SPEC,
        modules_to_not_convert=modules_to_not_convert,
    )
    text_encoder_cls = transformers.Qwen3_5TextModel if is_sdnq else transformers.Qwen3_5Model
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=text_encoder_cls,
        load_config=diffusers_load_config,
    )

    pipe = MOSSImagePipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=transformers.Qwen2Tokenizer.from_pretrained(repo_id, subfolder='tokenizer', cache_dir=shared.opts.diffusers_dir),
        vae=diffusers.AutoencoderKLFlux2.from_pretrained(repo_id, subfolder='vae', cache_dir=shared.opts.diffusers_dir, **load_args),
        scheduler=MOSSImageFlowMatchScheduler.from_pretrained(repo_id, subfolder='scheduler', cache_dir=shared.opts.diffusers_dir),
    )
    pipe.register_to_config(_name_or_path=repo_id)
    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer, text_encoder
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe


"""
def load_moss_image_modular(checkpoint_info, diffusers_load_config=None):
    class MossImageModularPipeline(diffusers.ModularPipeline): # empty pipeline so it has its own class
        pass

    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    generic.set_pipeline('MOSSImage', MossImageModularPipeline)

    if repo_id is None or repo_id.lower() == 'none':
        return None
    sd_models.hf_auth_check(checkpoint_info)

    check_linear_attention_kernels('Qwen3_5Model') # before the transformer load, so an unsupported setup fails fast
    # the repo's model_index names classes from its remote-code file, so the pipeline is assembled here instead of through from_pretrained
    sd_detect.get_model_index(repo_id) # the model list only shows cached repos that hold model_index.json

    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    log.debug(f'Load model: type=MossImage repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    from modules.modular_load import load_modular_pipe, missing_components
    from pipelines.moss.modular_blocks_moss_image import MOSSImageBlocks
    blocks = MOSSImageBlocks(components_repo=repo_id, trust_components_code=True)
    pipe = load_modular_pipe(
        MossImageModularPipeline,
        repo_id,
        blocks=blocks,
        offline_args=offline_args,
        base=True,
        load_config=diffusers_load_config,
        trust_remote_code=False,
        load_repo_config=False,
    )
    if pipe is None:
        return None

    missing = missing_components(pipe, workflow=None)
    if missing:
        raise RuntimeError(f'MOSS-Image components failed to load: {missing}')
    pipe.task_args = {'output_type': 'np'} # pylint: disable=attribute-defined-outside-init
    pipe.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    del blocks
    devices.torch_gc(force=True, reason='load')
    return pipe


def load_moss_image(checkpoint_info, diffusers_load_config=None):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    if 'sdnq' in repo_id.lower():
        return load_moss_image_modular(checkpoint_info, diffusers_load_config)
    else:
        return load_moss_image_standard(checkpoint_info, diffusers_load_config)
"""
