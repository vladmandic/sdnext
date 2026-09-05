import diffusers
from modules import shared, devices, sd_models, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_llada_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    log.debug(f'Load model: type=LLaDAImage repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    from pipelines.llada import LLaDAImagePipeline
    from pipelines.llada.transformer_llada_image import LLaDAImageTransformer2DModel
    from pipelines.llada.modeling_llada2uni_moe import LLaDA2MoeModelLM

    generic.set_pipeline('LLaDAImage', LLaDAImagePipeline)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    if 'Model' in shared.opts.sdnq_quantize_weights:
        if any(x in shared.opts.sdnq_quantize_weights_mode for x in ['2', '3', '4', '5', '6']):
            shared.opts.sdnq_quantize_weights_mode = 'uint8'
            log.warning('LLaDAImage: cls=LLaDAImageTransformer2DModel quant=uint8 override')
    if 'TE' in shared.opts.sdnq_quantize_weights:
        if any(x in shared.opts.sdnq_quantize_weights_mode_te for x in ['2', '3', '4', '5', '6']):
            shared.opts.sdnq_quantize_weights_mode_te = 'uint8'
            log.warning('LLaDAImage: cls=LLaDA2MoeModelLM quant=uint8 override')
        if shared.opts.sdnq_quantize_matmul_mode_te != 'disabled':
            shared.opts.sdnq_quantize_matmul_mode_te = 'disabled'
            log.warning('LLaDAImage: cls=LLaDA2MoeModelLM matmul=disabled override')

    transformer = generic.load_transformer(
        repo_id,
        cls_name=LLaDAImageTransformer2DModel,
        load_config=diffusers_load_config,
        modules_to_not_convert=[
            'all_x_embedder',
            'all_final_layer',
            't_embedder',
            'cap_embedder',
            'semantic_embedder',
            'sigvq_embedder',
        ],
    )
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=LLaDA2MoeModelLM,
        load_config=diffusers_load_config,
        allow_shared=False,
        trust_remote_code=True,
        modules_to_not_convert=[
            '.model.language_model.word_embeddings',
            '.model.language_model.norm',
            '.model.lm_head',
        ],
    )

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['llada-image'] = LLaDAImagePipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['llada-image'] = LLaDAImagePipeline

    pipe = LLaDAImagePipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        torch_dtype=devices.dtype,
        transformer=transformer,
        text_encoder=text_encoder,
    )
    pipe.task_args = {
        'output_type': 'np',
    }
    # generation_mode = "text", "vq", "editing"

    del transformer, text_encoder
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
