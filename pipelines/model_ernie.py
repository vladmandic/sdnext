import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_ernie_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=ERNIE-Image repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args} pe={shared.opts.model_ernie_enable_pe}')

    transformer = generic.load_transformer(
        repo_id,
        cls_name=diffusers.ErnieImageTransformer2DModel,
        load_config=diffusers_load_config,
    )
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.Mistral3Model,
        load_config=diffusers_load_config,
    )

    if not shared.opts.model_ernie_enable_pe:
        load_args['pe'] = None

    pipe = diffusers.ErnieImagePipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
        'use_pe': shared.opts.model_ernie_enable_pe,
    }

    from pipelines.ernie.ernie_image import ErnieImageImg2ImgPipeline, ErnieImageInpaintPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["ernieimage"] = diffusers.ErnieImagePipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["ernieimage"] = ErnieImageImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["ernieimage"] = ErnieImageInpaintPipeline

    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
