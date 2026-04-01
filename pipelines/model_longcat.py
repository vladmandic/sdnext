import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_longcat(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=LongCat repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.LongCatImageTransformer2DModel, load_config=diffusers_load_config)
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config)
    text_processor = transformers.Qwen2VLProcessor.from_pretrained(repo_id, subfolder='tokenizer', cache_dir=shared.opts.hfcache_dir)

    if 'edit' in repo_id.lower():
        cls = diffusers.LongCatImageEditPipeline
    else:
        cls = diffusers.LongCatImagePipeline

    pipe = cls.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        text_processor=text_processor,
        **load_args,
    )
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["longcat"] = cls
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["longcat"] = cls
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["longcat"] = cls

    del transformer
    del text_encoder
    del text_processor
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
