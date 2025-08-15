import transformers
import diffusers
from modules import shared, devices, modelloader, sd_models, shared_items, sd_hijack_te


def load_meissonic(checkpoint_info, diffusers_load_config={}):
    from pipelines.meissonic.transformer import Transformer2DModel as TransformerMeissonic
    from pipelines.meissonic.scheduler import Scheduler as MeissonicScheduler
    from pipelines.meissonic.pipeline import MeissonicPipeline
    from pipelines.meissonic.pipeline_img2img import MeissonicImg2ImgPipeline
    from pipelines.meissonic.pipeline_inpaint import MeissonicInpaintPipeline
    shared_items.pipelines['Meissonic'] = MeissonicPipeline

    modelloader.hf_login()
    fn = sd_models.path_to_repo(checkpoint_info)
    cache_dir = shared.opts.diffusers_dir

    diffusers_load_config['variant'] = 'fp16'
    diffusers_load_config['trust_remote_code'] = True

    model = TransformerMeissonic.from_pretrained(
        fn,
        subfolder="transformer",
        cache_dir=cache_dir,
        **diffusers_load_config,
    )
    vqvae = diffusers.VQModel.from_pretrained(
        fn,
        subfolder="vqvae",
        cache_dir=cache_dir,
        **diffusers_load_config,
    )
    text_encoder = transformers.CLIPTextModelWithProjection.from_pretrained(
        fn,
        subfolder="text_encoder",
        cache_dir=cache_dir,
    )
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        fn,
        subfolder="tokenizer",
        cache_dir=cache_dir,
    )
    scheduler = MeissonicScheduler.from_pretrained(fn, subfolder="scheduler", cache_dir=cache_dir)
    pipe = MeissonicPipeline(
            vqvae=vqvae.to(devices.dtype),
            text_encoder=text_encoder.to(devices.dtype),
            transformer=model.to(devices.dtype),
            tokenizer=tokenizer,
            scheduler=scheduler,
    )

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["meissonic"] = MeissonicPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["meissonic"] = MeissonicImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["meissonic"] = MeissonicInpaintPipeline
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
