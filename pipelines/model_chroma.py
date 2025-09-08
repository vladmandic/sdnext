import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from pipelines import generic


def load_chroma(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=Chroma repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.ChromaTransformer2DModel, load_config=diffusers_load_config, modules_to_not_convert=["distilled_guidance_layer"])
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config)

    pipe = diffusers.ChromaPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["chroma"] = diffusers.ChromaPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["chroma"] = diffusers.ChromaImg2ImgPipeline
    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
