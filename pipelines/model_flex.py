import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_flex(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Flex repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.FluxTransformer2DModel, load_config=diffusers_load_config)
    text_encoder_2 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder_2")

    from pipelines.flex2 import Flex2Pipeline
    pipe = Flex2Pipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["flex2"] = Flex2Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["flex2"] = Flex2Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["flex2"] = Flex2Pipeline

    del text_encoder_2
    del transformer
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc()
    return pipe
