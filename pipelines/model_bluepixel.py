import diffusers
import transformers

from modules import devices, model_quant, sd_hijack_te, sd_hijack_vae, sd_models, shared
from modules.logger import log
from pipelines import generic


def load_bluepixel(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    if repo_id is None or repo_id.lower() == 'none':
        return None
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    log.debug(f'Load model: type=BluePixel repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines import bluepixel
    from pipelines.flux2_klein import FLUX2_KLEIN_SPEC

    generic.set_pipeline('BluePixel', bluepixel.BluePixelPipeline)
    transformer = generic.load_transformer(
        repo_id,
        cls_name=diffusers.Flux2Transformer2DModel,
        load_config=diffusers_load_config,
        native_spec=FLUX2_KLEIN_SPEC,
    )
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.Qwen3ForCausalLM,
        load_config=diffusers_load_config,
    )
    pipe = bluepixel.BluePixelPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {'output_type': 'np'}
    pipe.patch_size = 2
    pipe.max_condition_images = 4
    pipe.sdnext_task_type = sd_models.DiffusersTaskType.TEXT_2_IMAGE
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['bluepixel'] = bluepixel.BluePixelPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['bluepixel'] = bluepixel.BluePixelPipeline

    generic.load_vae_override(pipe, diffusers_load_config)

    from pipelines.flux import flux2_lora
    flux2_lora.apply_patch()

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
