import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from pipelines import generic


def load_flux2_klein(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    shared.log.debug(f'Load model: type=Flux2Klein repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # Load transformer - Klein uses Flux2Transformer2DModel (same class as Flux2, different size)
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Flux2Transformer2DModel, load_config=diffusers_load_config)

    # Load text encoder - Klein uses Qwen3ForCausalLM (8B), shared across all Klein variants
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3ForCausalLM, load_config=diffusers_load_config)

    pipe = diffusers.Flux2KleinPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["flux2klein"] = diffusers.Flux2KleinPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["flux2klein"] = diffusers.Flux2KleinPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["flux2klein"] = diffusers.Flux2KleinPipeline

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
