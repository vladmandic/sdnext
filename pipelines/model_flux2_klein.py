import transformers
import diffusers
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic
from pipelines.native_transformer import TransformerSpec


# Klein shares Flux2Transformer2DModel with full Flux 2, but uses a smaller
# config (hidden_size and friends). diffusers' from_single_file picks the
# class default (= Flux 2 full), so loading a Klein-shaped community file
# crashes at load_model_dict_into_meta with a shape mismatch like
# "expected (36864, 6144), got (24576, 4096)". Routing through
# native_transformer pulls the Klein transformer/config.json from the base
# repo first and instantiates Flux2Transformer2DModel at the right size,
# then runs the diffusers Flux 2 converter to split fused QKV blocks and
# rename BFL keys into the diffusers-expected names.
FLUX2_KLEIN_SPEC = TransformerSpec(
    cls=diffusers.Flux2Transformer2DModel,
    converter=convert_flux2_transformer_checkpoint_to_diffusers,
)


def load_flux2_klein(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Flux2Klein repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # Load transformer - Klein uses Flux2Transformer2DModel (same class as Flux2, different size)
    transformer = generic.load_transformer(repo_id, cls_name=diffusers.Flux2Transformer2DModel, load_config=diffusers_load_config, native_spec=FLUX2_KLEIN_SPEC)

    # Load text encoder - Klein uses Qwen3 (4B for Klein-4B, 8B for Klein-9B)
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

    generic.load_vae_override(pipe, diffusers_load_config)

    from pipelines.flux import flux2_lora
    flux2_lora.apply_patch()

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
