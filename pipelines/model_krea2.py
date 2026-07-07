import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_krea2(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Krea2 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.krea2.transformer_krea2 import Krea2Transformer2DModel
    from pipelines.krea2.pipeline_krea2 import Krea2Pipeline, Krea2Img2ImgPipeline
    from pipelines.krea2 import KREA2_SPEC
    diffusers.Krea2Transformer2DModel = Krea2Transformer2DModel
    diffusers.Krea2Pipeline = Krea2Pipeline
    diffusers.Krea2Img2ImgPipeline = Krea2Img2ImgPipeline
    generic.set_pipeline('Krea2', Krea2Pipeline)
    # One class per task so get_diffusers_task defaults to text2image and set_diffuser_pipe switches
    # to the img2img variant cleanly (matches the Chroma/Qwen per-task-class pattern).
    from diffusers.pipelines import auto_pipeline
    auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['krea2'] = Krea2Pipeline
    auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['krea2'] = Krea2Img2ImgPipeline
    if repo_id is None or repo_id.lower() == 'none':
        return None

    # Keep small/sensitive layers in compute dtype. `first` (in=64) and `txtfusion.projector`
    # (in=12) are below the int8 GEMM's minimum K; `last` is the output projection; `tmlp`/`tproj`
    # produce the global per-block modulation, too int8-sensitive to quantize (its error compounds
    # across blocks and steps). All are tiny next to the 28 blocks, so the memory cost is small.
    transformer = generic.load_transformer(repo_id, cls_name=Krea2Transformer2DModel, load_config=diffusers_load_config, native_spec=KREA2_SPEC, modules_to_not_convert=['first', 'last', 'projector', 'tmlp', 'tproj'])
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3VLModel, load_config=diffusers_load_config)

    pipe = Krea2Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        **load_args,
    )

    generic.load_vae_override(pipe, diffusers_load_config)

    del transformer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
