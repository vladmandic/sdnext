import sys
import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_vibe(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=VIBE repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines.vibe import VIBESanaEditingModel, VIBESanaEditingPipeline, VIBESanaImagePipeline
    diffusers.VIBESanaEditingPipeline = VIBESanaEditingPipeline
    diffusers.VIBESanaEditingModel = VIBESanaEditingModel

    sys.modules['vibe.transformer.vibe_sana_editing'] = diffusers # monkey patch since hf model_index.json points to custom class path

    from pipelines.vibe import VIBE_SPEC
    transformer = generic.load_transformer(
        repo_id,
        cls_name=VIBESanaEditingModel,
        load_config=diffusers_load_config,
        allow_quant=False,
        native_spec=VIBE_SPEC,
    )
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.Qwen3VLForConditionalGeneration,
        load_config=diffusers_load_config,
        allow_quant=False,
        allow_shared=False,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(
        repo_id,
        subfolder='tokenizer',
        cache_dir=shared.opts.hfcache_dir,
    )

    pipe = VIBESanaEditingPipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=processor,
        **load_args,
    )

    del transformer
    del text_encoder
    del processor

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['vibe-sana'] = VIBESanaEditingPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['vibe-sana'] = VIBESanaImagePipeline

    pipe.task_args = {
        'output_type': 'np',
    }

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
