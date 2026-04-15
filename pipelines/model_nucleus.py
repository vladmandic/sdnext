import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_nucleus(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=NucleusMoEImage repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(
        repo_id,
        cls_name=diffusers.NucleusMoEImageTransformer2DModel,
        load_config=diffusers_load_config,
    )
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.Qwen3VLForConditionalGeneration,
        load_config=diffusers_load_config,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(
        repo_id,
        subfolder='processor',
        cache_dir=shared.opts.hfcache_dir,
    )

    pipe = diffusers.NucleusMoEImagePipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        processor=processor,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }

    del transformer
    del text_encoder
    del processor
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
