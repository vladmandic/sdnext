import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_step1x_edit(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Step1XEdit repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # Load text encoder (Qwen2.5-VL - available in transformers)
    text_encoder = generic.load_text_encoder(repo_id,
                                             cls_name=transformers.Qwen2_5_VLForConditionalGeneration, load_config=diffusers_load_config
                                            )

    # Load processor for Qwen2.5-VL
    processor = transformers.Qwen2_5_VLProcessor.from_pretrained(repo_id,
                                                                 cache_dir=shared.opts.hfcache_dir,
                                                                 subfolder='processor'
                                                                )

    # Step1XEditPipeline and Step1XEditTransformer2DModel are custom classes not in current diffusers
    # Try direct pipeline class first, fall back to trust_remote_code
    pipe_cls = getattr(diffusers, 'Step1XEditPipeline', None)
    if pipe_cls is not None:
        pipe = pipe_cls.from_pretrained(
            repo_id,
            cache_dir=shared.opts.diffusers_dir,
            text_encoder=text_encoder,
            processor=processor,
            **load_args,
        )
    else:
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            repo_id,
            cache_dir=shared.opts.diffusers_dir,
            text_encoder=text_encoder,
            processor=processor,
            trust_remote_code=True,
            **load_args,
        )

    del text_encoder
    del processor
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
