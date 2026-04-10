import sys

import diffusers
import transformers
from modules import devices, model_quant, sd_hijack_te, sd_models, shared
from modules.logger import log


TEXT_ENCODER_REPO = "Tongyi-MAI/Z-Image-Turbo"


def load_zetachroma(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    load_args.setdefault("torch_dtype", devices.dtype)
    log.debug(
        f'Load model: type=ZetaChroma repo="{repo_id}" config={diffusers_load_config} '
        f'offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}'
    )

    from pipelines import generic, zetachroma

    diffusers.ZetaChromaPipeline = zetachroma.ZetaChromaPipeline
    sys.modules["zetachroma"] = zetachroma

    text_encoder = generic.load_text_encoder(
        TEXT_ENCODER_REPO,
        cls_name=transformers.Qwen3ForCausalLM,
        load_config=diffusers_load_config,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TEXT_ENCODER_REPO,
        subfolder="tokenizer",
        cache_dir=shared.opts.hfcache_dir,
        trust_remote_code=True,
    )

    pipe = zetachroma.ZetaChromaPipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_args,
    )
    pipe.task_args = {
        "output_type": "np",
    }
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["zetachroma"] = zetachroma.ZetaChromaPipeline

    del tokenizer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason="load")
    return pipe
