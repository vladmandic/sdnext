import sys
import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_flite(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=FLite repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    from pipelines import f_lite
    diffusers.FLitePipeline = f_lite.FLitePipeline
    sys.modules['f_lite'] = f_lite

    dit_model = generic.load_transformer(repo_id, cls_name=f_lite.DiT, load_config=diffusers_load_config, subfolder="dit_model")
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder")

    pipe = f_lite.FLitePipeline.from_pretrained(
        "Freepik/F-Lite", # pr only exists on main repo
        revision="refs/pr/8",
        dit_model=dit_model,
        text_encoder=text_encoder,
        trust_remote_code=True,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    del text_encoder
    del dit_model
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc()
    return pipe
