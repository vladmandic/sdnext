import diffusers
import transformers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_sd3(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=SD3 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.SD3Transformer2DModel, load_config=diffusers_load_config)
    # text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.CLIPTextModelWithProjection, load_config=diffusers_load_config, subfolder="text_encoder")
    # text_encoder_2 = generic.load_text_encoder(repo_id, cls_name=transformers.CLIPTextModelWithProjection, load_config=diffusers_load_config, subfolder="text_encoder_2")
    if shared.opts.model_sd3_disable_te5:
        text_encoder_3 = None
    else:
        text_encoder_3 = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder_3")

    pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        # text_encoder=text_encoder,
        # text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    del text_encoder_3
    del transformer
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
