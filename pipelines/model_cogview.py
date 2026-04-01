import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_cogview3(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    log.debug(f'Load model: type=CogView3 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.CogView3PlusTransformer2DModel, load_config=diffusers_load_config, subfolder="transformer")
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder")

    pipe = diffusers.CogView3PlusPipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    sd_hijack_te.init_hijack(pipe)
    del transformer
    del text_encoder
    devices.torch_gc()
    return pipe


def load_cogview4(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    log.debug(f'Load model: type=CogView4 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.CogView4Transformer2DModel, load_config=diffusers_load_config, subfolder="transformer")
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.GlmModel, load_config=diffusers_load_config, subfolder="text_encoder", allow_quant=True)

    pipe = diffusers.CogView4Pipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    sd_hijack_te.init_hijack(pipe)
    del transformer
    del text_encoder

    devices.torch_gc()
    return pipe
