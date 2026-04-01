import transformers
import diffusers
from huggingface_hub import file_exists
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_pixart(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    repo_id_tenc = repo_id
    repo_id_pipe = repo_id

    # PixArt-alpha/PixArt-Sigma-XL-2-2K-MS only holds transformer
    if not file_exists(repo_id_tenc, "text_encoder/config.json"):
        repo_id_tenc = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    if not file_exists(repo_id_pipe, "model_index.json"):
        repo_id_pipe = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=PixArtSigma repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.PixArtTransformer2DModel, load_config=diffusers_load_config)
    text_encoder = generic.load_text_encoder(repo_id_tenc, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config)

    pipe = diffusers.PixArtSigmaPipeline.from_pretrained(
        repo_id_pipe,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    del text_encoder
    del transformer
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
