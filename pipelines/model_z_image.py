import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_nunchaku():
    import nunchaku
    if not hasattr(nunchaku, 'NunchakuZImageTransformer2DModel'): # not present in older versions of nunchaku
        return None
    nunchaku_precision = nunchaku.utils.get_precision()
    nunchaku_rank = 128
    nunchaku_repo = f"nunchaku-ai/nunchaku-z-image-turbo/svdq-{nunchaku_precision}_r{nunchaku_rank}-z-image-turbo.safetensors"
    log.debug(f'Load module: quant=Nunchaku module=transformer repo="{nunchaku_repo}" attention={shared.opts.nunchaku_attention}')
    transformer = nunchaku.NunchakuZImageTransformer2DModel.from_pretrained( # pylint: disable=no-member
        nunchaku_repo,
        torch_dtype=devices.dtype,
        cache_dir=shared.opts.hfcache_dir,
    )
    return transformer


def load_z_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=ZImage repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={diffusers_load_config}')

    if model_quant.check_nunchaku('Model'): # only available model
        transformer = load_nunchaku()
    if transformer is None:
        transformer = generic.load_transformer(repo_id, cls_name=diffusers.ZImageTransformer2DModel, load_config=diffusers_load_config)

    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3ForCausalLM, load_config=diffusers_load_config)

    pipe = diffusers.ZImagePipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        **load_args,
    )

    del transformer
    del text_encoder
    sd_hijack_te.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
