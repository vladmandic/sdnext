import transformers
import diffusers
from modules import shared, sd_models, devices, model_quant, sd_hijack_te
from modules.logger import log
from pipelines import generic


def load_auraflow(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    log.debug(f'Load model: type=AuraFlow repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.AuraFlowTransformer2DModel, load_config=diffusers_load_config)
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.UMT5EncoderModel, load_config=diffusers_load_config, allow_shared=False) # auraflow uses EleutherAI/pile-t5-xl

    pipe = diffusers.AuraFlowPipeline.from_pretrained(
        repo_id,
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
