import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic


def load_cosmos_t2i(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Cosmos repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.CosmosTransformer3DModel, load_config=diffusers_load_config, subfolder="transformer")
    repo_te = 'nvidia/Cosmos-Predict2-2B-Text2Image' if 'Cosmos-Predict2-14B-Text2Image' in repo_id else repo_id
    text_encoder = generic.load_text_encoder(repo_te, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder", allow_shared=False) # cosmos does use standard t5
    safety_checker = Fake_safety_checker()

    pipe = diffusers.Cosmos2TextToImagePipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        safety_checker=safety_checker,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    del text_encoder
    del transformer

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc()
    return pipe


class Fake_safety_checker:
    def __init__(self):
        from diffusers.utils import import_utils
        import_utils._cosmos_guardrail_available = True # pylint: disable=protected-access

    def __call__(self, *args, **kwargs): # pylint: disable=unused-argument
        return

    def to(self, _device):
        pass

    def check_text_safety(self, _prompt):
        return True

    def check_video_safety(self, vid):
        return vid
