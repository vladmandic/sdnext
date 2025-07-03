import os
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, modelloader, sd_hijack_te


def load_transformer(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    fn = None

    if shared.opts.sd_unet is not None and shared.opts.sd_unet != 'Default':
        from modules import sd_unet
        if shared.opts.sd_unet not in list(sd_unet.unet_dict):
            shared.log.error(f'Load module: type=Transformer not found: {shared.opts.sd_unet}')
            return None
        fn = sd_unet.unet_dict[shared.opts.sd_unet] if os.path.exists(sd_unet.unet_dict[shared.opts.sd_unet]) else None

    if fn is not None and 'gguf' in fn.lower():
        shared.log.error('Load model: type=Cosmos format="gguf" unsupported')
        transformer = None
    elif fn is not None and 'safetensors' in fn.lower():
        shared.log.debug(f'Load model: type=Cosmos transformer="{repo_id}" quant="{model_quant.get_quant(repo_id)}" args={load_args}')
        transformer = diffusers.CosmosTransformer3DModel.from_single_file(fn, cache_dir=shared.opts.hfcache_dir, **load_args)
    else:
        shared.log.debug(f'Load model: type=Cosmos transformer="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
        transformer = diffusers.CosmosTransformer3DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
    if shared.opts.diffusers_offload_mode != 'none' and transformer is not None:
        sd_models.move_model(transformer, devices.cpu)
    return transformer


def load_text_encoder(repo_id, diffusers_load_config={}):
    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    shared.log.debug(f'Load model: type=Cosmos te="{repo_id}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')
    text_encoder = transformers.T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.hfcache_dir,
        **load_args,
        **quant_args,
    )
    if shared.opts.diffusers_offload_mode != 'none' and text_encoder is not None:
        sd_models.move_model(text_encoder, devices.cpu)

    load_args, quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    llama_repo = shared.opts.model_h1_llama_repo if shared.opts.model_h1_llama_repo != 'Default' else 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    shared.log.debug(f'Load model: type=HiDream te4="{llama_repo}" quant="{model_quant.get_quant_type(quant_args)}" args={load_args}')

    return text_encoder


def load_cosmos_t2i(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    transformer = load_transformer(repo_id, diffusers_load_config)
    text_encoder = load_text_encoder(repo_id, diffusers_load_config)
    safety_checker = Fake_safety_checker()

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model')
    shared.log.debug(f'Load model: type=Cosmos model="{checkpoint_info.name}" repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    cls = diffusers.Cosmos2TextToImagePipeline
    pipe = cls.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        safety_checker=safety_checker,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    sd_hijack_te.init_hijack(pipe)
    del text_encoder
    del transformer

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
