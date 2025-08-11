import transformers
import diffusers
from modules import shared, sd_models, devices, model_quant, sd_hijack_te
from pipelines import generic


def load_hunyuandit(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    shared.log.debug(f'Load model: type=HunyuanDiT repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    transformer = generic.load_transformer(repo_id, cls_name=diffusers.HunyuanDiT2DModel, load_config=diffusers_load_config)
    repo_te = 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers' if 'HunyuanDiT-v1' in repo_id else repo_id
    text_encoder_2 = generic.load_text_encoder(repo_te, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder_2", allow_shared=False) # this is not normal t5

    pipe = diffusers.HunyuanDiTPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    del text_encoder_2
    del transformer
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
