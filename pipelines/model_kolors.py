import torch
import diffusers


repo_id = 'Kwai-Kolors/Kolors-diffusers'


def load_kolors(_checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices
    diffusers_load_config['variant'] = "fp16"
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = torch.float16

    # import torch
    # import transformers
    # encoder_id = 'THUDM/chatglm3-6b'
    # text_encoder = transformers.AutoModel.from_pretrained(encoder_id, torch_dtype=torch.float16, trust_remote_code=True, cache_dir=shared.opts.diffusers_dir)
    # text_encoder = transformers.AutoModel.from_pretrained("THUDM/chatglm3-6b", torch_dtype=torch.float16, trust_remote_code=True).quantize(4).cuda()
    # tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_id, trust_remote_code=True, cache_dir=shared.opts.diffusers_dir)
    pipe = diffusers.KolorsPipeline.from_pretrained(
        repo_id,
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    pipe.vae.config.force_upcast = True
    devices.torch_gc(force=True)
    return pipe
