import transformers
import diffusers
from modules import shared, sd_models, devices, model_quant, sd_hijack_te
from pipelines import generic


def load_kandinsky21(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    shared.log.debug(f'Load model: type=Kandinsky21 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')
    pipe = diffusers.KandinskyCombinedPipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe


def load_kandinsky22(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    shared.log.debug(f'Load model: type=Kandinsky22 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')
    pipe = diffusers.KandinskyV22CombinedPipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe


def load_kandinsky3(checkpoint_info, diffusers_load_config={}):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config)
    shared.log.debug(f'Load model: type=Kandinsky30 repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    unet = generic.load_transformer(repo_id, cls_name=diffusers.Kandinsky3UNet, load_config=diffusers_load_config, subfolder="unet", variant="fp16")
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.T5EncoderModel, load_config=diffusers_load_config, subfolder="text_encoder", variant="fp16")

    pipe = diffusers.Kandinsky3Pipeline.from_pretrained(
        repo_id,
        unet=unet,
        text_encoder=text_encoder,
        variant="fp16",
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )
    pipe.task_args = {
        'output_type': 'np',
    }

    del text_encoder
    del unet
    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
