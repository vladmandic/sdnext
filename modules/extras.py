import os
import html
import json
import time

from PIL import Image
import torch
import gradio as gr
import safetensors.torch
from modules.merging import merge, merge_utils, modules_sdxl
from modules import shared, images, sd_models, sd_vae, sd_samplers, devices
from modules.logger import log


def run_pnginfo(image):
    if image is None:
        return '', '', ''
    geninfo, items = images.read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}
    info = ''
    for key, text in items.items():
        if key != 'UserComment':
            info += f"<div><b>{html.escape(str(key))}</b>: {html.escape(str(text))}</div>"
    return '', geninfo, info


def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float:
        return tensor.half()
    return tensor


def run_modelmerger(id_task, **kwargs):  # pylint: disable=unused-argument
    from installer import install
    install('tensordict', quiet=True)
    try:
        pass # pylint: disable=unused-import
    except Exception as e:
        log.error(f"Merge: {e}")
        return [*[gr.update() for _ in range(4)], "tensordict not available"]

    jobid = shared.state.begin('Merge')
    t0 = time.time()

    def fail(message):
        shared.state.textinfo = message
        shared.state.end(jobid)
        return [*[gr.update() for _ in range(4)], message]

    kwargs["models"] = {
        "model_a": sd_models.get_closest_checkpoint_match(kwargs.get("primary_model_name", None)).filename,
        "model_b": sd_models.get_closest_checkpoint_match(kwargs.get("secondary_model_name", None)).filename,
    }

    if kwargs.get("primary_model_name", None) in [None, 'None']:
        return fail("Failed: Merging requires a primary model.")
    primary_model_info = sd_models.get_closest_checkpoint_match(kwargs.get("primary_model_name", None))
    if kwargs.get("secondary_model_name", None) in [None, 'None']:
        return fail("Failed: Merging requires a secondary model.")
    secondary_model_info = sd_models.get_closest_checkpoint_match(kwargs.get("secondary_model_name", None))
    if kwargs.get("tertiary_model_name", None) in [None, 'None'] and kwargs.get("merge_mode", None) in merge_utils.TRIPLE_METHODS:
        return fail(f"Failed: Interpolation method ({kwargs.get('merge_mode', None)}) requires a tertiary model.")
    tertiary_model_info = sd_models.get_closest_checkpoint_match(kwargs.get("tertiary_model_name", None)) if kwargs.get("merge_mode", None) in merge_utils.TRIPLE_METHODS else None

    del kwargs["primary_model_name"]
    del kwargs["secondary_model_name"]
    if kwargs.get("tertiary_model_name", None) is not None:
        kwargs["models"] |= {"model_c": sd_models.get_closest_checkpoint_match(kwargs.get("tertiary_model_name", None)).filename}
        del kwargs["tertiary_model_name"]

    if kwargs.get("alpha_base", None) and kwargs.get("alpha_in_blocks", None) and kwargs.get("alpha_mid_block", None) and kwargs.get("alpha_out_blocks", None):
        try:
            alpha = [float(x) for x in
                    [kwargs["alpha_base"]] + kwargs["alpha_in_blocks"].split(",") + [kwargs["alpha_mid_block"]] + kwargs["alpha_out_blocks"].split(",")]
            assert len(alpha) == 26 or len(alpha) == 20, "Alpha Block Weights are wrong length (26 or 20 for SDXL)"
            kwargs["alpha"] = alpha
        except KeyError as ke:
            log.warning(f"Merge: Malformed manual block weight: {ke}")
    elif kwargs.get("alpha_preset", None) or kwargs.get("alpha", None):
        kwargs["alpha"] = kwargs.get("alpha_preset", kwargs["alpha"])

    kwargs.pop("alpha_base", None)
    kwargs.pop("alpha_in_blocks", None)
    kwargs.pop("alpha_mid_block", None)
    kwargs.pop("alpha_out_blocks", None)
    kwargs.pop("alpha_preset", None)

    if kwargs.get("beta_base", None) and kwargs.get("beta_in_blocks", None) and kwargs.get("beta_mid_block", None) and kwargs.get("beta_out_blocks", None):
        try:
            beta = [float(x) for x in
                    [kwargs["beta_base"]] + kwargs["beta_in_blocks"].split(",") + [kwargs["beta_mid_block"]] + kwargs["beta_out_blocks"].split(",")]
            assert len(beta) == 26 or len(beta) == 20, "Beta Block Weights are wrong length (26 or 20 for SDXL)"
            kwargs["beta"] = beta
        except KeyError as ke:
            log.warning(f"Merge: Malformed manual block weight: {ke}")
    elif kwargs.get("beta_preset", None) or kwargs.get("beta", None):
        kwargs["beta"] = kwargs.get("beta_preset", kwargs["beta"])

    kwargs.pop("beta_base", None)
    kwargs.pop("beta_in_blocks", None)
    kwargs.pop("beta_mid_block", None)
    kwargs.pop("beta_out_blocks", None)
    kwargs.pop("beta_preset", None)

    if kwargs["device"] == "gpu":
        kwargs["device"] = devices.device
    elif kwargs["device"] == "shuffle":
        kwargs["device"] = torch.device("cpu")
        kwargs["work_device"] = devices.device
    else:
        kwargs["device"] = torch.device("cpu")
    if kwargs.pop("unload", False):
        sd_models.unload_model_weights()

    try:
        theta_0 = merge.merge_models(**kwargs)
    except Exception as e:
        return fail(f"{e}")

    try:
        theta_0 = theta_0.to_dict() #TensorDict -> Dict if necessary
    except Exception:
        pass

    bake_in_vae_filename = sd_vae.vae_dict.get(kwargs.get("bake_in_vae", None), None)
    if bake_in_vae_filename is not None:
        log.info(f"Merge VAE='{bake_in_vae_filename}'")
        shared.state.textinfo = 'Merge VAE'
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename)
        for key in vae_dict.keys():
            theta_0_key = 'first_stage_model.' + key
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = to_half(vae_dict[key], kwargs.get("precision", "fp16") == "fp16")
        del vae_dict

    ckpt_dir = shared.opts.ckpt_dir or sd_models.model_path
    filename = kwargs.get("custom_name", "Unnamed_Merge")
    filename += "." + kwargs.get("checkpoint_format", None)
    output_modelname = os.path.join(ckpt_dir, filename)
    shared.state.textinfo = "merge saving"
    metadata = None
    if kwargs.get("save_metadata", False):
        metadata = {"format": "pt", "sd_merge_models": {}}
        merge_recipe = {
            "type": "SDNext",  # indicate this model was merged with webui's built-in merger
            "primary_model_hash": primary_model_info.sha256,
            "secondary_model_hash": secondary_model_info.sha256 if secondary_model_info else None,
            "tertiary_model_hash": tertiary_model_info.sha256 if tertiary_model_info else None,
            "merge_mode": kwargs.get('merge_mode', None),
            "alpha": kwargs.get('alpha', None),
            "beta": kwargs.get('beta', None),
            "precision": kwargs.get('precision', None),
            "custom_name": kwargs.get("custom_name", "Unamed_Merge"),
        }
        metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

        def add_model_metadata(checkpoint_info):
            checkpoint_info.calculate_shorthash()
            metadata["sd_merge_models"][checkpoint_info.sha256] = {
                "name": checkpoint_info.name,
                "legacy_hash": checkpoint_info.hash,
                "sd_merge_recipe": checkpoint_info.metadata.get("sd_merge_recipe", None)
            }
            metadata["sd_merge_models"].update(checkpoint_info.metadata.get("sd_merge_models", {}))

        add_model_metadata(primary_model_info)
        if secondary_model_info:
            add_model_metadata(secondary_model_info)
        if tertiary_model_info:
            add_model_metadata(tertiary_model_info)
        metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

    _, extension = os.path.splitext(output_modelname)

    if os.path.exists(output_modelname) and not kwargs.get("overwrite", False):
        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_titles()) for _ in range(4)], f"Model alredy exists: {output_modelname}"]
    if extension.lower() == ".safetensors":
        safetensors.torch.save_file(theta_0, output_modelname, metadata=metadata)
    else:
        torch.save(theta_0, output_modelname)

    t1 = time.time()
    log.info(f"Merge complete: saved='{output_modelname}' time={t1-t0:.2f}")
    sd_models.list_models()
    created_model = next((ckpt for ckpt in sd_models.checkpoints_list.values() if ckpt.name == filename), None)
    if created_model:
        created_model.calculate_shorthash()
    devices.torch_gc(force=True, reason='merge')
    shared.state.end(jobid)
    return [*[gr.Dropdown.update(choices=sd_models.checkpoint_titles()) for _ in range(4)], f"Model saved to {output_modelname}"]


def run_model_modules(model_type:str, model_name:str, custom_name:str,
                      comp_unet:str, comp_vae:str, comp_te1:str, comp_te2:str,
                      precision:str, comp_scheduler:str, comp_prediction:str,
                      comp_lora:str, comp_fuse:float,
                      meta_author:str, meta_version:str, meta_license:str, meta_desc:str, meta_hint:str, meta_thumbnail:Image.Image,
                      create_diffusers:bool, create_safetensors:bool, debug:bool):

    status = ''
    def msg(text, err:bool=False):
        nonlocal status
        if err:
            log.error(f'Modules merge: {text}')
        else:
            log.info(f'Modules merge: {text}')
        status += text + '<br>'
        return status

    if model_type != 'sdxl':
        yield msg("only SDXL models are supported", err=True)
        return
    if len(custom_name) == 0:
        yield msg("output name is required", err=True)
        return
    checkpoint_info = sd_models.get_closest_checkpoint_match(model_name)
    if checkpoint_info is None:
        yield msg("input model not found", err=True)
        return
    fn = checkpoint_info.filename
    jobid = shared.state.begin('Merge')
    yield msg("modules merge starting")
    yield msg("unload current model")
    sd_models.unload_model_weights(op='model')

    modules_sdxl.recipe.name = custom_name
    modules_sdxl.recipe.author = meta_author
    modules_sdxl.recipe.version = meta_version
    modules_sdxl.recipe.desc = meta_desc
    modules_sdxl.recipe.hint = meta_hint
    modules_sdxl.recipe.license = meta_license
    modules_sdxl.recipe.thumbnail = meta_thumbnail
    modules_sdxl.recipe.base = fn
    modules_sdxl.recipe.unet = comp_unet
    modules_sdxl.recipe.vae = comp_vae
    modules_sdxl.recipe.te1 = comp_te1
    modules_sdxl.recipe.te2 = comp_te2
    modules_sdxl.recipe.prediction = comp_prediction
    modules_sdxl.recipe.diffusers = create_diffusers
    modules_sdxl.recipe.safetensors = create_safetensors
    modules_sdxl.recipe.fuse = float(comp_fuse)
    modules_sdxl.recipe.debug = debug

    loras = [l.strip() if ':' in l else f'{l.strip()}:1.0' for l in comp_lora.split(',') if len(l.strip()) > 0]
    for lora, strength in [l.split(':') for l in loras]:
        modules_sdxl.recipe.lora[lora] = float(strength)
    scheduler = sd_samplers.create_sampler(comp_scheduler, None)
    modules_sdxl.recipe.scheduler = scheduler.__class__.__name__ if scheduler is not None else None
    if precision == 'fp32':
        modules_sdxl.recipe.precision = torch.float32
    elif precision == 'bf16':
        modules_sdxl.recipe.precision = torch.bfloat16
    else:
        modules_sdxl.recipe.precision = torch.float16

    modules_sdxl.status = status
    yield from modules_sdxl.merge()
    status = modules_sdxl.status

    devices.torch_gc(force=True, reason='merge')
    yield msg("modules merge complete")
    if modules_sdxl.pipeline is not None:
        checkpoint_info = sd_models.CheckpointInfo(filename='None')
        shared.sd_model = modules_sdxl.pipeline
        sd_models.set_defaults(shared.sd_model, checkpoint_info)
        sd_models.set_diffuser_options(shared.sd_model, offload=False)
        sd_models.set_diffuser_offload(shared.sd_model)
        yield msg("pipeline loaded")
    shared.state.end(jobid)
