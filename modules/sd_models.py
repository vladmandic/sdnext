import sys
import time
import copy
import inspect
import logging
import os.path
from enum import Enum
import diffusers
import diffusers.loaders.single_file_utils
import torch
import huggingface_hub as hf
from installer import log
from modules import timer, paths, shared, shared_items, modelloader, devices, script_callbacks, sd_vae, sd_unet, errors, sd_models_compile, sd_hijack_accelerate, sd_detect, model_quant, sd_hijack_te
from modules.memstats import memory_stats
from modules.modeldata import model_data
from modules.sd_checkpoint import CheckpointInfo, select_checkpoint, list_models, checkpoints_list, checkpoint_titles, get_closet_checkpoint_match, model_hash, update_model_hashes, setup_model, write_metadata, read_metadata_from_safetensors # pylint: disable=unused-import
from modules.sd_offload import disable_offload, set_diffuser_offload, apply_balanced_offload, set_accelerate # pylint: disable=unused-import
from modules.sd_models_utils import NoWatermark, get_signature, get_call, path_to_repo, patch_diffuser_config, convert_to_faketensors, read_state_dict, get_state_dict_from_checkpoint, apply_function_to_model # pylint: disable=unused-import


model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0
sd_metadata_timer = 0
debug_move = log.trace if os.environ.get('SD_MOVE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_load = os.environ.get('SD_LOAD_DEBUG', None)
debug_process = log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
diffusers_version = int(diffusers.__version__.split('.')[1])
checkpoint_tiles = checkpoint_titles # legacy compatibility
allow_post_quant = None
pipe_switch_task_exclude = [
    'AnimateDiffPipeline', 'AnimateDiffSDXLPipeline',
    'FluxControlPipeline',
    'FluxFillPipeline',
    'InstantIRPipeline',
    'LTXConditionPipeline',
    'OmniGenPipeline',
    'OmniGen2Pipeline',
    'PhotoMakerStableDiffusionXLPipeline',
    'PixelSmithXLPipeline',
    'StableDiffusion3ControlNetPipeline',
    'StableDiffusionAdapterPipeline',
    'StableDiffusionAdapterPipeline', 'StableDiffusionXLAdapterPipeline',
    'StableDiffusionControlNetXSPipeline', 'StableDiffusionXLControlNetXSPipeline',
    'StableDiffusionReferencePipeline',
    'StableDiffusionXLInstantIDPipeline',
]
i2i_pipes = [
    'LEditsPPPipelineStableDiffusion',
    'LEditsPPPipelineStableDiffusionXL',
    'OmniGenPipeline',
    'OmniGen2Pipeline',
    'StableDiffusionAdapterPipeline', 'StableDiffusionXLAdapterPipeline',
    'StableDiffusionControlNetXSPipeline', 'StableDiffusionXLControlNetXSPipeline',
]


def copy_diffuser_options(new_pipe, orig_pipe):
    new_pipe.sd_checkpoint_info = getattr(orig_pipe, 'sd_checkpoint_info', None)
    new_pipe.sd_model_checkpoint = getattr(orig_pipe, 'sd_model_checkpoint', None)
    new_pipe.embedding_db = getattr(orig_pipe, 'embedding_db', None)
    new_pipe.sd_model_hash = getattr(orig_pipe, 'sd_model_hash', None)
    new_pipe.has_accelerate = getattr(orig_pipe, 'has_accelerate', False)
    new_pipe.current_attn_name = getattr(orig_pipe, 'current_attn_name', None)
    new_pipe.default_scheduler = getattr(orig_pipe, 'default_scheduler', None)
    new_pipe.is_sdxl = getattr(orig_pipe, 'is_sdxl', False) # a1111 compatibility item
    new_pipe.is_sd2 = getattr(orig_pipe, 'is_sd2', False)
    new_pipe.is_sd1 = getattr(orig_pipe, 'is_sd1', True)
    add_noise_pred_to_diffusers_callback(new_pipe)
    if new_pipe.has_accelerate:
        set_accelerate(new_pipe)


def set_huggingface_options():
    if shared.opts.diffusers_to_gpu: # and model_type.startswith('Stable Diffusion'):
        sd_hijack_accelerate.hijack_accelerate()
    else:
        sd_hijack_accelerate.restore_accelerate()


def set_vae_options(sd_model, vae=None, op:str='model', quiet:bool=False):
    ops = {}
    if hasattr(sd_model, "vae"):
        if vae is not None:
            sd_model.vae = vae
            ops['name'] = f"{sd_vae.loaded_vae_file}"
        if shared.opts.diffusers_vae_upcast != 'default':
            sd_model.vae.config.force_upcast = True if shared.opts.diffusers_vae_upcast == 'true' else False
            ops['upcast'] = sd_model.vae.config.force_upcast
        if shared.opts.no_half_vae and op not in {'decode', 'encode'}:
            devices.dtype_vae = torch.float32
            sd_model.vae.to(devices.dtype_vae)
            ops['no-half'] = True
    if hasattr(sd_model, "enable_vae_slicing") and hasattr(sd_model, "disable_vae_slicing"):
        ops['slicing'] = shared.opts.diffusers_vae_slicing
        if shared.opts.diffusers_vae_slicing:
            sd_model.enable_vae_slicing()
        else:
            sd_model.disable_vae_slicing()
    if hasattr(sd_model, "enable_vae_tiling") and hasattr(sd_model, "disable_vae_tiling"):
        ops['tiling'] = shared.opts.diffusers_vae_tiling
        if shared.opts.diffusers_vae_tiling:
            if hasattr(sd_model, 'vae') and hasattr(sd_model.vae, 'config') and hasattr(sd_model.vae.config, 'sample_size') and isinstance(sd_model.vae.config.sample_size, int):
                if getattr(sd_model.vae, "tile_sample_min_size_backup", None) is None:
                    sd_model.vae.tile_sample_min_size_backup = sd_model.vae.tile_sample_min_size
                    sd_model.vae.tile_latent_min_size_backup = sd_model.vae.tile_latent_min_size
                    sd_model.vae.tile_overlap_factor_backup = sd_model.vae.tile_overlap_factor
                if shared.opts.diffusers_vae_tile_size > 0:
                    sd_model.vae.tile_sample_min_size = int(shared.opts.diffusers_vae_tile_size)
                    sd_model.vae.tile_latent_min_size = int(shared.opts.diffusers_vae_tile_size / (2 ** (len(sd_model.vae.config.block_out_channels) - 1)))
                else:
                    sd_model.vae.tile_sample_min_size = getattr(sd_model.vae, "tile_sample_min_size_backup", sd_model.vae.tile_sample_min_size)
                    sd_model.vae.tile_latent_min_size = getattr(sd_model.vae, "tile_latent_min_size_backup", sd_model.vae.tile_latent_min_size)
                if shared.opts.diffusers_vae_tile_overlap != 0.25:
                    sd_model.vae.tile_overlap_factor = float(shared.opts.diffusers_vae_tile_overlap)
                else:
                    sd_model.vae.tile_overlap_factor = getattr(sd_model.vae, "tile_overlap_factor_backup", sd_model.vae.tile_overlap_factor)
                ops['tile'] = sd_model.vae.tile_sample_min_size
                ops['overlap'] = sd_model.vae.tile_overlap_factor
            sd_model.enable_vae_tiling()
        else:
            sd_model.disable_vae_tiling()
    if hasattr(sd_model, "vqvae"):
        ops['upcast'] = True
        sd_model.vqvae.to(torch.float32) # vqvae is producing nans in fp16
    if not quiet and len(ops) > 0:
        shared.log.quiet(quiet, f'Setting {op}: component=vae {ops}')


def set_diffuser_options(sd_model, vae=None, op:str='model', offload:bool=True, quiet:bool=False):
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return

    if hasattr(sd_model, "watermark"):
        sd_model.watermark = NoWatermark()
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False

    clear_caches()
    set_vae_options(sd_model, vae, op, quiet)
    set_diffusers_attention(sd_model, quiet)

    if shared.opts.diffusers_fuse_projections and hasattr(sd_model, 'fuse_qkv_projections'):
        try:
            sd_model.fuse_qkv_projections()
            shared.log.quiet(quiet, f'Setting {op}: fused-qkv=True')
        except Exception as e:
            shared.log.error(f'Setting {op}: fused-qkv=True {e}')
    if shared.opts.diffusers_fuse_projections and hasattr(sd_model, 'transformer') and hasattr(sd_model.transformer, 'fuse_qkv_projections'):
        try:
            sd_model.transformer.fuse_qkv_projections()
            shared.log.quiet(quiet, f'Setting {op}: fused-qkv=True')
        except Exception as e:
            shared.log.error(f'Setting {op}: fused-qkv=True {e}')
    if shared.opts.diffusers_eval:
        shared.log.debug(f'Setting {op}: eval=True')
        def eval_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
            if hasattr(model, "requires_grad_"):
                model.requires_grad_(False)
                model.eval()
            return model
        sd_model = apply_function_to_model(sd_model, eval_model, ["Model", "VAE", "TE"], op="eval")

    if shared.opts.opt_channelslast and hasattr(sd_model, 'unet'):
        shared.log.quiet(quiet, f'Setting {op}: channels-last=True')
        sd_model.unet.to(memory_format=torch.channels_last)

    if offload:
        set_diffuser_offload(sd_model, op, quiet)


def move_model(model, device=None, force=False):
    if model is None or device is None:
        return

    if hasattr(model, 'pipe'):
        move_model(model.pipe, device, force)

    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    if getattr(model, 'vae', None) is not None and get_diffusers_task(model) != DiffusersTaskType.TEXT_2_IMAGE:
        if device == devices.device and model.vae.device.type != "meta": # force vae back to gpu if not in txt2img mode
            model.vae.to(device)
            if hasattr(model.vae, '_hf_hook'):
                debug_move(f'Model move: to={device} class={model.vae.__class__} fn={fn}') # pylint: disable=protected-access
                model.vae._hf_hook.execution_device = device # pylint: disable=protected-access
    if hasattr(model, "components"): # accelerate patch
        for name, m in model.components.items():
            if not hasattr(m, "_hf_hook"): # not accelerate hook
                break
            if not isinstance(m, torch.nn.Module) or name in model._exclude_from_cpu_offload: # pylint: disable=protected-access
                continue
            for module in m.modules():
                if (hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None): # pylint: disable=protected-access
                    try:
                        module._hf_hook.execution_device = device # pylint: disable=protected-access
                    except Exception as e:
                        if os.environ.get('SD_MOVE_DEBUG', None):
                            shared.log.error(f'Model move execution device: device={device} {e}')
    if getattr(model, 'has_accelerate', False) and not force:
        return
    if hasattr(model, "device") and devices.normalize_device(model.device) == devices.normalize_device(device) and not force:
        return
    try:
        t0 = time.time()
        try:
            if hasattr(model, 'to'):
                model.to(device)
            if hasattr(model, "prior_pipe"):
                model.prior_pipe.to(device)
        except Exception as e0:
            if 'Cannot copy out of meta tensor' in str(e0) or 'must be Tensor, not NoneType' in str(e0):
                if hasattr(model, "components"):
                    for _name, component in model.components.items():
                        if hasattr(component, 'modules'):
                            for module in component.modules():
                                try:
                                    if hasattr(module, 'to'):
                                        module.to(device)
                                except Exception as e2:
                                    if 'Cannot copy out of meta tensor' in str(e2):
                                        if os.environ.get('SD_MOVE_DEBUG', None):
                                            shared.log.warning(f'Model move meta: module={module.__class__}')
                                        module.to_empty(device=device)
            elif 'enable_sequential_cpu_offload' in str(e0):
                pass # ignore model move if sequential offload is enabled
            elif 'Params4bit' in str(e0) or 'Params8bit' in str(e0):
                pass # ignore model move if quantization is enabled
            elif 'already been set to the correct devices' in str(e0):
                pass # ignore errors on pre-quant models
            elif 'Casting a quantized model to' in str(e0):
                pass # ignore errors on quantized models
            else:
                raise e0
        t1 = time.time()
    except Exception as e1:
        t1 = time.time()
        shared.log.error(f'Model move: device={device} {e1}')
    if 'move' not in timer.process.records:
        timer.process.records['move'] = 0
    timer.process.records['move'] += t1 - t0
    if os.environ.get('SD_MOVE_DEBUG', None) is not None or (t1-t0) > 2:
        shared.log.debug(f'Model move: device={device} class={model.__class__.__name__} accelerate={getattr(model, "has_accelerate", False)} fn={fn} time={t1-t0:.2f}') # pylint: disable=protected-access
    devices.torch_gc()


def move_base(model, device):
    if hasattr(model, 'transformer'):
        key = 'transformer'
    elif hasattr(model, 'unet'):
        key = 'unet'
    else:
        shared.log.warning(f'Model move: model={model.__class__} device={device} key=unknown')
        return None
    shared.log.debug(f'Model move: module={key} device={device}')
    model = getattr(model, key)
    R = model.device
    move_model(model, device)
    return R


def load_diffuser_initial(diffusers_load_config, op='model'):
    sd_model = None
    checkpoint_info = None
    ckpt_basename = os.path.basename(shared.cmd_opts.ckpt)
    model_name = modelloader.find_diffuser(ckpt_basename)
    if model_name is not None:
        shared.log.info(f'Load model {op}: path="{model_name}"')
        model_file = modelloader.download_diffusers_model(hub_id=model_name, variant=diffusers_load_config.get('variant', None))
        try:
            shared.log.debug(f'Load {op}: config={diffusers_load_config}')
            sd_model = diffusers.DiffusionPipeline.from_pretrained(model_file, **diffusers_load_config)
        except Exception as e:
            shared.log.error(f'Failed loading model: {model_file} {e}')
            errors.display(e, f'Load {op}: path="{model_file}"')
            return None, None
        list_models() # rescan for downloaded model
        checkpoint_info = CheckpointInfo(model_name)
    return sd_model, checkpoint_info


def load_diffuser_force(model_type, checkpoint_info, diffusers_load_config, op='model'):
    sd_model = None
    global allow_post_quant # pylint: disable=global-statement
    unload_model_weights(op=op)
    shared.sd_model = None
    try:
        if model_type in ['Stable Cascade']: # forced pipeline
            from pipelines.model_stablecascade import load_cascade_combined
            sd_model = load_cascade_combined(checkpoint_info, diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['InstaFlow']: # forced pipeline
            pipeline = diffusers.utils.get_class_from_dynamic_module('instaflow_one_step', module_file='pipeline.py')
            shared_items.pipelines['InstaFlow'] = pipeline
            sd_model = pipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['SegMoE']: # forced pipeline
            from pipelines.segmoe.segmoe_model import SegMoEPipeline
            sd_model = SegMoEPipeline(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
            sd_model = sd_model.pipe # segmoe pipe does its stuff in __init__ and __call__ is the original pipeline
            allow_post_quant = True
            shared_items.pipelines['SegMoE'] = SegMoEPipeline
        elif model_type in ['PixArt Sigma']: # forced pipeline
            from pipelines.model_pixart import load_pixart
            sd_model = load_pixart(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Sana']: # forced pipeline
            from pipelines.model_sana import load_sana
            sd_model = load_sana(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Lumina-Next']: # forced pipeline
            from pipelines.model_lumina import load_lumina
            sd_model = load_lumina(checkpoint_info, diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['Kolors']: # forced pipeline
            from pipelines.model_kolors import load_kolors
            sd_model = load_kolors(checkpoint_info, diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['AuraFlow']: # forced pipeline
            from pipelines.model_auraflow import load_auraflow
            sd_model = load_auraflow(checkpoint_info, diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['FLUX']:
            from pipelines.model_flux import load_flux
            sd_model = load_flux(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['FLEX']:
            from pipelines.model_flex import load_flex
            sd_model = load_flex(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Chroma']:
            from pipelines.model_chroma import load_chroma
            sd_model = load_chroma(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Lumina 2']:
            from pipelines.model_lumina import load_lumina2
            sd_model = load_lumina2(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Stable Diffusion 3']:
            from pipelines.model_sd3 import load_sd3
            sd_model = load_sd3(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['CogView 3']: # forced pipeline
            from pipelines.model_cogview import load_cogview3
            sd_model = load_cogview3(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['CogView 4']: # forced pipeline
            from pipelines.model_cogview import load_cogview4
            sd_model = load_cogview4(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Meissonic']: # forced pipeline
            from pipelines.model_meissonic import load_meissonic
            sd_model = load_meissonic(checkpoint_info, diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['OmniGen2']: # forced pipeline
            from pipelines.model_omnigen2 import load_omnigen2
            sd_model = load_omnigen2(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['OmniGen']: # forced pipeline
            from pipelines.model_omnigen import load_omnigen
            sd_model = load_omnigen(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['HiDream']:
            from pipelines.model_hidream import load_hidream
            sd_model = load_hidream(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Cosmos']:
            from pipelines.model_cosmos import load_cosmos_t2i
            sd_model = load_cosmos_t2i(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['FLite']:
            from pipelines.model_flite import load_flite
            sd_model = load_flite(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['WanAI']:
            from pipelines.model_wanai import load_wan
            sd_model = load_wan(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Bria']:
            from pipelines.model_bria import load_bria
            sd_model = load_bria(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Qwen']:
            from pipelines.model_qwen import load_qwen
            sd_model = load_qwen(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['HunyuanDiT']:
            from pipelines.model_hunyuandit import load_hunyuandit
            sd_model = load_hunyuandit(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Kandinsky 2.1']:
            from pipelines.model_kandinsky import load_kandinsky21
            sd_model = load_kandinsky21(checkpoint_info, diffusers_load_config)
            allow_post_quant = True
        elif model_type in ['Kandinsky 2.2']:
            from pipelines.model_kandinsky import load_kandinsky22
            sd_model = load_kandinsky22(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['Kandinsky 3.0']:
            from pipelines.model_kandinsky import load_kandinsky3
            sd_model = load_kandinsky3(checkpoint_info, diffusers_load_config)
            allow_post_quant = False
        elif model_type in ['NextStep']:
            from pipelines.model_nextstep import load_nextstep
            sd_model = load_nextstep(checkpoint_info, diffusers_load_config) # pylint: disable=assignment-from-none
            allow_post_quant = False
    except Exception as e:
        shared.log.error(f'Load {op}: path="{checkpoint_info.path}" {e}')
        if debug_load:
            errors.display(e, 'Load')
        return None
    return sd_model


def load_diffuser_folder(model_type, pipeline, checkpoint_info, diffusers_load_config, op='model'):
    sd_model = None
    files = shared.walk_files(checkpoint_info.path, ['.safetensors', '.bin', '.ckpt'])
    if 'variant' not in diffusers_load_config and any('diffusion_pytorch_model.fp16' in f for f in files): # deal with diffusers lack of variant fallback when loading
        diffusers_load_config['variant'] = 'fp16'
    if (model_type is not None) and (pipeline is not None) and ('ONNX' in model_type): # forced pipeline
        try:
            sd_model = pipeline.from_pretrained(checkpoint_info.path)
        except Exception as e:
            shared.log.error(f'Load {op}: type=ONNX path="{checkpoint_info.path}" {e}')
            if debug_load:
                errors.display(e, 'Load')
            return None
    else:
        err1, err2, err3 = None, None, None
        if os.path.exists(checkpoint_info.path) and os.path.isdir(checkpoint_info.path):
            if os.path.exists(os.path.join(checkpoint_info.path, 'unet', 'diffusion_pytorch_model.bin')):
                shared.log.debug(f'Load {op}: type=pickle')
                diffusers_load_config['use_safetensors'] = False
        if debug_load:
            shared.log.debug(f'Load {op}: args={diffusers_load_config}')
        try: # 1 - autopipeline, best choice but not all pipelines are available
            try:
                sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                sd_model.model_type = sd_model.__class__.__name__
            except ValueError as e:
                if 'no variant default' in str(e):
                    shared.log.warning(f'Load {op}: variant={diffusers_load_config["variant"]} model="{checkpoint_info.path}" using default variant')
                    diffusers_load_config.pop('variant', None)
                    sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                    sd_model.model_type = sd_model.__class__.__name__
                elif 'safetensors found in directory' in str(err1):
                    shared.log.warning(f'Load {op}: type=pickle')
                    diffusers_load_config['use_safetensors'] = False
                    sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                    sd_model.model_type = sd_model.__class__.__name__
                else:
                    raise ValueError from e # reraise
        except Exception as e:
            err1 = e
            if debug_load:
                errors.display(e, 'Load AutoPipeline')
            # shared.log.error(f'AutoPipeline: {e}')
        try: # 2 - diffusion pipeline, works for most non-linked pipelines
            if err1 is not None:
                sd_model = diffusers.DiffusionPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                sd_model.model_type = sd_model.__class__.__name__
        except Exception as e:
            err2 = e
            if debug_load:
                errors.display(e, "Load DiffusionPipeline")
            # shared.log.error(f'DiffusionPipeline: {e}')
        try: # 3 - try basic pipeline just in case
            if err2 is not None:
                sd_model = diffusers.StableDiffusionPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                sd_model.model_type = sd_model.__class__.__name__
        except Exception as e:
            err3 = e  # ignore last error
            shared.log.error(f"StableDiffusionPipeline: {e}")
            if debug_load:
                errors.display(e, "Load StableDiffusionPipeline")
        if err3 is not None:
            shared.log.error(f'Load {op}: {checkpoint_info.path} auto={err1} diffusion={err2}')
            return None
    return sd_model


def load_diffuser_file(model_type, pipeline, checkpoint_info, diffusers_load_config, op='model'):
    sd_model = None
    diffusers_load_config["extract_ema"] = shared.opts.diffusers_extract_ema
    if pipeline is None:
        shared.log.error(f'Load {op}: pipeline={shared.opts.diffusers_pipeline} not initialized')
        return None
    try:
        if model_type.startswith('Stable Diffusion'):
            if shared.opts.diffusers_force_zeros:
                diffusers_load_config['force_zeros_for_empty_prompt '] = shared.opts.diffusers_force_zeros
            else:
                model_config = sd_detect.get_load_config(checkpoint_info.path, model_type, config_type='json')
                if model_config is not None:
                    if debug_load:
                        shared.log.debug(f'Load {op}: config="{model_config}"')
                    diffusers_load_config['config'] = model_config
        if model_type.startswith('Stable Diffusion 3'):
            from pipelines.model_sd3 import load_sd3
            sd_model = load_sd3(checkpoint_info, diffusers_load_config)
        elif hasattr(pipeline, 'from_single_file'):
            diffusers.loaders.single_file_utils.CHECKPOINT_KEY_NAMES["clip"] = "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight" # patch for diffusers==0.28.0
            diffusers_load_config['use_safetensors'] = True
            diffusers_load_config['cache_dir'] = shared.opts.hfcache_dir # use hfcache instead of diffusers dir as this is for config only in case of single-file
            if shared.opts.stream_load:
                diffusers_load_config['disable_mmap'] = True
            if shared.opts.disable_accelerate:
                from diffusers.utils import import_utils
                import_utils._accelerate_available = False # pylint: disable=protected-access
            sd_model = pipeline.from_single_file(checkpoint_info.path, **diffusers_load_config)
            # sd_model = patch_diffuser_config(sd_model, checkpoint_info.path)
        elif hasattr(pipeline, 'from_ckpt'):
            diffusers_load_config['cache_dir'] = shared.opts.hfcache_dir
            sd_model = pipeline.from_ckpt(checkpoint_info.path, **diffusers_load_config)
        else:
            shared.log.error(f'Load {op}: file="{checkpoint_info.path}" {shared.opts.diffusers_pipeline} cannot load safetensor model')
            return None
        if shared.opts.diffusers_vae_upcast != 'default' and model_type in ['Stable Diffusion', 'Stable Diffusion XL']:
            diffusers_load_config['force_upcast'] = True if shared.opts.diffusers_vae_upcast == 'true' else False
        # if debug_load:
        #    shared.log.debug(f'Model args: {diffusers_load_config}')
        if sd_model is not None:
            diffusers_load_config.pop('vae', None)
            diffusers_load_config.pop('safety_checker', None)
            diffusers_load_config.pop('requires_safety_checker', None)
            diffusers_load_config.pop('config_files', None)
            diffusers_load_config.pop('local_files_only', None)
            shared.log.debug(f'Setting {op}: pipeline={sd_model.__class__.__name__} config={diffusers_load_config}') # pylint: disable=protected-access
    except Exception as e:
        shared.log.error(f'Load {op}: file="{checkpoint_info.path}" pipeline={shared.opts.diffusers_pipeline}/{sd_model.__class__.__name__} config={diffusers_load_config} {e}')
        if 'Weights for this component appear to be missing in the checkpoint' in str(e):
            shared.log.error(f'Load {op}: file="{checkpoint_info.path}" is not a complete model')
        else:
            errors.display(e, 'Load')
        return None
    return sd_model


def set_overrides(sd_model, checkpoint_info):
    if 'bigaspv25' in checkpoint_info.name.lower():
        scheduler_config = sd_model.scheduler.config
        scheduler_config['prediction_type'] = 'flow_prediction'
        sd_model.scheduler = diffusers.UniPCMultistepScheduler.from_config(scheduler_config)
        shared.log.info(f'Setting override: model="{checkpoint_info.name}" component=scheduler prediction="flow-prediction"')
    if 'vpred' in checkpoint_info.name.lower() or 'v-pred' in checkpoint_info.name.lower():
        scheduler_config = sd_model.scheduler.config
        scheduler_config['prediction_type'] = 'v_prediction'
        sd_model.scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_config)
        shared.log.info(f'Setting override: model="{checkpoint_info.name}" component=scheduler prediction="v-prediction"')


def set_defaults(sd_model, checkpoint_info):
    sd_model.sd_model_hash = checkpoint_info.calculate_shorthash() # pylint: disable=attribute-defined-outside-init
    sd_model.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
    sd_model.sd_model_checkpoint = checkpoint_info.filename # pylint: disable=attribute-defined-outside-init
    if hasattr(sd_model, "prior_pipe"):
        sd_model.default_scheduler = copy.deepcopy(sd_model.prior_pipe.scheduler) if hasattr(sd_model.prior_pipe, "scheduler") else None
    else:
        sd_model.default_scheduler = copy.deepcopy(sd_model.scheduler) if hasattr(sd_model, "scheduler") else None
    sd_model.is_sdxl = False # a1111 compatibility item
    sd_model.is_sd2 = hasattr(sd_model, 'cond_stage_model') and hasattr(sd_model.cond_stage_model, 'model') # a1111 compatibility item
    sd_model.is_sd1 = not sd_model.is_sd2 # a1111 compatibility item
    sd_model.logvar = sd_model.logvar.to(devices.device) if hasattr(sd_model, 'logvar') else None # fix for training
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
    if hasattr(sd_model, "set_progress_bar_config"):
        sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining}', ncols=80, colour='#327fba')


def load_diffuser(checkpoint_info=None, op='model', revision=None): # pylint: disable=unused-argument
    global allow_post_quant # pylint: disable=global-statement
    allow_post_quant = True # assume default
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    timer.load.record("diffusers")
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "load_connected_pipeline": True,
        "safety_checker": None, # sd15 specific but we cant know ahead of time
        "requires_safety_checker": False, # sd15 specific but we cant know ahead of time
        # "use_safetensors": True,
    }
    if revision is not None:
        diffusers_load_config['revision'] = revision
    if shared.opts.diffusers_model_load_variant != 'default':
        diffusers_load_config['variant'] = shared.opts.diffusers_model_load_variant
    if shared.opts.diffusers_pipeline == 'Custom Diffusers Pipeline' and len(shared.opts.custom_diffusers_pipeline) > 0:
        shared.log.debug(f'Model pipeline: pipeline="{shared.opts.custom_diffusers_pipeline}"')
        diffusers_load_config['custom_pipeline'] = shared.opts.custom_diffusers_pipeline
    if shared.opts.data.get('sd_model_checkpoint', '') == 'model.safetensors' or shared.opts.data.get('sd_model_checkpoint', '') == '':
        shared.opts.data['sd_model_checkpoint'] = "stabilityai/stable-diffusion-xl-base-1.0"

    if (op == 'model' or op == 'dict'):
        if (model_data.sd_model is not None) and (checkpoint_info is not None) and (getattr(model_data.sd_model, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (checkpoint_info is not None) and (getattr(model_data.sd_refiner, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return

    sd_model = None
    try:
        # initial load only
        if sd_model is None:
            if shared.cmd_opts.ckpt is not None and os.path.isdir(shared.cmd_opts.ckpt) and model_data.initial:
                sd_model, checkpoint_info = load_diffuser_initial(diffusers_load_config, op)

        # unload current model
        checkpoint_info = checkpoint_info or select_checkpoint(op=op)
        if checkpoint_info is None:
            unload_model_weights(op=op)
            return

        # detect pipeline
        pipeline, model_type = sd_detect.detect_pipeline(checkpoint_info.path, op)
        set_huggingface_options()

        # preload vae so it can be used as param
        vae = None
        sd_vae.loaded_vae_file = None
        if model_type is None:
            shared.log.error(f'Load {op}: pipeline={shared.opts.diffusers_pipeline} not detected')
            return
        vae_file = None
        if model_type.startswith('Stable Diffusion') and (op == 'model' or op == 'refiner'): # preload vae for sd models
            vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
            vae = sd_vae.load_vae_diffusers(checkpoint_info.path, vae_file, vae_source)
            if vae is not None:
                diffusers_load_config["vae"] = vae
                timer.load.record("vae")

        # load with custom loader
        if sd_model is None:
            sd_model = load_diffuser_force(model_type, checkpoint_info, diffusers_load_config, op)
            if sd_model is not None and not sd_model:
                shared.log.error(f'Load {op}: type="{model_type}" pipeline="{pipeline}" not loaded')
                return

        # load from hf folder-style
        if sd_model is None:
            if os.path.isdir(checkpoint_info.path) or checkpoint_info.type == 'huggingface' or checkpoint_info.type == 'transformer':
                sd_model = load_diffuser_folder(model_type, pipeline, checkpoint_info, diffusers_load_config, op)

        # load from single-file
        if sd_model is None:
            if os.path.isfile(checkpoint_info.path) and checkpoint_info.path.lower().endswith('.safetensors'):
                sd_model = load_diffuser_file(model_type, pipeline, checkpoint_info, diffusers_load_config, op)

        if sd_model is None:
            shared.log.error(f'Load {op}: name="{checkpoint_info.name if checkpoint_info is not None else None}" not loaded')
            return

        set_overrides(sd_model, checkpoint_info)
        set_defaults(sd_model, checkpoint_info)

        if "Kandinsky" in sd_model.__class__.__name__: # need a special case
            sd_model.scheduler.name = 'DDIM'

        if hasattr(sd_model, "unet") and model_type not in ['Stable Cascade']: # others calls load_diffuser again
            sd_unet.load_unet(sd_model)

        add_noise_pred_to_diffusers_callback(sd_model)

        timer.load.record("load")

        if op == 'refiner':
            model_data.sd_refiner = sd_model
        else:
            model_data.sd_model = sd_model

        reload_text_encoder(initial=True) # must be before embeddings
        timer.load.record("te")

        if debug_load:
            shared.log.trace(f'Model components: {list(get_signature(sd_model).values())}')

        from modules import textual_inversion
        sd_model.embedding_db = textual_inversion.EmbeddingDatabase()
        sd_model.embedding_db.add_embedding_dir(shared.opts.embeddings_dir)
        sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        timer.load.record("embeddings")

        from modules import prompt_parser_diffusers
        prompt_parser_diffusers.insert_parser_highjack(sd_model.__class__.__name__)
        prompt_parser_diffusers.cache.clear()

        set_diffuser_options(sd_model, vae, op, offload=False)
        sd_model = model_quant.do_post_load_quant(sd_model, allow=allow_post_quant) # run this before move model so it can be compressed in CPU
        timer.load.record("options")

        set_diffuser_offload(sd_model, op)

        if op == 'model' and not (os.path.isdir(checkpoint_info.path) or checkpoint_info.type == 'huggingface'):
            if getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None and vae_file is not None:
                sd_vae.apply_vae_config(shared.sd_model.sd_checkpoint_info.filename, vae_file, sd_model)
        if op == 'refiner' and shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving refiner model to CPU')
            move_model(sd_model, devices.cpu)
        else:
            move_model(sd_model, devices.device)
        timer.load.record("move")

        if shared.opts.ipex_optimize:
            sd_model = sd_models_compile.ipex_optimize(sd_model)

        if ('Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none'):
            sd_model = sd_models_compile.compile_diffusers(sd_model)
        timer.load.record("compile")

    except Exception as e:
        shared.log.error(f"Load {op}: {e}")
        errors.display(e, "Model")

    if shared.opts.diffusers_offload_mode != 'balanced':
        devices.torch_gc(force=True, reason='load')
    if sd_model is not None:
        script_callbacks.model_loaded_callback(sd_model)

    if debug_load:
        from modules import modelstats
        modelstats.analyze()

    shared.log.info(f"Load {op}: family={shared.sd_model_type} time={timer.load.dct()} native={get_native(sd_model)} memory={memory_stats()}")


class DiffusersTaskType(Enum):
    TEXT_2_IMAGE = 1
    IMAGE_2_IMAGE = 2
    INPAINTING = 3
    INSTRUCT = 4


def get_diffusers_task(pipe: diffusers.DiffusionPipeline) -> DiffusersTaskType:
    cls = pipe.__class__.__name__
    if cls in i2i_pipes: # special case
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif 'ImageToVideo' in cls or cls in ['LTXConditionPipeline', 'StableVideoDiffusionPipeline']: # i2v pipelines
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif 'Instruct' in cls:
        return DiffusersTaskType.INSTRUCT
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.values():
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.values():
        return DiffusersTaskType.INPAINTING
    else:
        return DiffusersTaskType.TEXT_2_IMAGE


def switch_pipe(cls: diffusers.DiffusionPipeline, pipeline: diffusers.DiffusionPipeline = None, force = False, args = {}):
    """
    args:
    - cls: can be pipeline class or a string from custom pipelines
      for example: diffusers.StableDiffusionPipeline or 'mixture_tiling'
    - pipeline: source model to be used, if not provided currently loaded model is used
    - args: any additional components to load into the pipeline
      for example: { 'vae': None }
    """
    try:
        if isinstance(cls, str):
            shared.log.debug(f'Pipeline switch: custom={cls}')
            cls = diffusers.utils.get_class_from_dynamic_module(cls, module_file='pipeline.py')
        if pipeline is None:
            pipeline = shared.sd_model
        new_pipe = None
        signature = get_signature(cls)
        possible = signature.keys()
        if not force and isinstance(pipeline, cls) and args == {}:
            return pipeline
        pipe_dict = {}
        components_used = []
        components_skipped = []
        components_missing = []
        switch_mode = 'none'
        if hasattr(pipeline, '_internal_dict'):
            for item in pipeline._internal_dict.keys(): # pylint: disable=protected-access
                if item in possible:
                    pipe_dict[item] = getattr(pipeline, item, None)
                    components_used.append(item)
                else:
                    components_skipped.append(item)
            for item in possible:
                if item in ['self', 'args', 'kwargs']: # skip
                    continue
                if signature[item].default != inspect._empty: # has default value so we dont have to worry about it # pylint: disable=protected-access
                    continue
                if item not in components_used:
                    shared.log.warning(f'Pipeling switch: missing component={item} type={signature[item].annotation}')
                    pipe_dict[item] = None # try but not likely to work
                    components_missing.append(item)
            new_pipe = cls(**pipe_dict)
            switch_mode = 'auto'
        elif 'tokenizer_2' in possible and hasattr(pipeline, 'tokenizer_2'):
            new_pipe = cls(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
            )
            move_model(new_pipe, pipeline.device)
            switch_mode = 'sdxl'
        elif 'tokenizer' in possible and hasattr(pipeline, 'tokenizer'):
            new_pipe = cls(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                requires_safety_checker=False,
                safety_checker=None,
            )
            move_model(new_pipe, pipeline.device)
            switch_mode = 'sd'
        else:
            shared.log.error(f'Pipeline switch error: {pipeline.__class__.__name__} unrecognized')
            return pipeline
        if new_pipe is not None:
            for k, v in args.items():
                if k in possible:
                    setattr(new_pipe, k, v)
                    components_used.append(k)
                else:
                    shared.log.warning(f'Pipeline switch skipping unknown: component={k}')
                    components_skipped.append(k)
        if new_pipe is not None:
            copy_diffuser_options(new_pipe, pipeline)
            sd_hijack_te.init_hijack(new_pipe)
            if hasattr(new_pipe, "watermark"):
                new_pipe.watermark = NoWatermark()
            if switch_mode == 'auto':
                shared.log.debug(f'Pipeline switch: from={pipeline.__class__.__name__} to={new_pipe.__class__.__name__} components={components_used} skipped={components_skipped} missing={components_missing}')
            else:
                shared.log.debug(f'Pipeline switch: from={pipeline.__class__.__name__} to={new_pipe.__class__.__name__} mode={switch_mode}')
            return new_pipe
        else:
            shared.log.error(f'Pipeline switch error: from={pipeline.__class__.__name__} to={cls.__name__} empty pipeline')
    except Exception as e:
        shared.log.error(f'Pipeline switch error: from={pipeline.__class__.__name__} to={cls.__name__} {e}')
        errors.display(e, 'Pipeline switch')
    return pipeline


def clean_diffuser_pipe(pipe):
    if pipe is not None and shared.sd_model_type == 'sdxl' and hasattr(pipe, 'config') and 'requires_aesthetics_score' in pipe.config and hasattr(pipe, '_internal_dict'):
        debug_process(f'Pipeline clean: {pipe.__class__.__name__}')
        # diffusers adds requires_aesthetics_score with img2img and complains if requires_aesthetics_score exist in txt2img
        internal_dict = dict(pipe._internal_dict) # pylint: disable=protected-access
        internal_dict.pop('requires_aesthetics_score', None)
        del pipe._internal_dict
        pipe.register_to_config(**internal_dict)


def backup_pipe_components(pipe):
    if pipe is None:
        return {}
    return {
        'sd_checkpoint_info': getattr(pipe, "sd_checkpoint_info", None),
        'sd_model_checkpoint': getattr(pipe, "sd_model_checkpoint", None),
        'embedding_db': getattr(pipe, "embedding_db", None),
        'loaded_loras': getattr(pipe, "loaded_loras", {}),
        'sd_model_hash': getattr(pipe, "sd_model_hash", None),
        'has_accelerate': getattr(pipe, "has_accelerate", None),
        'current_attn_name': getattr(pipe, "current_attn_name", None),
        'default_scheduler': getattr(pipe, "default_scheduler", None),
        'image_encoder': getattr(pipe, "image_encoder", None),
        'feature_extractor': getattr(pipe, "feature_extractor", None),
        'mask_processor': getattr(pipe, "mask_processor", None),
        'restore_pipeline': getattr(pipe, "restore_pipeline", None),
        'task_args': getattr(pipe, "task_args", None),
    }


def restore_pipe_components(pipe, components):
    if pipe is None or components is None:
        return
    pipe.sd_checkpoint_info = components['sd_checkpoint_info']
    pipe.sd_model_checkpoint = components['sd_model_checkpoint']
    pipe.embedding_db = components['embedding_db']
    pipe.loaded_loras = components['loaded_loras'] if components['loaded_loras'] is not None else {}
    pipe.sd_model_hash = components['sd_model_hash']
    pipe.has_accelerate = components['has_accelerate']
    pipe.current_attn_name = components['current_attn_name']
    pipe.default_scheduler = components['default_scheduler']

    if components['image_encoder'] is not None:
        pipe.image_encoder = components['image_encoder']
    if components['feature_extractor'] is not None:
        pipe.feature_extractor = components['feature_extractor']
    if components['mask_processor'] is not None:
        pipe.mask_processor = components['mask_processor']
    if components['restore_pipeline'] is not None:
        pipe.restore_pipeline = components['restore_pipeline']
    if components['task_args'] is not None:
        pipe.task_args = components['task_args']

    if pipe.__class__.__name__ in ['FluxPipeline', 'StableDiffusion3Pipeline']:
        pipe.register_modules(image_encoder = components['image_encoder'])
        pipe.register_modules(feature_extractor = components['feature_extractor'])


def set_diffuser_pipe(pipe, new_pipe_type):
    has_errors = False
    if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
        clean_diffuser_pipe(pipe)

    if hasattr(pipe, 'no_task_switch'):
        del pipe.no_task_switch
        return pipe
    if get_diffusers_task(pipe) == new_pipe_type:
        return pipe

    # skip specific pipelines
    cls = pipe.__class__.__name__
    if cls in pipe_switch_task_exclude:
        return pipe
    if 'Video' in cls:
        return pipe
    if 'Onnx' in cls:
        return pipe

    new_pipe = None
    # in some cases we want to reset the pipeline to parent as they dont have their own variants
    if new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE or new_pipe_type == DiffusersTaskType.INPAINTING:
        if cls == 'StableDiffusionPAGPipeline':
            pipe = switch_pipe(diffusers.StableDiffusionPipeline, pipe)
        if cls == 'StableDiffusionXLPAGPipeline':
            pipe = switch_pipe(diffusers.StableDiffusionXLPipeline, pipe)

    components_backup = backup_pipe_components(pipe)

    if new_pipe is None:
        if hasattr(pipe, 'config'): # real pipeline which can be auto-switched
            try:
                if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
                    new_pipe = diffusers.AutoPipelineForText2Image.from_pipe(pipe)
                elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
                    new_pipe = diffusers.AutoPipelineForImage2Image.from_pipe(pipe)
                elif new_pipe_type == DiffusersTaskType.INPAINTING:
                    new_pipe = diffusers.AutoPipelineForInpainting.from_pipe(pipe)
                else:
                    shared.log.warning(f'Pipeline class change failed: type={new_pipe_type} pipeline={cls}')
                    return pipe
            except Exception as e: # pylint: disable=unused-variable
                shared.log.warning(f'Pipeline class change failed: type={new_pipe_type} pipeline={cls} {e}')
                has_errors = True
        if not hasattr(pipe, 'config') or has_errors:
            try: # maybe a wrapper pipeline so just change the class
                if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
                    pipe.__class__ = diffusers.pipelines.auto_pipeline._get_task_class(diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING, cls) # pylint: disable=protected-access
                    new_pipe = pipe
                elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
                    pipe.__class__ = diffusers.pipelines.auto_pipeline._get_task_class(diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, cls) # pylint: disable=protected-access
                    new_pipe = pipe
                elif new_pipe_type == DiffusersTaskType.INPAINTING:
                    pipe.__class__ = diffusers.pipelines.auto_pipeline._get_task_class(diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING, cls) # pylint: disable=protected-access
                    new_pipe = pipe
                else:
                    shared.log.error(f'Pipeline class set failed: type={new_pipe_type} pipeline={cls}')
                    return pipe
            except Exception as e: # pylint: disable=unused-variable
                shared.log.warning(f'Pipeline class set failed: type={new_pipe_type} pipeline={cls} {e}')
                has_errors = True
                return pipe

    if new_pipe is None:
        return pipe

    restore_pipe_components(new_pipe, components_backup)
    components_backup = None # free memory

    new_pipe.is_sdxl = getattr(pipe, 'is_sdxl', False) # a1111 compatibility item
    new_pipe.is_sd2 = getattr(pipe, 'is_sd2', False)
    new_pipe.is_sd1 = getattr(pipe, 'is_sd1', True)
    if hasattr(new_pipe, 'watermark'):
        new_pipe.watermark = NoWatermark()
    add_noise_pred_to_diffusers_callback(new_pipe)

    if hasattr(new_pipe, 'pipe'): # also handle nested pipelines
        new_pipe.pipe = set_diffuser_pipe(new_pipe.pipe, new_pipe_type)
        add_noise_pred_to_diffusers_callback(new_pipe.pipe)

    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    shared.log.debug(f"Pipeline class change: original={cls} target={new_pipe.__class__.__name__} device={pipe.device} fn={fn}") # pylint: disable=protected-access
    pipe = new_pipe
    return pipe


def set_diffusers_attention(pipe, quiet:bool=False):
    import diffusers.models.attention_processor as p

    def set_attn(pipe, attention, name:str=None, quiet:bool=False):
        if attention is None:
            return
        # other models uses their own attention processor
        if pipe.__class__.__name__.startswith("StableDiffusion") and hasattr(pipe, "unet"):
            pipe.unet.set_attn_processor(attention)
        elif not quiet:
            shared.log.warning(f"Attention: {name if name is not None else attention.__class__.__name__} is not compatible with {pipe.__class__.__name__}")

    # if hasattr(pipe, 'pipe'):
    #    set_diffusers_attention(pipe.pipe)

    if 'Control' in pipe.__class__.__name__ or 'Adapter' in pipe.__class__.__name__ or not (pipe.__class__.__name__.startswith("StableDiffusion") and hasattr(pipe, "unet")):
        if shared.opts.cross_attention_optimization not in {"Scaled-Dot-Product", "Disabled"}:
            shared.log.warning(f"Attention: {shared.opts.cross_attention_optimization} is not compatible with {pipe.__class__.__name__}")
        else:
            pipe.current_attn_name = shared.opts.cross_attention_optimization
        return

    shared.log.quiet(quiet, f'Setting model: attention="{shared.opts.cross_attention_optimization}"')
    if shared.opts.cross_attention_optimization == "Disabled":
        pass # do nothing
    elif shared.opts.cross_attention_optimization == "Scaled-Dot-Product": # The default set by Diffusers
        set_attn(pipe, p.AttnProcessor2_0(), name="Scaled-Dot-Product", quiet=True)
    elif shared.opts.cross_attention_optimization == "xFormers":
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            pipe.enable_xformers_memory_efficient_attention()
        else:
            shared.log.warning(f"Attention: xFormers is not compatible with {pipe.__class__.__name__}")
    elif shared.opts.cross_attention_optimization == "Split attention":
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        else:
            shared.log.warning(f"Attention: Split attention is not compatible with {pipe.__class__.__name__}")
    elif shared.opts.cross_attention_optimization == "Batch matrix-matrix":
        set_attn(pipe, p.AttnProcessor(), name="Batch matrix-matrix")
    elif shared.opts.cross_attention_optimization == "Dynamic Attention BMM":
        from modules.sd_hijack_dynamic_atten import DynamicAttnProcessorBMM
        set_attn(pipe, DynamicAttnProcessorBMM(), name="Dynamic Attention BMM")

    pipe.current_attn_name = shared.opts.cross_attention_optimization


def add_noise_pred_to_diffusers_callback(pipe):
    if not hasattr(pipe, "_callback_tensor_inputs"):
        return pipe
    if pipe.__class__.__name__.startswith("StableDiffusion"):
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    elif pipe.__class__.__name__.startswith("StableCascade"):
        pipe.prior_pipe._callback_tensor_inputs.append("predicted_image_embedding") # pylint: disable=protected-access
    elif hasattr(pipe, "scheduler") and "flow" in pipe.scheduler.__class__.__name__.lower():
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    elif hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "config") and getattr(pipe.scheduler.config, "prediction_type", "none") == "flow_prediction":
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    elif hasattr(pipe, "default_scheduler") and "flow" in pipe.default_scheduler.__class__.__name__.lower():
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    elif hasattr(pipe, "default_scheduler") and hasattr(pipe.default_scheduler, "config") and getattr(pipe.default_scheduler.config, "prediction_type", "none") == "flow_prediction":
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    return pipe


def get_native(pipe: diffusers.DiffusionPipeline):
    if hasattr(pipe, "vae") and hasattr(pipe.vae.config, "sample_size"):
        size = pipe.vae.config.sample_size # Stable Diffusion
    elif hasattr(pipe, "movq") and hasattr(pipe.movq.config, "sample_size"):
        size = pipe.movq.config.sample_size # Kandinsky
    elif hasattr(pipe, "unet") and hasattr(pipe.unet.config, "sample_size"):
        size = pipe.unet.config.sample_size
    else:
        size = 0
    return size


def reload_text_encoder(initial=False):
    if initial and (shared.opts.sd_text_encoder is None or shared.opts.sd_text_encoder == 'Default'):
        return # dont unload
    signature = get_signature(shared.sd_model)
    t5 = [k for k, v in signature.items() if 'T5EncoderModel' in str(v)]
    if hasattr(shared.sd_model, 'text_encoder') and 'vit' in shared.opts.sd_text_encoder.lower():
        from modules.model_te import set_clip
        set_clip(pipe=shared.sd_model)
    elif len(t5) > 0:
        from modules.model_te import set_t5
        shared.log.debug(f'Load module: type=t5 path="{shared.opts.sd_text_encoder}" module="{t5[0]}"')
        set_t5(pipe=shared.sd_model, module=t5[0], t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    elif hasattr(shared.sd_model, 'text_encoder_3'):
        from modules.model_te import set_t5
        shared.log.debug(f'Load module: type=t5 path="{shared.opts.sd_text_encoder}" module="text_encoder_3"')
        set_t5(pipe=shared.sd_model, module='text_encoder_3', t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    clear_caches(full=True)
    apply_balanced_offload(shared.sd_model)


def reload_model_weights(sd_model=None, info=None, op='model', force=False, revision=None):
    checkpoint_info = info or select_checkpoint(op=op) # are we selecting model or dictionary
    if checkpoint_info is None:
        unload_model_weights(op=op)
        return None
    orig_state = copy.deepcopy(shared.state)
    shared.state.begin('Load')
    if sd_model is None:
        sd_model = model_data.sd_model if op == 'model' or op == 'dict' else model_data.sd_refiner
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = getattr(sd_model, 'sd_checkpoint_info', None)
        if current_checkpoint_info is not None and checkpoint_info is not None and current_checkpoint_info.filename == checkpoint_info.filename and not force:
            return None
        else:
            move_model(sd_model, devices.cpu)
        unload_model_weights(op=op)
        sd_model = None
    timer.load = timer.Timer()
    # TODO model load: implement model in-memory caching
    timer.load.record("config")
    if sd_model is None or force:
        sd_model = None
        load_diffuser(checkpoint_info, op=op, revision=revision)
        shared.state.end()
        shared.state = orig_state
        if op == 'model':
            shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
            return model_data.sd_model
        else:
            shared.opts.data["sd_model_refiner"] = checkpoint_info.title
            return model_data.sd_refiner
    return None # should not be here


def clear_caches(full:bool=False):
    from modules import prompt_parser_diffusers, memstats, sd_offload
    from modules.lora import lora_common, lora_load
    prompt_parser_diffusers.cache.clear()
    memstats.reset_stats()
    lora_common.loaded_networks.clear()
    lora_common.previously_loaded_networks.clear()
    lora_load.lora_cache.clear()
    if full:
        shared.log.debug('Cache clear')
        sd_offload.offload_hook_instance = None


def unload_model_weights(op='model'):
    clear_caches(full=True)
    if shared.compiled_model_state is not None:
        shared.compiled_model_state.compiled_cache.clear()
        shared.compiled_model_state.req_cache.clear()
        shared.compiled_model_state.partitioned_modules.clear()
    if (op == 'model' or op == 'dict') and model_data.sd_model:
        shared.log.debug(f'Current {op}: {memory_stats()}')
        if not ('Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx"):
            disable_offload(model_data.sd_model)
            move_model(model_data.sd_model, 'meta')
        model_data.sd_model = None
        devices.torch_gc(force=True, reason='unload')
        shared.log.debug(f'Unload {op}: {memory_stats()} after')
    elif (op == 'refiner') and model_data.sd_refiner:
        shared.log.debug(f'Current {op}: {memory_stats()}')
        disable_offload(model_data.sd_refiner)
        move_model(model_data.sd_refiner, 'meta')
        model_data.sd_refiner = None
        devices.torch_gc(force=True, reason='unload')
        shared.log.debug(f'Unload {op}: {memory_stats()}')


def hf_auth_check(checkpoint_info):
    login = None
    try:
        if (checkpoint_info.path.endswith('.safetensors') and os.path.isfile(checkpoint_info.path)) or (os.path.exists(checkpoint_info.path) and os.path.isdir(checkpoint_info.path) and os.path.isfile(os.path.join(checkpoint_info.path, 'model_index.json'))): # skip check for already downloaded models
            return True
    except Exception:
        pass
    try:
        login = modelloader.hf_login()
        repo_id = path_to_repo(checkpoint_info)
        return hf.auth_check(repo_id)
    except Exception as e:
        shared.log.error(f'Load model: repo="{repo_id}" login={login} {e}')
        return False
