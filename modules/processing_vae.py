import os
import time
import numpy as np
import torch
from modules import shared, devices, sd_models, sd_vae, errors
from modules.vae import sd_vae_taesd


debug = os.environ.get('SD_VAE_DEBUG', None) is not None
log_debug = shared.log.trace if debug else lambda *args, **kwargs: None
log_debug('Trace: VAE')


def create_latents(image, p, dtype=None, device=None):
    from modules.processing import create_random_tensors
    from PIL import Image
    if image is None:
        return image
    elif isinstance(image, Image.Image):
        latents = vae_encode(image, model=shared.sd_model, vae_type=p.vae_type)
    elif isinstance(image, list):
        latents = [vae_encode(i, model=shared.sd_model, vae_type=p.vae_type).squeeze(dim=0) for i in image]
        latents = torch.stack(latents, dim=0).to(shared.device)
    else:
        shared.log.warning(f'Latents: input type: {type(image)} {image}')
        return image
    noise = p.denoising_strength * create_random_tensors(latents.shape[1:], seeds=p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, p=p)
    latents = (1 - p.denoising_strength) * latents + noise
    if dtype is not None:
        latents = latents.to(dtype=dtype)
    if device is not None:
        latents = latents.to(device=device)
    return latents


def full_vqgan_decode(latents, model):
    t0 = time.time()
    if model is None or not hasattr(model, 'vqgan'):
        shared.log.error('VQGAN not found in model')
        return []
    if debug:
        devices.torch_gc(force=True)
        shared.mem_mon.reset()

    base_device = None
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
        base_device = sd_models.move_base(model, devices.cpu)

    if shared.opts.diffusers_offload_mode == "balanced":
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    elif shared.opts.diffusers_offload_mode != "sequential":
        sd_models.move_model(model.vqgan, devices.device)

    latents = latents.to(devices.device, dtype=model.vqgan.dtype)

    #normalize latents
    scaling_factor = model.vqgan.config.get("scale_factor", None)
    if scaling_factor:
        latents = latents * scaling_factor

    log_debug(f'VAE config: {model.vqgan.config}')
    try:
        decoded = model.vqgan.decode(latents).sample.clamp(0, 1)
    except Exception as e:
        shared.log.error(f'VAE decode: {e}')
        errors.display(e, 'VAE decode')
        decoded = []

    # delete vae after OpenVINO compile
    if 'VAE' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx" and shared.compiled_model_state.first_pass_vae:
        shared.compiled_model_state.first_pass_vae = False
        if not shared.opts.openvino_disable_memory_cleanup and hasattr(shared.sd_model, "vqgan"):
            model.vqgan.apply(sd_models.convert_to_faketensors)
            devices.torch_gc(force=True)

    if shared.opts.diffusers_offload_mode == "balanced":
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    elif shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and base_device is not None:
        sd_models.move_base(model, base_device)
    t1 = time.time()
    if debug:
        log_debug(f'VAE memory: {shared.mem_mon.read()}')
    vae_name = os.path.splitext(os.path.basename(sd_vae.loaded_vae_file))[0] if sd_vae.loaded_vae_file is not None else "default"
    shared.log.debug(f'VAE decode: vae="{vae_name}" type="vqgan" dtype={model.vqgan.dtype} device={model.vqgan.device} time={round(t1-t0, 3)}')
    return decoded


def full_vae_decode(latents, model):
    t0 = time.time()
    if not hasattr(model, 'vae') and hasattr(model, 'pipe'):
        model = model.pipe
    if model is None or not hasattr(model, 'vae'):
        shared.log.error('VAE not found in model')
        return []
    if debug:
        devices.torch_gc(force=True)
        shared.mem_mon.reset()

    base_device = None
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
        base_device = sd_models.move_base(model, devices.cpu)
    elif shared.opts.diffusers_offload_mode != "sequential":
        sd_models.move_model(model.vae, devices.device)

    sd_models.set_vae_options(model, vae=None, op='decode')
    upcast = (model.vae.dtype == torch.float16) and (getattr(model.vae.config, 'force_upcast', False) or shared.opts.no_half_vae)
    if upcast:
        if hasattr(model, 'upcast_vae'): # this is done by diffusers automatically if output_type != 'latent'
            model.upcast_vae()
        else: # manual upcast and we restore it later
            model.vae.orig_dtype = model.vae.dtype
            model.vae = model.vae.to(dtype=torch.float32)
    latents = latents.to(devices.device)

    # normalize latents
    latents_mean = model.vae.config.get("latents_mean", None)
    latents_std = model.vae.config.get("latents_std", None)
    scaling_factor = model.vae.config.get("scaling_factor", 1.0)
    shift_factor = model.vae.config.get("shift_factor", None)
    if latents_mean and latents_std:
        broadcast_shape = [1 for _ in range(latents.ndim)]
        broadcast_shape[1] = -1
        latents_mean = (torch.tensor(latents_mean).view(*broadcast_shape).to(latents.device, latents.dtype))
        latents_std = (torch.tensor(latents_std).view(*broadcast_shape).to(latents.device, latents.dtype))
        latents = ((latents * latents_std) / scaling_factor) + latents_mean
    else:
        latents = latents / scaling_factor
    if shift_factor:
        latents = latents + shift_factor

    # check dims
    if model.vae.__class__.__name__ in ['AutoencoderKLWan'] and latents.ndim == 4:
        latents = latents.unsqueeze(2) # wan is __nhw

    # handle quants
    if getattr(model.vae, "post_quant_conv", None) is not None:
        if getattr(model.vae.post_quant_conv, "bias", None) is not None:
            latents = latents.to(model.vae.post_quant_conv.bias.dtype)
        elif "VAE" in shared.opts.sdnq_quantize_weights:
            latents = latents.to(devices.dtype_vae)
        else:
            latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
        # if getattr(model.vae.post_quant_conv, "bias", None) is not None:
            # model.vae.post_quant_conv.bias = torch.nn.Parameter(model.vae.post_quant_conv.bias.to(devices.device), requires_grad=False)
        # if getattr(model.vae.post_quant_conv, "weight", None) is not None:
            # model.vae.post_quant_conv.weight = torch.nn.Parameter(model.vae.post_quant_conv.weight.to(devices.device), requires_grad=False)
    else:
        latents = latents.to(model.vae.dtype)

    log_debug(f'VAE config: {model.vae.config}')
    try:
        with devices.inference_context():
            decoded = model.vae.decode(latents, return_dict=False)[0]
    except Exception as e:
        shared.log.error(f'VAE decode: {e}')
        if 'out of memory' not in str(e) and 'no data' not in str(e):
            errors.display(e, 'VAE decode')
        decoded = []

    if hasattr(model.vae, "orig_dtype"):
        model.vae = model.vae.to(dtype=model.vae.orig_dtype)
        del model.vae.orig_dtype

    # delete vae after OpenVINO compile
    if 'VAE' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx" and shared.compiled_model_state.first_pass_vae:
        shared.compiled_model_state.first_pass_vae = False
        if not shared.opts.openvino_disable_memory_cleanup and hasattr(shared.sd_model, "vae"):
            model.vae.apply(sd_models.convert_to_faketensors)
            devices.torch_gc(force=True)

    elif shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and base_device is not None:
        sd_models.move_base(model, base_device)

    t1 = time.time()
    if debug:
        log_debug(f'VAE memory: {shared.mem_mon.read()}')
    vae_name = os.path.splitext(os.path.basename(sd_vae.loaded_vae_file))[0] if sd_vae.loaded_vae_file is not None else "default"
    vae_scale_factor = sd_vae.get_vae_scale_factor(model)
    shared.log.debug(f'Decode: vae="{vae_name}" scale={vae_scale_factor} upcast={upcast} slicing={getattr(model.vae, "use_slicing", None)} tiling={getattr(model.vae, "use_tiling", None)} latents={list(latents.shape)}:{latents.device} dtype={latents.dtype} time={t1-t0:.3f}')
    return decoded


def full_vae_encode(image, model):
    t0 = time.time()
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        log_debug('Moving to CPU: model=UNet')
        unet_device = model.unet.device
        sd_models.move_model(model.unet, devices.cpu)
    if not shared.opts.diffusers_offload_mode == "sequential" and hasattr(model, 'vae'):
        sd_models.move_model(model.vae, devices.device)
    vae_name = sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "default"
    log_debug(f'Encode vae="{vae_name}" dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)}')

    sd_models.set_vae_options(model, vae=None, op='encode')
    upcast = (model.vae.dtype == torch.float16) and (getattr(model.vae.config, 'force_upcast', False) or shared.opts.no_half_vae)
    if upcast:
        if hasattr(model, 'upcast_vae'): # this is done by diffusers automatically if output_type != 'latent'
            model.upcast_vae()
        else: # manual upcast and we restore it later
            model.vae.orig_dtype = model.vae.dtype
            model.vae = model.vae.to(dtype=torch.float32)

    encoded = model.vae.encode(image.to(model.vae.device, model.vae.dtype)).latent_dist.sample()

    if hasattr(model.vae, "orig_dtype"):
        model.vae = model.vae.to(dtype=model.vae.orig_dtype)
        del model.vae.orig_dtype

    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        sd_models.move_model(model.unet, unet_device)
    t1 = time.time()
    shared.log.debug(f'Encode: vae="{vae_name}" upcast={upcast} slicing={getattr(model.vae, "use_slicing", None)} tiling={getattr(model.vae, "use_tiling", None)} latents={encoded.shape}:{encoded.device}:{encoded.dtype} time={t1-t0:.3f}')
    return encoded


def taesd_vae_decode(latents):
    t0 = time.time()
    if len(latents) == 0:
        return []
    if len(latents) > 1:
        decoded = torch.zeros((len(latents), 3, latents.shape[2] * 8, latents.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        for i in range(latents.shape[0]):
            decoded[i] = sd_vae_taesd.decode(latents[i])
    else:
        decoded = sd_vae_taesd.decode(latents)
    t1 = time.time()
    shared.log.debug(f'Decode: vae="taesd" latents={latents.shape}:{latents.device} dtype={latents.dtype} time={t1-t0:.3f}')
    return decoded


def taesd_vae_encode(image):
    shared.log.debug(f'Encode: vae="taesd" image={image.shape}')
    encoded = sd_vae_taesd.encode(image)
    return encoded


def vae_postprocess(tensor, model, output_type='np'):
    images = []
    try:
        if isinstance(tensor, list) and len(tensor) > 0 and torch.is_tensor(tensor[0]):
            tensor = torch.stack(tensor)
        if torch.is_tensor(tensor):
            if tensor.ndim == 3 and tensor.shape[0] == 3:
                tensor = tensor.unsqueeze(0)
            if hasattr(model, 'video_processor'):
                if tensor.ndim == 6 and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(0)
                images = model.video_processor.postprocess_video(tensor, output_type='pil')
            elif hasattr(model, 'image_processor'):
                if tensor.ndim == 5 and tensor.shape[1] == 3: # Qwen Image
                    tensor = tensor[:, :, 0]
                images = model.image_processor.postprocess(tensor, output_type=output_type)
            elif hasattr(model, "vqgan"):
                images = tensor.permute(0, 2, 3, 1).cpu().float().numpy()
                if output_type == "pil":
                    images = model.numpy_to_pil(images)
            else:
                from diffusers.image_processor import VaeImageProcessor
                model.image_processor = VaeImageProcessor()
                if tensor.ndim == 5 and tensor.shape[1] == 3: # Qwen Image
                    tensor = tensor[:, :, 0]
                images = model.image_processor.postprocess(tensor, output_type=output_type)
        else:
            images = tensor if isinstance(tensor, list) or isinstance(tensor, np.ndarray) else [tensor]
    except Exception as e:
        shared.log.error(f'VAE postprocess: {e}')
        errors.display(e, 'VAE')
    return images


def vae_decode(latents, model, output_type='np', vae_type='Full', width=None, height=None, frames=None):
    t0 = time.time()
    model = model or shared.sd_model
    if not hasattr(model, 'vae') and hasattr(model, 'pipe'):
        model = model.pipe
    if latents is None or not torch.is_tensor(latents): # already decoded
        return latents

    if latents.shape[0] == 0:
        shared.log.error(f'VAE nothing to decode: {latents.shape}')
        return []
    if shared.state.interrupted or shared.state.skipped:
        return []
    if not hasattr(model, 'vae') and not hasattr(model, 'vqgan'):
        shared.log.error('VAE not found in model')
        return []

    if vae_type == 'Remote':
        jobid = shared.state.begin('Remote VAE')
        from modules.vae.sd_vae_remote import remote_decode
        tensors = remote_decode(latents=latents, width=width, height=height)
        shared.state.end(jobid)
        if tensors is not None and len(tensors) > 0:
            return vae_postprocess(tensors, model, output_type)
    if vae_type == 'Repa':
        from modules.vae.sd_vae_repa import repa_load
        vae = repa_load(latents)
        vae_type = 'Full'
        if vae is not None:
            model.vae = vae

    jobid = shared.state.begin('VAE Decode')
    if hasattr(model, '_unpack_latents') and hasattr(model, 'transformer_spatial_patch_size') and frames is not None: # LTX
        latent_num_frames = (frames - 1) // model.vae_temporal_compression_ratio + 1
        latents = model._unpack_latents(latents.unsqueeze(0), latent_num_frames, height // 32, width // 32, model.transformer_spatial_patch_size, model.transformer_temporal_patch_size) # pylint: disable=protected-access
        latents = model._denormalize_latents(latents, model.vae.latents_mean, model.vae.latents_std, model.vae.config.scaling_factor) # pylint: disable=protected-access
    elif hasattr(model, '_unpack_latents') and hasattr(model, "vae_scale_factor") and width is not None and height is not None and latents.ndim == 3: # FLUX
        latents = model._unpack_latents(latents, height, width, model.vae_scale_factor) # pylint: disable=protected-access

    if latents.ndim == 3: # lost a batch dim in hires
        latents = latents.unsqueeze(0)
    if latents.shape[-1] <= 4: # not a latent, likely an image
        decoded = latents.float().cpu().numpy()
    elif vae_type == 'Tiny':
        decoded = taesd_vae_decode(latents=latents)
        if torch.is_tensor(decoded):
            decoded = 2.0 * decoded - 1.0 # typical normalized range
    elif hasattr(model, "vqgan"):
        decoded = full_vqgan_decode(latents=latents, model=model)
    elif hasattr(model, "vae"):
        decoded = full_vae_decode(latents=latents, model=model)
    else:
        shared.log.error('VAE not found in model')
        decoded = []

    images = vae_postprocess(decoded, model, output_type)
    if shared.cmd_opts.profile or debug:
        t1 = time.time()
        shared.log.debug(f'Profile: VAE decode: {t1-t0:.2f}')
    devices.torch_gc()
    shared.state.end(jobid)
    return images


def vae_encode(image, model, vae_type='Full'): # pylint: disable=unused-variable
    jobid = shared.state.begin('VAE Encode')
    from modules import images_sharpfin
    if shared.state.interrupted or shared.state.skipped:
        return []
    if not hasattr(model, 'vae') and hasattr(model, 'pipe'):
        model = model.pipe
    if not hasattr(model, 'vae'):
        shared.log.error('VAE not found in model')
        return []
    tensor = images_sharpfin.to_tensor(image.convert("RGB")).unsqueeze(0).to(devices.device, devices.dtype_vae)
    if vae_type == 'Tiny':
        latents = taesd_vae_encode(image=tensor)
    elif vae_type == 'Full' and hasattr(model, 'vae'):
        tensor = tensor * 2 - 1
        latents = full_vae_encode(image=tensor, model=shared.sd_model)
    else:
        shared.log.error('VAE not found in model')
        latents = []
    devices.torch_gc()
    shared.state.end(jobid)
    return latents


def reprocess(gallery):
    from PIL import Image
    from modules import images
    latent, index = shared.history.selected
    if latent is None or gallery is None:
        return None
    shared.log.info(f'Reprocessing: latent={latent.shape}')
    reprocessed = vae_decode(latent, shared.sd_model, output_type='pil')
    outputs = []
    for i0, i1 in zip(gallery, reprocessed):
        if isinstance(i1, np.ndarray):
            i1 = Image.fromarray(i1)
        fn = i0['name']
        i0 = Image.open(fn)
        fn = os.path.splitext(os.path.basename(fn))[0] + '-re'
        i0.load() # wait for info to be populated
        i1.info = i0.info
        info, _params = images.read_info_from_image(i0)
        if shared.opts.samples_save:
            images.save_image(i1, info=info, forced_filename=fn)
            i1.already_saved_as = fn
        if index == -1:
            outputs.append(i0)
        outputs.append(i1)
    return outputs
