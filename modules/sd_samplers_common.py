import time
import threading
from collections import namedtuple
import torch
from PIL import Image
from modules import shared, devices, processing, images, sd_samplers, timer
from modules.logger import log
from modules.vae import sd_vae_approx, sd_vae_taesd, sd_vae_stablecascade
from modules.image import convert


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])
approximation_indexes = { "Simple": 0, "Approximate": 1, "TAESD": 2, "Full VAE": 3 }
flow_models = ['f1', 'f2', 'sd3', 'lumina', 'auraflow', 'sana', 'zimage', 'lumina2', 'cogview4', 'h1', 'cosmos', 'chroma', 'omnigen', 'omnigen2', 'longcat']
warned = False
queue_lock = threading.Lock()


def warn_once(message):
    global warned # pylint: disable=global-statement
    if not warned:
        log.warning(f'VAE: {message}')
        warned = True


def setup_img2img_steps(p, steps=None):
    if shared.opts.img2img_fix_steps or steps is not None:
        requested_steps = (steps or p.steps)
        steps = int(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc


def single_sample_to_image(sample, approximation=None):
    with queue_lock:
        t0 = time.time()
        if approximation is None:
            approximation = approximation_indexes.get(shared.opts.show_progress_type, None)
            if approximation is None:
                warn_once('Unknown decode type')
                approximation = 0
        try:
            if sample.dtype == torch.bfloat16 and (approximation == 0 or approximation == 1):
                sample = sample.to(torch.float16)
        except Exception as e:
            warn_once(f'Preview: {e}')

        if len(sample.shape) > 4: # likely unknown video latent (e.g. svd)
            return Image.new(mode="RGB", size=(512, 512))
        if len(sample.shape) == 4 and sample.shape[0]: # likely animatediff latent
            sample = sample.permute(1, 0, 2, 3)[0]
        if approximation == 2: # TAESD
            if (len(sample.shape) == 3 or len(sample.shape) == 4) and shared.opts.live_preview_downscale and (sample.shape[-1]*sample.shape[-2] > 128*128):
                try:
                    scale = (128 * 128) / (sample.shape[-1] * sample.shape[-2])
                    sample = torch.nn.functional.interpolate(sample.unsqueeze(0), scale_factor=[scale, scale], mode='bilinear', align_corners=False)[0]
                except Exception:
                    pass
            x_sample = sd_vae_taesd.decode(sample)
            # x_sample = (1.0 + x_sample) / 2.0 # preview requires smaller range
        elif shared.sd_model_type == 'sc' and approximation != 3:
            x_sample = sd_vae_stablecascade.decode(sample)
        elif approximation == 0: # Simple
            x_sample = sd_vae_approx.cheap_approximation(sample) * 0.5 + 0.5
        elif approximation == 1: # Approximate
            x_sample = sd_vae_approx.nn_approximation(sample) * 0.5 + 0.5
            if shared.sd_model_type == "sdxl":
                x_sample = x_sample[[2, 1, 0], :, :] # BGR to RGB
        elif approximation == 3: # Full VAE
            x_sample = processing.decode_first_stage(shared.sd_model, sample.unsqueeze(0))[0]
        else:
            warn_once(f"Unknown latent decode type: {approximation}")
            return Image.new(mode="RGB", size=(512, 512))
        try:
            if isinstance(x_sample, Image.Image):
                image = x_sample
            else:
                if x_sample.shape[0] > 4 or x_sample.shape[0] == 4:
                    return Image.new(mode="RGB", size=(512, 512))
                x_sample = torch.nan_to_num(x_sample, nan=0.0, posinf=1, neginf=0)
                x_sample = (255.0 * x_sample).to(torch.uint8)
                if len(x_sample.shape) == 4:
                    x_sample = x_sample[0]
                image = convert.to_pil(x_sample)
        except Exception as e:
            warn_once(f'Preview: {e}')
            image = Image.new(mode="RGB", size=(512, 512))
        t1 = time.time()
        timer.process.add('preview', t1 - t0)
        return image


def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


def images_tensor_to_samples(image, approximation=None, model=None):
    '''image[0, 1] -> latent'''
    if approximation is None:
        approximation = approximation_indexes.get(shared.opts.show_progress_type, 0)
    if approximation == 2:
        image = image.to(devices.device, devices.dtype)
        x_latent = sd_vae_taesd.encode(image)
    else:
        if model is None:
            model = shared.sd_model
        model.first_stage_model.to(devices.dtype_vae)
        image = image.to(shared.device, dtype=devices.dtype_vae)
        image = image * 2 - 1
        if len(image) > 1:
            image_latents = [model.get_first_stage_encoding(model.encode_first_stage(torch.unsqueeze(img, 0)))[0] for img in image]
            x_latent = torch.stack(image_latents)
        else:
            x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    return x_latent


def store_latent(decoded):
    shared.state.current_latent = decoded
    if shared.opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % shared.opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            image = sample_to_image(decoded)
            shared.state.assign_current_image(image)


def is_sampler_using_eta_noise_seed_delta(p):
    """returns whether sampler from config will use eta noise seed delta for image creation"""
    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)
    eta = 0
    if hasattr(p, "eta"):
        eta = p.eta
    if not hasattr(p.sampler, "eta"):
        return False
    if eta is None and p.sampler is not None:
        eta = p.sampler.eta
    if eta is None and sampler_config is not None:
        eta = 0 if sampler_config.options.get("default_eta_is_0", False) else 1.0
    if eta == 0:
        return False
    return True


class InterruptedException(BaseException):
    pass
