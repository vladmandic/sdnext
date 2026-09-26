import os
import time
import threading
from collections import namedtuple
import torch
from PIL import Image
from modules import shared, processing, images, sd_samplers, timer, errors
from modules.logger import log
from modules.image import convert


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])
flow_models = ['f1', 'f2', 'sd3', 'lumina', 'auraflow', 'sana', 'zimage', 'lumina2', 'cogview4', 'h1', 'cosmos', 'anima', 'chroma', 'omnigen', 'omnigen2', 'longcat', 'ideogram4', 'krea2', 'qwen21']
warned = False
queue_lock = threading.Lock()
debug = os.environ.get('SD_VAE_DEBUG', None) is not None


def warn_once(message='', e=None):
    global warned # pylint: disable=global-statement
    if not shared.sd_loaded:
        return
    if warned != shared.sd_model_type:
        log.warning(f'Decode: {message} {str(e) if e is not None else ""}')
        if e is not None:
            errors.display(e, 'Decode')
        warned = shared.sd_model_type


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
    with queue_lock: # only one preview can run at a time
        t0 = time.time()
        try:
            approximation = approximation or shared.opts.show_progress_type
            if debug:
                log.debug(f'Preview sample: shape={list(sample.shape)} dtype={sample.dtype} method={approximation}')

            if len(sample.shape) > 4: # likely unknown video latent (e.g. svd)
                return Image.new(mode="RGB", size=(512, 512))
            if len(sample.shape) == 4:
                sample = sample[0] # standard batch [B, C, H, W] -> [C, H, W]
            if shared.opts.live_preview_downscale and (len(sample.shape) == 3 or len(sample.shape) == 4) and (sample.shape[-1]*sample.shape[-2] > 128*128):
                try:
                    scale = (128 * 128) / (sample.shape[-1] * sample.shape[-2])
                    sample = torch.nn.functional.interpolate(sample.unsqueeze(0), scale_factor=[scale, scale], mode='bilinear', align_corners=False)[0]
                except Exception:
                    pass

            if approximation == "None":
                x_sample = Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))
            elif approximation == "Micro":
                from modules.vae import sd_vae_micro
                x_sample = sd_vae_micro.decode(sample)
            elif approximation == "Tiny":
                from modules.vae import sd_vae_taesd
                x_sample = sd_vae_taesd.decode(sample)
            elif approximation == "Full":
                x_sample = processing.decode_first_stage(shared.sd_model, sample.unsqueeze(0), output_type='pil', use_job=False)[0]
            else:
                warn_once(f"method={approximation} unknown")
                x_sample = Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))

            if isinstance(x_sample, Image.Image):
                image = x_sample
            else:
                if len(x_sample.shape) == 4:
                    x_sample = x_sample[0]
                if x_sample.shape[0] > 4:
                    image = Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))
                else:
                    x_sample = torch.nan_to_num(x_sample, nan=0.0, posinf=1, neginf=0)
                    x_sample = (255.0 * x_sample).to(torch.uint8)
                    image = convert.to_pil(x_sample)
        except Exception as e:
            warn_once('exception', e)
            image = Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))
        t1 = time.time()
        timer.process.add('preview', t1 - t0)
        return image


def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


def store_latent(decoded):
    shared.state.current_latent = decoded
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
