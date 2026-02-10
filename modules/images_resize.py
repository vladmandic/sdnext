from typing import Union
import sys
import time
import numpy as np
import torch
from PIL import Image
from modules import shared, upscaler, images_sharpfin


def resize_image(resize_mode: int, im: Union[Image.Image, torch.Tensor], width: int, height: int, upscaler_name: str=None, output_type: str='image', context: str=None):
    upscaler_name = upscaler_name or shared.opts.upscaler_for_img2img

    def verify_image(image):
        try:
            if isinstance(image, torch.Tensor):
                image = image.float().detach().cpu().numpy()
            if isinstance(image, np.ndarray):
                if np.issubdtype(image.dtype, np.floating):
                    image = (255.0 * image).astype(np.uint8)
                image = Image.fromarray(image)
        except Exception as e:
            shared.log.error(f"Image verification failed: {e}")
        return image

    def latent(im, scale: float, selected_upscaler: upscaler.UpscalerData):
        if isinstance(im, torch.Tensor):
            im = selected_upscaler.scaler.upscale(im, scale, selected_upscaler.name)
            return im
        else:
            from modules.processing_vae import vae_encode, vae_decode
            latents = vae_encode(im, shared.sd_model, vae_type='Tiny') # TODO resize image: enable full VAE mode for resize-latent
            latents = selected_upscaler.scaler.upscale(latents, scale, selected_upscaler.name)
            im = vae_decode(latents, shared.sd_model, output_type='pil', vae_type='Tiny')[0]
            return im

    def resize(im: Union[Image.Image, torch.Tensor], w, h):
        w, h = int(w), int(h)
        if upscaler_name is None or upscaler_name == "None" or (hasattr(im, 'mode') and im.mode == 'L'):
            return images_sharpfin.resize(im, (w, h), linearize=False) # force for mask
        if isinstance(im, torch.Tensor):
            scale = max(w // 8 / im.shape[-1] , h // 8 / im.shape[-2])
        else:
            scale = max(w / im.width, h / im.height)
        if scale > 1.0:
            upscalers = [x for x in shared.sd_upscalers if x.name.lower().replace('-', ' ') == upscaler_name.lower().replace('-', ' ')]
            if len(upscalers) > 0:
                selected_upscaler: upscaler.UpscalerData = upscalers[0]
                if selected_upscaler.name.lower().startswith('latent'):
                    im = latent(im, scale, selected_upscaler)
                else:
                    im = selected_upscaler.scaler.upscale(im, scale, selected_upscaler.name)
            else:
                shared.log.warning(f"Resize upscaler: invalid={upscaler_name} fallback={selected_upscaler.name}")
                shared.log.debug(f"Resize upscaler: available={[u.name for u in shared.sd_upscalers]}")
        if isinstance(im, Image.Image) and (im.width != w or im.height != h): # probably downsample after upscaler created larger image
            im = images_sharpfin.resize(im, (w, h))
        return im

    def crop(im: Image.Image):
        ratio = width / height
        src_ratio = im.width / im.height
        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width
        resized = resize(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    def fill(im: Image.Image, color=None):
        color = color or shared.opts.image_background
        """
        ratio = round(width / height, 1)
        src_ratio = round(im.width / im.height, 1)
        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width
        resized = resize(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if width > 0 and fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if height > 0 and fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))
        return res
        """
        ratio = min(width / im.width, height / im.height)
        im = resize(im, int(im.width * ratio), int(im.height * ratio))
        res = Image.new(im.mode, (width, height), color=color)
        res.paste(im, box=((width - im.width)//2, (height - im.height)//2))
        return res

    def context_aware(im: Image.Image, width, height, context):
        from installer import install
        install('seam-carving')
        width, height = int(width), int(height)
        import seam_carving # https://github.com/li-plus/seam-carving
        if 'forward' in context.lower():
            energy_mode = "forward"
        elif 'backward' in context.lower():
            energy_mode = "backward"
        else:
            return im
        if 'add' in context.lower():
            src_ratio = min(width / im.width, height / im.height)
            src_w = int(im.width * src_ratio)
            src_h = int(im.height * src_ratio)
            src_image = resize(im, src_w, src_h)
        elif 'remove' in context.lower():
            ratio = width / height
            src_ratio = im.width / im.height
            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width
            src_image = resize(im, src_w, src_h)
        else:
            return im
        np_image = seam_carving.resize(
            src_image, # source image (rgb or gray)
            size=(width, height),  # target size
            energy_mode=energy_mode,  # choose from {backward, forward}
            order="width-first",  # choose from {width-first, height-first}
            keep_mask=None,  # object mask to protect from removal
        )
        res = Image.fromarray(np_image)
        return res

    t0 = time.time()
    if resize_mode is None:
        resize_mode = 0
    if isinstance(im, torch.Tensor): # latent resize only supports fixed mode
        res = resize(im, width, height)
        return res
    im = verify_image(im)
    if not isinstance(im, Image.Image):
        shared.log.error(f'Image resize: image={type(im)} invalid type')
        return im
    if (resize_mode == 0) or ((im.width == width) and (im.height == height)) or (width == 0 and height == 0): # none
        res = im.copy()
    elif resize_mode == 1: # fixed
        res = resize(im, width, height)
    elif resize_mode == 2: # crop
        res = crop(im)
    elif resize_mode == 3: # fill
        res = fill(im)
    elif resize_mode == 4: # edge
        from modules import masking
        res = fill(im, color=0)
        res, _mask = masking.outpaint(res)
    elif resize_mode == 5: # context-aware
        res = context_aware(im, width, height, context)
    else:
        res = im.copy()
        shared.log.error(f'Invalid resize mode: {resize_mode}')
    t1 = time.time()
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    if im.width != width or im.height != height:
        shared.log.debug(f'Image resize: source={im.width}:{im.height} target={width}:{height} mode="{shared.resize_modes[resize_mode]}" upscaler="{upscaler_name}" type={output_type} time={t1-t0:.2f} fn={fn}') # pylint: disable=protected-access
    return np.array(res) if output_type == 'np' else res
