import time
import random
import numpy as np
from PIL import Image
from modules import shared
from modules.logger import log
from modules.api.helpers import decode_base64_to_image, encode_pil_to_base64
from installer import install


def set_image_watermark(image: Image.Image, wm_image: Image.Image | None = None, position: str | None = None, quiet: bool = False):
    position = (position or shared.opts.image_watermark_position).lower()
    if (position != 'none') and (wm_image is not None): # visible watermark
        t0 = time.perf_counter()
        if isinstance(wm_image, str):
            try:
                wm_image = Image.open(wm_image)
            except Exception as e:
                log.warning(f'Watermark: image={wm_image} {e}')
                return image
        if isinstance(wm_image, Image.Image):
            if wm_image.mode != 'RGBA':
                wm_image = wm_image.convert('RGBA')
        position = (position or shared.opts.image_watermark_position).lower()
        if position == 'top/left':
            coords = (0, 0)
        elif position == 'top/center':
            coords = ((image.width - wm_image.width) // 2, 0)
        elif position == 'top/right':
            coords = (image.width - wm_image.width, 0)
        elif position == 'bottom/left':
            coords = (0, image.height - wm_image.height)
        elif position == 'bottom/center':
            coords = ((image.width - wm_image.width) // 2, image.height - wm_image.height)
        elif position == 'bottom/right':
            coords = (image.width - wm_image.width, image.height - wm_image.height)
        elif position == 'center':
            coords = ((image.width - wm_image.width) // 2, (image.height - wm_image.height) // 2)
        elif position == 'random':
            coords = (random.randint(0, image.width - wm_image.width), random.randint(0, image.height - wm_image.height))
        else: # fallback to random if unknown position
            coords = (random.randint(0, image.width - wm_image.width), random.randint(0, image.height - wm_image.height))
        try:
            # original per-pixel implementation, kept for reference (slow, and crashes if base image mode is RGBA):
            # for x in range(wm_image.width):
            #     for y in range(wm_image.height):
            #         rgba = wm_image.getpixel((x, y))
            #         orig = image.getpixel((x+coords[0], y+coords[1]))
            #         # alpha blend
            #         a = rgba[3] / 255
            #         r = int(rgba[0] * a + orig[0] * (1 - a))
            #         g = int(rgba[1] * a + orig[1] * (1 - a))
            #         b = int(rgba[2] * a + orig[2] * (1 - a))
            #         if not a == 0:
            #             image.putpixel((x+coords[0], y+coords[1]), (r, g, b))
            alpha = wm_image.getchannel('A')
            min_a, max_a = alpha.getextrema()
            if max_a == 0:
                log.warning(f'Watermark: image={wm_image} has no opaque pixels, nothing to blend')
            elif min_a == 255:
                log.warning(f'Watermark: image={wm_image} has no transparency, will overwrite region opaquely')
            base_mode = image.mode
            image = image.convert('RGBA')
            image.paste(wm_image, coords, mask=alpha) # alpha-composite using native PIL blending
            image = image.convert(base_mode)
            t1 = time.perf_counter()
            if not quiet:
                log.debug(f'Watermark: image={image} watermark={wm_image} position={position} coords={coords} time={t1 - t0:.4f}')
        except Exception as e:
            log.warning(f'Watermark: image={image} watermark={wm_image} {e}')
    return image


def set_text_watermark(image: Image.Image, wm_text: str | None = None, quiet: bool = False, verify: bool = False):
    if wm_text is not None and len(wm_text) > 0: # invisible watermark
        install('invisible-watermark')
        try:
            from imwatermark import WatermarkEncoder, WatermarkDecoder
        except Exception as e:
            log.warning(f'Watermark: cannot import WatermarkEncoder: {e}')
            return image
        t0 = time.perf_counter()
        wm_type = 'bytes'
        wm_method = 'dwtDctSvd'
        wm_length = len(wm_text) * 8
        length = wm_length // 8
        info = image.info
        data = np.asarray(image)
        encoder = WatermarkEncoder()
        decoder = WatermarkDecoder(wm_type, wm_length)
        text = f"{wm_text:<{length}}"[:length]
        bytearr = text.encode(encoding='ascii', errors='ignore')
        try:
            encoder.set_watermark(wm_type, bytearr)
            encoded = encoder.encode(data, wm_method)
            image = Image.fromarray(encoded)
            if verify:
                decoded = decoder.decode(encoded, wm_method)
                if decoded != bytearr:
                    log.warning(f'Watermark: text="{wm_text}" method={wm_method} bits={wm_length} embedding failed to verify')
            image.info = info
            t1 = time.perf_counter()
            if not quiet:
                log.debug(f'Watermark: text="{wm_text}" method={wm_method} bits={wm_length} time={t1 - t0:.4f}')
        except Exception as e:
            log.warning(f'Watermark: text="{wm_text}" method={wm_method} bits={wm_length} {e}')
    return image


def set_watermark(image: Image.Image, wm_text: str | None = None, wm_image: Image.Image | None = None, position: str | None = None):
    image = set_image_watermark(image, wm_image, position)
    image = set_text_watermark(image, wm_text)
    return image


def get_watermark(image: Image.Image | str, wm_text: str | None = None, quiet: bool = False) -> str:
    install('invisible-watermark')
    try:
        from imwatermark import WatermarkDecoder
    except Exception as e:
        log.warning(f'Watermark: cannot import WatermarkDecoder: {e}')
        return ''
    wm_type = 'bytes'
    wm_method = 'dwtDctSvd'
    wm_length = len(wm_text) * 8 if wm_text is not None else 32 # set based on desired wm length
    if isinstance(image, str):
        image = decode_base64_to_image(image, keep_alpha=True)
    data = np.asarray(image)
    decoder = WatermarkDecoder(wm_type, wm_length)
    try:
        decoded = decoder.decode(data, wm_method)
        if not quiet:
            log.debug(f'Watermark: image={image} wm="{decoded}" method={wm_method} bits={wm_length}')
    except Exception:
        decoded = ''
    return decoded


def post_watermark(image: str, wm_text: str | None = None, wm_image: Image.Image | None = None, position: str = 'none', fmt: str = 'PNG', quiet: bool = False) -> str:
    img = decode_base64_to_image(image)
    wm = decode_base64_to_image(wm_image, keep_alpha=True) if wm_image is not None else None # preserve alpha for watermark blending
    img = set_image_watermark(img, wm, position, quiet=quiet)
    img = set_text_watermark(img, wm_text, quiet=quiet)
    encoded = encode_pil_to_base64(img, ext=fmt)
    return encoded
