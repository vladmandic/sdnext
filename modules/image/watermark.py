import random
import numpy as np
from PIL import Image
from modules import shared
from modules.logger import log


def set_watermark(image, wm_text: str | None = None, wm_image: Image.Image | None = None):
    if shared.opts.image_watermark_position != 'none' and wm_image is not None: # visible watermark
        if isinstance(wm_image, str):
            try:
                wm_image = Image.open(wm_image)
            except Exception as e:
                log.warning(f'Set image watermark: image={wm_image} {e}')
                return image
        if isinstance(wm_image, Image.Image):
            if wm_image.mode != 'RGBA':
                wm_image = wm_image.convert('RGBA')
        if shared.opts.image_watermark_position == 'top/left':
            position = (0, 0)
        elif shared.opts.image_watermark_position == 'top/right':
            position = (image.width - wm_image.width, 0)
        elif shared.opts.image_watermark_position == 'bottom/left':
            position = (0, image.height - wm_image.height)
        elif shared.opts.image_watermark_position == 'bottom/right':
            position = (image.width - wm_image.width, image.height - wm_image.height)
        elif shared.opts.image_watermark_position == 'center':
            position = ((image.width - wm_image.width) // 2, (image.height - wm_image.height) // 2)
        else:
            position = (random.randint(0, image.width - wm_image.width), random.randint(0, image.height - wm_image.height))
        try:
            for x in range(wm_image.width):
                for y in range(wm_image.height):
                    rgba = wm_image.getpixel((x, y))
                    orig = image.getpixel((x+position[0], y+position[1]))
                    # alpha blend
                    a = rgba[3] / 255
                    r = int(rgba[0] * a + orig[0] * (1 - a))
                    g = int(rgba[1] * a + orig[1] * (1 - a))
                    b = int(rgba[2] * a + orig[2] * (1 - a))
                    if not a == 0:
                        image.putpixel((x+position[0], y+position[1]), (r, g, b))
            log.debug(f'Set image watermark: image={wm_image} position={position}')
        except Exception as e:
            log.warning(f'Set image watermark: image={wm_image} {e}')

    if shared.opts.image_watermark_enabled and wm_text is not None: # invisible watermark
        from imwatermark import WatermarkEncoder
        wm_type = 'bytes'
        wm_method = 'dwtDctSvd'
        wm_length = 32
        length = wm_length // 8
        info = image.info
        data = np.asarray(image)
        encoder = WatermarkEncoder()
        text = f"{wm_text:<{length}}"[:length]
        bytearr = text.encode(encoding='ascii', errors='ignore')
        try:
            encoder.set_watermark(wm_type, bytearr)
            encoded = encoder.encode(data, wm_method)
            image = Image.fromarray(encoded)
            image.info = info
            log.debug(f'Set invisible watermark: {wm_text} method={wm_method} bits={wm_length}')
        except Exception as e:
            log.warning(f'Set invisible watermark error: {wm_text} method={wm_method} bits={wm_length} {e}')

    return image


def get_watermark(image):
    from imwatermark import WatermarkDecoder
    wm_type = 'bytes'
    wm_method = 'dwtDctSvd'
    wm_length = 32
    data = np.asarray(image)
    decoder = WatermarkDecoder(wm_type, wm_length)
    try:
        decoded = decoder.decode(data, wm_method)
        wm = decoded.decode(encoding='ascii', errors='ignore')
    except Exception:
        wm = ''
    return wm
