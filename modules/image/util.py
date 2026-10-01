from PIL import Image, ImageDraw
from modules.image.grid import get_font
from modules import shared


def draw_text(im, text: str = '', y_offset: int = 0, fontsize: int = 0, color: str | None = None):
    d = ImageDraw.Draw(im)
    if fontsize == 0:
        fontsize = (im.width + im.height) // 50
    font = get_font(fontsize)
    d.text((fontsize//2, fontsize//2 + y_offset), text, font=font, fill=color or shared.opts.font_color)
    return im


def collapse_alpha(img):
    """returns RGB for an RGBA image whose alpha never drops below half, since decoders leave near-opaque alpha on opaque images"""
    if img.mode == 'RGBA' and img.getextrema()[3][0] >= 128:
        return img.convert('RGB')
    return img


def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""
    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background
    return img.convert('RGB')
