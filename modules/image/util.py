from PIL import Image, ImageDraw
from modules.image.grid import get_font
from modules import shared


def draw_text(im, text: str = '', y_offset: int = 0):
    d = ImageDraw.Draw(im)
    fontsize = (im.width + im.height) // 50
    font = get_font(fontsize)
    d.text((fontsize//2, fontsize//2 + y_offset), text, font=font, fill=shared.opts.font_color)
    return im


def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""
    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background
    return img.convert('RGB')
