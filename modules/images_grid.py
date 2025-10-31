import math
from collections import namedtuple
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from modules import shared, script_callbacks


Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def check_grid_size(imgs):
    if imgs is None or len(imgs) == 0:
        return False
    mp = 0
    for img in imgs:
        if isinstance(img, list):
            for im in img:
                mp += im.width * im.height if im is not None else 0
        else:
            mp += img.width * img.height if img is not None else 0
    mp = round(mp / 1000000)
    ok = mp <= shared.opts.img_max_size_mp
    if not ok:
        shared.log.warning(f'Maximum image size exceded: size={mp} maximum={shared.opts.img_max_size_mp} MPixels')
    return ok


def get_grid_size(imgs, batch_size=1, rows=None, cols=None):
    if rows and rows > len(imgs):
        rows = len(imgs)
    if cols and cols > len(imgs):
        cols = len(imgs)
    if rows is None and cols is None:
        if shared.opts.n_rows > 0:
            rows = shared.opts.n_rows
            cols = math.ceil(len(imgs) / rows)
        elif shared.opts.n_rows == 0:
            rows = batch_size
            cols = math.ceil(len(imgs) / rows)
        elif shared.opts.n_cols > 0:
            cols = shared.opts.n_cols
            rows = math.ceil(len(imgs) / cols)
        elif shared.opts.n_cols == 0:
            cols = batch_size
            rows = math.ceil(len(imgs) / cols)
        else:
            rows = math.floor(math.sqrt(len(imgs)))
            while len(imgs) % rows != 0:
                rows -= 1
            cols = math.ceil(len(imgs) / rows)
    elif cols is None:
        cols = math.ceil(len(imgs) / rows)
    elif rows is None:
        rows = math.ceil(len(imgs) / cols)
    else:
        pass
    return rows, cols


def image_grid(imgs, batch_size:int=1, rows:int=None, cols:int=None):
    rows, cols = get_grid_size(imgs, batch_size, rows=rows, cols=cols)
    params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
    script_callbacks.image_grid_callback(params)
    imgs = [i for i in imgs if i is not None] if imgs is not None else []
    if len(imgs) == 0:
        return None
    w, h = max(i.width for i in imgs if i is not None), max(i.height for i in imgs if i is not None)
    grid = Image.new('RGB', size=(params.cols * w, params.rows * h), color=shared.opts.grid_background)
    for i, img in enumerate(params.imgs):
        if img is not None:
            grid.paste(img, box=(i % params.cols * w, i // params.cols * h))
    return grid


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height
    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap
    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0
    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []
        y = int(row * dy)
        if y + tile_h >= h:
            y = h - tile_h
        for col in range(cols):
            x = int(col * dx)
            if x + tile_w >= w:
                x = w - tile_w
            tile = image.crop((x, y, x + tile_w, y + tile_h))
            row_images.append([x, tile_w, tile])
        grid.tiles.append([y, tile_h, row_images])
    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))
    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue
            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))
        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue
        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))
    return combined_image


class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = str(text)
        self.is_active = is_active
        self.size = None


def get_font(fontsize):
    try:
        return ImageFont.truetype(shared.opts.font or "javascript/notosans-nerdfont-regular.ttf", fontsize)
    except Exception:
        return ImageFont.truetype("javascript/notosans-nerdfont-regular.ttf", fontsize)


def draw_grid_annotations(im, width, height, x_texts, y_texts, margin=0, title=None):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing: ImageDraw, draw_x, draw_y, lines, initial_fnt, initial_fontsize):
        for line in lines:
            font = initial_fnt
            fontsize = initial_fontsize
            while drawing.multiline_textbbox((0,0), text=line.text, font=font)[2] > line.allowed_width and fontsize > 0:
                fontsize -= 1
                font = get_font(fontsize)
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=font, fill=shared.opts.font_color if line.is_active else color_inactive, anchor="mm", align="center")
            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)
            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    font = get_font(fontsize)
    color_inactive = (127, 127, 127)
    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in y_texts]) == 0 else width * 3 // 4
    cols = len(x_texts)
    rows = len(y_texts)
    # assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    # assert rows == len(hor_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'
    calc_img = Image.new("RGB", (1, 1), shared.opts.grid_background)
    calc_d = ImageDraw.Draw(calc_img)
    title_texts = [title] if title else [[GridAnnotation()]]
    for texts, allowed_width in zip(x_texts + y_texts + title_texts, [width] * len(x_texts) + [pad_left] * len(y_texts) + [(width+margin)*cols]):
        items = [] + texts
        texts.clear()
        for line in items:
            wrapped = wrap(calc_d, line.text, font, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]
        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=font)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            line.allowed_width = allowed_width
    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in x_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in y_texts]
    pad_top = 0 if sum(hor_text_heights) == 0 else max(hor_text_heights) + line_spacing * 2
    title_pad = 0
    if title:
        title_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in title_texts] # pylint: disable=unsubscriptable-object
        title_pad = 0 if sum(title_text_heights) == 0 else max(title_text_heights) + line_spacing * 2
    result = Image.new("RGB", (im.width + pad_left + margin * (cols-1), im.height + pad_top + title_pad + margin * (rows-1)), shared.opts.grid_background)
    for row in range(rows):
        for col in range(cols):
            cell = im.crop((width * col, height * row, width * (col+1), height * (row+1)))
            result.paste(cell, (pad_left + (width + margin) * col, pad_top + title_pad + (height + margin) * row))
    d = ImageDraw.Draw(result)
    if title:
        x = pad_left + ((width+margin)*cols) / 2
        y = title_pad / 2 - title_text_heights[0] / 2
        draw_texts(d, x, y, title_texts[0], font, fontsize)
    for col in range(cols):
        x = pad_left + (width + margin) * col + width / 2
        y = (pad_top / 2 - hor_text_heights[col] / 2) + title_pad
        draw_texts(d, x, y, x_texts[col], font, fontsize)
    for row in range(rows):
        x = pad_left / 2
        y = (pad_top + (height + margin) * row + height / 2 - ver_text_heights[row] / 2) + title_pad
        draw_texts(d, x, y, y_texts[row], font, fontsize)
    return result


def draw_prompt_matrix(im, width, height, all_prompts, margin=0):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]
    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]
    return draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin)
