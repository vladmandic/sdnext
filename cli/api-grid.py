#!/usr/bin/env python
from dataclasses import dataclass
import io
import os
import time
import math
import base64
import logging
import argparse
import requests
import urllib3
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Options: # set default parameters here
    prompt: str = ''
    negative_prompt: str = ''
    seed: int = -1
    steps: int = 20
    sampler_name: str = "Default"
    width: int = 1024
    height: int = 1024
    save_images: bool = False
    send_images: bool = True


@dataclass
class Server: # set server and save options here or use command line arguments
    url: str = 'http://127.0.0.1:7860'
    api: str = '/sdapi/v1/txt2img'
    user: str = None
    password: str = None
    folder: str = '/tmp'
    images: bool = False
    grids: bool = False
    labels: bool = False


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
server = Server()
options = Options()


def post():
    req = requests.post(f'{server.url}{server.api}',
                        json=vars(options),
                        timeout=300,
                        verify=False,
                        auth=requests.auth.HTTPBasicAuth(server.user, server.password) if (server.user is not None) and (server.password is not None) else None)
    return { 'error': req.status_code, 'reason': req.reason, 'url': req.url } if req.status_code != 200 else req.json()


def generate(ts: float, x: int, y: int): # pylint: disable=redefined-outer-name
    t0 = time.time()
    log.info(f'x={x} y={y} {options}')
    data = post()
    t1 = time.time()
    images = []
    if 'images' in data:
        for i in range(len(data['images'])):
            b64 = data['images'][i].split(',',1)[0]
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
            images.append(image)
            info = data['info']
            fn = os.path.join(server.folder, f'{round(ts)}-{x}-{y}.jpg') if server.images else None
            log.info(f'image: time={t1-t0:.2f} size={image.size} fn="{fn}" info="{info}"')
            if fn is not None:
                image.save(fn)
    else:
        log.warning(data)
    return images


def merge(images: list[Image.Image], horizontal: bool, labels: list[str] = None):
    rows = 1 if horizontal else len(images)
    cols = math.ceil(len(images) / rows)
    w = max([i.size[0] for i in images])
    h = max([i.size[1] for i in images])
    image = Image.new('RGB', size = (cols * w, rows * h), color = 'black')
    font = ImageFont.truetype('DejaVuSansMono', 1024 // 32)
    for i, img in enumerate(images):
        x = i % cols * w
        y = i // cols * h
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        image.paste(img, box=(x, y))
        if labels is not None and len(images) == len(labels):
            ctx = ImageDraw.Draw(image)
            ctx.text((x + 1, y + 1), labels[i], font = font, fill = (0, 0, 0))
            ctx.text((x, y), labels[i], font = font, fill = (255, 255, 255))
    # log.info({ 'grid': { 'images': len(images), 'rows': rows, 'cols': cols, 'cell': [w, h] } })
    return image


def grid(x_file: str, y_file: str):
    def set_param(line):
        param = line.split(':', maxsplit=1)
        if param[0] == 'prompt':
            options.prompt += f'{param[1]} ' # prompt is appended so its not overwritten
        elif param[0] == 'lora':
            options.prompt += f'<lora:{param[1]}> ' # lora is appended to prompt
        else:
            setattr(options, param[0].strip(), param[1].strip())

    log.info(server)
    x = open(x_file, encoding='utf8').read().splitlines() if x_file is not None else []
    y = open(y_file, encoding='utf8').read().splitlines() if y_file is not None else []
    t0 = time.time()
    log.info(f'grid: x={len(x)} y={len(y)} prefix={round(t0)}')
    vertical = []
    Image.MAX_IMAGE_PIXELS = None
    for j in range(max(1, len(y))):
        horizontal = []
        labels = []
        for i in range(max(1, len(x))):
            if len(x) > i:
                set_param(x[i])
            if len(y) > i:
                set_param(y[i])
            images = generate(t0, i, j)
            if images is not None and len(images) > 0:
                horizontal.extend(images)
                labels.append(f'{x[i] if len(x) > i else ""}\n{y[j] if len(y) > j else ""}')
            options.prompt = '' # reset prompt
        if server.grids:
            if len(horizontal) == 0:
                log.warning(f'grid: empty row={j}')
                continue
            merged = merge(horizontal, horizontal=True, labels=labels if server.labels else None)
            vertical.append(merged)
    if server.grids:
        if len(vertical) == 0:
            log.warning('grid: empty grid')
            return
        merged = merge(vertical, horizontal=False)
        fn = os.path.join(server.folder, f'{round(t0)}.jpg')
        merged.save(fn)
        log.info(f'grid: size={merged.size} fn="{fn}"')
    t1 = time.time()
    log.info(f'done: time={t1-t0:.2f}')


if __name__ == "__main__":
    log.info(__file__)
    parser = argparse.ArgumentParser(description = 'api-txt2img')
    parser.add_argument('--x', required=False, default=None, help='file to use for x-axis values')
    parser.add_argument('--y', required=False, default=None, help='file to use for y-axis values')
    parser.add_argument('--folder', required=False, default='/tmp', help='folder to use for saving images')
    parser.add_argument('--image', required=False, default=False, help='save individual images')
    parser.add_argument('--grid', required=False, default=True, help='save image grids')
    parser.add_argument('--labels', required=False, default=True, help='draw image labels')
    parser.add_argument('--url', required=False, default='http://127.0.0.1:7860', help='server url')
    parser.add_argument('--user', required=False, default=None, help='server user')
    parser.add_argument('--password', required=False, default=None, help='server password')
    args = parser.parse_args()
    log.info(args)
    server.folder = args.folder
    server.images = args.image
    server.grids = args.grid
    server.labels = args.labels
    server.url = args.url
    server.user = args.user
    server.password = args.password
    grid(args.x, args.y)
