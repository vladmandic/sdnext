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
    cfg_scale: float = 6.0
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
    name: str = str(round(time.time()))
    images: bool = False
    grids: bool = False
    labels: bool = False


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
server = Server()
options = Options()


def post():
    try:
        req = requests.post(f'{server.url}{server.api}',
                            json=vars(options),
                            timeout=300,
                            verify=False,
                            auth=requests.auth.HTTPBasicAuth(server.user, server.password) if (server.user is not None) and (server.password is not None) else None)
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url } if req.status_code != 200 else req.json()
    except Exception as e:
        return { 'error': 0, 'reason': str(e), 'url': server.url }


def generate(x: int, y: int): # pylint: disable=redefined-outer-name
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
            fn = os.path.join(server.folder, f'{server.name}-{x}-{y}.jpg') if server.images else None
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
        k = param[0].strip()
        v = param[1].strip() if len(param) > 1 else ''
        if k == 'prompt':
            options.prompt += f'{v} ' # prompt is appended so its not overwritten
        elif k == 'lora':
            options.prompt += f'<lora:{v}> ' # lora is appended to prompt
        else:
            setattr(options, k, v)

    log.info(server)
    os.makedirs(server.folder, exist_ok=True)
    try:
        x = open(x_file, encoding='utf8').read().splitlines() if x_file is not None else []
        y = open(y_file, encoding='utf8').read().splitlines() if y_file is not None else []
    except Exception as e:
        log.error(f'read file: x={x_file} y={y_file} {e}')
        return
    x = [line for line in x if ':' in line]
    y = [line for line in y if ':' in line]
    t0 = time.time()
    log.info(f'grid: x={len(x)} y={len(y)} prefix={server.name}')
    vertical = []
    Image.MAX_IMAGE_PIXELS = None
    for j in range(max(1, len(y))):
        horizontal = []
        labels = []
        for i in range(max(1, len(x))):
            if len(x) > i:
                set_param(x[i])
            if len(y) > j:
                set_param(y[j])
            images = generate(i, j)
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
        fn = os.path.join(server.folder, f'{server.name}.jpg')
        merged.save(fn)
        log.info(f'grid: size={merged.size} fn="{fn}"')
    t1 = time.time()
    log.info(f'done: time={t1-t0:.2f}')


if __name__ == "__main__":
    log.info(__file__)
    parser = argparse.ArgumentParser(description = 'api-grid')
    parser.add_argument('--x', type=str, required=False, default=None, help='file to use for x-axis values')
    parser.add_argument('--y', type=str, required=False, default=None, help='file to use for y-axis values')
    parser.add_argument('--folder', type=str, required=False, default='/tmp', help='folder to use for saving images')
    parser.add_argument('--name', type=str, required=False, default=str(round(time.time())), help='name prefix to use for saving images and grids')
    parser.add_argument('--image', type=bool, required=False, default=False, help='save individual images')
    parser.add_argument('--grid', type=bool, required=False, default=True, help='save image grids')
    parser.add_argument('--labels', type=bool, required=False, default=True, help='draw image labels')
    parser.add_argument('--url', type=str, required=False, default='http://127.0.0.1:7860', help='server url')
    parser.add_argument('--user', type=str, required=False, default=None, help='server user')
    parser.add_argument('--password', type=str, required=False, default=None, help='server password')
    parser.add_argument('--prompt', type=str, required=False, default='', help='generate prompt')
    parser.add_argument('--negative', type=str, required=False, default='', help='generate negative prompt')
    parser.add_argument('--sampler', type=str, required=False, default='Default', help='generate sampler')
    parser.add_argument('--width', type=int, required=False, default=1024, help='generate width')
    parser.add_argument('--height', type=int, required=False, default=1024, help='generate height')
    parser.add_argument('--steps', type=int, required=False, default=20, help='generate steps')
    parser.add_argument('--cfg', type=float, required=False, default=6.0, help='generate guidance scale')
    parser.add_argument('--seed', type=int, required=False, default=-1, help='generate seed')
    args = parser.parse_args()
    log.info(args)
    server.folder = args.folder
    server.name = args.name
    server.images = bool(args.image)
    server.grids = bool(args.grid)
    server.labels = bool(args.labels)
    server.url = args.url
    server.user = args.user
    server.password = args.password
    options.prompt = args.prompt
    options.negative_prompt = args.negative
    options.width = int(args.width)
    options.height = int(args.height)
    options.sampler_name = args.sampler
    options.seed = int(args.seed)
    options.steps = int(args.steps)
    options.cfg_scale = float(args.cfg)
    grid(args.x, args.y)
