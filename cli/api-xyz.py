#!/usr/bin/env python
# example: api-control.py --prompt "anime girl" --control "Canny:Canny:1.0:0.1:0.9:/home/vlado/generative/Samples/anime1.jpg,None:Depth:0.9:0.0:1.0:/home/vlado/generative/Samples/anime1.jpg" --hires --detailer --output /tmp/anime.jpg
import os
import io
import time
import base64
import logging
import argparse
import requests
import urllib3
from PIL import Image

sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

options = {
    "save_images": False,
    "send_images": True,
}


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def generate(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    options['prompt'] = args.prompt
    options['negative_prompt'] = args.negative

    options['xyz'] = {
        'draw_legend': args.legend,
        'include_grid': args.grid,
        'include_subgrids': args.subgrids,
        'include_images': args.images,
        'include_time': args.time,
        'include_text': args.text,
    }
    if args.x_type and args.x_values:
        options['xyz']['x_type'] = args.x_type
        options['xyz']['x_values'] = args.x_values

    if args.y_type and args.y_values:
        options['xyz']['y_type'] = args.y_type
        options['xyz']['y_values'] = args.y_values

    if args.z_type and args.z_values:
        options['xyz']['z_type'] = args.z_type
        options['xyz']['z_values'] = args.z_values

    data = post('/sdapi/v1/control', options)
    t1 = time.time()
    if 'info' in data:
        log.info(f'info: {data["info"]}')

    def get_image(encoded, output):
        if not isinstance(encoded, list):
            return
        for i in range(len(encoded)):
            b64 = encoded[i].split(',',1)[0]
            info = data['info']
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
            log.info(f'received image: size={image.size} time={t1-t0:.2f} info="{info}"')
            if output:
                image.save(output)
                log.info(f'image saved: size={image.size} filename={output}')

    if 'images' in data:
        get_image(data['images'], args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-control')
    parser.add_argument('--output', required=False, default=None, help='output filename')
    parser.add_argument('--prompt', required=False, default='', help='prompt text')
    parser.add_argument('--negative', required=False, default='', help='negative prompt text')
    parser.add_argument('--x-type', required=False, default=None, help='x axis type')
    parser.add_argument('--y-type', required=False, default=None, help='y axis type')
    parser.add_argument('--z-type', required=False, default=None, help='z axis type')
    parser.add_argument('--x-values', required=False, default=None, help='x axis values')
    parser.add_argument('--y-values', required=False, default=None, help='y axis values')
    parser.add_argument('--z-values', required=False, default=None, help='z axis values')
    parser.add_argument('--legend', required=False, default=True, help='Draw legend')
    parser.add_argument('--grid', required=False, default=True, help='Include grid')
    parser.add_argument('--subgrids', required=False, default=False, help='Include subgrids')
    parser.add_argument('--images', required=False, default=True, help='Include images')
    parser.add_argument('--time', required=False, default=True, help='Include time')
    parser.add_argument('--text', required=False, default=True, help='Include text')
    args = parser.parse_args()
    log.info(f'api-control: {args}')
    generate(args)
