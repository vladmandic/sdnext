#!/usr/bin/env python
import io
import os
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
options = {
    "save_images": True,
    "send_images": True,
}

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    if args.model is not None:
        post('/sdapi/v1/options', { 'sd_model_checkpoint': args.model })
        post('/sdapi/v1/reload-checkpoint') # needed if running in api-only to trigger new model load
    options['prompt'] = args.prompt
    options['negative_prompt'] = args.negative
    options['steps'] = int(args.steps)
    options['seed'] = int(args.seed)
    options['sampler_name'] = args.sampler
    options['width'] = int(args.width)
    options['height'] = int(args.height)
    if args.detailer:
        options['detailer'] = args.detailer
        options['denoising_strength'] = 0.5
        options['hr_sampler_name'] = args.sampler
    data = post('/sdapi/v1/txt2img', options)
    t1 = time.time()
    images = []
    if 'images' in data:
        for i in range(len(data['images'])):
            b64 = data['images'][i].split(',',1)[0]
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
            images.append(image)
            info = data['info']
            log.info(f'image received: size={image.size} time={t1-t0:.2f} info="{info}"')
            if args.output:
                image.save(args.output, exif=image._getexif())
                log.info(f'image saved: size={image.size} filename={args.output}')
    else:
        log.warning(f'no images received: {data}')
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-txt2img')
    parser.add_argument('--prompt', required=False, default='', help='prompt text')
    parser.add_argument('--negative', required=False, default='', help='negative prompt text')
    parser.add_argument('--width', required=False, default=512, help='image width')
    parser.add_argument('--height', required=False, default=512, help='image height')
    parser.add_argument('--steps', required=False, default=20, help='number of steps')
    parser.add_argument('--seed', required=False, default=-1, help='initial seed')
    parser.add_argument('--detailer', action='store_true', help='run detailer')
    parser.add_argument('--sampler', required=False, default='Euler a', help='sampler name')
    parser.add_argument('--output', required=False, default=None, help='output image file')
    parser.add_argument('--model', required=False, help='model name')
    args = parser.parse_args()
    log.info(f'api-txt2img: {args}')
    generate(args)
