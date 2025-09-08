#!/usr/bin/env python
import os
import io
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


def encode(f):
    if f is not None and os.path.exists(f):
        image = Image.open(f)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        log.info(f'encoding image: {image}')
        with io.BytesIO() as stream:
            image.save(stream, 'JPEG')
            image.close()
            values = stream.getvalue()
            encoded = base64.b64encode(values).decode()
            return encoded
    else:
        return None


def enhance(args): # pylint: disable=redefined-outer-name
    options = {
        'prompt': str(args.prompt),
        'seed': int(args.seed),
        'type': str(args.type),
        'nsfw': bool(args.nsfw),
    }
    if args.model:
        options['model'] = str(args.model)
    if args.image:
        options['image'] = encode(args.image)
    response = post('/sdapi/v1/prompt-enhance', options)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-enhance')
    parser.add_argument('--prompt', type=str, default='', required=False, help='prompt')
    parser.add_argument('--seed', type=int, default=-1, required=False, help='seed')
    parser.add_argument('--type', type=str, default='text', choices=['text', 'image', 'video'], required=False, help='enhance type')
    parser.add_argument('--model', type=str, default=None, required=False, help='model name')
    parser.add_argument('--image', type=str, default=None, required=False, help='optional input image')
    parser.add_argument('--nsfw', type=bool, action=argparse.BooleanOptionalAction, required=False, help='nsfw allowed')
    args = parser.parse_args()
    log.info(f'api-upscale: {args}')
    result = enhance(args)
    log.info(result)
