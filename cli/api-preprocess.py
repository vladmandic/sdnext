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


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def get(endpoint: str, dct: dict | None = None):
    req = requests.get(f'{sd_url}{endpoint}', json=dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def post(endpoint: str, dct: dict | None = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def info(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    with open(args.input, 'rb') as f:
        content = f.read()
    models = get('/sdapi/v1/preprocessors')
    log.info(f'models: {models}')
    req = {
        'model': args.model or 'Canny',
        'image': base64.b64encode(content).decode(),
        'config': { 'low_threshold': 50 },
    }
    data = post('/sdapi/v1/preprocess', req)
    t1 = time.time()
    if 'image' in data:
        b64 = data['image'].split(',',1)[0]
        image = Image.open(io.BytesIO(base64.b64decode(b64)))
        log.info(f'received image: size={image.size} time={t1-t0:.2f}')
        if args.output:
            image.save(args.output)
            log.info(f'saved image: fn={args.output}')
    else:
        log.info(f'received: {data} time={t1-t0:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-preprocess')
    parser.add_argument('--input', required=True, help='input image')
    parser.add_argument('--model', required=True, help='preprocessing model')
    parser.add_argument('--output', required=False, help='output image')
    args = parser.parse_args()
    log.info(f'api-preprocess: {args}')
    info(args)
