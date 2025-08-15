#!/usr/bin/env python
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


def encode(f):
    image = Image.open(f)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        image.close()
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def generate(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    if args.model is not None:
        post('/sdapi/v1/options', { 'sd_model_checkpoint': args.model })
        post('/sdapi/v1/reload-checkpoint') # needed if running in api-only to trigger new model load
    if args.init is not None:
        options['inits'] = [encode(args.init)]
        image = Image.open(args.init)
        options['width'] = image.width
        options['height'] = image.height
        image.close()
    if args.input is not None:
        options['inputs'] = [encode(args.input)]
        image = Image.open(args.input)
        options['width'] = image.width
        options['height'] = image.height
        image.close()
    options['prompt'] = args.prompt
    options['negative_prompt'] = args.negative
    options['steps'] = int(args.steps)
    options['seed'] = int(args.seed)
    options['sampler_name'] = args.sampler

    if args.control is not None:
        options['unit_type'] = args.type
        options['control'] = []
        for control in args.control.split(','):
            u = control.split(':')
            if len(u) < 2:
                log.error(f'invalid control: {control}')
                continue
            options['control'].append({
                'process': u[0].strip(),
                'model': u[1].strip(),
                'strength': float(u[2].strip()) if len(u) > 2 else 1.0,
                'start': float(u[3].strip()) if len(u) > 3 else 0.0,
                'end': float(u[4].strip()) if len(u) > 4 else 1.0,
            })

    if args.ipadapter is not None:
        options['ip_adapter'] = []
        for ipadapter in args.ipadapter.split(','):
            u = ipadapter.split(':')
            if len(u) < 2:
                log.error(f'invalid ipadapter: {ipadapter}')
                continue
            if not os.path.exists(u[1].strip()):
                log.error(f'invalid ipadapter image: {u[1]}')
                continue
            options['ip_adapter'].append({
                'adapter': u[0].strip(),
                'images': [encode(u[1].strip())],
                'scale': float(u[2].strip()) if len(u) > 2 else 1.0,
                'start': float(u[3].strip()) if len(u) > 3 else 0.1,
                'end': float(u[4].strip()) if len(u) > 4 else 1.0,
            })

    if args.mask is not None:
        options['mask'] = encode(args.mask)

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
    if 'processed' in data:
        get_image(data['processed'], args.processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-control')
    parser.add_argument('--init', required=False, default=None, help='init image')
    parser.add_argument('--input', required=False, default=None, help='input image')
    parser.add_argument('--mask', required=False, help='mask image')
    parser.add_argument('--prompt', required=False, default='', help='prompt text')
    parser.add_argument('--negative', required=False, default='', help='negative prompt text')
    parser.add_argument('--steps', required=False, default=20, help='number of steps')
    parser.add_argument('--seed', required=False, default=-1, help='initial seed')
    parser.add_argument('--sampler', required=False, default='UniPC', help='sampler name')
    parser.add_argument('--output', required=False, default=None, help='output image file')
    parser.add_argument('--processed', required=False, default=None, help='processed output file')
    parser.add_argument('--model', required=False, help='model name')
    parser.add_argument('--type', required=False, help='control type')
    parser.add_argument('--control', required=False, help='control units')
    parser.add_argument('--ipadapter', required=False, help='ipadapter units')
    args = parser.parse_args()
    log.info(f'api-control: {args}')
    generate(args)
