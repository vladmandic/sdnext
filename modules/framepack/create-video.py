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


def get(endpoint: str, dct: dict = None):
    req = requests.get(f'{sd_url}{endpoint}', json=dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=None, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def encode(f):
    if not os.path.exists(f):
        log.error(f'file not found: {f}')
        os._exit(1)
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
    request = {
        'variant': args.variant,
        'prompt': args.prompt,
        'section_prompt': args.sections,
        'init_image': encode(args.init),
        'end_image': encode(args.end) if args.end else None,
        'resolution': int(args.resolution),
        'duration': float(args.duration),
        'mp4_fps': int(args.fps),
        'seed': int(args.seed),
        'steps': int(args.steps),
        'shift': float(args.shift),
        'cfg_scale': float(args.scale),
        'cfg_rescale': float(args.rescale),
        'cfg_distilled': float(args.distilled),
        'use_teacache': bool(args.teacache),
        'vlm_enhance': bool(args.enhance),
    }
    log.info(f'request: {args}')
    result = post('/sdapi/v1/framepack', request) # can abandon request here and not wait for response or wait synchronously
    log.info(f'response: {result}')

    progress = get('/sdapi/v1/progress?skip_current_image=true', None) # monitor progress of the current task
    task_id = progress.get('id', None)
    log.info(f'id: {task_id}')
    log.info(f'progress: {progress}')

    outputs = []
    history = get(f'/sdapi/v1/history?id={task_id}') # get history for the task
    for event in history:
        log.info(f'history: {event}')
        outputs = event.get('outputs', [])

    log.info(f'outputs: {outputs}') # you can download output files using /file={filename} endpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-framepack')
    parser.add_argument('--init', required=True, help='init image')
    parser.add_argument('--end', required=False, help='init image')
    parser.add_argument('--prompt', required=False, default='', help='prompt text')
    parser.add_argument('--sections', required=False, default='', help='per-section prompts')
    parser.add_argument('--resolution', type=int, required=False, default=640, help='video resolution')
    parser.add_argument('--duration', type=float, required=False, default=4.0, help='video duration')
    parser.add_argument('--fps', type=int, required=False, default=30, help='video frames per second')
    parser.add_argument('--seed', type=int, required=False, default=-1, help='random seed')
    parser.add_argument('--enhance', required=False, action='store_true', help='enable prompt enhancer')
    parser.add_argument('--teacache', required=False, action='store_true', help='enable teacache')
    parser.add_argument('--steps', type=int, default=25, help='steps')
    parser.add_argument('--scale', type=float, default=1.0, help='cfg scale')
    parser.add_argument('--rescale', type=float, default=0.0, help='cfg rescale')
    parser.add_argument('--distilled', type=float, default=10.0, help='cfg distilled')
    parser.add_argument('--shift', type=float, default=3.0, help='sampler shift')
    parser.add_argument('--variant', type=str, default='bi-directional', choices=['bi-directional', 'forward-only'], help='model variant')
    args = parser.parse_args()
    log.info(f'api-framepack: {args}')
    generate(args)
