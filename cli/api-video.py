#!/usr/bin/env python
# python cli/api-video.py --prompt "a paper boat drifting down a rain gutter" --frames 17 --steps 8 --output /tmp/video.mp4
import os
import time
import base64
import logging
import argparse
import threading
import requests
import urllib3

sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)
options = {
    "send_video": True,
    "send_thumbnail": False,
}

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def get(endpoint: str, params: dict | None = None, timeout: int = 60):
    req = requests.get(f'{sd_url}{endpoint}', params=params, timeout=timeout, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    return req.json()


def post(endpoint: str, dct: dict | None = None, timeout: int = 3600):
    req = requests.post(f'{sd_url}{endpoint}', json=dct, timeout=timeout, verify=False, auth=auth())
    if req.status_code != 200:
        res = { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
        try:
            res['detail'] = req.json().get('detail', None)
        except Exception:
            pass
        return res
    return req.json()


def encode(f: str):
    with open(f, 'rb') as file:
        return base64.b64encode(file.read()).decode()


def list_models():
    data = get('/sdapi/v1/video/models')
    if isinstance(data, dict) and 'error' in data:
        log.error(f'video models: {data}')
        return
    for item in data:
        loaded = ' loaded=true' if item.get('loaded') else ''
        log.info(f'engine="{item["engine"]}" model="{item["name"]}" mode={item["mode"]}{loaded}')
    log.info(f'video models: {len(data)}')


def watch_progress(stop_event: threading.Event):
    while not stop_event.is_set():
        status = get('/sdapi/v1/progress', params={ 'skip_current_image': True })
        if 'error' not in status:
            state = status.get('state') or {}
            log.info(f'progress={status.get("progress", 0):.2f} eta={status.get("eta_relative", 0):.1f} step={state.get("sampling_step", 0)}/{state.get("sampling_steps", 0)} info="{status.get("textinfo") or ""}"')
        stop_event.wait(5)


def save_output(data: dict, output: str):
    if data.get('video'):
        with open(output, 'wb') as f:
            f.write(base64.b64decode(data['video']))
        log.info(f'video saved: filename={output}')
    elif data.get('still') and data.get('frames'):
        with open(output, 'wb') as f:
            f.write(base64.b64decode(data['frames'][0]))
        log.info(f'still saved: filename={output}')
    elif data.get('video_path'):
        req = requests.get(f'{sd_url}/sdapi/v1/video/file', params={ 'file': data['video_path'] }, timeout=300, verify=False, auth=auth())
        if req.status_code == 200:
            with open(output, 'wb') as f:
                f.write(req.content)
            log.info(f'video fetched: filename={output} size={len(req.content)}')
        else:
            log.error(f'video fetch failed: code={req.status_code} reason={req.reason}')
    else:
        log.warning('no video output received')


def generate(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    if args.engine:
        options['engine'] = args.engine
    if args.model:
        options['model'] = args.model
    options['prompt'] = args.prompt
    options['negative_prompt'] = args.negative
    options['width'] = int(args.width)
    options['height'] = int(args.height)
    options['frames'] = int(args.frames)
    options['steps'] = int(args.steps)
    options['seed'] = int(args.seed)
    options['sampler_name'] = args.sampler
    options['mp4_fps'] = int(args.fps)
    options['mp4_interpolate'] = int(args.interpolate)
    options['audio'] = bool(args.audio)
    if args.init:
        options['init_image'] = encode(args.init)
    if args.last:
        options['last_image'] = encode(args.last)
    if args.reference:
        options['references'] = [encode(f) for f in args.reference]
    stop_event = threading.Event()
    if args.progress:
        threading.Thread(target=watch_progress, args=(stop_event,), daemon=True).start()
    data = post('/sdapi/v1/video', options, timeout=int(args.timeout))
    stop_event.set()
    t1 = time.time()
    if 'error' in data:
        log.error(f'generate failed: {data}')
        return
    log.info(f'video received: frames={data.get("frames_count")} fps={data.get("fps")} duration={data.get("duration")} audio={data.get("has_audio")} still={data.get("still")} path={data.get("video_path")} time={t1-t0:.2f}')
    if args.output:
        save_output(data, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-video')
    parser.add_argument('--list', action='store_true', help='list video engines and models')
    parser.add_argument('--engine', required=False, default=None, help='video engine; omit with --model to use the loaded checkpoint')
    parser.add_argument('--model', required=False, default=None, help='video model name within the engine')
    parser.add_argument('--prompt', required=False, default='', help='prompt text')
    parser.add_argument('--negative', required=False, default='', help='negative prompt text')
    parser.add_argument('--width', required=False, default=832, help='video width')
    parser.add_argument('--height', required=False, default=480, help='video height')
    parser.add_argument('--frames', required=False, default=17, help='number of frames; 1 for a still image')
    parser.add_argument('--steps', required=False, default=20, help='number of steps')
    parser.add_argument('--seed', required=False, default=-1, help='initial seed')
    parser.add_argument('--sampler', required=False, default='Default', help='sampler name')
    parser.add_argument('--fps', required=False, default=24, help='frames per second')
    parser.add_argument('--interpolate', required=False, default=0, help='rife interpolation passes')
    parser.add_argument('--audio', action=argparse.BooleanOptionalAction, default=True, help='generate audio on supported models')
    parser.add_argument('--init', required=False, default=None, help='init image file')
    parser.add_argument('--last', required=False, default=None, help='last frame image file')
    parser.add_argument('--reference', required=False, default=None, action='append', help='reference image file for reference workflows; repeat in the order the model should read them')
    parser.add_argument('--output', required=False, default=None, help='output video file')
    parser.add_argument('--progress', action='store_true', help='poll and log progress during generation')
    parser.add_argument('--timeout', required=False, default=3600, help='request timeout in seconds')
    args = parser.parse_args()
    log.info(f'api-video: {args}')
    if args.list:
        list_models()
    else:
        generate(args)
