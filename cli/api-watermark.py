#!/usr/bin/env python
"""
Example:
cli/api-watermark.py --input ~/generative/Samples --wm-image ui/assets/logo-transparent.png --position center --wm-text "sdnext-test-invisible-watermark-string" --verify
"""
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


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def get(endpoint: str, dct: dict | None = None):
    req = requests.get(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.text.strip('"')


def post(endpoint: str, dct: dict | None = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def display(image: Image.Image):
    try:
        import subprocess
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        subprocess.run(["timg", "--pixelation", "sixel", "-"], input=buf.getvalue())
    except Exception:
        pass


def encode(img: str | Image.Image):
    image = Image.open(img) if isinstance(img, str) else img
    with io.BytesIO() as stream:
        image.save(stream, 'PNG')
        image.close()
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def decode(encoding):
    if encoding.startswith("data:image/"):
        parts = encoding.split(";", 1)
        if len(parts) == 2:
            parts2 = parts[1].split(",", 1)
            encoding = parts2[1] if len(parts2) == 2 else parts2[0]
    return Image.open(io.BytesIO(base64.b64decode(encoding)))


def watermark(args): # pylint: disable=redefined-outer-name
    if os.path.isdir(args.input):
        filenames = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    elif os.path.isfile(args.input):
        filenames = [args.input]
    else:
        filenames = []
    if not args.quiet:
        log.info(f'Inputs: {filenames}')
    stats = { 'verified': 0, 'failed': 0, 'skipped': 0 }

    t0 = time.perf_counter()
    for fn in filenames:
        t1 = time.perf_counter()
        try:
            encoded = encode(fn)
        except Exception as e:
            log.error(f'Encode: file={fn}: {e}')
            stats['skipped'] += 1
            continue
        options = { 'image': encoded }
        if args.wm_text:
            options['wm_text'] = args.wm_text
        if args.wm_image:
            options['wm_image'] = encode(args.wm_image)
        if args.position:
            options['position'] = args.position
        response = post('/sdapi/v1/watermark', options)
        image = decode(response)
        t2 = time.perf_counter()
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            out = os.path.join(args.output, os.path.basename(fn))
            image.save(out)
        if not args.quiet:
            log.info(f'Watermarked: input="{fn}" output="{out}" image={image.size} watermark="{args.wm_image}" time={t2-t1:.4f}')

        if args.display and image is not None:
            display(image)

        if args.verify and args.wm_text:
            t3 = time.perf_counter()
            options = { 'image': encode(image), 'wm_text': '*' * len(args.wm_text) }
            wm = get('/sdapi/v1/watermark', options)
            t4 = time.perf_counter()
            if wm == args.wm_text:
                if not args.quiet:
                    log.info(f'Verified: input="{fn}" watermark="{wm}" time={t4-t3:.4f}')
                stats['verified'] += 1
            else:
                log.error(f'Failed: input="{fn}" expected="{args.wm_text}" got="{wm}" time={t4-t3:.4f}')
                stats['failed'] += 1

    t5 = time.perf_counter()
    log.info(f'Processed: images={len(filenames)} time={t5-t0:.4f} images/sec={len(filenames)/(t5-t0):.4f} avg={(t5-t0)/len(filenames):.4f} {stats}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-watermark')
    parser.add_argument('--input', required=True, default=None, help='input image or folder')
    parser.add_argument('--output', required=False, default=None, help='output folder')
    parser.add_argument('--wm-image', required=False, default=None, help='watermark image')
    parser.add_argument('--wm-text', required=False, default=None, help='watermark text')
    parser.add_argument('--position', required=False, default=None, help='watermark position')
    parser.add_argument('--display', default=False, action='store_true', help='Display images')
    parser.add_argument('--verify', default=False, action='store_true', help='Verify watermarks')
    parser.add_argument('--quiet', default=False, action='store_true', help='Suppress debug output')

    args = parser.parse_args()
    log.info(f'api-watermark: {args}')
    watermark(args)
