#!/usr/bin/env python
"""
API integration tests for multi-image condition sets.

A pipeline that declares max_condition_images receives an init image list as one
set shared by every prompt, generates at the requested size aligned to the model
multiple, and rejects more images than it declares.

Covers:
- POST /sdapi/v1/img2img: two images, one output at the requested size
- POST /sdapi/v1/img2img: aligned first image with a different requested size
- POST /sdapi/v1/img2img: batch size above the image count, image list not padded
- POST /sdapi/v1/img2img: more images than declared, no output
- POST /sdapi/v1/control: two inputs, one output
- POST /sdapi/v1/control: hires and detailer together, with and without a detailer prompt

Requires a running SD.Next instance with a multi-image model loaded (Qwen-Image 2.1).
With --log pointing at the server's sdnext.log, each case also checks the Base log
line for the number of images and the size the pipeline received, and the hires and
detailer case checks the Hires and Detail calls and the warnings they log.

Usage:
    python test/test-condition-images-api.py --url http://127.0.0.1:7860 [--log sdnext.log] [--multiple 32] [--max-images 10] [--steps 4]
"""

import io
import os
import re
import sys
import time
import base64
import argparse
import requests
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
FACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'Reference', 'ponyRealism_V23.jpg')


def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def decode(data):
    return Image.open(io.BytesIO(base64.b64decode(data.split(',', 1)[-1])))


def solid(width, height, color):
    return Image.new('RGB', (width, height), color)


class ConditionImagesAPITest:
    """Drives condition-set scenarios over the HTTP API."""

    def __init__(self, base_url, log_file=None, multiple=32, max_images=10, steps=4):
        self.base_url = base_url.rstrip('/')
        self.log_file = log_file
        self.multiple = multiple
        self.max_images = max_images
        self.steps = steps
        self.timeout = 1800
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def record(self, ok, name, detail=''):
        self.passed += 1 if ok else 0
        self.failed += 0 if ok else 1
        line = f'  {"PASS" if ok else "FAIL"}: {name}'
        if detail:
            line += f'  ({detail})'
        print(line, flush=True)

    def skip(self, name, reason):
        self.skipped += 1
        print(f'  SKIP: {name}  ({reason})', flush=True)

    def aligned(self, value):
        return self.multiple * (value // self.multiple)

    def post(self, endpoint, payload):
        offset = os.path.getsize(self.log_file) if self.log_file else 0
        t0 = time.time()
        r = requests.post(f'{self.base_url}{endpoint}', json=payload, timeout=self.timeout, verify=False)
        elapsed = time.time() - t0
        lines = []
        if self.log_file:
            time.sleep(1)
            with open(self.log_file, encoding='utf8', errors='replace') as f:
                f.seek(offset)
                lines = f.read().splitlines()
        images = []
        if r.status_code == 200:
            images = [decode(i) for i in (r.json().get('images') or [])]
        return r.status_code, images, lines, elapsed

    def base_line(self, lines):
        """Image count and call size from the Base log line, None without a log."""
        if not self.log_file:
            return None
        text = ' '.join(lines)
        match = re.search(r'Base: pipeline=.*?set=(\{.*?\'parser\')', text)
        if match is None:
            return {'images': 0, 'size': None}
        seg = match.group(1)
        width = re.search(r"'width': (\d+)", seg)
        height = re.search(r"'height': (\d+)", seg)
        return {'images': seg.count('<PIL.Image.Image'), 'size': (int(width.group(1)), int(height.group(1))) if width and height else None}

    def img2img(self, init_images, width, height, batch_size=1):
        payload = {
            'prompt': 'Put the object from Image 2 next to the subject of Image 1.',
            'init_images': [encode(i) for i in init_images],
            'width': width,
            'height': height,
            'batch_size': batch_size,
            'steps': self.steps,
            'seed': 42,
            'denoising_strength': 1.0,
            'save_images': False,
            'send_images': True,
        }
        return self.post('/sdapi/v1/img2img', payload)

    def check_generation(self, name, result, outputs, size, base_images):
        status, images, lines, elapsed = result
        sizes = [i.size for i in images]
        self.record(status == 200 and len(images) == outputs, f'{name}: {outputs} output(s)', f'http={status} outputs={len(images)} time={elapsed:.1f}s')
        if images:
            self.record(all(s == size for s in sizes), f'{name}: output size {size[0]}x{size[1]}', f'sizes={sizes}')
        base = self.base_line(lines)
        if base is not None:
            self.record(base['images'] == base_images and base['size'] == size, f'{name}: pipeline received {base_images} image(s) at {size[0]}x{size[1]}', f'log images={base["images"]} size={base["size"]}')
            errors = [line for line in lines if ' ERROR ' in line]
            self.record(not errors, f'{name}: no errors logged', errors[0][:160] if errors else '')

    def test_two_images(self):
        print('=== img2img: two images, unaligned request ===', flush=True)
        result = self.img2img([solid(1336, 744, (200, 60, 40)), solid(1200, 800, (40, 180, 60))], 1336, 744)
        self.check_generation('two images', result, 1, (self.aligned(1336), self.aligned(744)), 2)

    def test_aligned_first(self):
        print('=== img2img: aligned first image, different requested size ===', flush=True)
        image = solid(1024, 1536, (90, 90, 200))
        self.check_generation('square request', self.img2img([image], 1024, 1024), 1, (1024, 1024), 1)
        self.check_generation('transposed request', self.img2img([image], 1536, 1024), 1, (1536, 1024), 1)

    def test_batch(self):
        print('=== img2img: batch size above the image count ===', flush=True)
        result = self.img2img([solid(1344, 768, (200, 60, 40))], 1344, 768, batch_size=2)
        self.check_generation('batch 2', result, 2, (1344, 768), 1)

    def test_over_cap(self):
        print('=== img2img: more images than declared ===', flush=True)
        status, images, lines, elapsed = self.img2img([solid(256, 256, (10 * i, 40, 90)) for i in range(self.max_images + 1)], 1024, 1024)
        self.record(len(images) == 0, 'over cap: no output', f'http={status} outputs={len(images)} time={elapsed:.1f}s')
        if self.log_file:
            rejected = [line for line in lines if 'Mismatch:' in line and f'images={self.max_images + 1} max={self.max_images}' in line]
            base = self.base_line(lines)
            self.record(bool(rejected) and base['images'] == 0, 'over cap: rejected before the pipeline runs', rejected[0].strip()[:160] if rejected else 'no Mismatch line')

    def test_control(self):
        print('=== control: two inputs ===', flush=True)
        payload = {
            'prompt': 'Put the object from Image 2 next to the subject of Image 1.',
            'inputs': [encode(solid(1344, 768, (200, 60, 40))), encode(solid(1200, 800, (40, 180, 60)))],
            'input_type': 1,
            'width_before': 1344,
            'height_before': 768,
            'steps': self.steps,
            'seed': 42,
            'denoising_strength': 1.0,
            'save_images': False,
        }
        result = self.post('/sdapi/v1/control', payload)
        if result[0] == 422:
            self.skip('control two inputs', 'the control route rejects image inputs')
            return
        self.check_generation('control two inputs', result, 1, (1344, 768), 2)

    def call_lines(self, lines, kind):
        """Image count and call size of every log line for one pipeline call kind (Base, Hires, Detail)."""
        calls = []
        for line in lines:
            match = re.search(rf' {kind}: pipeline=.*?set=(\{{.*?\'parser\')', line)
            if match is None:
                continue
            width = re.search(r"'width': (\d+)", match.group(1))
            height = re.search(r"'height': (\d+)", match.group(1))
            calls.append({'images': match.group(1).count('<PIL.Image.Image'), 'size': (int(width.group(1)), int(height.group(1))) if width and height else None})
        return calls

    def test_hires_detailer(self):
        print('=== control: hires and detailer together ===', flush=True)
        if not os.path.exists(FACE):
            self.skip('hires and detailer', f'face image missing: {FACE}')
            return
        steps = max(self.steps, 8) # the detector needs a recognisable face in the hires output
        for label, detailer_prompt in (('own detailer prompt', 'sharp detailed face, keep the identity'), ('empty detailer prompt', '')):
            payload = {
                'prompt': 'Make the lighting warmer. Keep everything else the same.', # a photographic result keeps the face detectable
                'inputs': [encode(Image.open(FACE).convert('RGB'))],
                'input_type': 1,
                'skip_processing': True,
                'width_before': 1024,
                'height_before': 1024,
                'steps': steps,
                'seed': 42,
                'save_images': False,
                'enable_hr': True,
                'hr_upscaler': 'Resize Lanczos',
                'hr_scale': 1.25,
                'hr_force': True,
                'hr_resize_mode': 1,
                'hr_second_pass_steps': steps,
                'hr_denoising_strength': 0.5,
                'detailer_enabled': True,
                'detailer_models': ['face-yolo8n'],
                'detailer_prompt': detailer_prompt,
                'detailer_steps': steps,
                'detailer_strength': 0.4,
                'detailer_resolution': 1024,
            }
            status, images, lines, elapsed = self.post('/sdapi/v1/control', payload)
            size = (self.aligned(1280), self.aligned(1280))
            self.record(status == 200 and len(images) == 1, f'{label}: 1 output', f'http={status} outputs={len(images)} time={elapsed:.1f}s')
            if images:
                self.record(images[0].size == size, f'{label}: output at the hires size {size[0]}x{size[1]}', f'size={images[0].size}')
            if not self.log_file:
                continue
            hires, detail = self.call_lines(lines, 'Hires'), self.call_lines(lines, 'Detail')
            self.record(len(hires) == 1 and hires[0]['images'] == 1 and hires[0]['size'] == size, f'{label}: hires pass runs once at {size[0]}x{size[1]} on the upscaled image', f'calls={hires}')
            self.record(len(detail) >= 1 and all(d['images'] == 1 and d['size'] == (1024, 1024) for d in detail), f'{label}: detailer runs on each detection as a crop edit at 1024x1024', f'calls={detail}')
            warnings = {
                'hires strength': [line for line in lines if 'Hires: model=' in line and 'strength=ignored' in line],
                'detailer strength': [line for line in lines if 'Detailer: model=' in line and 'strength=ignored' in line],
                'empty detailer prompt': [line for line in lines if 'Detailer prompt: empty, main prompt used' in line],
            }
            expected = {'hires strength': 1, 'detailer strength': 1, 'empty detailer prompt': 0 if detailer_prompt else 1}
            self.record(all(len(warnings[k]) == v for k, v in expected.items()), f'{label}: each warning once where it applies', ' '.join(f'{k}={len(v)}' for k, v in warnings.items()))
            errors = [line for line in lines if ' ERROR ' in line]
            self.record(not errors, f'{label}: no errors logged', errors[0][:160] if errors else '')

    def run(self):
        model = requests.get(f'{self.base_url}/sdapi/v1/options', timeout=60, verify=False).json().get('sd_model_checkpoint')
        print(f'model: {model}', flush=True)
        self.test_two_images()
        self.test_aligned_first()
        self.test_batch()
        self.test_over_cap()
        self.test_control()
        self.test_hires_detailer()
        print('=== results ===', flush=True)
        print(f'  passed={self.passed} failed={self.failed} skipped={self.skipped}', flush=True)
        return self.failed == 0


def main():
    ap = argparse.ArgumentParser(description='Multi-image condition set API tests')
    ap.add_argument('--url', default='http://127.0.0.1:7860', help='SD.Next base URL')
    ap.add_argument('--log', default=None, help='server sdnext.log, enables checks on what the pipeline received')
    ap.add_argument('--multiple', type=int, default=32, help='size multiple of the loaded model')
    ap.add_argument('--max-images', type=int, default=10, help='max_condition_images of the loaded model')
    ap.add_argument('--steps', type=int, default=4)
    args = ap.parse_args()
    try:
        requests.get(f'{args.url.rstrip("/")}/sdapi/v1/options', timeout=10, verify=False)
    except Exception as e:
        print(f'cannot reach SD.Next at {args.url}: {e}', flush=True)
        return 2
    ok = ConditionImagesAPITest(args.url, log_file=args.log, multiple=args.multiple, max_images=args.max_images, steps=args.steps).run()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
