#!/usr/bin/env python
"""Force activation-statistics calibration for the currently loaded model.

Statistics normally accumulate passively during generations (see
modules/lora/lora_calib.py) and persist once per checkpoint; this tool runs
a few generations against a live server so the cache completes now instead
of during regular use. Progress is read from the server log, so it works
against remote instances as well.

Example:
    python cli/sdnq-calibrate.py --url http://127.0.0.1:7860
"""

import argparse
import json
import sys
import urllib.request


PROMPTS = [
    'photo of a woman in a park, detailed face, natural light',
    'cinematic street scene at night, rain, neon reflections, people with umbrellas',
    'still life of fruit and glassware on a wooden table, window light',
]


def api(base, path, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f'{base}{path}', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def calib_lines(base, clear=False):
    try:
        lines = api(base, f'/sdapi/v1/log?lines=300&clear={str(clear).lower()}')
        return [ln for ln in lines if 'Network calibration' in ln]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description='sdnq-calibrate')
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--prompt', action='append', default=None, help='calibration prompt; repeatable, defaults to a builtin set')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--gens', type=int, default=3)
    args = parser.parse_args()

    base = args.url.rstrip('/')
    opts = api(base, '/sdapi/v1/options', timeout=60)
    model = opts.get('sd_model_checkpoint')
    print(f'model: {model}')

    backlog = calib_lines(base)
    for ln in backlog:
        print(f'  {ln.strip()}')
    if any('saved=' in ln or 'loaded=' in ln for ln in backlog):
        print('statistics already cached for this model; nothing to do')
        return 0

    prompts = args.prompt or PROMPTS
    for i in range(args.gens):
        prompt = prompts[i % len(prompts)]
        print(f'gen {i + 1}/{args.gens}: steps={args.steps} size={args.size}')
        api(base, '/sdapi/v1/txt2img', dict(prompt=prompt, steps=args.steps, width=args.size, height=args.size,
                                            seed=1000 + i, save_images=False, send_images=False))
        lines = calib_lines(base, clear=True)
        for ln in lines:
            print(f'  {ln.strip()}')
        if any('saved=' in ln for ln in lines):
            print('calibration complete')
            return 0

    print('no calibration save observed; possible reasons: statistics already cached, '
          '"LoRA quantized host calibration" disabled, model not quantized below 8 bits, or model compile enabled')
    return 1


if __name__ == '__main__':
    sys.exit(main())
