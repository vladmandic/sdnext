#!/usr/bin/env python

import os
import logging
import argparse
import cv2
import torch
import torchvision
from safetensors.torch import safe_open
from tqdm.rich import trange

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger("sd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'framepack-cli')
    parser.add_argument('--input', required=True, help='input safetensors')
    parser.add_argument('--cv2', required=False, help='encode video file using cv2')
    parser.add_argument('--tv', required=False, help='encode video file using torchvision')
    parser.add_argument('--codec', default='libx264', help='specify video codec')
    parser.add_argument('--export', required=False, help='export frames as images to folder')
    parser.add_argument('--fps', default=30, help='frames-per-second')
    args = parser.parse_args()

    log.info(f'framepack-cli: {args}')
    log.info(f'torch={torch.__version__} torchvision={torchvision.__version__}')

    with safe_open(args.input, framework="pt", device="cpu") as f:
        frames = f.get_tensor('frames')
        metadata = f.metadata()
    n, h, w, _c = frames.shape
    log.info(f'file: metadata={metadata}')
    log.info(f'tensor: frames={n} shape={frames.shape} dtype={frames.dtype} device={frames.device}')
    fn = os.path.splitext(os.path.basename(args.input))[0]

    if args.export:
        log.info(f'export: folder="{args.export}" prefix="{fn}" frames={n} width={w} height={h}')
        os.makedirs(args.export, exist_ok=True)
        for i in trange(n):
            image = cv2.cvtColor(frames[i].numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.export, f'{fn}-{i:05d}.jpg'), image)

    if args.cv2:
        log.info(f'encode: file={args.cv2} frames={n} width={w} height={h} fps={args.fps} method=cv2')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(args.cv2, fourcc, args.fps, (w, h))
        for i in trange(n):
            image = cv2.cvtColor(frames[i].numpy(), cv2.COLOR_RGB2BGR)
            video.write(image)
        video.release()

    if args.tv:
        log.info(f'encode: file={args.tv} frames={n} width={w} height={h} fps={args.fps} method=tv ')
        torchvision.io.write_video(args.tv, video_array=frames, fps=args.fps, video_codec=args.codec)
