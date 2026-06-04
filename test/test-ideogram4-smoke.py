#!/usr/bin/env python
"""Standalone end-to-end smoke for diffusers-native Ideogram 4.

Loads the split-projection bf16 diffusers folder and quantizes both transformers
with SDNQ at load (the same path SD.Next uses), loads the shared Qwen3-VL text
encoder, builds the diffusers ``Ideogram4Pipeline``, and generates an image. This
exercises the real pipeline: both transformers under SDNQ, the Qwen3-VL 13-layer
tap, the dual-branch asymmetric CFG loop, the logit-normal schedule, the vae.bn
latent denorm, and VAE decode. SDNQ int4 fits the two towers plus the encoder on a
24GB GPU.

Usage:
    python test/test-ideogram4-smoke.py --model /path/to/Ideogram-4-bf16-split --output out.png
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="split-projection bf16 diffusers folder")
parser.add_argument("--output", default="ideogram4_smoke.png")
parser.add_argument("--prompt", default="a ginger cat wearing a tiny wizard hat reading a glowing spellbook, detailed digital illustration")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--weights-dtype", default="uint4", help="SDNQ weights dtype (uint4, int8, ...)")
parser.add_argument("--hf-cache", default=None, help="HF cache_dir for the shared Qwen3-VL encoder (default: HF default cache)")
args = parser.parse_args()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ["SD_INSTALL_QUIET"] = "1"

# Our own args are already parsed; clear argv (and leave it cleared) so sdnext's
# shared.py / devices, which re-parse argv on import, don't see this script's flags.
sys.argv = [sys.argv[0]]

import modules.cmd_args
import installer

modules.cmd_args.parse_args()
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

import torch
import diffusers
from transformers import AutoTokenizer
from transformers.models.qwen3_vl import Qwen3VLModel

from modules import devices
from modules.sdnq import SDNQConfig

TE_REPO = "Qwen/Qwen3-VL-8B-Instruct"


def main() -> int:
    device = devices.device
    cfg = SDNQConfig(weights_dtype=args.weights_dtype)

    print(f"loading transformer (sdnq {args.weights_dtype}) ...", flush=True)
    transformer = diffusers.Ideogram4Transformer2DModel.from_pretrained(args.model, subfolder="transformer", quantization_config=cfg, torch_dtype=torch.bfloat16).to(device)
    print("loading unconditional_transformer ...", flush=True)
    uncond = diffusers.Ideogram4Transformer2DModel.from_pretrained(args.model, subfolder="unconditional_transformer", quantization_config=cfg, torch_dtype=torch.bfloat16).to(device)
    print("loading text encoder Qwen3-VL ...", flush=True)
    te_kwargs = {"cache_dir": args.hf_cache} if args.hf_cache else {}
    text_encoder = Qwen3VLModel.from_pretrained(TE_REPO, quantization_config=SDNQConfig(weights_dtype=args.weights_dtype), torch_dtype=torch.bfloat16, **te_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    print("loading vae ...", flush=True)
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")

    pipe = diffusers.Ideogram4Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        unconditional_transformer=uncond,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(f"generating {args.width}x{args.height} steps={args.steps} ...", flush=True)
    start = time.time()
    out = pipe(prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=7.0, guidance_schedule=None, width=args.width, height=args.height, generator=generator)
    elapsed = time.time() - start

    image = out.images[0]
    image.save(args.output)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    print(f"PASS: generated {args.output} in {elapsed:.1f}s, size={image.size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
