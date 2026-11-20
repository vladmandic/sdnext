#!/usr/bin/env python
"""Apply a watermark to an image or video through the SD.Next API."""
import argparse
import base64
import logging
import os
from pathlib import Path

import requests


SD_URL = os.environ.get("SDAPI_URL", "http://127.0.0.1:7860")
SD_USERNAME = os.environ.get("SDAPI_USR")
SD_PASSWORD = os.environ.get("SDAPI_PWD")
VIDEO_FORMATS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
log = logging.getLogger(__name__)


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def apply_watermark(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    is_video = input_path.suffix.lower() in VIDEO_FORMATS
    if is_video and (args.wm_text is not None or args.verify):
        raise ValueError("Text watermarking and verification are supported for images only")
    options = {
        "wm_image": encode_file(Path(args.wm_image)) if args.wm_image else None,
        "wm_text": args.wm_text,
        "position": args.position,
        "mask": args.mask,
        "scale": args.scale,
        "opacity": args.opacity,
        "pad": args.pad,
        "verify": args.verify,
    }
    key = "video" if is_video else "image"
    options[key] = encode_file(input_path)
    if is_video:
        options["video_format"] = input_path.suffix.lstrip(".")
    else:
        options["image_format"] = args.format

    auth = requests.auth.HTTPBasicAuth(SD_USERNAME, SD_PASSWORD) if SD_USERNAME and SD_PASSWORD else None
    response = requests.post(f"{SD_URL}/sdapi/v1/watermark", json=options, timeout=args.timeout, auth=auth)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.reason)
        except ValueError:
            detail = response.reason
        raise RuntimeError(f"Watermark API returned {response.status_code}: {detail}")
    encoded = response.json()
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}-watermarked{input_path.suffix}")
    output_path.write_bytes(base64.b64decode(encoded))
    log.info(f"Watermarked: input={input_path} output={output_path}")

    if args.verify:
        if args.wm_text:
            verify_options = {
                "image": encoded,
                "wm_text": "*" * len(args.wm_text),
            }
            verified = requests.get(f"{SD_URL}/sdapi/v1/watermark", json=verify_options, timeout=args.timeout, auth=auth)
            verified.raise_for_status()
            if verified.json() != args.wm_text:
                raise RuntimeError(f"Watermark verification failed: received {verified.json()!r}")
            log.info("Verified text watermark")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Apply an SD.Next API watermark")
    parser.add_argument("--input", required=True, help="Input image or video file")
    parser.add_argument("--output", help="Output file (defaults to <input>-watermarked.<ext>)")
    parser.add_argument("--wm-image", help="Watermark image file")
    parser.add_argument("--wm-text", help="Invisible watermark text (images only)")
    parser.add_argument("--position", default="bottom/right", help="Watermark position")
    parser.add_argument("--mask", action="store_true", help="Remove watermark image background")
    parser.add_argument("--scale", type=float, default=1.0, help="Watermark image scale")
    parser.add_argument("--opacity", type=float, default=1.0, help="Watermark image opacity (0-1)")
    parser.add_argument("--pad", type=int, default=0, help="Padding from the selected position")
    parser.add_argument("--format", default="PNG", help="Output image format")
    parser.add_argument("--verify", action="store_true", help="Verify invisible text watermark on images")
    parser.add_argument("--timeout", type=int, default=3600, help="API request timeout in seconds")
    apply_watermark(parser.parse_args())
