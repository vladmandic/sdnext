from functools import lru_cache
import time
import random
import numpy as np
from PIL import Image
from modules.logger import log
from modules.api.helpers import decode_base64_to_image, encode_pil_to_base64
from installer import install


@lru_cache(maxsize=1)
def resolve_xy(pos: str, w_main: int, h_main: int, w_wm: int, h_wm: int, pad: int = 0) -> tuple[int, int]:
    if pos == "top/left":
        return pad, pad
    if pos == "top/center":
        return max(0, (w_main - w_wm) // 2), pad
    if pos == "top/right":
        return max(0, w_main - w_wm - pad), pad
    if pos == "bottom/left":
        return pad, max(0, h_main - h_wm - pad)
    if pos == "bottom/center":
        return max(0, (w_main - w_wm) // 2), max(0, h_main - h_wm - pad)
    if pos == "bottom/right":
        return max(0, w_main - w_wm - pad), max(0, h_main - h_wm - pad)
    if pos == "center":
        return max(0, (w_main - w_wm) // 2), max(0, (h_main - h_wm) // 2)
    if pos == "random":
        return (random.randint(0, max(0, w_main - w_wm)), random.randint(0, max(0, h_main - h_wm)))
    raise ValueError(f"Unsupported position: {pos}")


@lru_cache(maxsize=1)
def load_watermark(path: str | Image.Image, mask: bool, scale: float): # pylint: disable=unused-argument
    try:
        if path is None:
            return None
        if hasattr(path, "name"): # gr.tempfile
            path = path.name
        if isinstance(path, Image.Image):
            image = path
        else:
            image = Image.open(path)
            image.load()
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        if mask and not has_alpha:
            from modules.rembg import lucida
            image = lucida.remove(image.convert("RGB"))
        if scale != 1.0:
            width, height = image.size
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        log.error(f"Watermark: {e}")
        return None


def set_video_watermark(
    video,
    wm_image: Image.Image | None = None,
    position: str = "none",
    mask: bool = False,
    scale: float = 1.0,
    opacity: float = 1.0,
    pad: int = 0,
    quiet: bool = False,
):
    if video is None or not position or position.lower() == "none":
        return video
    t0 = time.perf_counter()
    wm_image = load_watermark(wm_image, mask=mask, scale=scale)
    if wm_image is None:
        return video
    try:
        from modules import video as v
        info = v.get_video_info(video) # {'file': 'furiousa-test.mp4', 'size': 3581130, 'container': 'mp4', 'frames': 221, 'fps': 24.0, 'width': 1422, 'height': 800, 'duration': 9.208, 'codec': 'h264', 'format': 'yuv420p', 'bitrate_kbps': 2976, 'audio_streams': 1, 'audio_channels': 2, 'audio_sample_rate': 44100, 'audio': '1x2x44100'}
        meta = v.get_video_metadata(video)
        reader = v.read_generator(video)
        first_frame = next(reader)
        first_frame = np.asarray(set_image_watermark(Image.fromarray(first_frame), wm_image=wm_image, position=position, opacity=opacity, pad=pad, quiet=True))
        filename, writer = v.save_video(p=None,
                                         images=[first_frame],
                                         filename=None,
                                         metadata=meta,
                                         video_type=info['container'],
                                         duration=info['duration'],
                                         sync=True,
                                         stream=True,
                                         frames_hint=info['frames'],
                                         mp4_codec=info['codec'],
                                         mp4_ext=info['container'],
                                         mp4_pix=info['format'],
                                         mp4_opt=None,
                                         source=video,
                                         fps=info['fps'],
                                         audio_rate=info.get('audio_sample_rate') if info.get('audio_streams') else None,
                                         audio_layout='stereo' if info.get('audio_channels') == 2 else 'mono')
        for frame in reader: # reader is generator/iterator that yields av.VideoFrame converted to RGB numpy arrays
            wm = Image.fromarray(frame)
            wm = set_image_watermark(wm, wm_image=wm_image, position=position, opacity=opacity, pad=pad, quiet=True) # returns PIL.Image, do not pass mask/scale here as wm_image is already processed
            wm = np.asarray(wm)
            writer.write(wm)

        reader.close()
        v.write_video_audio(video, writer)
        writer.release()

        if not quiet:
            t1 = time.perf_counter()
            log.debug(f'Watermark: video="{filename}" watermark={wm_image} position="{position}" time={t1 - t0:.4f}')
        return filename

    except Exception as e:
        log.error(f"Watermark: video={video} watermark={wm_image} {e}")
        from modules import errors
        errors.display(e, 'watermark')
        return video


@lru_cache(maxsize=1)
def get_rgba(wm_image: Image.Image, opacity: float) -> tuple[Image.Image, Image.Image]:
    wm_rgba = wm_image.convert("RGBA")
    wm_alpha = wm_rgba.getchannel("A")
    min_a, max_a = wm_alpha.getextrema()
    if max_a == 0:
        log.warning(f"Watermark: image={wm_image} has no opaque pixels, nothing to blend")
    elif min_a == 255:
        log.warning(f"Watermark: image={wm_image} has no transparency, will overwrite region opaquely")

    opacity = max(0.0, min(1.0, float(opacity)))
    if opacity < 1.0:
        lut = [min(255, max(0, int(i * opacity))) for i in range(256)]
        wm_alpha = wm_alpha.point(lut)
        wm_rgba.putalpha(wm_alpha)
    return wm_rgba, wm_alpha


def set_image_watermark(
    image: Image.Image,
    wm_image: Image.Image | None = None,
    position: str = "none",
    mask: bool = False,
    scale: float = 1.0,
    opacity: float = 1.0,
    pad: int = 0,
    quiet: bool = False,
) -> Image.Image:
    if image is None or not position or position.lower() == "none":
        return image
    t0 = time.perf_counter()
    wm_image = load_watermark(wm_image, mask=mask, scale=scale)
    if wm_image is None:
        return image

    try:
        coords = resolve_xy(position, image.width, image.height, wm_image.width, wm_image.height, pad=pad)
        wm_rgba, wm_alpha = get_rgba(wm_image, opacity)

        base_mode = image.mode
        work = image if base_mode == "RGBA" else image.convert("RGBA")
        work.paste(wm_rgba, coords, mask=wm_alpha)
        result = work if base_mode == "RGBA" else work.convert(base_mode)

        if not quiet:
            t1 = time.perf_counter()
            log.debug(f'Watermark: image={result} watermark={wm_image} position="{position}" coords={coords} time={t1 - t0:.2f}')
        return result
    except Exception as e:
        log.warning(f"Watermark: image={image} watermark={wm_image} {e}")
        return image


def set_text_watermark(image: Image.Image, wm_text: str | None = None, quiet: bool = False, verify: bool = False):
    if wm_text is not None and len(wm_text) > 0: # invisible watermark
        install('invisible-watermark')
        try:
            from imwatermark import WatermarkEncoder, WatermarkDecoder
        except Exception as e:
            log.warning(f'Watermark: cannot import WatermarkEncoder: {e}')
            return image
        t0 = time.perf_counter()
        wm_type = 'bytes'
        wm_method = 'dwtDctSvd'
        wm_length = len(wm_text) * 8
        length = wm_length // 8
        info = image.info
        data = np.asarray(image)
        encoder = WatermarkEncoder()
        decoder = WatermarkDecoder(wm_type, wm_length)
        text = f"{wm_text:<{length}}"[:length]
        bytearr = text.encode(encoding='ascii', errors='ignore')
        try:
            encoder.set_watermark(wm_type, bytearr)
            encoded = encoder.encode(data, wm_method)
            image = Image.fromarray(encoded)
            if verify:
                decoded = decoder.decode(encoded, wm_method)
                if decoded != bytearr:
                    log.warning(f'Watermark: text="{wm_text}" method={wm_method} bits={wm_length} embedding failed to verify')
            image.info = info
            t1 = time.perf_counter()
            if not quiet:
                log.debug(f'Watermark: text="{wm_text}" method={wm_method} bits={wm_length} time={t1 - t0:.4f}')
        except Exception as e:
            log.warning(f'Watermark: text="{wm_text}" method={wm_method} bits={wm_length} {e}')
    return image


def set_watermark(image: Image.Image, wm_text: str | None = None, wm_image: Image.Image | None = None, position: str | None = None):
    image = set_image_watermark(image, wm_image, position)
    image = set_text_watermark(image, wm_text)
    return image


def get_watermark(image: Image.Image | str, wm_text: str | None = None, quiet: bool = False) -> str:
    install('invisible-watermark')
    try:
        from imwatermark import WatermarkDecoder
    except Exception as e:
        log.warning(f'Watermark: cannot import WatermarkDecoder: {e}')
        return ''
    wm_type = 'bytes'
    wm_method = 'dwtDctSvd'
    wm_length = len(wm_text) * 8 if wm_text is not None else 32 # set based on desired wm length
    if isinstance(image, str):
        image = decode_base64_to_image(image, keep_alpha=True)
    data = np.asarray(image)
    decoder = WatermarkDecoder(wm_type, wm_length)
    try:
        decoded = decoder.decode(data, wm_method)
        if not quiet:
            log.debug(f'Watermark: image={image} wm="{decoded}" method={wm_method} bits={wm_length}')
    except Exception:
        decoded = ''
    return decoded


def post_watermark(
    image: str | None = None,
    wm_text: str | None = None,
    wm_image: Image.Image | str | None = None,
    position: str = 'none',
    fmt: str = 'PNG',
    quiet: bool = False,
    mask: bool = False,
    scale: float = 1.0,
    opacity: float = 1.0,
    pad: int = 0,
    verify: bool = False,
    video: str | None = None,
    video_format: str | None = None,
) -> str:
    from fastapi.exceptions import HTTPException
    if (image is None) == (video is None):
        raise HTTPException(status_code=400, detail="Provide exactly one of image or video")
    if video is not None and (wm_text is not None or verify):
        raise HTTPException(status_code=400, detail="Text watermarking and verification are supported for images only")

    if image is not None:
        img = decode_base64_to_image(image)
        wm = decode_base64_to_image(wm_image, preserve_mode=True) if wm_image is not None else None # preserve source mode and alpha to support background masking
        img = set_image_watermark(img, wm, position, mask=mask, scale=scale, opacity=opacity, pad=pad, quiet=quiet)
        img = set_text_watermark(img, wm_text, quiet=quiet, verify=verify)
        return encode_pil_to_base64(img, ext=fmt)

    import base64
    import binascii
    import os
    import re
    import tempfile
    from modules.api import helpers
    from modules.api.helpers import MAX_B64_BYTES
    video_data = video.strip()
    container = video_format or "mp4"
    if video_data.startswith("data:"):
        header, separator, video_data = video_data.partition(",")
        if not separator or ";base64" not in header:
            raise HTTPException(status_code=400, detail="Video data URI must be base64 encoded")
        mime_type = header[5:].split(";", 1)[0]
        if video_format is None and mime_type.startswith("video/"):
            container = mime_type.split("/", 1)[1]
    container = container.lower().lstrip(".")
    if not re.fullmatch(r"[a-z0-9]{1,8}", container):
        raise HTTPException(status_code=400, detail="Invalid video format")
    if len(video_data) > (MAX_B64_BYTES * 4 // 3) + 4:
        raise HTTPException(status_code=413, detail="Encoded video exceeds the size limit")
    try:
        video_bytes = base64.b64decode(video_data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail="Invalid base64-encoded video") from e
    if len(video_bytes) > MAX_B64_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds the size limit")

    wm = decode_base64_to_image(wm_image, preserve_mode=True) if isinstance(wm_image, str) and wm_image else wm_image
    with tempfile.TemporaryDirectory(prefix="sdnext-watermark-") as temp_dir:
        input_path = os.path.join(temp_dir, f"input.{container}")
        with open(input_path, "wb") as file:
            file.write(video_bytes)
        output_path = set_video_watermark(
            video=input_path,
            wm_image=wm,
            position=position,
            mask=mask,
            scale=scale,
            opacity=opacity,
            pad=pad,
            quiet=quiet,
        )
        if wm_image and position.lower() != "none" and output_path == input_path:
            raise HTTPException(status_code=500, detail="Unable to apply watermark to video")
        if not output_path or not os.path.isfile(output_path):
            raise HTTPException(status_code=500, detail="Unable to process video")
        try:
            encoded = helpers.encode_file_to_base64(output_path)
            if encoded is None:
                raise HTTPException(status_code=413, detail="Watermarked video exceeds the response size limit")
            return encoded
        finally:
            if output_path != input_path and os.path.isfile(output_path):
                os.remove(output_path)
