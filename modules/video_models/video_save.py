import os
import re
import time
import cv2
import numpy as np
import torch
import einops
from PIL import Image
from modules import shared, errors ,timer, rife, processing
from modules.logger import log
from modules.video_models.video_utils import check_av
from modules.video_models.video_upscale import upscale_video
from modules.video_models.video_writer import open_video_writer, rejected_options


def get_audio_rate(p=None, default: int = 24000) -> int:
    # pipeline output wins when it reports a rate, else the loaded vocoder: LTX-2.0 runs at 24k,
    # 2.3 and 2.5 at 48k, and muxing at the wrong rate shifts the pitch
    rate = getattr(p, 'audio_sampling_rate', None) if p is not None else None
    if not rate:
        vocoder = getattr(shared.sd_model, 'vocoder', None)
        rate = getattr(getattr(vocoder, 'config', None), 'output_sampling_rate', None)
    return int(rate) if rate else default


def get_video_filename(p:processing.StableDiffusionProcessingVideo):
    from modules.image.namegen import FilenameGenerator
    from modules.paths import resolve_output_path
    namegen = FilenameGenerator(p, seed=p.seed if p is not None else 0, prompt=p.prompt if p is not None else '')
    filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
    base_path = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)
    if shared.opts.save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]")
        dirfile = os.path.dirname(filename)
        dirname = os.path.join(base_path, dirname, dirfile)
    else:
        dirname = base_path
    filename = os.path.join(dirname, filename)
    filename = namegen.sanitize(filename, base=base_path) # before numbering and folder creation, so both see the folder that is used
    filename = namegen.sequence(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename


def save_params(p, filename: str | None = None):
    from modules.paths import params_path
    if p is None:
        dct = {}
    else:
        # sampler_index, sampler_shift, dynamic_shift, guidance_scale, guidance_true, init_image, init_strength, last_image, vae_type, vae_tile_frames, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, mp4_thumb, vlm_enhance, vlm_model, vlm_system_prompt, override_settings = args
        dct = {
            "Prompt": p.prompt,
            "Negative prompt": p.negative_prompt,
            "Steps": p.steps,
            "Sampler": p.sampler_name,
            "Seed": p.seed,
            "Engine": p.video_engine,
            "Model": p.video_model,
            "Frames": p.frames,
            "Size": f"{p.width}x{p.height}",
            "Styles": ','.join(p.styles) if isinstance(p.styles, list) else p.styles,
        }
    params = ', '.join([f'{k}: {v}' for k, v in dct.items() if v is not None and v != ''])
    fn = filename if filename is not None else params_path
    with open(fn, "w", encoding="utf8") as file:
        file.write(params)


def create_video_metadata(p: processing.StableDiffusionProcessingVideo | None, metadata: dict | None = None, filename: str | None = None):
    metadata = metadata.copy() if metadata is not None else {}
    if not shared.opts.image_metadata:
        return metadata
    title = os.path.basename(filename) if filename else 'SD.Next video'
    metadata.setdefault('title', title)
    metadata.setdefault('encoder', 'SD.Next')
    if p is None:
        return metadata
    try:
        info = processing.create_infotext(p)
    except Exception as e:
        log.debug(f'Video metadata: infotext failed: {e}')
        info = ''
    if len(info) == 0:
        info = getattr(p, 'prompt', '') or ''
    if len(info) == 0:
        return metadata
    metadata.setdefault('comment', info)
    metadata.setdefault('description', info)
    return metadata


def images_to_tensor(images):
    if images is None or len(images) == 0:
        return None
    array = [torch.from_numpy(np.array(image)) for image in images]
    tensor = torch.stack(array, dim=0) # n h w c
    tensor = tensor.unsqueeze(0) # 1, n, h, w, c
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous() # 1, c, n, h, w
    tensor = (tensor.float() / 127.5) - 1.0 # from [0,255] to [-1,1]
    # log.debug(f'Video output: images={len(images)} tensor={tensor.shape}')
    return tensor


def numpy_to_tensor(images):
    if images is None or len(images) == 0:
        return None
    images = (2.0 * images) - 1.0 # from [0,1] to [-1,1]
    array = [torch.from_numpy(images[i]) for i in range(images.shape[0])]
    tensor = torch.stack(array, dim=0) # n h w c
    tensor = tensor.unsqueeze(0) # 1, n, h, w, c
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous() # 1, c, n, h, w
    # tensor = (tensor.float() / 127.5) - 1.0 # from [0,255] to [-1,1]
    # log.debug(f'Video output: images={len(images)} tensor={tensor.shape}')
    return tensor


def atomic_save_video(
    filename: str,
    tensor: torch.Tensor,
    audio: torch.Tensor | None = None,
    fps: float = 24,
    codec: str = "libx264",
    pix_fmt: str | None = None,
    options: str = "",
    sample_rate: int = 24000,
    metadata: dict | None = None,
    pbar=None,
):
    if check_av() is None:
        log.error('Video: ffmpeg/av not available')
        return False
    frames, height, width, _channels = tensor.shape
    has_audio = (audio is not None) and ((torch.is_tensor(audio) or isinstance(audio, np.ndarray)) or (isinstance(audio, dict) and len(audio.get('frames', [])) > 0))
    audio_rate = None
    audio_layout = 'stereo'
    if has_audio:
        audio_rate = audio.get('sr', sample_rate) if isinstance(audio, dict) else sample_rate
        audio_layout = audio.get('layout', 'stereo') if isinstance(audio, dict) else 'stereo'
    savejob = shared.state.record('Save video')
    writer = None
    try:
        writer = open_video_writer(filename, width, height, fps, codec, pix_fmt, options, metadata, audio_rate=audio_rate, audio_layout=audio_layout)
        log.info(f'Video: file="{filename}" codec={writer.codec} pix_fmt={writer.pix_fmt} frames={frames} width={writer.width} height={writer.height} fps={writer.rate} audio={audio is not None} sample_rate={sample_rate} options={writer.options}')
        video_array = torch.as_tensor(tensor, dtype=torch.uint8).numpy(force=True)
        task = pbar.add_task('encoding', total=frames) if pbar is not None else None
        if task is not None:
            pbar.update(task, description='video encoding')
        for img in video_array:
            writer.write(img)
            if task is not None:
                pbar.update(task, advance=1)
        if has_audio:
            writer.write_audio(audio, sample_rate)
        writer.release()
        savejob.outputs.append(filename)
    except Exception:
        if writer is not None:
            try:
                writer.release()
            except Exception as cleanup_error:
                log.error(f'Video: file="{filename}" cleanup failed: {cleanup_error}')
        raise
    finally:
        savejob.end()
    return True


def save_thumbnail(video_path, tensor=None):
    try:
        base = os.path.splitext(video_path)[0]
        thumb_path = f'{base}.thumb.jpg'
        if tensor is not None and len(tensor) > 0:
            frame = Image.fromarray(tensor[0].numpy())
        else:
            from modules.video import get_video_params
            _frames, _fps, _dur, _w, _h, _codec, frame = get_video_params(video_path, capture=True)
        if frame is not None:
            frame.thumbnail((512, 512), Image.Resampling.LANCZOS)
            frame.save(thumb_path, quality=80)
            return thumb_path
    except Exception as e:
        log.debug(f'Video thumbnail: {e}')
    return None


def save_video(
    p: processing.StableDiffusionProcessingVideo,
    pixels: torch.Tensor | None = None,
    audio: torch.Tensor | None = None,
    binary: bytes | None = None,
    mp4_fps: float = 24,
    mp4_codec: str = "libx264",
    mp4_opt: str = "",
    mp4_ext: str = "mp4",
    mp4_sf: bool = False,  # save safetensors
    mp4_video: bool = True,  # save video
    mp4_frames: bool = False,  # save frames
    mp4_thumb: bool = True,  # save thumbnail
    mp4_interpolate: int = 0,  # rife interpolation
    aac_sample_rate: int = 24000,  # audio sample rate
    upscale_scale: float = 1.0,  # upscale scale
    upscale_upscaler: str = "",  # upscale upscaler
    stream=None,  # async progress reporting stream
    metadata: dict | None = None,  # metadata for video
    pbar=None,  # progress bar for video
    reclamp: bool = True,  # reclamp pixels to [-1, 1] range
):
    if metadata is None:
        metadata = {}
    output_video = None
    if not re.fullmatch(r'[A-Za-z0-9]+', mp4_ext or ''): # the extension is appended to the output path
        log.error(f'Video: ext="{mp4_ext}" invalid')
        return 0, output_video, None
    rejected = rejected_options(mp4_opt)
    if rejected:
        log.error(f'Video: options={rejected} not allowed')
        return 0, output_video, None

    if binary is not None:
        output_filename = get_video_filename(p)
        output_video = f'{output_filename}.{mp4_ext}'
        try:
            try:
                with open(output_video, 'wb') as f:
                    f.write(binary)
                log.info(f'Video output: file="{output_video}" size={len(binary)}')
                shared.state.outputs(output_video)
            except Exception as e:
                log.error(f'Video output: file="{output_video}" {e}')
        except Exception as e:
            log.error(f'Video output: file="{output_video}" write error {e}')
            errors.display(e, 'video')
        thumb = save_thumbnail(output_video) if mp4_thumb else None
        return 0, output_video, thumb

    if pixels is None:
        return 0, output_video, None
    if isinstance(pixels, np.ndarray):
        pixels = numpy_to_tensor(pixels)
    if isinstance(pixels, list) and isinstance(pixels[0], Image.Image):
        pixels = images_to_tensor(pixels)
    if not torch.is_tensor(pixels):
        log.error(f'Video: type={type(pixels)} not a tensor')
        return 0, output_video, None

    try:
        if mp4_interpolate > 0 and not getattr(p, 'video_interpolated', False):
            t_interpolate = time.time()
            x = pixels.squeeze(0).permute(1, 0, 2, 3)
            x = (x.clamp(-1., 1.) + 1.0) * 0.5  # RIFE expects [0, 1]; video pixels are [-1, 1]
            interpolated = rife.interpolate_nchw(x, count=mp4_interpolate+1)
            pixels = torch.stack(interpolated, dim=0)
            pixels = pixels.permute(1, 2, 0, 3, 4)
            pixels = pixels * 2.0 - 1.0
            timer.process.ts('interpolate', t_interpolate)
            if p is not None:
                p.video_interpolated = True
    except Exception as e:
        log.error(f'Video interpolate: {e}')
        errors.display(e, 'video')

    try:
        if upscale_upscaler is not None and len(upscale_upscaler) > 0 and not getattr(p, 'video_upscaled', False):
            t_upscale = time.time()
            pixels = upscale_video(pixels, scale=upscale_scale, upscaler_name=upscale_upscaler)
            timer.process.ts('upscale', t_upscale)
            if p is not None:
                p.video_upscaled = True
    except Exception as e:
        log.error(f'Video upscale: {e}')
        errors.display(e, 'video')

    t_save = time.time()
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(0)
    n, _c, t, h, w = pixels.shape
    size = pixels.element_size() * pixels.numel()
    log.debug(f'Video: video={mp4_video} export={mp4_frames} safetensors={mp4_sf} interpolate={mp4_interpolate}')
    if hasattr(audio, 'shape'):
        audio_txt = f'audio={audio.shape} aac={aac_sample_rate}' if audio is not None else 'no audio'
    elif isinstance(audio, dict):
        audio_txt = f'audio={audio.get("format", None)} packets={len(audio.get("frames", []))} '
    else:
        audio_txt = None
    log.debug(f'Video: encode={t} tensor={pixels.shape} bytes={size} {audio_txt} fps={mp4_fps} codec={mp4_codec} ext={mp4_ext} options="{mp4_opt}"')
    try:
        preparejob = shared.state.record('Prepare video')
        if stream is not None:
            stream.output_queue.push(('progress', (None, 'Saving video...')))

        if reclamp:
            x = torch.clamp(pixels.float(), -1., 1.) * 127.5 + 127.5
        else:
            x = pixels.float() * 255.0
        x = x.detach().cpu().to(torch.uint8)
        x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=n)
        x = x.contiguous()

        output_filename = get_video_filename(p)
        if shared.opts.save_txt:
            save_params(p, f'{output_filename}.txt')
        save_params(p)

        if mp4_sf:
            fn = f'{output_filename}.safetensors'
            log.info(f'Video export: file="{fn}" type=savetensors shape={x.shape}')
            from safetensors.torch import save_file
            preparejob.outputs.append(fn)
            save_file({ 'frames': x }, fn, metadata={'format': 'video', 'frames': str(t), 'width': str(w), 'height': str(h), 'fps': str(mp4_fps), 'codec': mp4_codec, 'options': mp4_opt, 'ext': mp4_ext, 'interpolate': str(mp4_interpolate)})

        if mp4_frames:
            log.info(f'Video frames: files="{output_filename}-00000.jpg" frames={t} width={w} height={h}')
            for i in range(t):
                image = cv2.cvtColor(x[i].numpy(), cv2.COLOR_RGB2BGR)
                fn = f'{output_filename}-{i:05d}.jpg'
                preparejob.outputs.append(fn)
                cv2.imwrite(fn, image)

        preparejob.end()

        if mp4_video and (mp4_codec != 'none'):
            target = f'{output_filename}.{mp4_ext}'
            metadata = create_video_metadata(p, metadata, output_filename)
            if atomic_save_video(target, tensor=x, audio=audio, fps=mp4_fps, codec=mp4_codec, options=mp4_opt, sample_rate=aac_sample_rate, metadata=metadata, pbar=pbar):
                output_video = target
            if stream is not None and output_video is not None:
                stream.output_queue.push(('progress', (None, f'Video {os.path.basename(output_video)} | Codec {mp4_codec} | Size {w}x{h}x{t} | FPS {mp4_fps}')))
                stream.output_queue.push(('file', output_video))
        else:
            if stream is not None:
                stream.output_queue.push(('progress', (None, '')))

    except Exception as e:
        log.error(f'Video save: raw={size} {e}')
        errors.display(e, 'video')
    timer.process.add('save', time.time()-t_save)
    thumb = save_thumbnail(output_video) if mp4_thumb and output_video is not None else None
    return t, output_video, thumb
