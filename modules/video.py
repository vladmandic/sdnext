import os
import threading
import numpy as np
from PIL import Image
from modules import shared, errors
from modules.logger import log
from modules.image.namegen import FilenameGenerator # pylint: disable=unused-import
from modules.paths import resolve_output_path


def interpolate_frames(images, count: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3):
    if images is None:
        return []
    if not isinstance(images, list):
        images = [images]
    if count > 0:
        try:
            import modules.rife
            frames = modules.rife.interpolate(images, count=count, scale=scale, pad=pad, change=change)
            if len(frames) > 0:
                images = frames
        except Exception as e:
            log.error(f'RIFE interpolation: {e}')
            errors.display(e, 'RIFE interpolation')
    return [np.array(image) for image in images]


def save_video_atomic(images, filename, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3):
    try:
        import cv2
    except Exception as e:
        log.error(f'Save video: cv2: {e}')
        return
    savejob = shared.state.begin('Save video')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if video_type.lower() in ['gif', 'png']:
        append = images.copy()
        image = append.pop(0)
        if loop:
            append += append[::-1]
        frames=len(append) + 1
        image.save(
            filename,
            save_all = True,
            append_images = append,
            optimize = False,
            duration = 1000.0 * duration / frames,
            loop = 0 if loop else 1,
        )
        size = os.path.getsize(filename)
        log.info(f'Save video: file="{filename}" frames={len(append) + 1} duration={duration} loop={loop} size={size}')
    elif video_type.lower() != 'none':
        frames = interpolate_frames(images, count=interpolate, scale=scale, pad=pad, change=change)
        fourcc = "mp4v"
        h, w, _c = frames[0].shape
        fps = max(1.0, len(frames) / duration) if duration > 0 else 30.0
        video_writer = cv2.VideoWriter(filename, fourcc=cv2.VideoWriter_fourcc(*fourcc), fps=fps, frameSize=(w, h))
        for i in range(len(frames)):
            img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        size = os.path.getsize(filename)
        log.info(f'Save video: file="{filename}" frames={len(frames)} duration={duration} fourcc={fourcc} size={size}')
    shared.state.end(savejob)


def save_video(p, images, filename = None, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3, sync: bool = False):
    if images is None or len(images) < 2 or video_type is None or video_type.lower() == 'none':
        return None
    if interpolate > 0 and getattr(p, 'video_interpolated', False):
        interpolate = 0
    image = images[0]
    if p is not None:
        seed = p.all_seeds[0] if getattr(p, 'all_seeds', None) is not None else p.seed
        prompt = p.all_prompts[0] if getattr(p, 'all_prompts', None) is not None else p.prompt
        namegen = FilenameGenerator(p, seed=seed, prompt=prompt, image=image)
    else:
        namegen = FilenameGenerator(None, seed=0, prompt='', image=image)
    base_path = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)
    if filename is None and p is not None:
        filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
        filename = os.path.join(base_path, filename)
        filename = namegen.sequence(filename)
    else:
        if os.path.sep not in filename:
            filename = os.path.join(base_path, filename)
    ext = video_type.lower().split('/')[0] if '/' in video_type else video_type.lower()
    if not filename.lower().endswith(ext):
        filename += f'.{ext}'
    filename = namegen.sanitize(filename)
    shared.state.outputs(filename)
    if not sync:
        threading.Thread(target=save_video_atomic, args=(images, filename, video_type, duration, loop, interpolate, scale, pad, change)).start()
    else:
        save_video_atomic(images, filename, video_type, duration, loop, interpolate, scale, pad, change)
    return filename


def get_video_params(filepath: str, capture: bool = False):
    import cv2
    from modules.control.util import decode_fourcc
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        msg = f'Video open failed: path="{filepath}"'
        log.error(msg)
        raise RuntimeError(msg)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(video.get(cv2.CAP_PROP_FPS), 2)
    if fps > 0:
        duration = round(float(frames) / fps, 2)
    else:
        duration = 0
    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = decode_fourcc(video.get(cv2.CAP_PROP_FOURCC))
    frame = None
    if capture:
        _status, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
    video.release()
    return frames, fps, duration, w, h, codec, frame


def get_video_info(filepath: str):
    import cv2
    from modules.control.util import decode_fourcc

    try:
        info = {
            'file': os.path.basename(filepath),
            'size': os.path.getsize(filepath),
            'container': os.path.splitext(filepath)[1].lower().lstrip('.'),
        }
    except Exception as e:
        log.error(f'Video probe failed: path="{filepath}" {e}')
        return {}

    def get_prop(video, name: str, label: str | None, cast: type = float):
        prop_id = getattr(cv2, name, None)
        if prop_id is None:
            return None
        try:
            value = video.get(prop_id)
            if value is None:
                return None
            value = cast(value)
            if value == 0:
                return None
            if label is not None:
                info[label] = value
            return value
        except Exception:
            return None

    video = cv2.VideoCapture(filepath)
    try:
        if not video.isOpened():
            msg = f'Video open failed: path="{filepath}"'
            info['error'] = msg
            log.error(msg)
            return info
        try:
            backend = video.getBackendName()
            if backend:
                info['backend'] = backend
        except Exception:
            pass

        frames = get_prop(video, 'CAP_PROP_FRAME_COUNT', 'frames', int) or 0
        fps = get_prop(video, 'CAP_PROP_FPS', 'fps', float) or 0
        _width = get_prop(video, 'CAP_PROP_FRAME_WIDTH', 'width', int) or 0
        _height = get_prop(video, 'CAP_PROP_FRAME_HEIGHT', 'height', int) or 0
        _bitrate = get_prop(video, 'CAP_PROP_BITRATE', 'bitrate_kbps', float)
        _orientation = get_prop(video, 'CAP_PROP_ORIENTATION_META', 'rotation', float)

        if frames > 0 and fps > 0:
            info['duration'] = round(float(frames) / fps, 3)

        fourcc = get_prop(video, 'CAP_PROP_FOURCC', None, int) or 0
        info['codec'] = decode_fourcc(fourcc)

        pix = get_prop(video, 'CAP_PROP_CODEC_PIXEL_FORMAT', None, int) or 0
        info['format'] = decode_fourcc(pix)

        sar_num = get_prop(video, 'CAP_PROP_SAR_NUM', None, int) or 1
        sar_den = get_prop(video, 'CAP_PROP_SAR_DEN', None, int) or 1
        if (sar_num > 0 and sar_den > 0) and (sar_num != 1 or sar_den != 1):
            info['sar'] = f'{sar_num}/{sar_den}'

        audio_streams = get_prop(video, 'CAP_PROP_AUDIO_TOTAL_STREAMS', 'audio_streams', int) or 0
        audio_channels = get_prop(video, 'CAP_PROP_AUDIO_TOTAL_CHANNELS', 'audio_channels', int) or 0
        audio_sample_rate = get_prop(video, 'CAP_PROP_AUDIO_SAMPLES_PER_SECOND', 'audio_sample_rate', int) or 0
        if audio_streams > 0 or audio_channels > 0 or audio_sample_rate > 0:
            info['audio'] = f'{audio_streams}x{audio_channels}x{audio_sample_rate}'

        return info
    except Exception as e:
        log.error(f'Video probe failed: path="{filepath}" {e}')
        return info
    finally:
        video.release()


def get_video_metadata(video):
    if not video or not isinstance(video, str) or not os.path.isfile(video):
        return {}
    from modules.video_models.video_utils import check_av
    av = check_av()
    with av.open(video, mode="r") as container:
        return dict(container.metadata)
