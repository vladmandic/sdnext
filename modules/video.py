import os
import threading
import numpy as np
from PIL import Image
import cv2
from modules import shared, errors
from modules.logger import log
from modules.image.namegen import FilenameGenerator # pylint: disable=unused-import
from modules.paths import resolve_output_path
from modules.video_models.video_writer import VideoWriteError, VideoWriter, open_video_writer, rejected_options

os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '8') # fatal only; OpenCV reads it once, at its first capture


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


def save_video_frames(video_writer: VideoWriter, frames):
    """RGB frames into an open writer; DLSS calls it per chunk in stream mode."""
    for frame in frames:
        video_writer.write(frame)


DEFAULT_VIDEO_CODEC = 'libx264'
VIDEO_TYPE_CODECS = {'mp4v': 'mpeg4', 'divx': 'mpeg4', 'avc1': 'libx264', 'h264': 'libx264', 'jvt3': 'libx264', 'mjpg': 'mjpeg'} # codec half of the "Video format" labels


def codec_from_video_type(video_type: str) -> str:
    """Encoder named by the codec half of a "Video format" label such as MP4/AVC1; the default when the label has none."""
    label = str(video_type or '')
    if '/' not in label:
        return DEFAULT_VIDEO_CODEC
    half = label.split('/', 1)[1].strip().lower()
    codec = VIDEO_TYPE_CODECS.get(half)
    if codec is None:
        log.warning(f'Video: format="{label}" codec={half} has no encoder, using {DEFAULT_VIDEO_CODEC}')
        return DEFAULT_VIDEO_CODEC
    return codec


def resolve_codec(video_type: str, mp4_codec: str | None, mp4_opt: str | None) -> tuple[str, str]:
    """Encoder and options for a save: an explicit codec wins over the label; unset options take the codec's preset, '' means none."""
    codec = mp4_codec if mp4_codec is not None else codec_from_video_type(video_type)
    if mp4_opt is None:
        from modules.video_models.video_codecs import get_codec_options
        mp4_opt = get_codec_options(codec)
    return codec, mp4_opt


def save_video_atomic(images,
                      filename,
                      video_type: str = 'none',
                      duration: float = 2.0,
                      loop: bool = False,
                      interpolate: int = 0,
                      scale: float = 1.0,
                      pad: int = 1,
                      change: float = 0.3,
                      stream: bool = False,
                      frames_hint: int = 0,
                      metadata: dict | None = None,
                      mp4_codec: str | None = None,
                      mp4_pix: str | None = None,
                      mp4_opt: str | None = None,
                      *,
                      fps: float | None = None,
                     ) -> VideoWriter | None:
    """Write the clip; with stream=True the open writer is returned for more frames.

    fps: output rate; None derives it from the frame count and duration.
    Raises VideoWriteError when the save is refused or fails, after logging the reason once."""
    mp4_codec, mp4_opt = resolve_codec(video_type, mp4_codec, mp4_opt)
    rejected = rejected_options(mp4_opt)
    if rejected:
        log.error(f'Video: options={rejected} not allowed')
        raise VideoWriteError(f'options {rejected} not allowed')
    savejob = shared.state.record('Save video')
    video_writer = None
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if video_type.lower() in ['gif', 'png']:
            append = images.copy()
            image = append.pop(0)
            if loop:
                append += append[::-1]
            num_frames=len(append) + 1
            image.save(
                filename,
                save_all = True,
                append_images = append,
                optimize = False,
                duration = 1000.0 * duration / num_frames,
                loop = 0 if loop else 1,
            )
            log.info(f'Save video: file="{filename}" frames={num_frames} duration={duration} loop={loop} size={os.path.getsize(filename)}')
        elif video_type.lower() != 'none':
            frames = interpolate_frames(images, count=interpolate, scale=scale, pad=pad, change=change)
            h, w = frames[0].shape[:2]
            num_frames = frames_hint if frames_hint > 0 else len(frames)
            if fps is None:
                fps = max(1.0, num_frames / duration) if duration > 0 else 30.0
            writer = open_video_writer(filename, w, h, fps, mp4_codec, mp4_pix, mp4_opt, metadata)
            video_writer = writer
            save_video_frames(writer, frames)
            if not stream:
                writer.release()
                video_writer = None
            hint = 'stream=True' if stream else f'size={os.path.getsize(filename)}'
            log.info(f'Save video: file="{filename}" frames={num_frames} duration={duration} fps={fps:.2f} resolution={(writer.width, writer.height)} codec={writer.codec} fmt={writer.pix_fmt} options={writer.options} {hint}')
            frames.clear()
    except Exception as e:
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception as cleanup_error:
                log.error(f'Save video: cleanup failed: {cleanup_error}')
            video_writer = None
        log.error(f'Save video: file="{filename}" {e}')
        if isinstance(e, VideoWriteError):
            raise
        errors.display(e, 'Video')
        raise VideoWriteError(str(e)) from e
    finally:
        savejob.end()
    return video_writer


def save_video_thread(*args, **kwargs):
    try:
        save_video_atomic(*args, **kwargs)
    except VideoWriteError:
        pass # logged by save_video_atomic


def save_video(p,
               images,
               filename = None,
               video_type: str = 'none',
               duration: float = 2.0,
               loop: bool = False,
               interpolate: int = 0,
               scale: float = 1.0,
               pad: int = 1,
               change: float = 0.3,
               sync: bool = False,
               stream: bool = False,
               frames_hint: int = 0,
               metadata: dict | None = None,
               mp4_codec: str | None = None,
               mp4_ext: str | None = None,
               mp4_pix: str | None = None,
               mp4_opt: str | None = None,
               *,
               source: str | None = None,
               fps: float | None = None,
              ):
    """Save frames as a video named by the output pattern.

    mp4_codec: None takes the codec half of video_type (MP4/AVC1), else libx264. mp4_opt: None takes the codec's preset, '' means no options.
    mp4_pix: None takes the codec's table value. mp4_ext: None takes the video_type extension when the codec fits that container, else the codec's own.
    fps: output rate; None derives it from the frame count and duration.
    sync: returns the filename, or None when the save is refused or fails.
    stream: returns (filename, writer) for more frames, or raises VideoWriteError with the reason.
    otherwise: encodes in a thread and returns the filename at once; the file is registered before the thread runs and a failure is only logged.
    source: the input being processed; a save that would overwrite it is refused.
    """
    if (video_type is None) or (video_type.lower() == 'none') or (images is None) or (len(images) < 1):
        return None
    if interpolate > 0 and getattr(p, 'video_interpolated', False):
        interpolate = 0
    if video_type.lower() not in ['gif', 'png']:
        mp4_codec, mp4_opt = resolve_codec(video_type, mp4_codec, mp4_opt)
    image = images[0]
    if p is not None:
        seed = p.all_seeds[0] if getattr(p, 'all_seeds', None) is not None else p.seed
        prompt = p.all_prompts[0] if getattr(p, 'all_prompts', None) is not None else p.prompt
        namegen = FilenameGenerator(p, seed=seed, prompt=prompt, image=image)
    else:
        namegen = FilenameGenerator(None, seed=0, prompt='', image=image)
    base_path = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_video)
    if filename is None:
        filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
        filename = os.path.join(base_path, filename)
    else:
        if os.path.sep not in filename:
            filename = os.path.join(base_path, filename)
    if mp4_ext is not None and video_type.lower() not in ['gif', 'png']:
        ext = mp4_ext.lower().lstrip('.')
        if not filename.lower().endswith(f'.{ext}'):
            root, current_ext = os.path.splitext(filename)
            filename = f'{root if current_ext else filename}.{ext}'
    else:
        ext = video_type.lower().split('/')[0] if '/' in video_type else video_type.lower()
        if video_type.lower() not in ['gif', 'png']:
            from modules.video_models.video_codecs import get_codec_allowed_exts, get_codec_ext
            allowed = get_codec_allowed_exts(mp4_codec)
            if allowed and ext not in allowed:
                log.warning(f'Video: codec={mp4_codec} does not fit container={ext}, using {get_codec_ext(mp4_codec)}')
                ext = get_codec_ext(mp4_codec)
        if not filename.lower().endswith(ext):
            filename += f'.{ext}'
    filename = namegen.sanitize(filename, base=base_path) # before numbering, so the scan sees the folder that is used
    filename = namegen.sequence(filename)
    if source is not None and os.path.realpath(filename) == os.path.realpath(source):
        reason = f'output would overwrite the input "{source}"'
        log.error(f'Save video: file="{filename}" {reason}')
        if stream:
            raise VideoWriteError(reason)
        return None
    from modules.video_models.video_save import create_video_metadata
    metadata = create_video_metadata(p, metadata, filename)
    if not sync:
        shared.state.outputs(filename)
        threading.Thread(target=save_video_thread, args=(images, filename, video_type, duration, loop, interpolate, scale, pad, change), kwargs={'metadata': metadata, 'mp4_codec': mp4_codec, 'mp4_pix': mp4_pix, 'mp4_opt': mp4_opt, 'fps': fps}).start()
        return filename
    try:
        video_writer = save_video_atomic(images, filename, video_type, duration, loop, interpolate, scale, pad, change, stream, frames_hint, metadata, mp4_codec, mp4_pix, mp4_opt, fps=fps)
    except VideoWriteError:
        if stream:
            raise
        return None
    shared.state.outputs(filename)
    if stream:
        return filename, video_writer
    return filename


def read_video(filepath: str):
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        msg = f'Video open failed: path="{filepath}"'
        log.error(msg)
        raise RuntimeError(msg)
    frames = []
    while True:
        ok, frame = video.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    video.release()
    return frames


def read_generator(filepath: str):
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        msg = f'Video open failed: path="{filepath}"'
        log.error(msg)
        raise RuntimeError(msg)
    frames = []
    while True:
        ok, frame = video.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
    video.release()
    return frames


def get_video_params(filepath: str, capture: bool = False):
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
        ok, raw = video.read()
        if ok and raw is not None:
            frame = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    video.release()
    return frames, fps, duration, w, h, codec, frame


def get_video_info(filepath: str):
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
    if av is None:
        return {}
    try:
        with av.open(video, mode="r") as container:
            return dict(container.metadata)
    except Exception as e:
        log.warning(f'Video metadata: file="{video}" {e}')
        return {}
