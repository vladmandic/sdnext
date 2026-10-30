import os
import threading
import numpy as np
from PIL import Image
from modules import shared, errors
from modules.logger import log
from modules.image.namegen import FilenameGenerator # pylint: disable=unused-import
from modules.paths import resolve_output_path
from modules.video_models.video_codecs import codec_from_video_type, get_codec_options, resolve_video_type
from modules.video_models.video_writer import VideoWriteError, VideoWriter, open_video_writer, rejected_options


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


def resolve_codec(video_type: str, mp4_codec: str | None, mp4_opt: str | None) -> tuple[str, str]:
    """Encoder and options for a save: an explicit codec wins over the label; unset options take the codec's preset, '' means none."""
    codec = mp4_codec if mp4_codec is not None else codec_from_video_type(video_type)
    if mp4_opt is None:
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

    video_type: none, GIF, PNG, a codecs_config key, a bare container (MP4) or a legacy EXT/FOURCC label (MP4/AVC1); see resolve_video_type.
    mp4_codec: None takes the codec video_type names, else libx264. mp4_opt: None takes the codec's preset, '' means no options.
    mp4_pix: None takes the codec's table value. mp4_ext: None takes the video_type container when the codec fits it, else the codec's own.
    fps: output rate; None derives it from the frame count and duration.
    sync: returns the filename, or None when the save is refused or fails.
    stream: returns (filename, writer) for more frames, or raises VideoWriteError with the reason.
    otherwise: encodes in a thread and returns the filename at once; the file is registered before the thread runs and a failure is only logged.
    source: the input being processed; a save that would overwrite it is refused.
    """
    target = resolve_video_type(video_type, mp4_codec, mp4_ext)
    if target.kind == 'none' or (images is None) or (len(images) < 1):
        return None
    if interpolate > 0 and getattr(p, 'video_interpolated', False):
        interpolate = 0
    if target.kind == 'video':
        mp4_codec = target.codec
        if mp4_opt is None:
            mp4_opt = get_codec_options(mp4_codec)
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
    if not filename.lower().endswith(f'.{target.ext}'): # appended, never replaced: a name such as v1.5 keeps its dot
        filename = f'{filename}.{target.ext}'
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


def read_video(filepath: str) -> list[np.ndarray]:
    """Every frame as HxWx3 uint8 RGB in display orientation; raises RuntimeError when the file cannot be opened."""
    from modules.video_models.video_reader import read_frames
    return read_frames(filepath)


def read_generator(filepath: str):
    """read_video one frame at a time; the file stays open until the generator is exhausted or closed."""
    from modules.video_models.video_reader import iter_frames
    return iter_frames(filepath)


def get_video_params(filepath: str, capture: bool = False):
    """(frames, fps, duration, width, height, codec, first frame as a PIL image when capture), numbers never None.

    Width and height are the display size. Without capture only headers are read, unless the installed av cannot report the display rotation
    from them (before av 19). Logs and raises RuntimeError when the file cannot be opened."""
    from modules.video_models.video_reader import probe_header, probe_video, count_frames, display_size
    first = None
    probe = None if capture else probe_header(filepath)
    if probe is None or probe.rotation is None:
        probe, first = probe_video(filepath)
    frames = probe.frames or count_frames(filepath)
    fps = probe.fps or 0.0
    duration = probe.duration or (frames / fps if frames and fps else 0.0)
    if first is not None:
        h, w = first.shape[:2]
    else:
        w, h = display_size(probe)
    frame = Image.fromarray(first) if capture and first is not None else None
    return frames, fps, duration, w, h, probe.codec or '', frame


def get_video_info(filepath: str) -> dict:
    """File, stream and audio properties for display; 'error' when the file cannot be opened as a video."""
    try:
        info = {
            'file': os.path.basename(filepath),
            'size': os.path.getsize(filepath),
            'container': os.path.splitext(filepath)[1].lower().lstrip('.'),
        }
    except Exception as e:
        log.error(f'Video probe failed: path="{filepath}" {e}')
        return {}
    from modules.video_models.video_reader import probe_video, count_frames
    try:
        probe, first = probe_video(filepath)
    except RuntimeError as e:
        info['error'] = str(e)
        return info
    frames = probe.frames or count_frames(filepath)
    if first is not None:
        height, width = first.shape[:2]
    else:
        width, height = probe.width, probe.height
    values = {
        'frames': frames,
        'fps': probe.fps,
        'width': width,
        'height': height,
        'duration': round(probe.duration or (frames / probe.fps if frames and probe.fps else 0), 3),
        'codec': probe.codec,
        'format': probe.pix_fmt,
        'bitrate_kbps': round(probe.bit_rate / 1000) if probe.bit_rate else None,
        'rotation': probe.rotation,
        'sar': probe.sar,
        'audio_streams': probe.audio_streams,
        'audio_channels': probe.channels,
        'audio_sample_rate': probe.sample_rate,
    }
    info.update({key: value for key, value in values.items() if value}) # zero and unknown values are left out
    if probe.audio_streams:
        info['audio'] = f'{probe.audio_streams}x{probe.channels or 0}x{probe.sample_rate or 0}'
    return info


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
