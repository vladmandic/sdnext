"""Video reading through PyAV: header probes, frame counts and decoded RGB frames in display orientation."""
import os
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np
from modules.logger import log


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


@dataclass
class MediaProbe:
    """What a container header reports, without decoding any of it; width and height are the coded size."""
    kind: str
    fps: float | None = None
    frames: int | None = None
    frames_source: str | None = None # 'header' when the container counts the frames, 'duration' when estimated from the video stream's own duration
    duration: float | None = None # seconds
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    pix_fmt: str | None = None
    bit_rate: int | None = None
    sar: str | None = None
    rotation: int | None = None # display rotation in degrees: from the header on av 19 and later, from the first decoded frame in probe_video
    channels: int | None = None
    sample_rate: int | None = None
    audio_streams: int = 0


def header_rotation(stream) -> int | None:
    """Display rotation in degrees read from the stream header, as av_display_rotation_get computes it; None before av 19, which does not expose the header's side data."""
    side_data = getattr(stream.codec_context, 'coded_side_data', None)
    if side_data is None:
        return None
    matrix = side_data.get('display_matrix')
    if not matrix or len(matrix) < 36:
        return 0
    m = struct.unpack('=9i', bytes(matrix[:36])) # 3x3 of 16.16 fixed point, row major
    scale_x, scale_y = math.hypot(m[0], m[3]), math.hypot(m[1], m[4])
    if scale_x == 0 or scale_y == 0:
        return 0
    return round(-math.degrees(math.atan2(m[1] / scale_y, m[0] / scale_x)))


def display_size(probe: MediaProbe) -> tuple[int, int]:
    """Width and height as players show the clip: the coded size, swapped for a quarter turn."""
    width, height = probe.width or 0, probe.height or 0
    return (height, width) if (probe.rotation or 0) % 180 == 90 else (width, height)


def probe_container(container, kind: str) -> MediaProbe:
    probe = MediaProbe(kind=kind)
    if kind == 'video' and container.streams.video: # an audio file with cover art carries a video stream that is not frames
        stream = container.streams.best('video')
        rate = stream.average_rate or stream.guessed_rate # average_rate is a Fraction and can be a falsy 0/1, which is why the decoder falls back the same way
        probe.fps = float(rate) if rate else None
        context = stream.codec_context
        probe.width, probe.height = context.width, context.height
        probe.codec = context.codec.canonical_name # context.name is the decoder, libdav1d for av1
        probe.pix_fmt = context.pix_fmt
        probe.rotation = header_rotation(stream)
        probe.bit_rate = stream.bit_rate or container.bit_rate or None
        sar = stream.sample_aspect_ratio
        if sar is not None and sar.numerator > 0 and sar.numerator != sar.denominator:
            probe.sar = f'{sar.numerator}/{sar.denominator}'
        stream_duration = float(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base is not None else None # stream durations are in time_base units
        if stream.frames: # 0 means the container carries no count, not an empty file
            probe.frames, probe.frames_source = stream.frames, 'header'
        elif stream_duration and probe.fps: # the container duration is not used: a longer soundtrack inflates it
            probe.frames, probe.frames_source = round(stream_duration * probe.fps), 'duration'
        if stream_duration is not None:
            probe.duration = stream_duration
        elif container.duration is not None:
            probe.duration = container.duration / 1000000 # container durations are in AV_TIME_BASE units
        elif probe.frames and probe.fps:
            probe.duration = probe.frames / probe.fps
    probe.audio_streams = len(container.streams.audio)
    if container.streams.audio:
        stream = container.streams.audio[0]
        # the soundtrack decoder converts to planar float keeping the container's own rate and layout, so these are the values it yields
        probe.channels = getattr(stream, 'channels', None) or getattr(getattr(stream, 'layout', None), 'nb_channels', None)
        probe.sample_rate = int(stream.codec_context.sample_rate)
    return probe


def probe_media(fn: str, kind: str = 'video') -> MediaProbe | None:
    """Container metadata for a media file, None when it cannot be opened. Reads headers only, so
    a file too large or too short to use is rejected before anything decodes it."""
    from modules.video_models.video_utils import check_av
    av = check_av()
    if not av:
        return None
    try:
        with av.open(fn) as container:
            return probe_container(container, kind)
    except Exception as e:
        debug(f'Video probe: file="{fn}" {e}')
        return None


def count_frames(fn: str) -> int:
    """Frames of the video stream counted from its packets, nothing decoded; for containers whose header has no count and no duration."""
    from modules.video_models.video_utils import check_av
    av = check_av()
    count = 0
    if av is None:
        return count
    try:
        with av.open(fn) as container:
            stream = container.streams.best('video')
            if stream is None:
                return count
            for packet in container.demux(stream):
                if packet.size:
                    count += 1
    except Exception as e:
        debug(f'Video count: file="{fn}" packets={count} {e}')
    return count


def open_video(fn: str):
    """av, the open container and its video stream; logs and raises RuntimeError when the file cannot be opened or has no video stream."""
    from modules.video_models.video_utils import check_av
    av = check_av()
    container = None
    try:
        if av is None:
            raise RuntimeError('av package not available')
        container = av.open(fn)
        stream = container.streams.best('video')
        if stream is None:
            raise RuntimeError('no video stream')
    except Exception as e:
        if container is not None:
            container.close()
        msg = f'Video open failed: path="{fn}"'
        log.error(f'{msg} {e}')
        raise RuntimeError(msg) from e
    return av, container, stream


def probe_header(fn: str) -> MediaProbe:
    """probe_media for a file that must hold a video: logs and raises RuntimeError like the frame readers."""
    _av, container, _stream = open_video(fn)
    try:
        return probe_container(container, 'video')
    finally:
        container.close()


def frame_rgb(av, frame) -> np.ndarray:
    """A decoded frame as contiguous HxWx3 uint8 RGB, rotated to its display orientation; the matrix and range come from the frame's own tags."""
    try:
        rgb = frame.to_ndarray(format='rgb24')
    except OSError:
        # av 17 refuses frames tagged with a reserved transfer or primaries (PyAV issue 2208, fixed in 18); clear the two tags as av 18 does
        # remove when av 17 is no longer installed, which is when Python 3.10 support ends
        if not av.__version__.startswith('17.'):
            raise
        frame.color_trc = 2 # unspecified
        frame.color_primaries = 2 # unspecified
        rgb = frame.to_ndarray(format='rgb24')
    turns = (frame.rotation // 90) % 4 # display matrix rotation, counter-clockwise, as OpenCV and the ffmpeg CLI apply it
    if turns:
        rgb = np.rot90(rgb, k=turns)
    return np.ascontiguousarray(rgb)


def probe_video(fn: str) -> tuple[MediaProbe, np.ndarray | None]:
    """Header probe and first decoded frame of a video from one open, None for a frame that does not decode.

    Raises RuntimeError when the file cannot be opened or has no video stream."""
    av, container, stream = open_video(fn)
    try:
        probe = probe_container(container, 'video')
        first = None
        try:
            frame = next(container.decode(stream), None)
            if frame is not None:
                probe.rotation = int(frame.rotation)
                first = frame_rgb(av, frame)
        except av.error.FFmpegError as e:
            log.warning(f'Video decode: file="{fn}" frame=0 {e}')
        return probe, first
    finally:
        container.close()


def iter_frames(fn: str, num_frames: int = -1, skip_frames: int = 0) -> Iterator[np.ndarray]:
    """Decoded frames as contiguous HxWx3 uint8 RGB in display orientation.

    Keeps frame 0 and every (skip_frames + 1)th frame after it, and stops after num_frames kept frames when that is positive.
    Raises RuntimeError when the file cannot be opened or has no video stream; a decode error later ends the frames with a warning.
    The file stays open until the generator is exhausted or closed, and Windows refuses to rename or remove an open file."""
    av, container, stream = open_video(fn)
    stream.thread_type = 'AUTO' # libav defaults to slice threads, one slice per frame for x264 output
    try:
        decoded = container.decode(stream)
        index = 0
        kept = 0
        while True:
            try:
                frame = next(decoded)
            except StopIteration:
                break
            except av.error.FFmpegError as e:
                log.warning(f'Video decode: file="{fn}" frame={index} {e}')
                break
            if index % (skip_frames + 1) == 0:
                yield frame_rgb(av, frame)
                kept += 1
                if 0 < num_frames <= kept:
                    break
            index += 1
    finally:
        container.close()


def read_frames(fn: str, num_frames: int = -1, skip_frames: int = 0, as_pil: bool = False) -> list:
    """Every frame iter_frames keeps, as arrays or as PIL images."""
    frames = iter_frames(fn, num_frames=num_frames, skip_frames=skip_frames)
    if as_pil:
        from PIL import Image
        return [Image.fromarray(frame) for frame in frames]
    return list(frames)
