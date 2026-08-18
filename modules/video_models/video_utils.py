import os
import sys
import time
import inspect
import importlib.util
from dataclasses import dataclass
from PIL import Image
from installer import install
from modules import shared, sd_models, timer, errors, devices
from modules.logger import log
from modules.video_models.video_codecs import codecs_config


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
MEDIA_EXTENSIONS = {
    'image': ('.png', '.jpg', '.jpeg', '.webp'),
    'video': ('.mp4', '.mov', '.avi'),
    'audio': ('.wav', '.mp3', '.flac', '.aac'),
}


@dataclass
class MediaProbe:
    """What a container header reports, without decoding any of it."""
    kind: str
    fps: float | None = None
    frames: int | None = None
    duration: float | None = None # seconds
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    sample_rate: int | None = None


def queue_err(msg):
    log.error(f'Video: {msg}')
    return [], None, '', '', f'Error: {msg}'


def get_url(url):
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="video-model-link">{url}</a><br><br>' if url else '<br><br>'


def supports_last_frame(model):
    # last-frame (FLF2V) conditioning needs a pipeline whose __call__ accepts `last_image`.
    # wan 2.2 5b accepts the arg but masks timesteps from the first frame only, so it drops the last frame.
    try:
        params = list(inspect.signature(type(model).__call__, follow_wrapped=True).parameters)
    except (ValueError, TypeError):
        return False
    if 'last_image' not in params:
        return False
    return not getattr(getattr(model, 'config', None), 'expand_timesteps', False)


def check_av():
    """The av module, or None when it is unavailable; callers guard on the None."""
    install('av')
    try:
        import av
        av.logging.set_level(av.logging.ERROR) # pylint: disable=c-extension-no-member
    except Exception as e:
        log.error(f'av package: {e}')
        return None
    return av


def has_torchaudio():
    # never installed on demand: torchaudio wheels pin a torch build and would replace it under the running server
    try:
        return importlib.util.find_spec('torchaudio') is not None
    except Exception:
        return False


def classify_extension(fn: str):
    """Media kind of a filename, None when the extension is not one sdnext reads."""
    lower = str(fn).lower()
    for kind, extensions in MEDIA_EXTENSIONS.items():
        if lower.endswith(extensions):
            return kind
    return None


def probe_media(fn: str, kind: str):
    """Container metadata for a media file, None when it cannot be opened. Reads headers only, so
    a file too large or too short to use is rejected before anything decodes it."""
    av = check_av()
    if not av:
        return None
    probe = MediaProbe(kind=kind)
    try:
        with av.open(fn) as container:
            if kind == 'video' and container.streams.video: # an audio file with cover art carries a video stream that is not frames
                stream = container.streams.video[0]
                rate = stream.average_rate or stream.guessed_rate # average_rate is a Fraction and can be a falsy 0/1, which is why the decoder falls back the same way
                probe.fps = float(rate) if rate else None
                probe.frames = stream.frames or None # 0 means the container carries no count, not an empty file
                probe.width, probe.height = stream.codec_context.width, stream.codec_context.height
                if stream.duration is not None and stream.time_base is not None:
                    probe.duration = float(stream.duration * stream.time_base) # stream durations are in time_base units
                elif container.duration is not None:
                    probe.duration = container.duration / 1000000 # container durations are in AV_TIME_BASE units
                elif probe.frames and probe.fps:
                    probe.duration = probe.frames / probe.fps
            if container.streams.audio:
                stream = container.streams.audio[0]
                # the soundtrack decoder converts to planar float keeping the container's own rate and layout, so these are the values it yields
                probe.channels = getattr(stream, 'channels', None) or getattr(getattr(stream, 'layout', None), 'nb_channels', None)
                probe.sample_rate = int(stream.codec_context.sample_rate)
    except Exception as e:
        debug(f'Video probe: file="{fn}" {e}')
        return None
    return probe


def hijack_encode_image(*args, **kwargs):
    t0 = time.time()
    try:
        sd_models.move_model(shared.sd_model.image_encoder, devices.device)
        res = shared.sd_model.orig_encode_image(*args, **kwargs)
    except Exception as e:
        log.error(f'Video encode image: {e}')
        errors.display(e, 'Video encode image')
        res = None
    t1 = time.time()
    timer.process.add('te', t1-t0)
    debug(f'Video encode image: te={shared.sd_model.image_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


def get_codecs():
    av = check_av()
    if av is None:
        return []
    codecs = []
    practical_codecs = codecs_config.keys()
    rejected = 0
    for codec in av.codecs_available:
        if codec not in practical_codecs:
            rejected += 1
            continue
        try:
            c = av.Codec(codec, mode='w')
            if c.type == 'video' and c.is_encoder and len(c.video_formats) > 0:
                if not any(c.name == ca.name for ca in codecs):
                    codecs.append(c)
        except Exception:
            pass
    hw_codecs = [c for c in codecs if (c.capabilities & 0x40000 > 0) or (c.capabilities & 0x80000 > 0)]
    sw_codecs = [c for c in codecs if c not in hw_codecs]
    log.debug(f'Video codecs enum: hardware={len(hw_codecs)} software={len(sw_codecs)} rejected={rejected}')
    """
    for c in hw_codecs:
        log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=True')
    for c in sw_codecs:
        log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=False')
    """
    return ['none'] + [c.name for c in hw_codecs + sw_codecs]


def decode_fourcc(cc):
    cc_bytes = int(cc).to_bytes(4, byteorder=sys.byteorder) # convert code to a bytearray
    cc_str = cc_bytes.decode() # decode byteaarray to a string
    return cc_str


def get_video_frames(fn: str, num_frames: int = -1, skip_frames: int = 0):
    import cv2
    frames = []
    try:
        video = cv2.VideoCapture(fn)
        if not video.isOpened():
            return frames
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = decode_fourcc(video.get(cv2.CAP_PROP_FOURCC))
        skip = 0
        while True:
            status, frame = video.read()
            if skip_frames > 0:
                if skip < skip_frames:
                    skip += 1
                    _status, _frame = video.read()
                    continue
                skip = 0
            if status:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            else:
                break
            if len(frames) >= num_frames > 0:
                break
        video.release()
        log.debug(f'Video open: file="{fn}" frames={len(frames)} total={frame_count} skip={skip} fps={fps} size={w}x{h} codec={codec}')
    except Exception as e:
        log.error(f'Video open: file="{fn}" {e}')
        return frames
    return frames
