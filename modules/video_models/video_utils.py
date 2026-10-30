import os
import time
import inspect
import importlib.util
from contextlib import contextmanager
from modules import shared, sd_models, timer, errors, devices
from modules.logger import log
from modules.video_models.video_codecs import codecs_config


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
MEDIA_EXTENSIONS = {
    'image': ('.png', '.jpg', '.jpeg', '.webp'),
    'video': ('.mp4', '.mov', '.avi'),
    'audio': ('.wav', '.mp3', '.flac', '.aac'),
}


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


HARDWARE_CAPABILITIES = 0x40000 | 0x80000 # AV_CODEC_CAP_HARDWARE, AV_CODEC_CAP_HYBRID


def check_av():
    """The av module, or None when it is unavailable; callers guard on the None."""
    try:
        import av
        av.logging.set_level(av.logging.ERROR) # pylint: disable=c-extension-no-member
    except Exception as e:
        log.error(f'av package: {e}')
        return None
    return av


def is_hardware_codec(codec) -> bool:
    """Encoder backed by a device or driver: present in any av build, opens only with that hardware."""
    return bool(codec.capabilities & HARDWARE_CAPABILITIES)


def open_failure_reason(av, codec_name: str, pix_fmt: str, e: Exception) -> str:
    """Why an encoder did not open, in terms the user can act on; libav's own last log line when nothing more specific applies."""
    try:
        codec = av.Codec(codec_name, mode='w')
    except Exception:
        return f'encoder {codec_name} is not in this av build'
    formats = [f.name for f in (codec.video_formats or [])]
    if formats and pix_fmt not in formats:
        return f'pixel format {pix_fmt} not supported, {codec_name} accepts {", ".join(formats)}'
    entry = getattr(e, 'log', None)
    reason = entry[2].strip() if entry and len(entry) > 2 and entry[2] else str(e)
    if is_hardware_codec(codec):
        reason += '; hardware encoder, device or driver unavailable'
    return reason


def has_torchaudio():
    # never installed on demand: torchaudio wheels pin a torch build and would replace it under the running server
    try:
        return importlib.util.find_spec('torchaudio') is not None
    except Exception:
        return False


@contextmanager
def phase(title: str):
    """Scoped generation phase, so an abort cannot leave the job holding a stage that never ended."""
    jobid = shared.state.begin(title)
    try:
        yield jobid
    finally:
        shared.state.end(jobid)


def pixel_size(pixels, fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    """Width and height of decoded frames, whether they arrive as PIL images or as a tensor.

    The runners request a resolution and the model answers with another one often enough that
    the request is not a usable substitute, so read it off the pixels and keep the fallback for
    the case where nothing was decoded at all.
    """
    if isinstance(pixels, list):
        return pixels[0].size if len(pixels) > 0 and hasattr(pixels[0], 'size') else fallback
    ndim = getattr(pixels, 'ndim', None)
    if ndim == 5: # NCTHW
        return int(pixels.shape[-1]), int(pixels.shape[-2])
    if ndim == 4: # NHWC
        return int(pixels.shape[2]), int(pixels.shape[1])
    shape = getattr(pixels, 'shape', None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    return fallback


def classify_extension(fn: str):
    """Media kind of a filename, None when the extension is not one sdnext reads."""
    lower = str(fn).lower()
    for kind, extensions in MEDIA_EXTENSIONS.items():
        if lower.endswith(extensions):
            return kind
    return None


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
    hw_codecs = [c for c in codecs if is_hardware_codec(c)]
    sw_codecs = [c for c in codecs if c not in hw_codecs]
    log.debug(f'Video codecs enum: hardware={len(hw_codecs)} software={len(sw_codecs)} rejected={rejected}')
    """
    for c in hw_codecs:
        log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=True')
    for c in sw_codecs:
        log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=False')
    """
    return ['none'] + [c.name for c in hw_codecs + sw_codecs]
