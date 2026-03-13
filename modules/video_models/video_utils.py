import os
import sys
import time
from PIL import Image
from installer import install
from modules import shared, sd_models, timer, errors, devices
from modules.logger import log


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def queue_err(msg):
    log.error(f'Video: {msg}')
    return [], None, '', '', f'Error: {msg}'


def get_url(url):
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="video-model-link">{url}</a><br><br>' if url else '<br><br>'


def check_av():
    install('av')
    try:
        import av
        av.logging.set_level(av.logging.ERROR) # pylint: disable=c-extension-no-member
    except Exception as e:
        log.error(f'av package: {e}')
        return False
    return av


def set_prompt(p):
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    shared.prompt_styles.apply_styles_to_extra(p)
    p.styles = []
    p.task_args['prompt'] = p.prompt
    p.task_args['negative_prompt'] = p.negative_prompt


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
    for codec in av.codecs_available:
        try:
            c = av.Codec(codec, mode='w')
            if c.type == 'video' and c.is_encoder and len(c.video_formats) > 0:
                if not any(c.name == ca.name for ca in codecs):
                    codecs.append(c)
        except Exception:
            pass
    hw_codecs = [c for c in codecs if (c.capabilities & 0x40000 > 0) or (c.capabilities & 0x80000 > 0)]
    sw_codecs = [c for c in codecs if c not in hw_codecs]
    log.debug(f'Video codecs: hardware={len(hw_codecs)} software={len(sw_codecs)}')
    # for c in hw_codecs:
    #     log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=True')
    # for c in sw_codecs:
    #     log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=False')
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
