import os
import threading
import numpy as np
from PIL import Image
from modules import shared, errors
from modules.logger import log
from modules.image.namegen import FilenameGenerator # pylint: disable=unused-import


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
        video_writer = cv2.VideoWriter(filename, fourcc=cv2.VideoWriter_fourcc(*fourcc), fps=len(frames)/duration, frameSize=(w, h))
        for i in range(len(frames)):
            img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        size = os.path.getsize(filename)
        log.info(f'Save video: file="{filename}" frames={len(frames)} duration={duration} fourcc={fourcc} size={size}')
    shared.state.end(savejob)


def save_video(p, images, filename = None, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3, sync: bool = False):
    if images is None or len(images) < 2 or video_type is None or video_type.lower() == 'none':
        return None
    image = images[0]
    if p is not None:
        seed = p.all_seeds[0] if getattr(p, 'all_seeds', None) is not None else p.seed
        prompt = p.all_prompts[0] if getattr(p, 'all_prompts', None) is not None else p.prompt
        namegen = FilenameGenerator(p, seed=seed, prompt=prompt, image=image)
    else:
        namegen = FilenameGenerator(None, seed=0, prompt='', image=image)
    if filename is None and p is not None:
        filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
        filename = os.path.join(shared.opts.outdir_video, filename)
        filename = namegen.sequence(filename)
    else:
        if os.path.sep not in filename:
            filename = os.path.join(shared.opts.outdir_video, filename)
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
    duration = round(float(frames) / fps, 2)
    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = decode_fourcc(video.get(cv2.CAP_PROP_FOURCC))
    frame = None
    if capture:
        _status, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
    video.release()
    return frames, fps, duration, w, h, codec, frame
