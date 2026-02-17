import os
from modules import shared


def save_video_frame(filepath: str):
    from modules import video
    try:
        frames, fps, duration, w, h, codec, frame = video.get_video_params(filepath, capture=True)
    except Exception as e:
        shared.log.error(f'Video: file={filepath} {e}')
        return None
    if frame is not None:
        basename = os.path.splitext(filepath)
        thumb = f'{basename[0]}.thumb.jpg'
        shared.log.debug(f'Video: file={filepath} frames={frames} fps={fps} size={w}x{h} codec={codec} duration={duration} thumb={thumb}')
        frame.save(thumb)
    else:
        shared.log.error(f'Video: file={filepath} no frames found')
    return frame
