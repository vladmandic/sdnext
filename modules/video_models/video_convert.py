"""Browser previews of saved videos, probed and converted in-process with av; no ffmpeg binary is needed."""
import os
import time
import uuid
from modules.logger import log


def is_playable(av, path: str) -> bool:
    """True when browsers play the file as written: 8-bit 4:2:0 H.264 in mp4, VP8, VP9 or AV1 in webm, AV1 in mp4, Theora in ogg.

    A file av cannot read is reported playable, so the browser shows it as is instead of a conversion failing on every request."""
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    try:
        with av.open(path) as container:
            if not container.streams.video:
                return True
            context = container.streams.video[0].codec_context
            codec, pix_fmt = context.codec.canonical_name, context.pix_fmt
    except Exception as e:
        log.info(f'Video preview: file="{path}" not readable, shown as is: {e}')
        return True
    if codec == 'h264':
        return ext == 'mp4' and pix_fmt in ('yuv420p', 'yuvj420p') # High 4:4:4, 10-bit and RGB H.264 are refused or shown with wrong colours
    if codec in ('vp8', 'vp9'):
        return ext == 'webm'
    if codec == 'av1':
        return ext in ('mp4', 'webm')
    if codec == 'theora':
        return ext == 'ogg'
    return False


def convert_for_preview(av, path: str, temp_dir: str) -> str:
    """Write an H.264 4:2:0 BT.709 copy of a video into temp_dir and return its path; the source is only read.

    AAC audio is copied, other audio is dropped. On failure the partial copy is removed and the source path returned."""
    from modules.video_models.video_codecs import stream_colour, chroma_crop_size
    from modules.video_models.video_save import video_rate
    target = os.path.join(temp_dir, f'{uuid.uuid4().hex}.mp4')
    t0 = time.time()
    frames = 0
    try:
        with av.open(path) as source, av.open(target, mode='w') as output:
            video_in = source.streams.video[0]
            context = video_in.codec_context
            fps = video_in.average_rate or video_in.guessed_rate or 24
            width, height = chroma_crop_size('yuv420p', context.width, context.height)
            video_out = output.add_stream('libx264', rate=video_rate(fps), options={'preset': 'veryfast', 'crf': '23'})
            video_out.width, video_out.height, video_out.pix_fmt = width, height, 'yuv420p'
            video_out.codec_context.time_base = video_in.time_base # encode() rebases each decoded frame from its source time base into this one
            tags, reformat_args = stream_colour('yuv420p')
            for key, value in tags.items():
                setattr(video_out.codec_context, key, value)
            audio_in = source.streams.audio[0] if source.streams.audio else None
            audio_out = None
            if audio_in is not None:
                if audio_in.codec_context.name == 'aac':
                    audio_out = output.add_stream_from_template(audio_in)
                else:
                    log.info(f'Video preview: file="{path}" audio={audio_in.codec_context.name} dropped, only aac is copied')
            streams = [video_in] + ([audio_in] if audio_out is not None else [])
            for packet in source.demux(*streams):
                if packet.stream.type == 'audio':
                    if packet.dts is not None:
                        packet.stream = audio_out
                        output.mux(packet)
                    continue
                for frame in packet.decode():
                    frame = frame.reformat(width=width, height=height, format='yuv420p', **reformat_args)
                    frames += 1
                    for out_packet in video_out.encode(frame):
                        output.mux(out_packet)
            for out_packet in video_out.encode():
                output.mux(out_packet)
    except Exception as e:
        log.error(f'Video preview: file="{path}" conversion failed, shown as is: {e}')
        if os.path.exists(target):
            os.remove(target)
        return path
    log.info(f'Video preview: file="{path}" codec={context.codec.canonical_name} pix_fmt={context.pix_fmt} frames={frames} converted="{target}" time={time.time() - t0:.2f}')
    return target
