"""Video encoding through PyAV: the one writer behind every video save, and the option, rate and audio helpers it uses."""
import os
from fractions import Fraction
import numpy as np
from modules import errors
from modules.logger import log


class VideoWriteError(RuntimeError):
    """A video save was refused or failed; the message is the reason shown to the user."""


extra_option_keys = {'tune', 'maxrate', 'minrate', 'bufsize', 'g', 'bf', 'cq', 'lossless'} # rate and quality keys beyond the codec defaults


def parse_options(options):
    if isinstance(options, dict):
        return options
    if not isinstance(options, str) or not options.strip():
        return {}
    parsed_options = {}
    normalized = options.replace(',', ':') # Standardize delimiters by replacing commas with colons
    for item in normalized.split(':'):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            key, value = item.split('=', 1)
            parsed_options[key.strip()] = value.strip()
        else:
            parsed_options[item] = '1' # Handle flag options without explicit '=' (e.g., 'fastseek')
    return parsed_options


def rejected_options(options) -> list[str]:
    # options reach the encoder as-is, and keys such as stats or x264-params write files at any path
    from modules.video_models.video_codecs import codecs_config
    allowed = extra_option_keys | {key for codec in codecs_config.values() for key in parse_options(codec['options'])}
    return sorted(set(parse_options(options)) - allowed)


def video_rate(fps) -> Fraction:
    """Frame rate as a fraction whose time base fits a 16-bit denominator, the mpeg4 limit; rates such as 30000/1001 come through exact."""
    value = fps if isinstance(fps, Fraction) else Fraction(fps)
    if value <= 0:
        raise ValueError(f'fps must be positive, got {fps}')
    return 1 / (1 / value).limit_denominator(65535)


def add_audio_packets(container, audio_stream, audio: dict):
    if not audio or "frames" not in audio:
        return
    try:
        from modules.video_models.video_utils import check_av
        av = check_av()
        sr = audio.get("sr", 44100)
        layout = audio.get("layout", "stereo")
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sr)
        fifo = av.AudioFifo()
        for raw_frame in audio.get("frames", []):
            for resampled in resampler.resample(raw_frame):
                fifo.write(resampled)
        for resampled in resampler.resample(None):
            fifo.write(resampled)
        pts_counter = 0
        frame_size = audio_stream.codec_context.frame_size or 1024
        while fifo.samples >= frame_size:
            frame = fifo.read(frame_size)
            frame.pts = pts_counter
            pts_counter += frame.samples
            for packet in audio_stream.encode(frame):
                packet.stream = audio_stream
                container.mux_one(packet)
        if fifo.samples > 0:
            frame = fifo.read(fifo.samples)
            frame.pts = pts_counter
            pts_counter += frame.samples
            for packet in audio_stream.encode(frame):
                packet.stream = audio_stream
                container.mux_one(packet)
        for packet in audio_stream.encode():
            packet.stream = audio_stream
            container.mux_one(packet)
    except Exception as e:
        log.error(f"Video audio encoding: type=packets {e}")
        errors.display(e, "Audio")


def add_audio_tensor(container, audio_stream, audio, sample_rate: int):
    """Samples as a torch tensor or numpy array, mono or stereo, float in [-1, 1] or int16."""
    from modules.video_models.video_utils import check_av
    av = check_av()
    if hasattr(audio, 'detach'): # torch tensor
        audio = audio.detach().float().cpu().numpy()
    if audio.ndim > 2:
        audio = np.squeeze(audio)
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1] and audio.shape[1] in (1, 2):
        audio = audio.T
    channels = audio.shape[0] if audio.shape[0] in (1, 2) else 1
    layout = "stereo" if channels == 2 else "mono"
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    audio_frame = av.AudioFrame.from_ndarray(audio, format="s16p", layout=layout)
    audio_frame.sample_rate = sample_rate
    add_audio_packets(container, audio_stream, {"sr": sample_rate, "layout": layout, "frames": [audio_frame]})


class VideoWriter:
    """An open encoder: write() takes RGB frames, write_audio() fills the audio stream declared at open, release() flushes and closes once."""

    def __init__(self, av, container, stream, codec: str, pix_fmt: str, options: dict, rate: Fraction, reformat_args: dict, audio_stream=None):
        self.av = av
        self.container = container
        self.stream = stream
        self.audio_stream = audio_stream
        self.codec = codec
        self.pix_fmt = pix_fmt
        self.options = options
        self.rate = rate
        self.width: int = stream.width
        self.height: int = stream.height
        self.reformat_args = reformat_args # colour matrix and range for the RGB to YUV conversion
        self.frame_index = 0
        self.closed = False

    def write(self, image: np.ndarray):
        """One uint8 RGB frame, HxW, HxWx3 or HxWx4; alpha is dropped and a frame larger than the stream is cropped, never rescaled."""
        if self.closed:
            return
        try:
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=2)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            image = np.ascontiguousarray(image[:self.height, :self.width])
            frame = self.av.VideoFrame.from_ndarray(image, format='rgb24')
            frame = frame.reformat(width=self.width, height=self.height, format=self.pix_fmt, **self.reformat_args)
            frame.pts = self.frame_index
            self.frame_index += 1
            for packet in self.stream.encode(frame):
                self.container.mux_one(packet)
        except Exception:
            self.release()
            raise

    def write_audio(self, audio, sample_rate: int):
        """The soundtrack: samples as a tensor or array, or a dict of decoded frames with its rate and layout."""
        if self.audio_stream is None:
            raise VideoWriteError('no audio stream was declared when the writer was opened')
        if isinstance(audio, dict):
            add_audio_packets(self.container, self.audio_stream, audio)
        else:
            add_audio_tensor(self.container, self.audio_stream, audio, sample_rate)

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.stream.codec_context.is_open: # flushing an encoder that never opened crashes libav
                for packet in self.stream.encode():
                    self.container.mux_one(packet)
        finally:
            self.container.close()


def open_video_writer(filename: str, width: int, height: int, fps: float | Fraction, codec: str, pix_fmt: str | None = None, options: str | dict | None = None, metadata: dict | None = None, audio_rate: int | None = None, audio_layout: str = 'stereo') -> VideoWriter:
    """Resolve the pixel format, size, rate and options, then open the container and encoders before any frame, so a refusal leaves no file.

    pix_fmt: None takes the codec's table value. audio_rate: declares an aac stream, filled later with write_audio().
    Raises VideoWriteError with the reason."""
    from modules.video_models.video_utils import check_av, open_failure_reason
    from modules.video_models.video_codecs import resolve_pix_fmt, chroma_crop_size, stream_colour
    av = check_av()
    if av is None:
        raise VideoWriteError('av package not available')
    rejected = rejected_options(options)
    if rejected:
        raise VideoWriteError(f'options {rejected} not allowed')
    parsed = dict(parse_options(options))
    if str(codec).lower() == 'libx265':
        parsed['x265-params'] = 'log-level=error'
    try:
        pix_fmt = resolve_pix_fmt(codec, pix_fmt)
        crop_w, crop_h = chroma_crop_size(pix_fmt, width, height)
        rate = video_rate(fps)
    except ValueError as e:
        raise VideoWriteError(f'codec={codec} pix_fmt={pix_fmt}: {e}') from e
    if (crop_w, crop_h) != (width, height):
        log.warning(f'Video: file="{filename}" size={width}x{height} cropped to {crop_w}x{crop_h} for pix_fmt={pix_fmt}')
    existed = os.path.exists(filename)
    container = None
    try:
        container = av.open(filename, mode='w')
        for key, value in (metadata or {}).items():
            if value is not None:
                container.metadata[str(key)] = str(value)
        stream = container.add_stream(codec, rate=rate, options=parsed)
        stream.width = crop_w
        stream.height = crop_h
        stream.pix_fmt = pix_fmt
        tags, reformat_args = stream_colour(pix_fmt)
        for key, value in tags.items(): # read by the encoder at open: x264 writes them as VUI, the mp4 muxer as a colr box
            setattr(stream.codec_context, key, value)
        audio_stream = None
        if audio_rate is not None:
            audio_stream = container.add_stream('aac', rate=audio_rate)
            audio_stream.layout = audio_layout
            audio_stream.time_base = Fraction(1, audio_rate)
        container.start_encoding()
    except Exception as e:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        if not existed and os.path.exists(filename):
            os.remove(filename)
        raise VideoWriteError(f'codec={codec} pix_fmt={pix_fmt}: {open_failure_reason(av, codec, pix_fmt, e)}') from e
    return VideoWriter(av, container, stream, codec, pix_fmt, parsed, rate, reformat_args, audio_stream)
