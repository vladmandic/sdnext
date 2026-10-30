from dataclasses import dataclass


codecs_config = {
    # --- Modern / Standard Distribution (CPU) ---
    'libx264': {
        'name': 'H.264 / AVC',
        'desc': 'Standard for web and streaming compatibility.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'crf=18:preset=medium',
        'pix_fmt': 'yuv420p',
    },
    'libx264rgb': {
        'name': 'H.264 Lossless RGB',
        'desc': 'Lossless recording for screen capture / pure RGB.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov'],
        'options': 'crf=0:preset=fast',
        'pix_fmt': 'rgb24',
    },
    'libx265': {
        'name': 'HEVC / H.265',
        'desc': 'High efficiency compression for 4K video.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'crf=22:preset=medium',
        'pix_fmt': 'yuv420p',
    },
    'libvpx': {
        'name': 'VP8 Video',
        'desc': 'Legacy open video format for WebM.',
        'ext': 'webm',
        'allowed_exts': ['webm', 'mkv'],
        'options': 'crf=10:b=0',
        'pix_fmt': 'yuv420p',
    },
    'libvpx-vp9': {
        'name': 'VP9 Video',
        'desc': 'Royalty-free web video format.',
        'ext': 'webm',
        'allowed_exts': ['webm', 'mkv'],
        'options': 'crf=23:b=0',
        'pix_fmt': 'yuv420p',
    },
    'libsvtav1': {
        'name': 'AV1 (SVT-AV1)',
        'desc': 'Next-gen open codec with high compression.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'preset=5:crf=26',
        'pix_fmt': 'yuv420p',
    },
    'mpeg4': {
        'name': 'MPEG-4 Part 2',
        'desc': 'Legacy format for old media players.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'avi', 'mkv'],
        'options': 'qmin=3:qmax=3',
        'pix_fmt': 'yuv420p',
    },
    'mjpeg': {
        'name': 'Motion JPEG',
        'desc': 'Sequence of JPEG frames; minimal CPU load.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mov', 'mkv'],
        'options': 'qmin=3:qmax=3',
        'pix_fmt': 'yuvj420p',
    },

    # --- Hardware Accelerated: NVIDIA NVENC ---
    'h264_nvenc': {
        'name': 'H.264 (NVIDIA)',
        'desc': 'Fast NVIDIA GPU encoding.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=constqp:qp=19:preset=p5',
        'pix_fmt': 'yuv420p',
    },
    'hevc_nvenc': {
        'name': 'HEVC / H.265 (NVIDIA)',
        'desc': 'Fast high-efficiency NVIDIA GPU encoding.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=constqp:qp=23:preset=p5',
        'pix_fmt': 'yuv420p',
    },
    'av1_nvenc': {
        'name': 'AV1 (NVIDIA)',
        'desc': 'Fast AV1 encoding for NVIDIA RTX 40+.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'rc=constqp:qp=26:preset=p5',
        'pix_fmt': 'yuv420p',
    },

    # --- Hardware Accelerated: Intel QSV ---
    'h264_qsv': {
        'name': 'H.264 (Intel QuickSync)',
        'desc': 'Hardware encoding via Intel GPU.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'global_quality=20:preset=medium',
        'pix_fmt': 'nv12',
    },
    'hevc_qsv': {
        'name': 'HEVC / H.265 (Intel QuickSync)',
        'desc': 'Hardware HEVC encoding via Intel GPU.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'global_quality=23:preset=medium',
        'pix_fmt': 'nv12',
    },
    'av1_qsv': {
        'name': 'AV1 (Intel QuickSync)',
        'desc': 'Hardware AV1 encoding via Intel Arc/iGPU.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'global_quality=26:preset=medium',
        'pix_fmt': 'nv12',
    },
    'vp9_qsv': {
        'name': 'VP9 (Intel QuickSync)',
        'desc': 'Hardware VP9 encoding via Intel GPU.',
        'ext': 'webm',
        'allowed_exts': ['webm', 'mkv'],
        'options': 'global_quality=24',
        'pix_fmt': 'nv12',
    },

    # --- Hardware Accelerated: AMD AMF (Windows) ---
    'h264_amf': {
        'name': 'H.264 (AMD AMF)',
        'desc': 'Hardware encoding for AMD GPUs on Windows.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=cqp:qp_i=18:qp_p=18:quality=balanced',
        'pix_fmt': 'yuv420p',
    },
    'hevc_amf': {
        'name': 'HEVC / H.265 (AMD AMF)',
        'desc': 'Hardware HEVC encoding for AMD GPUs on Windows.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=cqp:qp_i=22:qp_p=22:quality=balanced',
        'pix_fmt': 'yuv420p',
    },
    'av1_amf': {
        'name': 'AV1 (AMD AMF)',
        'desc': 'Hardware AV1 encoding for AMD RX 7000+.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'rc=cqp:qp_i=26:qp_p=26:quality=balanced',
        'pix_fmt': 'yuv420p',
    },

    # --- Professional / Editing / Archival ---
    'prores_ks': {
        'name': 'Apple ProRes (iCodec)',
        'desc': 'Universal ProRes encoder for editing suites.',
        'ext': 'mov',
        'allowed_exts': ['mov', 'mkv'],
        'options': 'profile=3',  # 3 = ProRes 422 HQ
        'pix_fmt': 'yuv422p10le',
    },
    'ffv1': {
        'name': 'Lossless Archival (FFV1)',
        'desc': 'Lossless codec for long-term storage.',
        'ext': 'mkv',
        'allowed_exts': ['mkv', 'avi'],
        'options': 'level=3:coder=1:context=1:slices=16',
        'pix_fmt': 'yuv420p',
    },
    'huffyuv': {
        'name': 'HuffYUV Lossless',
        'desc': 'Fast, simple intra-frame lossless codec.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mkv'],
        'options': '',
        'pix_fmt': 'rgb24',
    },
    'utvideo': {
        'name': 'Ut Video Lossless',
        'desc': 'Efficient lossless format for editing.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mkv', 'mov'],
        'options': 'pred=left',
        'pix_fmt': 'yuv420p',
    },
    'magicyuv': {
        'name': 'MagicYUV Lossless',
        'desc': 'Real-time lossless recording codec.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mkv'],
        'options': '',
        'pix_fmt': 'yuv420p',
    },
}


def get_codec_list():
    return ['none'] + list(codecs_config.keys())

def get_codec_name(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return ''
    return codec_info['name']

def get_codec_ext(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return ''
    return codec_info['ext']

def get_codec_options(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return ''
    return codec_info['options']

def get_codec_allowed_exts(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return []
    return codec_info['allowed_exts']

def get_codec_pix_fmt(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return ''
    return codec_info['pix_fmt']

def resolve_pix_fmt(codec_name: str, pix_fmt: str | None) -> str:
    """Pixel format for an encode: the codec's table value when unset, yuv420p for a codec outside the table."""
    default = get_codec_pix_fmt(codec_name)
    if pix_fmt is None:
        if default:
            return default
        from modules.logger import log
        log.warning(f'Video: codec={codec_name} is not in the codec table, using pix_fmt=yuv420p')
        return 'yuv420p'
    if not pix_fmt.strip():
        raise ValueError(f'pixel format is empty, {codec_name} uses {default or "yuv420p"}')
    return pix_fmt.strip()

def chroma_crop_size(pix_fmt: str, width: int, height: int) -> tuple[int, int]:
    """Largest size within width x height that the pixel format's chroma subsampling can represent."""
    import av
    fmt = av.VideoFormat(pix_fmt)
    factor_w = 16 // max(1, fmt.chroma_width(16))
    factor_h = 16 // max(1, fmt.chroma_height(16))
    return width - width % max(1, factor_w), height - height % max(1, factor_h)

def stream_colour(pix_fmt: str) -> tuple[dict, dict]:
    """Colour tags for the codec context and reformat() arguments for converting RGB frames into pix_fmt.

    YUV is BT.709 limited range, JPEG-range yuvj is BT.601 full range, RGB needs no matrix. All four tags are set
    because browsers differ in which ones they read."""
    import av
    from av.video.reformatter import Colorspace, ColorRange # pylint: disable=no-name-in-module
    if av.VideoFormat(pix_fmt).is_rgb:
        return {'colorspace': 0, 'color_range': 2, 'color_primaries': 1, 'color_trc': 1}, {}
    if pix_fmt.startswith('yuvj'):
        return {'colorspace': 6, 'color_range': 2, 'color_primaries': 1, 'color_trc': 1}, {'dst_colorspace': Colorspace.ITU601, 'dst_color_range': ColorRange.JPEG}
    return {'colorspace': 1, 'color_range': 1, 'color_primaries': 1, 'color_trc': 1}, {'dst_colorspace': Colorspace.ITU709, 'dst_color_range': ColorRange.MPEG}

def get_codec_dict(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return {}
    return codec_info


DEFAULT_VIDEO_CODEC = 'libx264'
VIDEO_TYPE_CODECS = {'mp4v': 'mpeg4', 'divx': 'mpeg4', 'avc1': 'libx264', 'h264': 'libx264', 'jvt3': 'libx264', 'mjpg': 'mjpeg'} # codec half of the legacy EXT/FOURCC labels


@dataclass
class VideoTarget:
    kind: str # 'none', 'gif', 'png' or 'video'
    codec: str | None = None
    ext: str | None = None


def resolve_video_type(video_type: str | None, mp4_codec: str | None = None, mp4_ext: str | None = None) -> VideoTarget:
    """What a "Video format" value saves: nothing, an animated gif or png, or a video with its encoder and container.

    video_type: a codecs_config key (any case), a bare container such as MP4, or a legacy EXT/FOURCC label such as MP4/AVC1.
    mp4_codec wins over the codec video_type names; mp4_ext is used as given. Otherwise a container the codec cannot be written to
    is replaced by the codec's own, with a warning."""
    from modules.logger import log
    label = str(video_type or 'none').strip()
    lower = label.lower()
    if lower in ('', 'none'):
        return VideoTarget('none')
    if lower in ('gif', 'png'):
        return VideoTarget(lower, ext=lower)
    keys = {key.lower(): key for key in codecs_config}
    if lower in keys:
        named, ext = keys[lower], get_codec_ext(keys[lower])
    elif '/' in label:
        container, half = label.split('/', 1)
        ext = container.strip().lower()
        named = VIDEO_TYPE_CODECS.get(half.strip().lower()) if mp4_codec is None else None
        if named is None and mp4_codec is None:
            log.warning(f'Video: format="{label}" codec={half.strip().lower()} has no encoder, using {DEFAULT_VIDEO_CODEC}')
            named = DEFAULT_VIDEO_CODEC
    else:
        named, ext = DEFAULT_VIDEO_CODEC, lower
    codec = mp4_codec if mp4_codec is not None else named
    if mp4_ext is not None:
        return VideoTarget('video', codec, mp4_ext.lower().lstrip('.'))
    allowed = get_codec_allowed_exts(codec)
    if allowed and ext not in allowed:
        log.warning(f'Video: codec={codec} does not fit container={ext}, using {get_codec_ext(codec)}')
        ext = get_codec_ext(codec)
    return VideoTarget('video', codec, ext)


def codec_from_video_type(video_type: str | None) -> str:
    """Encoder a "Video format" value names; the default for none, gif and png."""
    return resolve_video_type(video_type).codec or DEFAULT_VIDEO_CODEC
