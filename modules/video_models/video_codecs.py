codecs_config = {
    # --- Modern / Standard Distribution (CPU) ---
    'libx264': {
        'name': 'H.264 / AVC',
        'desc': 'Standard for web and streaming compatibility.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'crf=18:preset=p4',
    },
    'libx264rgb': {
        'name': 'H.264 Lossless RGB',
        'desc': 'Lossless recording for screen capture.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov'],
        'options': 'crf=0:preset=p4',
    },
    'libx265': {
        'name': 'HEVC / H.265',
        'desc': 'High efficiency compression for 4K video.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'crf=22:preset=medium',
    },
    'libvpx': {
        'name': 'VP8 Video',
        'desc': 'Legacy open video format for WebM.',
        'ext': 'webm',
        'allowed_exts': ['webm', 'mkv'],
        'options': 'crf=10:b=0',
    },
    'libvpx-vp9': {
        'name': 'VP9 Video',
        'desc': 'Royalty-free web video format.',
        'ext': 'webm',
        'allowed_exts': ['webm', 'mkv'],
        'options': 'crf=23:b=0',
    },
    'libsvtav1': {
        'name': 'AV1',
        'desc': 'Next-gen open codec with high compression.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'crf=26:preset=5',
    },
    'mpeg4': {
        'name': 'MPEG-4 Part 2',
        'desc': 'Legacy format for old media players.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'avi', 'mkv'],
        'options': 'qscale:v=3',
    },
    'mjpeg': {
        'name': 'Motion JPEG',
        'desc': 'Sequence of JPEG frames; minimal CPU load.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mov', 'mkv'],
        'options': 'qscale:v=3',
    },

    # --- Hardware Accelerated: NVIDIA NVENC ---
    'h264_nvenc': {
        'name': 'H.264 (NVIDIA)',
        'desc': 'Fast NVIDIA GPU encoding.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=vbr:cq=19:preset=p5',
    },
    'hevc_nvenc': {
        'name': 'HEVC / H.265 (NVIDIA)',
        'desc': 'Fast high-efficiency NVIDIA GPU encoding.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=vbr:cq=23:preset=p5',
    },
    'av1_nvenc': {
        'name': 'AV1 (NVIDIA)',
        'desc': 'Fast AV1 encoding for NVIDIA RTX 40+.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'rc=vbr:cq=26:preset=p5',
    },

    # --- Hardware Accelerated: Intel QSV ---
    'h264_qsv': {
        'name': 'H.264 (Intel QuickSync)',
        'desc': 'Hardware encoding via Intel GPU.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'global_quality=20:preset=medium',
    },
    'hevc_qsv': {
        'name': 'HEVC / H.265 (Intel QuickSync)',
        'desc': 'Hardware HEVC encoding via Intel GPU.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'global_quality=23:preset=medium',
    },
    'av1_qsv': {
        'name': 'AV1 (Intel QuickSync)',
        'desc': 'Hardware AV1 encoding via Intel Arc/iGPU.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'global_quality=26:preset=medium',
    },
    'vp9_qsv': {
        'name': 'VP9 (Intel QuickSync)',
        'desc': 'Hardware VP9 encoding via Intel GPU.',
        'ext': 'webm',
        'allowed_exts': ['webm', 'mkv'],
        'options': 'global_quality=24',
    },

    # --- Hardware Accelerated: AMD AMF (Windows) ---
    'h264_amf': {
        'name': 'H.264 (AMD AMF)',
        'desc': 'Hardware encoding for AMD GPUs on Windows.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=cqp:qp_i=18:qp_p=18:quality=quality',
    },
    'hevc_amf': {
        'name': 'HEVC / H.265 (AMD AMF)',
        'desc': 'Hardware HEVC encoding for AMD GPUs on Windows.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov', 'ts'],
        'options': 'rc=cqp:qp_i=22:qp_p=22:quality=quality',
    },
    'av1_amf': {
        'name': 'AV1 (AMD AMF)',
        'desc': 'Hardware AV1 encoding for AMD RX 7000+.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'rc=cqp:qp_i=26:qp_p=26:quality=quality',
    },

    # --- Hardware Accelerated: AMD / Linux (VAAPI) ---
    'h264_vaapi': {
        'name': 'H.264 (VAAPI Linux)',
        'desc': 'Hardware encoding via Linux VAAPI.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov'],
        'options': 'qp=20',
    },
    'hevc_vaapi': {
        'name': 'HEVC / H.265 (VAAPI Linux)',
        'desc': 'Hardware HEVC encoding via Linux VAAPI.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'mov'],
        'options': 'qp=25',
    },
    'av1_vaapi': {
        'name': 'AV1 (VAAPI Linux)',
        'desc': 'Hardware AV1 encoding via Linux VAAPI.',
        'ext': 'mp4',
        'allowed_exts': ['mp4', 'mkv', 'webm'],
        'options': 'qp=28',
    },

    # --- Professional / Editing / Archival ---
    'prores': {
        'name': 'Apple ProRes',
        'desc': 'High-performance video editing standard.',
        'ext': 'mov',
        'allowed_exts': ['mov', 'mkv'],
        'options': 'profile=3',
    },
    'prores_ks': {
        'name': 'Apple ProRes (iCodec)',
        'desc': 'Open ProRes implementation for editing.',
        'ext': 'mov',
        'allowed_exts': ['mov', 'mkv'],
        'options': 'profile=3',
    },
    'dnxhd': {
        'name': 'Avid DNxHD / DNxHR',
        'desc': 'Broadcast editing format for Avid workflows.',
        'ext': 'mov',
        'allowed_exts': ['mov', 'mkv', 'mxf'],
        'options': 'b=185M',
    },
    'ffv1': {
        'name': 'Lossless Archival (FFV1)',
        'desc': 'Lossless codec for long-term storage.',
        'ext': 'mkv',
        'allowed_exts': ['mkv', 'avi'],
        'options': 'level=3:coder=1:context=1:slices=16',
    },
    'huffyuv': {
        'name': 'HuffYUV Lossless',
        'desc': 'Fast, simple intra-frame lossless codec.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mkv'],
        'options': '',
    },
    'utvideo': {
        'name': 'Ut Video Lossless',
        'desc': 'Efficient lossless format for editing.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mkv', 'mov'],
        'options': 'pred=left',
    },
    'magicyuv': {
        'name': 'MagicYUV Lossless',
        'desc': 'Real-time lossless recording codec.',
        'ext': 'avi',
        'allowed_exts': ['avi', 'mkv'],
        'options': '',
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

def get_codec_dict(codec_name):
    codec_info = codecs_config.get(codec_name)
    if codec_info is None:
        return {}
    return codec_info
