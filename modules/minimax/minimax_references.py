import os
import math
from dataclasses import dataclass
from PIL import Image
from modules.logger import log


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
REFERENCE_FPS = 24.0 # references are resampled onto the model's own frame rate before anything reads them


@dataclass(frozen=True)
class ReferenceCaps:
    """Limits a reference workflow enforces, mirrored from the pipeline's own setup step so a bad
    request is rejected before the model load instead of after it. Frozen because the registry rows
    these describe are rewritten in place by the loader."""
    max_images: int = 9
    max_videos: int = 3
    max_audios: int = 3 # audio references only: a video's own soundtrack takes an <Audio i> label but no slot here
    max_references: int = 12 # not the sum of the three: 9 images and 3 videos is legal, 9 and 3 and 3 is not
    image_aspect: float = 4.0 # widest side ratio, rejected strictly so exactly 4:1 passes
    video_min_frames: int = 13 # after the resample: the conditioner samples at 2 fps and merges frames in pairs, so a shorter reference has no pair to merge
    video_max_seconds: float = 16.0 # a reference is truncated to the generated video, which tops out at 14.375s, so beyond this the decode is thrown away
    video_max_bytes: int = 2 * 1024 * 1024 * 1024 # decoding costs frames*width*height*3 plus a stacking copy, and the result is held across the model load
    audio_max_channels: int = 2 # mono is upmixed to stereo, more than two channels has no downmix
    audio_sample_rate: int = 32000 # the audio vae's own rate; any other rate resamples through torchaudio, which sdnext does not install


# limits are per workflow rather than per model: the registry rows that carry ref2va differ only in which repo they load
REFERENCE_CAPS = {
    'ref2va': ReferenceCaps(),
}


def get_reference_caps(workflow):
    """Reference limits of a workflow, None for the workflows that condition on none."""
    return REFERENCE_CAPS.get(workflow, None)


def reference_error(msg: str, code: int = 400):
    from modules.video_models.video_run import VideoError
    return VideoError(msg, code)


def source_of(entry) -> str:
    return f'file="{entry}"' if isinstance(entry, (str, os.PathLike)) else f'type={type(entry).__name__}'


def classify_entries(entries: list) -> list:
    """(label, kind, entry) per reference in the order given, labelled from 1 to match the model's own numbering."""
    from modules.video_models import video_utils
    items = []
    for index, entry in enumerate(entries):
        label = index + 1
        if isinstance(entry, Image.Image):
            items.append((label, 'image', entry))
            continue
        if not isinstance(entry, (str, os.PathLike)):
            raise reference_error(f'reference {label} unsupported input: type={type(entry).__name__} expected an image or a local file path')
        fn = str(entry)
        if fn.lower().startswith(('http://', 'https://')):
            # the reference classes fetch a url and decode whatever comes back, so a request never gets to name one
            raise reference_error(f'reference {label} not a local file: url="{fn}"')
        if not os.path.isfile(fn):
            raise reference_error(f'reference {label} file not found: file="{fn}"')
        kind = video_utils.classify_extension(fn)
        if kind is None:
            supported = [ext for extensions in video_utils.MEDIA_EXTENSIONS.values() for ext in extensions]
            raise reference_error(f'reference {label} unsupported media type: file="{fn}" supported={supported}')
        items.append((label, kind, fn))
    return items


def check_counts(caps: ReferenceCaps, kinds: list):
    """Per kind, then the total, then the pairing rule: the order the setup step itself checks in."""
    for kind, limit in (('image', caps.max_images), ('video', caps.max_videos), ('audio', caps.max_audios)):
        count = kinds.count(kind)
        if count > limit:
            raise reference_error(f'too many {kind} references: count={count} max={limit}')
    if len(kinds) > caps.max_references:
        raise reference_error(f'too many references: count={len(kinds)} max={caps.max_references}')
    if set(kinds) == {'audio'}: # an audio reference goes to the audio vae alone and conditions no picture on its own
        raise reference_error(f'audio references must be paired with an image or video reference: count={len(kinds)}')


def check_video_probe(caps: ReferenceCaps, label: int, fn: str, probe):
    if not probe.fps:
        raise reference_error(f'reference {label} video frame rate unknown: file="{fn}"')
    if probe.duration is not None and probe.duration > caps.video_max_seconds:
        raise reference_error(f'reference {label} video too long: seconds={probe.duration:.1f} max={caps.video_max_seconds:g}')
    frames = probe.frames or (math.ceil(probe.duration * probe.fps) if probe.duration else None)
    if frames and probe.width and probe.height:
        size = frames * probe.width * probe.height * 3
        if size > caps.video_max_bytes:
            raise reference_error(f'reference {label} video too large to decode: estimate={size / 1024 ** 3:.1f}GB max={caps.video_max_bytes / 1024 ** 3:.1f}GB size={probe.width}x{probe.height} frames={frames}')
    if probe.width and probe.height and (probe.width > caps.image_aspect * probe.height or probe.height > caps.image_aspect * probe.width):
        raise reference_error(f'reference {label} aspect ratio out of range: size={probe.width}x{probe.height} supported=1:{caps.image_aspect:g}..{caps.image_aspect:g}:1')


def check_audio_probe(caps: ReferenceCaps, label: int, probe):
    from modules.video_models import video_utils
    if probe.channels is not None and probe.channels > caps.audio_max_channels:
        raise reference_error(f'reference {label} too many audio channels: channels={probe.channels} max={caps.audio_max_channels}')
    if probe.sample_rate != caps.audio_sample_rate and not video_utils.has_torchaudio():
        raise reference_error(f'reference {label} resampling {probe.sample_rate}Hz to {caps.audio_sample_rate}Hz requires torchaudio')


def preflight_probe(caps: ReferenceCaps, items: list):
    """Header checks for the file entries, each one standing in for a failure that would otherwise
    land after the model load. A container that reports no duration is decoded unbounded."""
    from modules.video_models import video_utils
    for label, kind, entry in items:
        if kind == 'image':
            continue
        probe = video_utils.probe_media(entry, kind)
        if probe is None:
            raise reference_error(f'reference {label} unreadable: file="{entry}"')
        debug(f'Video: op=reference probe={label} kind={kind} file="{entry}" fps={probe.fps} frames={probe.frames} seconds={probe.duration} channels={probe.channels} rate={probe.sample_rate}')
        if kind == 'video':
            check_video_probe(caps, label, entry, probe)
        elif probe.sample_rate is None:
            raise reference_error(f'reference {label} has no audio stream: file="{entry}"')
        if probe.sample_rate is not None: # a video's own soundtrack goes through the same normalization an audio reference does
            check_audio_probe(caps, label, probe)


def build_reference_objects(items: list) -> list:
    from diffusers.utils import load_image
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference
    references = []
    for label, kind, entry in items:
        try:
            if kind == 'image':
                # the reference encoder reads the image array raw, and load_image is what applies the exif transpose and the rgb conversion
                references.append(MiniMaxH3ImageReference(image=load_image(entry)))
            elif kind == 'video':
                references.append(MiniMaxH3VideoReference.from_file(entry))
            else:
                references.append(MiniMaxH3AudioReference.from_file(entry))
        except Exception as e:
            raise reference_error(f'reference {label} decode failed: {source_of(entry)} {e}') from e
    return references


def check_decoded(caps: ReferenceCaps, items: list, references: list):
    """The checks that need the decoded media: an image's real size, and a video's frame count at the
    rate it is resampled to, which is what the conditioner counts."""
    for (label, kind, _entry), reference in zip(items, references):
        if kind == 'image':
            width, height = reference.image.size
            if width <= 0 or height <= 0:
                raise reference_error(f'reference {label} image size invalid: size={width}x{height}')
            if width > caps.image_aspect * height or height > caps.image_aspect * width:
                raise reference_error(f'reference {label} aspect ratio out of range: size={width}x{height} supported=1:{caps.image_aspect:g}..{caps.image_aspect:g}:1')
        elif kind == 'video':
            frames, fps = len(reference.frames), float(reference.fps or REFERENCE_FPS)
            resampled = math.floor(frames * REFERENCE_FPS / fps + 0.5) if fps > 0 else 0 # the rounding the resample itself uses
            if resampled < caps.video_min_frames:
                raise reference_error(f'reference {label} video too short: frames={frames} fps={fps:g} min={caps.video_min_frames}@{REFERENCE_FPS:g}fps')


def resolve(workflow: str, references: list | None, init_image=None) -> list:
    """The ordered reference objects a reference workflow conditions on, from decoded images and
    local file paths. Order is preserved exactly: it fixes the <Picture i>, <Video i> and <Audio i>
    labels a prompt addresses, and the shared clock the references are laid on."""
    caps = get_reference_caps(workflow)
    if caps is None:
        raise reference_error(f'workflow conditions on no references: workflow={workflow}')
    entries = list(references) if references else ([init_image] if init_image is not None else [])
    if len(entries) == 0:
        # keep the workflow named here: the api test probes for a reference server by matching it in the rejection
        raise reference_error(f'No reference media provided. The {workflow} workflow conditions on references, so at least one is required.')
    items = classify_entries(entries)
    kinds = [kind for _label, kind, _entry in items]
    check_counts(caps, kinds)
    if 'video' in kinds or 'audio' in kinds:
        from modules.video_models import video_utils
        if not video_utils.check_av():
            raise reference_error('video and audio references require the av package', 500)
    preflight_probe(caps, items)
    built = build_reference_objects(items)
    check_decoded(caps, items, built)
    log.debug(f'Video: op=reference workflow={workflow} images={kinds.count("image")} videos={kinds.count("video")} audio={kinds.count("audio")} total={len(built)}')
    return built
