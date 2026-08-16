#!/usr/bin/env python
"""
Offline unit tests for mixed-media reference validation in modules.minimax.minimax_references.

The resolver turns decoded images and local file paths into the reference objects a ref2va
workflow conditions on, rejecting a bad request before the model load. Its checks run cheapest
first, so the count rules never open a file and the decode rules never run on a request the
counts already rejected.

Covers:

- ``ReferenceCaps`` values and immutability, and the workflow lookup that serves them
- classification of every extension sdnext reads, plus the rejections: an unknown type, a url,
  a missing file, an unsupported extension
- the count rules in the order the pipeline itself checks them, per kind before the total
- the pairing rule that an all-audio request has nothing to condition
- order preservation across a mixed request
- the aspect and frame-count rules that need the decoded media
- container header probing, and the guards for a video too long, too large, or too short
- the two soft dependencies, forced to be absent so their rejections are observed rather than
  assumed

No running server required, and no model is loaded. The decode cases need ``av`` and the
construction cases need ``diffusers``; both skip with a reason when absent.

Usage:
    python test/test-video-references.py
"""

import os
import sys
import math
import wave
import struct
import types
import tempfile

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from PIL import Image  # pylint: disable=wrong-import-position
from modules.errors import log  # pylint: disable=wrong-import-position
from modules import shared  # pylint: disable=wrong-import-position,unused-import
from modules.video_models import video_utils  # pylint: disable=wrong-import-position
from modules.minimax import minimax_references as refs  # pylint: disable=wrong-import-position


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}
tmpdir = None


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, passed: bool, name: str, detail: str = ''):
    status = 'PASS' if passed else 'FAIL'
    results[cat]['passed' if passed else 'failed'] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if passed:
        log.info(msg)
    else:
        log.error(msg)


def skip(cat: str, name: str, reason: str):
    results[cat]['skipped'] += 1
    results[cat]['tests'].append(('SKIP', name))
    log.warning(f'  SKIP: {name} ({reason})')


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        elif isinstance(ok, str):
            skip(cat, name, ok)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e: # pylint: disable=broad-except
        record(cat, False, name, f'exception: {type(e).__name__}: {e}')


def expect_error(fn, fragment: str, code: int = 400):
    """Run fn, assert it rejects with the given code and a message naming the reason."""
    try:
        fn()
    except Exception as e: # pylint: disable=broad-except
        assert getattr(e, 'code', None) == code, f'expected code={code} got code={getattr(e, "code", None)}: {e}'
        assert fragment in str(e), f'expected "{fragment}" in "{e}"'
        return True
    raise AssertionError(f'expected a rejection containing "{fragment}"')


def touch(name: str) -> str:
    """An empty file with a real extension: enough for every check that runs before the decode."""
    fn = os.path.join(tmpdir, name)
    with open(fn, 'wb'):
        pass
    return fn


def image(width: int = 64, height: int = 64):
    return Image.new('RGB', (width, height))


def caps():
    return refs.get_reference_caps('ref2va')


def stub_image_reference(width: int, height: int):
    """What check_decoded reads off an image reference, without constructing the real one."""
    return types.SimpleNamespace(image=image(width, height), kind='image')


def stub_video_reference(frames: int, fps: float):
    return types.SimpleNamespace(frames=[None] * frames, fps=fps, kind='video')


def make_wav(name: str, seconds: float = 1.0, rate: int = 32000, channels: int = 1) -> str:
    fn = os.path.join(tmpdir, name)
    with wave.open(fn, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        count = int(seconds * rate) * channels
        handle.writeframes(struct.pack('<' + 'h' * count, *([0] * count)))
    return fn


def make_mp4(name: str, frames: int = 20, width: int = 64, height: int = 64, fps: int = 24) -> str | None:
    av = video_utils.check_av()
    if not av:
        return None
    import numpy as np
    fn = os.path.join(tmpdir, name)
    try:
        with av.open(fn, mode='w') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
            for index in range(frames):
                array = np.full((height, width, 3), (index * 8) % 256, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as e: # pylint: disable=broad-except
        log.warning(f'test fixture: file="{fn}" {e}')
        return None
    return fn


def has_av() -> bool:
    return bool(video_utils.check_av())


def has_diffusers() -> bool:
    try:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference # pylint: disable=unused-import
        return True
    except Exception: # pylint: disable=broad-except
        return False


# ============================================================
# Caps
# ============================================================

def test_caps_mirror_the_pipeline_limits():
    c = caps()
    assert c is not None, 'ref2va has no caps'
    assert (c.max_images, c.max_videos, c.max_audios, c.max_references) == (9, 3, 3, 12), f'{c}'
    assert c.image_aspect == 4.0, f'{c.image_aspect}'
    assert c.video_min_frames == 13, f'{c.video_min_frames}'
    assert c.audio_sample_rate == 32000, f'{c.audio_sample_rate}'
    assert c.audio_max_channels == 2, f'{c.audio_max_channels}'


def test_caps_total_is_not_the_sum_of_the_kinds():
    c = caps()
    assert c.max_references < c.max_images + c.max_videos + c.max_audios, 'the total limit has to bind'


def test_caps_are_immutable():
    try:
        caps().max_images = 99
    except Exception: # pylint: disable=broad-except
        return True
    raise AssertionError('caps accepted a write')


def test_caps_lookup_misses_on_keyframe_workflows():
    assert refs.get_reference_caps('fl2va') is None, 'fl2va reported reference limits'
    assert refs.get_reference_caps(None) is None, 'a missing workflow reported reference limits'


def test_resolve_rejects_a_workflow_without_caps():
    return expect_error(lambda: refs.resolve('fl2va', [image()]), 'conditions on no references')


# ============================================================
# Classification
# ============================================================

def test_every_supported_extension_classifies():
    for kind, extensions in video_utils.MEDIA_EXTENSIONS.items():
        for ext in extensions:
            got = video_utils.classify_extension(f'sample{ext}')
            assert got == kind, f'{ext} classified as {got}, expected {kind}'


def test_classification_ignores_case():
    assert video_utils.classify_extension('SAMPLE.MP4') == 'video', 'uppercase extension missed'


def test_unknown_extension_has_no_kind():
    assert video_utils.classify_extension('notes.txt') is None, 'txt classified as media'


def test_decoded_image_classifies_without_a_file():
    items = refs.classify_entries([image()])
    assert [kind for _label, kind, _entry in items] == ['image'], f'{items}'


def test_paths_classify_by_extension():
    entries = [touch('a.png'), touch('b.mp4'), touch('c.wav')]
    items = refs.classify_entries(entries)
    assert [kind for _label, kind, _entry in items] == ['image', 'video', 'audio'], f'{items}'


def test_unsupported_extension_is_rejected():
    return expect_error(lambda: refs.classify_entries([touch('notes.txt')]), 'unsupported media type')


def test_url_is_rejected_before_any_fetch():
    # the reference classes download a url and decode whatever comes back, so a request never names one
    return expect_error(lambda: refs.classify_entries(['https://example.com/clip.mp4']), 'not a local file')


def test_missing_file_is_rejected():
    return expect_error(lambda: refs.classify_entries([os.path.join(tmpdir, 'absent.png')]), 'file not found')


def test_unsupported_input_type_is_rejected():
    return expect_error(lambda: refs.classify_entries([7]), 'unsupported input')


def test_labels_are_one_based_and_follow_the_request():
    items = refs.classify_entries([image(), touch('d.mp4'), image()])
    assert [label for label, _kind, _entry in items] == [1, 2, 3], f'{items}'


def test_order_is_preserved_across_kinds():
    entries = [touch('o1.mp4'), image(), touch('o2.wav'), touch('o3.png'), image(), touch('o4.mov')]
    items = refs.classify_entries(entries)
    assert [kind for _label, kind, _entry in items] == ['video', 'image', 'audio', 'image', 'image', 'video'], f'{items}'


# ============================================================
# Counts
# ============================================================

def test_too_many_images():
    return expect_error(lambda: refs.check_counts(caps(), ['image'] * 10), 'too many image references')


def test_too_many_videos():
    return expect_error(lambda: refs.check_counts(caps(), ['video'] * 4), 'too many video references')


def test_too_many_audio():
    return expect_error(lambda: refs.check_counts(caps(), ['audio'] * 4), 'too many audio references')


def test_total_limit_binds_when_every_kind_is_legal():
    kinds = ['image'] * 9 + ['video'] * 3 + ['audio'] # 13 references, no kind over its own limit
    return expect_error(lambda: refs.check_counts(caps(), kinds), 'too many references')


def test_per_kind_is_reported_before_the_total():
    kinds = ['image'] * 9 + ['video'] * 4 # over both, and the pipeline names the kind first
    return expect_error(lambda: refs.check_counts(caps(), kinds), 'too many video references')


def test_the_kind_limits_accept_their_boundary():
    refs.check_counts(caps(), ['image'] * 9)
    refs.check_counts(caps(), ['image'] * 9 + ['video'] * 3) # exactly the total


def test_audio_alone_is_rejected():
    for count in (1, 2, 3):
        expect_error(lambda n=count: refs.check_counts(caps(), ['audio'] * n), 'must be paired')


def test_audio_paired_with_a_picture_passes():
    refs.check_counts(caps(), ['image', 'audio', 'audio', 'audio'])


def test_counts_run_before_any_file_is_opened():
    # every entry is an empty file, so reaching the decode would fail differently than the count rule
    entries = [touch(f'count{index}.mp4') for index in range(4)]
    return expect_error(lambda: refs.resolve('ref2va', entries), 'too many video references')


# ============================================================
# Nothing to condition on
# ============================================================

def test_empty_request_names_the_workflow():
    # the api test detects a reference server by matching the workflow in this rejection
    return expect_error(lambda: refs.resolve('ref2va', []), 'ref2va')


def test_init_image_stands_in_for_a_single_reference():
    items = refs.classify_entries([image()])
    assert len(items) == 1, f'{items}'
    return expect_error(lambda: refs.resolve('ref2va', None), 'No reference media provided')


# ============================================================
# Checks that need the decoded media
# ============================================================

def test_image_aspect_accepts_the_boundary():
    refs.check_decoded(caps(), [(1, 'image', None)], [stub_image_reference(64, 16)]) # exactly 4:1


def test_image_aspect_rejects_beyond_the_boundary():
    expect_error(lambda: refs.check_decoded(caps(), [(1, 'image', None)], [stub_image_reference(8, 64)]), 'aspect ratio out of range')
    expect_error(lambda: refs.check_decoded(caps(), [(1, 'image', None)], [stub_image_reference(64, 8)]), 'aspect ratio out of range')


def test_video_frame_floor_counts_at_the_resampled_rate():
    # 13 frames at 24 fps is the floor; the same 13 frames at 12 fps resample up to 26 and clear it
    refs.check_decoded(caps(), [(1, 'video', None)], [stub_video_reference(13, 24)])
    refs.check_decoded(caps(), [(1, 'video', None)], [stub_video_reference(7, 12)])
    expect_error(lambda: refs.check_decoded(caps(), [(1, 'video', None)], [stub_video_reference(12, 24)]), 'video too short')


def test_video_frame_floor_uses_the_rounding_the_resample_uses():
    c = caps()
    for frames, fps in ((13, 24), (7, 12), (26, 48)):
        resampled = math.floor(frames * refs.REFERENCE_FPS / fps + 0.5)
        assert resampled >= c.video_min_frames, f'{frames}@{fps} resampled to {resampled}'


# ============================================================
# Container probing
# ============================================================

def test_probe_reads_video_headers():
    if not has_av():
        return 'av not installed'
    fn = make_mp4('probe.mp4', frames=20)
    if fn is None:
        return 'no h264 encoder to build a fixture'
    probe = video_utils.probe_media(fn, 'video')
    assert probe is not None, 'probe returned nothing'
    assert abs(probe.fps - 24.0) < 0.01, f'fps={probe.fps}'
    assert (probe.width, probe.height) == (64, 64), f'{probe.width}x{probe.height}'
    assert probe.duration is not None and probe.duration < 2.0, f'duration={probe.duration}'


def test_probe_reads_audio_headers():
    if not has_av():
        return 'av not installed'
    probe = video_utils.probe_media(make_wav('probe.wav', seconds=0.5, rate=44100), 'audio')
    assert probe is not None, 'probe returned nothing'
    assert probe.sample_rate == 44100, f'rate={probe.sample_rate}'
    assert probe.channels == 1, f'channels={probe.channels}'


def test_probe_returns_nothing_for_an_unreadable_file():
    if not has_av():
        return 'av not installed'
    assert video_utils.probe_media(touch('broken.mp4'), 'video') is None, 'an empty container probed clean'


def test_unreadable_video_is_rejected():
    if not has_av():
        return 'av not installed'
    return expect_error(lambda: refs.resolve('ref2va', [image(), touch('empty.mp4')]), 'unreadable')


def test_video_too_long_is_rejected():
    if not has_av():
        return 'av not installed'
    fn = make_mp4('long.mp4', frames=24 * 20, width=32, height=32) # 20 seconds, past what a generation can use
    if fn is None:
        return 'no h264 encoder to build a fixture'
    return expect_error(lambda: refs.resolve('ref2va', [image(), fn]), 'video too long')


def test_video_aspect_is_rejected_from_the_header():
    if not has_av():
        return 'av not installed'
    fn = make_mp4('wide.mp4', frames=20, width=16, height=128)
    if fn is None:
        return 'no h264 encoder to build a fixture'
    return expect_error(lambda: refs.resolve('ref2va', [image(), fn]), 'aspect ratio out of range')


def test_video_too_large_to_decode_is_rejected():
    if not has_av():
        return 'av not installed'
    probe = types.SimpleNamespace(kind='video', fps=24.0, frames=24 * 15, duration=15.0, width=7680, height=4320, channels=None, sample_rate=None)
    return expect_error(lambda: refs.check_video_probe(caps(), 1, 'huge.mp4', probe), 'too large to decode')


def test_audio_channels_over_stereo_are_rejected():
    if not has_av():
        return 'av not installed'
    fn = make_wav('surround.wav', seconds=0.5, channels=6)
    probe = video_utils.probe_media(fn, 'audio')
    if probe is None or probe.channels != 6:
        return 'no six channel probe available'
    return expect_error(lambda: refs.check_audio_probe(caps(), 1, probe), 'too many audio channels')


def test_audio_file_without_a_stream_is_rejected():
    if not has_av():
        return 'av not installed'
    fn = make_mp4('silent_as_audio.mp4', frames=20)
    if fn is None:
        return 'no h264 encoder to build a fixture'
    target = os.path.join(tmpdir, 'silent.wav')
    os.replace(fn, target) # a video container behind an audio extension: classified audio, and it carries no soundtrack
    return expect_error(lambda: refs.resolve('ref2va', [image(), target]), 'has no audio stream')


# ============================================================
# Soft dependencies, forced absent
# ============================================================

def test_missing_torchaudio_rejects_a_resample():
    if not has_av():
        return 'av not installed'
    fn = make_wav('offrate.wav', seconds=0.5, rate=44100)
    original = video_utils.has_torchaudio
    video_utils.has_torchaudio = lambda: False
    try:
        return expect_error(lambda: refs.resolve('ref2va', [image(), fn]), 'requires torchaudio')
    finally:
        video_utils.has_torchaudio = original


def test_matching_rate_needs_no_torchaudio():
    if not has_av():
        return 'av not installed'
    fn = make_wav('onrate.wav', seconds=0.5, rate=32000)
    probe = video_utils.probe_media(fn, 'audio')
    assert probe is not None, 'probe returned nothing'
    original = video_utils.has_torchaudio
    video_utils.has_torchaudio = lambda: False
    try:
        refs.check_audio_probe(caps(), 1, probe) # the vae's own rate, so nothing resamples
    finally:
        video_utils.has_torchaudio = original


def test_missing_av_rejects_media_references():
    original = video_utils.check_av
    video_utils.check_av = lambda: False
    try:
        return expect_error(lambda: refs.resolve('ref2va', [image(), touch('noav.mp4')]), 'require the av package', code=500)
    finally:
        video_utils.check_av = original


def test_missing_av_leaves_image_requests_alone():
    original = video_utils.check_av
    video_utils.check_av = lambda: False
    try:
        items = refs.classify_entries([image(), image()])
        refs.check_counts(caps(), [kind for _label, kind, _entry in items])
    finally:
        video_utils.check_av = original


# ============================================================
# Construction
# ============================================================

def test_built_references_keep_the_request_order():
    if not has_av():
        return 'av not installed'
    if not has_diffusers():
        return 'diffusers not installed'
    video = make_mp4('build.mp4', frames=30)
    if video is None:
        return 'no h264 encoder to build a fixture'
    audio = make_wav('build.wav', seconds=1.0, rate=32000)
    built = refs.resolve('ref2va', [video, image(), audio])
    assert [reference.kind for reference in built] == ['video', 'image', 'audio'], f'{[r.kind for r in built]}'


def test_built_video_carries_its_frame_rate():
    if not has_av():
        return 'av not installed'
    if not has_diffusers():
        return 'diffusers not installed'
    fn = make_mp4('rate.mp4', frames=30)
    if fn is None:
        return 'no h264 encoder to build a fixture'
    built = refs.resolve('ref2va', [image(), fn])
    reference = built[1]
    assert abs(float(reference.fps) - 24.0) < 0.01, f'fps={reference.fps}'
    assert len(reference.frames) == 30, f'frames={len(reference.frames)}'
    assert reference.audio is None, 'a silent video reported a soundtrack' # silent containers are legal


def test_built_image_is_rgb():
    if not has_diffusers():
        return 'diffusers not installed'
    # the reference encoder reads the array raw, so a non-rgb upload has to be converted on the way in
    built = refs.resolve('ref2va', [Image.new('RGBA', (64, 64))])
    assert built[0].image.mode == 'RGB', f'mode={built[0].image.mode}'


def test_short_video_is_rejected_after_the_decode():
    if not has_av():
        return 'av not installed'
    if not has_diffusers():
        return 'diffusers not installed'
    fn = make_mp4('short.mp4', frames=8)
    if fn is None:
        return 'no h264 encoder to build a fixture'
    return expect_error(lambda: refs.resolve('ref2va', [image(), fn]), 'video too short')


# ============================================================
# Runner
# ============================================================

def run_all():
    global tmpdir # pylint: disable=global-statement
    with tempfile.TemporaryDirectory(prefix='sdnext-refs-') as path:
        tmpdir = path

        log.warning('=== caps ===')
        cat = category('caps')
        for fn in [
            test_caps_mirror_the_pipeline_limits,
            test_caps_total_is_not_the_sum_of_the_kinds,
            test_caps_are_immutable,
            test_caps_lookup_misses_on_keyframe_workflows,
            test_resolve_rejects_a_workflow_without_caps,
        ]:
            run_test(cat, fn)

        log.warning('=== classification ===')
        cat = category('classification')
        for fn in [
            test_every_supported_extension_classifies,
            test_classification_ignores_case,
            test_unknown_extension_has_no_kind,
            test_decoded_image_classifies_without_a_file,
            test_paths_classify_by_extension,
            test_unsupported_extension_is_rejected,
            test_url_is_rejected_before_any_fetch,
            test_missing_file_is_rejected,
            test_unsupported_input_type_is_rejected,
            test_labels_are_one_based_and_follow_the_request,
            test_order_is_preserved_across_kinds,
        ]:
            run_test(cat, fn)

        log.warning('=== counts ===')
        cat = category('counts')
        for fn in [
            test_too_many_images,
            test_too_many_videos,
            test_too_many_audio,
            test_total_limit_binds_when_every_kind_is_legal,
            test_per_kind_is_reported_before_the_total,
            test_the_kind_limits_accept_their_boundary,
            test_audio_alone_is_rejected,
            test_audio_paired_with_a_picture_passes,
            test_counts_run_before_any_file_is_opened,
            test_empty_request_names_the_workflow,
            test_init_image_stands_in_for_a_single_reference,
        ]:
            run_test(cat, fn)

        log.warning('=== decoded media ===')
        cat = category('decoded')
        for fn in [
            test_image_aspect_accepts_the_boundary,
            test_image_aspect_rejects_beyond_the_boundary,
            test_video_frame_floor_counts_at_the_resampled_rate,
            test_video_frame_floor_uses_the_rounding_the_resample_uses,
        ]:
            run_test(cat, fn)

        log.warning('=== probing ===')
        cat = category('probing')
        for fn in [
            test_probe_reads_video_headers,
            test_probe_reads_audio_headers,
            test_probe_returns_nothing_for_an_unreadable_file,
            test_unreadable_video_is_rejected,
            test_video_too_long_is_rejected,
            test_video_aspect_is_rejected_from_the_header,
            test_video_too_large_to_decode_is_rejected,
            test_audio_channels_over_stereo_are_rejected,
            test_audio_file_without_a_stream_is_rejected,
        ]:
            run_test(cat, fn)

        log.warning('=== soft dependencies ===')
        cat = category('dependencies')
        for fn in [
            test_missing_torchaudio_rejects_a_resample,
            test_matching_rate_needs_no_torchaudio,
            test_missing_av_rejects_media_references,
            test_missing_av_leaves_image_requests_alone,
        ]:
            run_test(cat, fn)

        log.warning('=== construction ===')
        cat = category('construction')
        for fn in [
            test_built_references_keep_the_request_order,
            test_built_video_carries_its_frame_rate,
            test_built_image_is_rgb,
            test_short_video_is_rejected_after_the_decode,
        ]:
            run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
        total_skipped += info['skipped']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    success = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if success else 1)
