#!/usr/bin/env python
"""
API tests for video generation.

Tests:
- GET /sdapi/v1/video/models — engine/model enumeration and mode derivation
- POST /sdapi/v1/video — request validation errors (partial pair, unknown model/sampler, checkpoint override, unknown script)
- POST /sdapi/v1/video — still mode (frames=1) against the currently loaded model
- POST /sdapi/v1/video — video generation against the currently loaded model
- POST /sdapi/v1/video — wire switches and GET /sdapi/v1/video/file serving

Requires a running SD.Next instance. Generation categories require a video-capable
model loaded (for example MiniMax-H3 via the base checkpoint dropdown) and are
skipped otherwise; enumeration and validation run against any instance.

Usage:
    python test/test-video-api.py [--url URL] [--steps STEPS] [--frames FRAMES]
"""

import os
import sys
import base64
import time
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

VALID_MODES = {'workflow', 't2v', 'i2v', 'flf2v', 'vace', 'animate'}


class VideoAPITest:
    """Test harness for the video generation API."""

    def __init__(self, base_url, steps=8, frames=17, timeout=3600):
        self.base_url = base_url.rstrip('/')
        self.steps = steps
        self.frames = frames
        self.timeout = timeout
        self.video_capable = None # set by the still-mode probe
        self.results = {
            'enumerate': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'validation': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'still': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'generation': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'wire': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
        }
        self._category = 'enumerate'

    def _get(self, endpoint, params=None):
        try:
            r = requests.get(f'{self.base_url}{endpoint}', params=params, timeout=self.timeout, verify=False)
            if r.status_code != 200:
                res = {'error': r.status_code, 'reason': r.reason}
                try:
                    res['detail'] = r.json().get('detail', None)
                except Exception:
                    pass
                return res
            return r.json()
        except requests.exceptions.ConnectionError:
            return {'error': 'connection_refused', 'reason': 'Server not running'}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    def _post(self, endpoint, data):
        try:
            r = requests.post(f'{self.base_url}{endpoint}', json=data, timeout=self.timeout, verify=False)
            if r.status_code != 200:
                res = {'error': r.status_code, 'reason': r.reason}
                try:
                    res['detail'] = r.json().get('detail', None)
                except Exception:
                    pass
                return res
            return r.json()
        except requests.exceptions.ConnectionError:
            return {'error': 'connection_refused', 'reason': 'Server not running'}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    def record(self, passed, name, detail=''):
        status = 'PASS' if passed else 'FAIL'
        self.results[self._category]['passed' if passed else 'failed'] += 1
        self.results[self._category]['tests'].append((status, name))
        msg = f'  {status}: {name}'
        if detail:
            msg += f' ({detail})'
        print(msg)

    def skip(self, name, reason):
        self.results[self._category]['skipped'] += 1
        self.results[self._category]['tests'].append(('SKIP', name))
        print(f'  SKIP: {name} ({reason})')

    def _video(self, extra_params=None, prompt='a red fox in the snow'):
        payload = {
            'prompt': prompt,
            'steps': self.steps,
            'frames': self.frames,
            'width': 640,
            'height': 384,
            'seed': 42,
        }
        if extra_params:
            payload.update(extra_params)
        t0 = time.time()
        data = self._post('/sdapi/v1/video', payload)
        return data, time.time() - t0

    # =========================================================================
    # Tests: Enumeration
    # =========================================================================

    def test_enumerate(self):
        self._category = 'enumerate'
        print("\n--- Enumeration Tests ---")
        data = self._get('/sdapi/v1/video/models')
        if isinstance(data, dict) and 'error' in data:
            self.record(False, 'models_list', f'error: {data}')
            return []
        self.record(len(data) > 0, 'models_list', f'{len(data)} models')
        bad_modes = [item['name'] for item in data if item.get('mode') not in VALID_MODES]
        self.record(len(bad_modes) == 0, 'models_modes', 'all valid' if not bad_modes else f'invalid: {bad_modes}')
        minimax = [item for item in data if item['engine'] == 'MiniMax']
        if minimax:
            self.record(all(item['base'] for item in minimax), 'models_minimax_base', f'{len(minimax)} rows')
            self.record(all(item['mode'] == 'workflow' for item in minimax), 'models_minimax_workflow')
        else:
            self.skip('models_minimax', 'no MiniMax rows in registry')
        filtered = self._get('/sdapi/v1/video/models', params={'engine': 'MiniMax'})
        if isinstance(filtered, list):
            self.record(all(item['engine'] == 'MiniMax' for item in filtered), 'models_engine_filter', f'{len(filtered)} rows')
        else:
            self.record(False, 'models_engine_filter', f'error: {filtered}')
        return data

    # =========================================================================
    # Tests: Validation
    # =========================================================================

    def test_validation(self, models):
        self._category = 'validation'
        print("\n--- Validation Tests ---")
        data, _elapsed = self._video({'engine': 'MiniMax'})
        self.record(data.get('error') == 400, 'partial_pair_rejected', f'code={data.get("error")}')
        data, _elapsed = self._video({'engine': 'NoSuchEngine', 'model': 'NoSuchModel'})
        self.record(data.get('error') == 404, 'unknown_model_rejected', f'code={data.get("error")} detail={data.get("detail")}')
        # a valid registry pair fails on the sampler before any model load happens
        if models:
            pair = {'engine': models[0]['engine'], 'model': models[0]['name']}
            data, _elapsed = self._video({**pair, 'sampler_name': 'NoSuchSampler'})
            self.record(data.get('error') == 404, 'unknown_sampler_rejected', f'code={data.get("error")}')
            data, _elapsed = self._video({**pair, 'override_settings': {'sd_model_checkpoint': 'other-model'}})
            self.record(data.get('error') == 400, 'checkpoint_override_rejected', f'code={data.get("error")}')
            data, _elapsed = self._video({**pair, 'alwayson_scripts': {'no-such-script': {'args': []}}})
            self.record(data.get('error') == 422, 'unknown_script_rejected', f'code={data.get("error")}')
        else:
            self.skip('unknown_sampler_rejected', 'no registry models to pair with')

    # =========================================================================
    # Tests: Still mode (doubles as the video-capability probe)
    # =========================================================================

    def test_still(self):
        self._category = 'still'
        print("\n--- Still Mode Tests ---")
        data, elapsed = self._video({'frames': 1})
        if data.get('error') == 400:
            self.video_capable = False
            self.skip('still_generation', f'no video-capable model loaded: {data.get("detail")}')
            return
        if 'error' in data:
            self.video_capable = False
            self.record(False, 'still_generation', f'error: {data}')
            return
        self.video_capable = True
        self.record(data.get('still') is True, 'still_flag', f'time={elapsed:.1f}s')
        self.record(data.get('video') is None, 'still_no_video')
        self.record(len(data.get('frames') or []) == 1, 'still_single_frame', f'frames={len(data.get("frames") or [])}')

    # =========================================================================
    # Tests: Generation with the loaded model
    # =========================================================================

    def test_generation(self):
        self._category = 'generation'
        print("\n--- Generation Tests ---")
        if not self.video_capable:
            self.skip('video_generation', 'no video-capable model loaded')
            return None
        data, elapsed = self._video()
        if 'error' in data:
            self.record(False, 'video_generation', f'error: {data}')
            return None
        self.record(data.get('frames_count', 0) > 0, 'video_frames_count', f'frames={data.get("frames_count")} time={elapsed:.1f}s')
        video_b64 = data.get('video')
        decoded = len(base64.b64decode(video_b64)) if video_b64 else 0
        self.record(decoded > 1000, 'video_payload', f'bytes={decoded}')
        self.record(data.get('fps', 0) > 0 and data.get('duration', 0) > 0, 'video_timing', f'fps={data.get("fps")} duration={data.get("duration")}')
        self.record(isinstance(data.get('has_audio'), bool), 'video_audio_flag', f'has_audio={data.get("has_audio")}')
        self.record(bool(data.get('info')), 'video_info')
        return data

    # =========================================================================
    # Tests: Wire switches and file serving
    # =========================================================================

    def test_wire(self):
        self._category = 'wire'
        print("\n--- Wire Tests ---")
        if not self.video_capable:
            self.skip('wire_all', 'no video-capable model loaded')
            return
        data, _elapsed = self._video({'send_video': False, 'send_thumbnail': False})
        if 'error' in data:
            self.record(False, 'wire_send_video_off', f'error: {data}')
            return
        self.record(data.get('video') is None, 'wire_send_video_off')
        path = data.get('video_path')
        self.record(bool(path), 'wire_video_path', f'path={path}')
        if path:
            r = requests.get(f'{self.base_url}/sdapi/v1/video/file', params={'file': path}, timeout=300, verify=False)
            ctype = r.headers.get('content-type', '')
            self.record(r.status_code == 200 and ctype.startswith('video/'), 'wire_file_endpoint', f'code={r.status_code} type={ctype} bytes={len(r.content)}')
        r = requests.get(f'{self.base_url}/sdapi/v1/video/file', params={'file': '/etc/passwd'}, timeout=60, verify=False)
        self.record(r.status_code == 403, 'wire_file_jail', f'code={r.status_code}')

    # =========================================================================
    # Runner
    # =========================================================================

    def run_all(self):
        print("=" * 60)
        print("Video API Test Suite")
        print(f"Server: {self.base_url}")
        print(f"Steps: {self.steps} Frames: {self.frames}")
        print("=" * 60)

        models = self.test_enumerate()
        self.test_validation(models)
        self.test_still()
        self.test_generation()
        self.test_wire()

        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        for cat, data in self.results.items():
            total_passed += data['passed']
            total_failed += data['failed']
            total_skipped += data['skipped']
            status = 'PASS' if data['failed'] == 0 else 'FAIL'
            print(f"  {cat}: {data['passed']} passed, {data['failed']} failed, {data['skipped']} skipped [{status}]")
        print(f"  Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
        print("=" * 60)
        return total_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video API Tests (enumeration, validation, generation, file serving)')
    parser.add_argument('--url', default=os.environ.get('SDAPI_URL', 'http://127.0.0.1:7860'), help='server URL')
    parser.add_argument('--steps', type=int, default=8, help='generation steps (lower = faster tests)')
    parser.add_argument('--frames', type=int, default=17, help='frame count for video tests')
    args = parser.parse_args()
    test = VideoAPITest(args.url, args.steps, args.frames)
    success = test.run_all()
    sys.exit(0 if success else 1)
