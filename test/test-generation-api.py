#!/usr/bin/env python
"""
API tests for generation with scheduler params, color grading, and latent corrections.

Tests:
- GET /sdapi/v1/samplers — sampler enumeration and config
- POST /sdapi/v1/txt2img — generation with various samplers
- POST /sdapi/v1/txt2img — generation with color grading params
- POST /sdapi/v1/txt2img — generation with latent correction params

Requires a running SD.Next instance with a model loaded.

Usage:
    python test/test-generation-api.py [--url URL] [--steps STEPS]
"""

import io
import os
import sys
import json
import time
import base64
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GenerationAPITest:
    """Test harness for generation API with scheduler and grading params."""

    # Samplers to test — a representative subset covering different scheduler families
    TEST_SAMPLERS = [
        'Euler a',
        'Euler',
        'DPM++ 2M',
        'UniPC',
        'DDIM',
        'DPM++ 2M SDE',
    ]

    def __init__(self, base_url, steps=10, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.steps = steps
        self.timeout = timeout
        self.results = {
            'samplers': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'generation': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'grading': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'correction': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'param_validation': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
        }
        self._category = 'samplers'
        self._critical_error = None

    def _get(self, endpoint):
        try:
            r = requests.get(f'{self.base_url}{endpoint}', timeout=self.timeout, verify=False)
            if r.status_code != 200:
                return {'error': r.status_code, 'reason': r.reason}
            return r.json()
        except requests.exceptions.ConnectionError:
            return {'error': 'connection_refused', 'reason': 'Server not running'}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    def _post(self, endpoint, data):
        try:
            r = requests.post(f'{self.base_url}{endpoint}', json=data, timeout=self.timeout, verify=False)
            if r.status_code != 200:
                return {'error': r.status_code, 'reason': r.reason}
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

    def _txt2img(self, extra_params=None, prompt='a cat'):
        """Helper: run txt2img with base params + overrides. Returns (data, time)."""
        payload = {
            'prompt': prompt,
            'steps': self.steps,
            'width': 512,
            'height': 512,
            'seed': 42,
            'save_images': False,
            'send_images': True,
        }
        if extra_params:
            payload.update(extra_params)
        t0 = time.time()
        data = self._post('/sdapi/v1/txt2img', payload)
        return data, time.time() - t0

    def _check_generation(self, data, test_name, elapsed):
        """Validate a generation response has images."""
        if 'error' in data:
            self.record(False, test_name, f"error: {data}")
            return False
        has_images = 'images' in data and len(data['images']) > 0
        self.record(has_images, test_name, f"time={elapsed:.1f}s")
        return has_images

    def _get_info(self, data):
        """Extract info string from generation response."""
        if 'info' not in data:
            return ''
        info = data['info']
        return info if isinstance(info, str) else json.dumps(info)

    def _decode_image(self, data):
        """Decode first image from generation response into numpy array."""
        import numpy as np
        from PIL import Image
        if 'images' not in data or len(data['images']) == 0:
            return None
        img_data = data['images'][0].split(',', 1)[0]
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('RGB')
        return np.array(img, dtype=np.float32)

    def _pixel_diff(self, arr_a, arr_b):
        """Mean absolute pixel difference between two images (0-255 scale)."""
        import numpy as np
        if arr_a is None or arr_b is None:
            return -1.0
        if arr_a.shape != arr_b.shape:
            return -1.0
        return float(np.abs(arr_a - arr_b).mean())

    def _channel_means(self, arr):
        """Return per-channel means [R, G, B]."""
        if arr is None:
            return [0, 0, 0]
        return [float(arr[:, :, c].mean()) for c in range(3)]

    # =========================================================================
    # Tests: Sampler Enumeration
    # =========================================================================

    def test_samplers_list(self):
        """GET /sdapi/v1/samplers returns available samplers with config."""
        self._category = 'samplers'
        print("\n--- Sampler Enumeration ---")

        data = self._get('/sdapi/v1/samplers')
        if 'error' in data:
            self.record(False, 'samplers_list', f"error: {data}")
            self._critical_error = f"Server error: {data}"
            return []

        if not isinstance(data, list):
            self.record(False, 'samplers_list', f"expected list, got {type(data).__name__}")
            return []

        self.record(True, 'samplers_list', f"{len(data)} samplers available")

        # Check that each sampler has a name
        sampler_names = []
        for s in data:
            name = s.get('name', '')
            if name:
                sampler_names.append(name)

        self.record(len(sampler_names) == len(data), 'samplers_have_names',
                    f"{len(sampler_names)}/{len(data)} have names")

        # Check for our test samplers
        for test_sampler in self.TEST_SAMPLERS:
            found = test_sampler in sampler_names
            if not found:
                self.skip(f'sampler_available_{test_sampler}', 'not in server sampler list')
            else:
                self.record(True, f'sampler_available_{test_sampler}')

        return sampler_names

    # =========================================================================
    # Tests: Generation with Different Samplers
    # =========================================================================

    def test_samplers_generate(self, available_samplers):
        """Generate with each test sampler and verify success."""
        self._category = 'generation'
        print("\n--- Generation with Different Samplers ---")

        if self._critical_error:
            for s in self.TEST_SAMPLERS:
                self.skip(f'generate_{s}', self._critical_error)
            return

        for sampler in self.TEST_SAMPLERS:
            if sampler not in available_samplers:
                self.skip(f'generate_{sampler}', 'sampler not available')
                continue
            data, elapsed = self._txt2img({'sampler_name': sampler})
            self._check_generation(data, f'generate_{sampler}', elapsed)

    # =========================================================================
    # Tests: Color Grading Params
    # =========================================================================

    def test_grading_brightness_contrast(self):
        """Generate with grading brightness and contrast."""
        data, elapsed = self._txt2img({
            'grading_brightness': 0.2,
            'grading_contrast': 0.3,
        })
        self._check_generation(data, 'grading_brightness_contrast', elapsed)

    def test_grading_saturation_hue(self):
        """Generate with grading saturation and hue shift."""
        data, elapsed = self._txt2img({
            'grading_saturation': 0.5,
            'grading_hue': 0.1,
        })
        self._check_generation(data, 'grading_saturation_hue', elapsed)

    def test_grading_gamma_sharpness(self):
        """Generate with gamma correction and sharpness."""
        data, elapsed = self._txt2img({
            'grading_gamma': 0.8,
            'grading_sharpness': 0.5,
        })
        self._check_generation(data, 'grading_gamma_sharpness', elapsed)

    def test_grading_color_temp(self):
        """Generate with warm color temperature."""
        data, elapsed = self._txt2img({
            'grading_color_temp': 3500,
        })
        self._check_generation(data, 'grading_color_temp', elapsed)

    def test_grading_tone(self):
        """Generate with shadows/midtones/highlights adjustments."""
        data, elapsed = self._txt2img({
            'grading_shadows': 0.3,
            'grading_midtones': -0.1,
            'grading_highlights': 0.2,
        })
        self._check_generation(data, 'grading_tone', elapsed)

    def test_grading_effects(self):
        """Generate with vignette and grain."""
        data, elapsed = self._txt2img({
            'grading_vignette': 0.5,
            'grading_grain': 0.3,
        })
        self._check_generation(data, 'grading_effects', elapsed)

    def test_grading_split_toning(self):
        """Generate with split toning colors."""
        data, elapsed = self._txt2img({
            'grading_shadows_tint': '#003366',
            'grading_highlights_tint': '#ffcc00',
            'grading_split_tone_balance': 0.6,
        })
        self._check_generation(data, 'grading_split_toning', elapsed)

    def test_grading_combined(self):
        """Generate with multiple grading params at once."""
        data, elapsed = self._txt2img({
            'grading_brightness': 0.1,
            'grading_contrast': 0.2,
            'grading_saturation': 0.3,
            'grading_gamma': 0.9,
            'grading_color_temp': 5000,
            'grading_vignette': 0.3,
        })
        self._check_generation(data, 'grading_combined', elapsed)

    def run_grading_tests(self):
        """Run all grading tests."""
        self._category = 'grading'
        print("\n--- Color Grading Tests ---")

        if self._critical_error:
            self.skip('grading_all', self._critical_error)
            return

        self.test_grading_brightness_contrast()
        self.test_grading_saturation_hue()
        self.test_grading_gamma_sharpness()
        self.test_grading_color_temp()
        self.test_grading_tone()
        self.test_grading_effects()
        self.test_grading_split_toning()
        self.test_grading_combined()

    # =========================================================================
    # Tests: Latent Correction Params
    # =========================================================================

    def test_correction_brightness(self):
        """Generate with latent brightness correction."""
        data, elapsed = self._txt2img({'hdr_brightness': 1.5})
        ok = self._check_generation(data, 'correction_brightness', elapsed)
        if ok:
            info = self._get_info(data)
            has_param = 'Latent brightness' in info
            self.record(has_param, 'correction_brightness_metadata',
                        'found in info' if has_param else 'not found in info')

    def test_correction_color(self):
        """Generate with latent color centering."""
        data, elapsed = self._txt2img({'hdr_color': 0.5, 'hdr_mode': 1})
        ok = self._check_generation(data, 'correction_color', elapsed)
        if ok:
            info = self._get_info(data)
            has_param = 'Latent color' in info
            self.record(has_param, 'correction_color_metadata',
                        'found in info' if has_param else 'not found in info')

    def test_correction_clamp(self):
        """Generate with latent clamping."""
        data, elapsed = self._txt2img({
            'hdr_clamp': True,
            'hdr_threshold': 0.8,
            'hdr_boundary': 4.0,
        })
        ok = self._check_generation(data, 'correction_clamp', elapsed)
        if ok:
            info = self._get_info(data)
            has_param = 'Latent clamp' in info
            self.record(has_param, 'correction_clamp_metadata',
                        'found in info' if has_param else 'not found in info')

    def test_correction_sharpen(self):
        """Generate with latent sharpening."""
        data, elapsed = self._txt2img({'hdr_sharpen': 1.0})
        ok = self._check_generation(data, 'correction_sharpen', elapsed)
        if ok:
            info = self._get_info(data)
            has_param = 'Latent sharpen' in info
            self.record(has_param, 'correction_sharpen_metadata',
                        'found in info' if has_param else 'not found in info')

    def test_correction_maximize(self):
        """Generate with latent maximize/normalize."""
        data, elapsed = self._txt2img({
            'hdr_maximize': True,
            'hdr_max_center': 0.6,
            'hdr_max_boundary': 2.0,
        })
        ok = self._check_generation(data, 'correction_maximize', elapsed)
        if ok:
            info = self._get_info(data)
            has_param = 'Latent max' in info
            self.record(has_param, 'correction_maximize_metadata',
                        'found in info' if has_param else 'not found in info')

    def test_correction_combined(self):
        """Generate with multiple correction params."""
        data, elapsed = self._txt2img({
            'hdr_brightness': 1.0,
            'hdr_color': 0.3,
            'hdr_sharpen': 0.5,
            'hdr_clamp': True,
        })
        ok = self._check_generation(data, 'correction_combined', elapsed)
        if ok:
            info = self._get_info(data)
            # At least some correction params should appear
            found = [k for k in ['Latent brightness', 'Latent color', 'Latent sharpen', 'Latent clamp'] if k in info]
            self.record(len(found) > 0, 'correction_combined_metadata', f"found: {found}")

    def run_correction_tests(self):
        """Run all latent correction tests."""
        self._category = 'correction'
        print("\n--- Latent Correction Tests ---")

        if self._critical_error:
            self.skip('correction_all', self._critical_error)
            return

        self.test_correction_brightness()
        self.test_correction_color()
        self.test_correction_clamp()
        self.test_correction_sharpen()
        self.test_correction_maximize()
        self.test_correction_combined()

    # =========================================================================
    # Tests: Per-Request Param Validation (baseline comparison)
    # =========================================================================

    def _generate_baseline(self):
        """Generate a baseline image with no grading/correction. Cache and reuse."""
        if hasattr(self, '_baseline_arr') and self._baseline_arr is not None:
            return self._baseline_arr, self._baseline_data
        data, elapsed = self._txt2img()
        if 'error' in data or 'images' not in data:
            return None, data
        self._baseline_arr = self._decode_image(data)
        self._baseline_data = data
        print(f'  Baseline generated: time={elapsed:.1f}s mean={self._channel_means(self._baseline_arr)}')
        return self._baseline_arr, data

    def _compare_param(self, name, params, check_fn=None):
        """Generate with params and compare to baseline. Optionally run check_fn(baseline, result)."""
        baseline, _ = self._generate_baseline()
        if baseline is None:
            self.skip(f'param_{name}', 'baseline generation failed')
            return

        data, elapsed = self._txt2img(params)
        if 'error' in data:
            self.record(False, f'param_{name}', f"generation error: {data}")
            return

        result = self._decode_image(data)
        if result is None:
            self.record(False, f'param_{name}', 'no image in response')
            return

        diff = self._pixel_diff(baseline, result)
        differs = diff > 0.5  # more than 0.5/255 mean difference
        self.record(differs, f'param_{name}_differs',
                    f"mean_diff={diff:.2f}" if differs else f"images identical (diff={diff:.4f})")

        if check_fn and differs:
            try:
                ok, detail = check_fn(baseline, result, data)
                self.record(ok, f'param_{name}_direction', detail)
            except Exception as e:
                self.record(False, f'param_{name}_direction', f"check error: {e}")

    def run_param_validation_tests(self):
        """Verify per-request grading/correction params actually change the output."""
        self._category = 'param_validation'
        print("\n--- Per-Request Param Validation ---")

        if self._critical_error:
            self.skip('param_validation_all', self._critical_error)
            return

        import numpy as np

        # -- Grading params --

        # Brightness: positive should increase mean pixel value
        def check_brightness(base, result, _data):
            base_mean = float(base.mean())
            result_mean = float(result.mean())
            return result_mean > base_mean, f"baseline={base_mean:.1f} graded={result_mean:.1f}"
        self._compare_param('grading_brightness', {'grading_brightness': 0.3}, check_brightness)

        # Contrast: should increase standard deviation
        def check_contrast(base, result, _data):
            return float(result.std()) > float(base.std()), \
                f"baseline_std={float(base.std()):.1f} graded_std={float(result.std()):.1f}"
        self._compare_param('grading_contrast', {'grading_contrast': 0.5}, check_contrast)

        # Saturation: desaturation should reduce color channel spread
        def check_desaturation(base, result, _data):
            base_spread = max(self._channel_means(base)) - min(self._channel_means(base))
            result_spread = max(self._channel_means(result)) - min(self._channel_means(result))
            return result_spread < base_spread, \
                f"baseline_spread={base_spread:.1f} graded_spread={result_spread:.1f}"
        self._compare_param('grading_saturation_neg', {'grading_saturation': -0.5}, check_desaturation)

        # Hue shift: just verify it changes
        self._compare_param('grading_hue', {'grading_hue': 0.2})

        # Gamma < 1: should brighten (raise values that are < 1)
        def check_gamma(base, result, _data):
            return float(result.mean()) > float(base.mean()), \
                f"baseline={float(base.mean()):.1f} gamma={float(result.mean()):.1f}"
        self._compare_param('grading_gamma', {'grading_gamma': 0.7}, check_gamma)

        # Sharpness: just verify it changes
        self._compare_param('grading_sharpness', {'grading_sharpness': 0.8})

        # Color temperature warm: red channel mean should increase relative to blue
        def check_warm(base, result, _data):
            base_r, _, base_b = self._channel_means(base)
            res_r, _, res_b = self._channel_means(result)
            base_rb = base_r - base_b
            res_rb = res_r - res_b
            return res_rb > base_rb, f"baseline R-B={base_rb:.1f} warm R-B={res_rb:.1f}"
        self._compare_param('grading_color_temp_warm', {'grading_color_temp': 3000}, check_warm)

        # Color temperature cool: blue should increase relative to red
        def check_cool(base, result, _data):
            base_r, _, base_b = self._channel_means(base)
            res_r, _, res_b = self._channel_means(result)
            base_rb = base_r - base_b
            res_rb = res_r - res_b
            return res_rb < base_rb, f"baseline R-B={base_rb:.1f} cool R-B={res_rb:.1f}"
        self._compare_param('grading_color_temp_cool', {'grading_color_temp': 10000}, check_cool)

        # Vignette: corners should be darker than baseline corners
        def check_vignette(base, result, _data):
            h, w = base.shape[:2]
            corner_size = h // 8
            base_corners = np.concatenate([
                base[:corner_size, :corner_size].flatten(),
                base[:corner_size, -corner_size:].flatten(),
                base[-corner_size:, :corner_size].flatten(),
                base[-corner_size:, -corner_size:].flatten(),
            ])
            result_corners = np.concatenate([
                result[:corner_size, :corner_size].flatten(),
                result[:corner_size, -corner_size:].flatten(),
                result[-corner_size:, :corner_size].flatten(),
                result[-corner_size:, -corner_size:].flatten(),
            ])
            return float(result_corners.mean()) < float(base_corners.mean()), \
                f"baseline_corners={float(base_corners.mean()):.1f} vignette_corners={float(result_corners.mean()):.1f}"
        self._compare_param('grading_vignette', {'grading_vignette': 0.8}, check_vignette)

        # Grain: just verify it changes (stochastic)
        self._compare_param('grading_grain', {'grading_grain': 0.5})

        # Shadows/midtones/highlights: verify changes
        self._compare_param('grading_shadows', {'grading_shadows': 0.5})
        self._compare_param('grading_highlights', {'grading_highlights': -0.3})

        # CLAHE: should increase local contrast
        self._compare_param('grading_clahe', {'grading_clahe_clip': 2.0})

        # Split toning: verify changes
        self._compare_param('grading_split_toning', {
            'grading_shadows_tint': '#003366',
            'grading_highlights_tint': '#ffcc00',
        })

        # -- Correction params --

        # Latent brightness: should change output and appear in metadata
        def check_correction_meta(key):
            def _check(_base, _result, data):
                info = self._get_info(data)
                return key in info, f"'{key}' {'found' if key in info else 'missing'} in info"
            return _check
        self._compare_param('hdr_brightness', {'hdr_brightness': 2.0}, check_correction_meta('Latent brightness'))
        self._compare_param('hdr_color', {'hdr_color': 0.8, 'hdr_mode': 1}, check_correction_meta('Latent color'))
        self._compare_param('hdr_sharpen', {'hdr_sharpen': 1.5}, check_correction_meta('Latent sharpen'))
        self._compare_param('hdr_clamp', {'hdr_clamp': True, 'hdr_threshold': 0.7}, check_correction_meta('Latent clamp'))

        # Isolation: verify params from one request don't leak to the next
        data_after, _ = self._txt2img()
        arr_after = self._decode_image(data_after)
        baseline, _ = self._generate_baseline()
        if baseline is not None and arr_after is not None:
            leak_diff = self._pixel_diff(baseline, arr_after)
            no_leak = leak_diff < 0.5
            self.record(no_leak, 'param_isolation',
                        f"post-grading baseline diff={leak_diff:.4f}" if no_leak
                        else f"LEAK: baseline changed after grading requests (diff={leak_diff:.2f})")

    # =========================================================================
    # Runner
    # =========================================================================

    def run_all(self):
        print("=" * 60)
        print("Generation API Test Suite")
        print(f"Server: {self.base_url}")
        print(f"Steps: {self.steps}")
        print("=" * 60)

        # Samplers
        available = self.test_samplers_list()
        self.test_samplers_generate(available)

        # Grading
        self.run_grading_tests()

        # Corrections
        self.run_correction_tests()

        # Per-request param validation (baseline comparison)
        self.run_param_validation_tests()

        # Summary
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
    parser = argparse.ArgumentParser(description='Generation API Tests (samplers, grading, correction)')
    parser.add_argument('--url', default=os.environ.get('SDAPI_URL', 'http://127.0.0.1:7860'), help='server URL')
    parser.add_argument('--steps', type=int, default=10, help='generation steps (lower = faster tests)')
    args = parser.parse_args()
    test = GenerationAPITest(args.url, args.steps)
    success = test.run_all()
    sys.exit(0 if success else 1)
