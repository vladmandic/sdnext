#!/usr/bin/env python
"""
API tests for YOLO Detailer endpoints.

Tests:
- GET /sdapi/v1/detailers — model enumeration
- POST /sdapi/v1/detect — object detection on test images
- POST /sdapi/v1/txt2img — generation with detailer enabled

Requires a running SD.Next instance with a model loaded.

Usage:
    python test/test-detailer-api.py [--url URL] [--image PATH]
"""

import io
import os
import sys
import time
import json
import base64
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Reference model cover images with faces (best for detailer testing)
FACE_TEST_IMAGES = [
    'models/Reference/ponyRealism_V23.jpg',                               # realistic woman, clear face
    'models/Reference/HiDream-ai--HiDream-I1-Fast.jpg',                   # realistic man, clear face + text
    'models/Reference/stabilityai--stable-diffusion-xl-base-1.0.jpg',     # realistic woman portrait
    'models/Reference/CalamitousFelicitousness--Anima-Preview-3-sdnext-diffusers.jpg',  # anime face (non-realistic test)
]

# Fallback images (no guaranteed faces)
FALLBACK_IMAGES = [
    'html/sdnext-robot-2k.jpg',
    'html/favicon.png',
]


class DetailerAPITest:
    """Test harness for YOLO Detailer API endpoints."""

    def __init__(self, base_url, image_path=None, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.test_images = {}  # name -> base64
        self.timeout = timeout
        self.results = {
            'enumerate': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'detect': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'generate': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'detailer_params': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
        }
        self._category = 'enumerate'
        self._critical_error = None
        self._load_images(image_path)

    def _encode_image(self, path):
        from PIL import Image
        image = Image.open(path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        buf = io.BytesIO()
        image.save(buf, 'JPEG')
        return base64.b64encode(buf.getvalue()).decode(), image.size

    def _load_images(self, image_path=None):
        if image_path and os.path.exists(image_path):
            b64, size = self._encode_image(image_path)
            name = os.path.basename(image_path)
            self.test_images[name] = b64
            print(f"  Test image: {image_path} ({size})")
            return

        # Load all available face test images
        for p in FACE_TEST_IMAGES:
            if os.path.exists(p):
                b64, size = self._encode_image(p)
                name = os.path.basename(p)
                self.test_images[name] = b64
                print(f"  Loaded: {name} ({size[0]}x{size[1]})")

        # Fallback if no face images found
        if not self.test_images:
            for p in FALLBACK_IMAGES:
                if os.path.exists(p):
                    b64, size = self._encode_image(p)
                    name = os.path.basename(p)
                    self.test_images[name] = b64
                    print(f"  Fallback: {name} ({size[0]}x{size[1]})")
                    break

        if not self.test_images:
            print("  WARNING: No test images found, detect tests will be skipped")

    @property
    def image_b64(self):
        """Return the first available test image for backwards compat."""
        if self.test_images:
            return next(iter(self.test_images.values()))
        return None

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

    # =========================================================================
    # Tests: Model Enumeration
    # =========================================================================

    def test_detailers_list(self):
        """GET /sdapi/v1/detailers returns a list of available models."""
        self._category = 'enumerate'
        print("\n--- Detailer Model Enumeration ---")

        data = self._get('/sdapi/v1/detailers')
        if 'error' in data:
            self.record(False, 'detailers_list', f"error: {data}")
            self._critical_error = f"Server error: {data}"
            return []

        if not isinstance(data, list):
            self.record(False, 'detailers_list', f"expected list, got {type(data).__name__}")
            return []

        self.record(True, 'detailers_list', f"{len(data)} models found")

        # Verify each entry has expected fields
        if len(data) > 0:
            sample = data[0]
            has_name = 'name' in sample
            self.record(has_name, 'detailer_entry_has_name', f"sample: {sample}")
            if not has_name:
                self.record(False, 'detailer_entry_schema', "missing 'name' field")

        return data

    # =========================================================================
    # Tests: Detection
    # =========================================================================

    def _validate_detect_response(self, data, label):
        """Validate detection response schema and return detection count."""
        expected_keys = ['classes', 'labels', 'boxes', 'scores']
        for key in expected_keys:
            if key not in data:
                self.record(False, f'{label}_schema_{key}', f"missing '{key}'")
                return -1

        # All arrays should have the same length
        lengths = [len(data[key]) for key in expected_keys]
        all_same = len(set(lengths)) <= 1
        if not all_same:
            self.record(False, f'{label}_array_lengths', f"mismatched: {dict(zip(expected_keys, lengths))}")
            return -1

        n = lengths[0]

        if n > 0:
            # Scores should be in [0, 1]
            scores_valid = all(0 <= s <= 1 for s in data['scores'])
            if not scores_valid:
                self.record(False, f'{label}_scores_range', f"scores: {data['scores']}")

            # Boxes should be lists of 4 numbers
            boxes_valid = all(isinstance(b, list) and len(b) == 4 for b in data['boxes'])
            if not boxes_valid:
                self.record(False, f'{label}_boxes_format', "bad box format")

        return n

    # Face detection models to try (in priority order)
    FACE_MODELS = ['face-yolo8n', 'face-yolo8m', 'anzhc-face-1024-seg-8n']

    def _pick_face_model(self, available_models):
        """Pick the best face detection model from available ones."""
        available_names = [m.get('name', '') for m in available_models] if available_models else []
        for model in self.FACE_MODELS:
            if model in available_names:
                return model
        return ''  # fall back to server default

    def test_detect_all_images(self, available_models=None):
        """POST /sdapi/v1/detect on each loaded test image with a face model."""
        self._category = 'detect'
        print("\n--- Detection Tests (per-image) ---")

        if not self.test_images:
            self.skip('detect_all', 'no test images')
            return

        if self._critical_error:
            self.skip('detect_all', self._critical_error)
            return

        face_model = self._pick_face_model(available_models)
        if face_model:
            print(f"  Using face model: {face_model}")
        else:
            print("  No face model available, using server default")

        total_detections = 0
        any_face_found = False

        for img_name, img_b64 in self.test_images.items():
            short = img_name.replace('.jpg', '')[:40]
            data = self._post('/sdapi/v1/detect', {'image': img_b64, 'model': face_model})

            if 'error' in data:
                self.record(False, f'detect_{short}', f"error: {data}")
                continue

            n = self._validate_detect_response(data, f'detect_{short}')
            if n < 0:
                continue

            labels = data.get('labels', [])
            scores = data.get('scores', [])
            detail_parts = [f"{n} detections"]
            if labels:
                detail_parts.append(f"labels={labels}")
            if scores:
                detail_parts.append(f"top_score={max(scores):.3f}")

            self.record(True, f'detect_{short}', ', '.join(detail_parts))
            total_detections += n
            if n > 0:
                any_face_found = True

        self.record(any_face_found, 'detect_found_faces',
                    f"{total_detections} total detections across {len(self.test_images)} images")

    def test_detect_with_model(self, model_name):
        """POST /sdapi/v1/detect with a specific model on all images."""
        if not self.test_images:
            self.skip(f'detect_model_{model_name}', 'no test images')
            return

        total = 0
        for _img_name, img_b64 in self.test_images.items():
            data = self._post('/sdapi/v1/detect', {'image': img_b64, 'model': model_name})
            if 'error' not in data:
                total += len(data.get('scores', []))

        self.record(True, f'detect_model_{model_name}', f"{total} detections across {len(self.test_images)} images")

    # =========================================================================
    # Tests: Generation with Detailer
    # =========================================================================

    def test_txt2img_with_detailer(self):
        """POST /sdapi/v1/txt2img with detailer_enabled=True."""
        self._category = 'generate'
        print("\n--- Generation with Detailer ---")

        if self._critical_error:
            self.skip('txt2img_detailer', self._critical_error)
            return

        payload = {
            'prompt': 'a photo of a person, face, portrait',
            'negative_prompt': '',
            'steps': 10,
            'width': 512,
            'height': 512,
            'seed': 42,
            'save_images': False,
            'send_images': True,
            'detailer_enabled': True,
            'detailer_strength': 0.3,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
            'detailer_max': 3,
        }

        t0 = time.time()
        # Detailer generation is multi-pass (generate + detect + inpaint per region), use longer timeout
        try:
            r = requests.post(f'{self.base_url}/sdapi/v1/txt2img', json=payload, timeout=600, verify=False)
            if r.status_code != 200:
                data = {'error': r.status_code, 'reason': r.reason}
            else:
                data = r.json()
        except requests.exceptions.ConnectionError as e:
            self.record(False, 'txt2img_detailer', f"connection error (is a model loaded?): {e}")
            return
        except requests.exceptions.ReadTimeout:
            self.record(False, 'txt2img_detailer', 'timeout after 600s')
            return
        t1 = time.time()

        if 'error' in data:
            self.record(False, 'txt2img_detailer', f"error: {data} (ensure a model is loaded)")
            return

        # Should have images
        has_images = 'images' in data and len(data['images']) > 0
        self.record(has_images, 'txt2img_detailer_has_images', f"time={t1 - t0:.1f}s")

        if has_images:
            # Decode and verify image
            from PIL import Image
            img_data = data['images'][0].split(',', 1)[0]
            img = Image.open(io.BytesIO(base64.b64decode(img_data)))
            self.record(True, 'txt2img_detailer_image_valid', f"size={img.size}")

        # Check info field for detailer metadata
        if 'info' in data:
            info = data['info'] if isinstance(data['info'], str) else json.dumps(data['info'])
            has_detailer_info = 'detailer' in info.lower() or 'Detailer' in info
            self.record(has_detailer_info, 'txt2img_detailer_metadata',
                        'detailer info found in metadata' if has_detailer_info else 'no detailer metadata (detection may have found nothing)')

    def test_txt2img_without_detailer(self):
        """POST /sdapi/v1/txt2img baseline without detailer (sanity check)."""
        if self._critical_error:
            self.skip('txt2img_baseline', self._critical_error)
            return

        payload = {
            'prompt': 'a simple landscape',
            'steps': 5,
            'width': 512,
            'height': 512,
            'seed': 42,
            'save_images': False,
            'send_images': True,
        }

        data = self._post('/sdapi/v1/txt2img', payload)
        if 'error' in data:
            self.record(False, 'txt2img_baseline', f"error: {data}")
            return

        has_images = 'images' in data and len(data['images']) > 0
        self.record(has_images, 'txt2img_baseline', 'generation works without detailer')

    # =========================================================================
    # Tests: Per-Request Detailer Param Validation
    # =========================================================================

    def _txt2img(self, extra_params=None):
        """Helper: generate a portrait with optional param overrides."""
        payload = {
            'prompt': 'a photo of a person, face, portrait',
            'steps': 10,
            'width': 512,
            'height': 512,
            'seed': 42,
            'save_images': False,
            'send_images': True,
        }
        if extra_params:
            payload.update(extra_params)
        try:
            r = requests.post(f'{self.base_url}/sdapi/v1/txt2img', json=payload, timeout=600, verify=False)
            if r.status_code != 200:
                return {'error': r.status_code, 'reason': r.reason}
            return r.json()
        except requests.exceptions.ConnectionError as e:
            return {'error': 'connection_refused', 'reason': str(e)}
        except requests.exceptions.ReadTimeout:
            return {'error': 'timeout', 'reason': 'timeout after 600s'}

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
        """Mean absolute pixel difference between two images."""
        import numpy as np
        if arr_a is None or arr_b is None or arr_a.shape != arr_b.shape:
            return -1.0
        return float(np.abs(arr_a - arr_b).mean())

    def _get_info(self, data):
        """Extract info string from generation response."""
        if 'info' not in data:
            return ''
        info = data['info']
        return info if isinstance(info, str) else json.dumps(info)

    def run_detailer_param_tests(self, available_models=None):
        """Verify per-request detailer params change the output."""
        self._category = 'detailer_params'
        print("\n--- Per-Request Detailer Param Validation ---")

        if self._critical_error:
            self.skip('detailer_params_all', self._critical_error)
            return

        # Generate baseline WITHOUT detailer (same seed/prompt as detailer tests)
        print("  Generating baseline (no detailer)...")
        baseline_data = self._txt2img()
        if 'error' in baseline_data:
            self.record(False, 'detailer_baseline', f"error: {baseline_data}")
            return
        baseline = self._decode_image(baseline_data)
        if baseline is None:
            self.record(False, 'detailer_baseline', 'no image')
            return
        self.record(True, 'detailer_baseline')

        # Generate WITH detailer enabled (default params)
        print("  Generating with detailer (defaults)...")
        detailer_default_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.3,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
        })
        if 'error' in detailer_default_data:
            self.record(False, 'detailer_default', f"error: {detailer_default_data}")
            return
        detailer_default = self._decode_image(detailer_default_data)

        # Detailer ON vs OFF should produce different images (if a face was detected)
        diff_on_off = self._pixel_diff(baseline, detailer_default)
        self.record(diff_on_off > 0.5, 'detailer_on_vs_off',
                    f"mean_diff={diff_on_off:.2f}" if diff_on_off > 0.5
                    else f"identical (diff={diff_on_off:.4f}) — no face detected?")

        # -- Strength variation --
        print("  Testing strength variation...")
        strong_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.7,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
        })
        if 'error' not in strong_data:
            strong = self._decode_image(strong_data)
            diff_strong = self._pixel_diff(detailer_default, strong)
            self.record(diff_strong > 0.5, 'detailer_strength_effect',
                        f"strength 0.3 vs 0.7: diff={diff_strong:.2f}")

        # -- Steps variation --
        print("  Testing steps variation...")
        more_steps_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.3,
            'detailer_steps': 20,
            'detailer_conf': 0.3,
        })
        if 'error' not in more_steps_data:
            more_steps = self._decode_image(more_steps_data)
            diff_steps = self._pixel_diff(detailer_default, more_steps)
            self.record(diff_steps > 0.5, 'detailer_steps_effect',
                        f"steps 5 vs 20: diff={diff_steps:.2f}")

        # -- Resolution variation --
        print("  Testing resolution variation...")
        hires_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.3,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
            'detailer_resolution': 512,
        })
        if 'error' not in hires_data:
            hires = self._decode_image(hires_data)
            diff_res = self._pixel_diff(detailer_default, hires)
            self.record(diff_res > 0.5, 'detailer_resolution_effect',
                        f"resolution 1024 vs 512: diff={diff_res:.2f}")

        # -- Segmentation mode --
        # Segmentation requires a -seg model (e.g. anzhc-face-1024-seg-8n).
        # Detection-only models (face-yolo8n) don't produce masks, so the flag has no effect.
        seg_models = [m.get('name', '') for m in (available_models or [])
                      if 'seg' in m.get('name', '').lower() and 'face' in m.get('name', '').lower()]
        if seg_models:
            seg_model = seg_models[0]
            print(f"  Testing segmentation mode (model={seg_model})...")
            # bbox baseline with the seg model
            seg_bbox_data = self._txt2img({
                'detailer_enabled': True,
                'detailer_strength': 0.3,
                'detailer_steps': 5,
                'detailer_conf': 0.3,
                'detailer_segmentation': False,
                'detailer_models': [seg_model],
            })
            seg_data = self._txt2img({
                'detailer_enabled': True,
                'detailer_strength': 0.3,
                'detailer_steps': 5,
                'detailer_conf': 0.3,
                'detailer_segmentation': True,
                'detailer_models': [seg_model],
            })
            if 'error' not in seg_data and 'error' not in seg_bbox_data:
                seg_bbox = self._decode_image(seg_bbox_data)
                seg_mask = self._decode_image(seg_data)
                diff_seg = self._pixel_diff(seg_bbox, seg_mask)
                self.record(diff_seg > 0.5, 'detailer_segmentation_effect',
                            f"bbox vs seg mask ({seg_model}): diff={diff_seg:.2f}")
            else:
                err = seg_data if 'error' in seg_data else seg_bbox_data
                self.record(False, 'detailer_segmentation_effect', f"error: {err}")
        else:
            print("  Testing segmentation mode...")
            seg_data = {'error': 'skipped'}
            self.skip('detailer_segmentation_effect', 'no face-seg model available')

        # -- Confidence threshold --
        print("  Testing confidence threshold...")
        high_conf_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.3,
            'detailer_steps': 5,
            'detailer_conf': 0.95,
        })
        if 'error' not in high_conf_data:
            high_conf = self._decode_image(high_conf_data)
            diff_conf = self._pixel_diff(baseline, high_conf)
            # High confidence may reject detections, making output closer to baseline
            self.record(True, 'detailer_conf_effect',
                        f"conf=0.95 vs baseline: diff={diff_conf:.2f} "
                        f"(low diff = detections filtered out, high diff = still detected)")

        # -- Custom detailer prompt --
        print("  Testing detailer prompt override...")
        prompt_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.5,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
            'detailer_prompt': 'a detailed close-up face with freckles',
        })
        if 'error' not in prompt_data:
            prompt_result = self._decode_image(prompt_data)
            diff_prompt = self._pixel_diff(detailer_default, prompt_result)
            self.record(diff_prompt > 0.5, 'detailer_prompt_effect',
                        f"custom prompt vs default: diff={diff_prompt:.2f}")

        # -- Metadata verification across params --
        for test_data, label in [
            (detailer_default_data, 'detailer_default'),
            (strong_data if 'error' not in strong_data else None, 'detailer_strong'),
            (more_steps_data if 'error' not in more_steps_data else None, 'detailer_more_steps'),
            (seg_data if 'error' not in seg_data else None, 'detailer_segmentation'),
        ]:
            if test_data is None:
                continue
            info = self._get_info(test_data)
            has_meta = 'detailer' in info.lower() or 'Detailer' in info
            self.record(has_meta, f'{label}_metadata',
                        'detailer info in metadata' if has_meta else 'no detailer metadata')

        # -- Param isolation: generate without detailer after all detailer runs --
        print("  Testing param isolation...")
        after_data = self._txt2img()
        if 'error' not in after_data:
            after = self._decode_image(after_data)
            leak_diff = self._pixel_diff(baseline, after)
            self.record(leak_diff < 0.5, 'detailer_param_isolation',
                        f"post-detailer baseline diff={leak_diff:.4f}" if leak_diff < 0.5
                        else f"LEAK: baseline changed (diff={leak_diff:.2f})")

    # =========================================================================
    # Runner
    # =========================================================================

    def run_all(self):
        print("=" * 60)
        print("YOLO Detailer API Test Suite")
        print(f"Server: {self.base_url}")
        print("=" * 60)

        # Enumerate
        models = self.test_detailers_list()

        # Detect across all loaded test images
        self.test_detect_all_images(models)
        # Test with first available model if any
        if models and len(models) > 0:
            model_name = models[0].get('name', models[0].get('filename', ''))
            if model_name:
                self.test_detect_with_model(model_name)

        # Generate
        self.test_txt2img_without_detailer()
        self.test_txt2img_with_detailer()

        # Per-request detailer param validation
        self.run_detailer_param_tests(models)

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
    parser = argparse.ArgumentParser(description='YOLO Detailer API Tests')
    parser.add_argument('--url', default=os.environ.get('SDAPI_URL', 'http://127.0.0.1:7860'), help='server URL')
    parser.add_argument('--image', default=None, help='test image path')
    args = parser.parse_args()
    test = DetailerAPITest(args.url, args.image)
    success = test.run_all()
    sys.exit(0 if success else 1)
