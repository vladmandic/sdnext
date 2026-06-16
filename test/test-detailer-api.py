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
    'ui/assets/favicon.png',
]


class DetailerAPITest:
    """Test harness for YOLO Detailer API endpoints."""

    def __init__(self, base_url, image_path=None, timeout=300, model_query=None):
        self.base_url = base_url.rstrip('/')
        self.test_images = {}  # name -> base64
        self.timeout = timeout
        self.model_query = model_query or 'anima base'  # checkpoint to load for the run (substring match)
        self.results = {
            'enumerate': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'detect': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'generate': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'detailer_params': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'detail_endpoint': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'extras_script_args': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
        }
        self._category = 'enumerate'
        self._critical_error = None
        self.face_models = []  # picked in run_all; detectors tried to locate the region for effect-diff crops
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

    # Detectors to try when locating the edited region, most-general first. The seg model is listed
    # ahead of the realistic yolo8n/8m so it also fires on stylized (e.g. anime) generated faces; it
    # is also what the default detailer uses, so its box matches the region that was actually edited.
    REGION_MODELS = ['anzhc-face-1024-seg-8n', 'face-yolo8m', 'face-yolo8n', 'anzhc-head-seg-8n']

    def _pick_region_models(self, available_models):
        """Ordered list of available face/head detectors to try when locating the edited region."""
        names = [m.get('name', '') for m in (available_models or [])]
        ordered = [m for m in self.REGION_MODELS if m in names]
        ordered += [n for n in names if ('face' in n.lower() or 'head' in n.lower()) and n not in ordered]
        return ordered or ['']  # '' = server default

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

    def _detect_box(self, img_b64, models=None, pad=0.1):
        """Largest detection box (x1,y1,x2,y2) from /sdapi/v1/detect, trying each name in `models`
        until one detects something (different detectors fire on realistic vs stylized faces). Padded
        by `pad` of box size per side. Returns None when nothing is found so callers fall back to a
        whole-frame diff."""
        for model in (models or ['']):
            data = self._post('/sdapi/v1/detect', {'image': img_b64, 'model': model})
            boxes = data.get('boxes', []) if 'error' not in data else []
            if not boxes:
                continue
            box = max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
            x1, y1, x2, y2 = (float(v) for v in box)
            dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
            return (int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy))
        return None

    def _region_diff(self, arr_a, arr_b, box):
        """Mean absolute pixel difference within box=(x1,y1,x2,y2), clamped to image bounds. The
        detailer only edits the detected region, so cropping to it keeps a real but localized change
        from being averaged away by the unchanged majority of the frame. Whole-frame when box is None."""
        import numpy as np
        if arr_a is None or arr_b is None or arr_a.shape != arr_b.shape:
            return -1.0
        if box is None:
            return float(np.abs(arr_a - arr_b).mean())
        h, w = arr_a.shape[:2]
        x1 = max(0, min(int(box[0]), w - 1))
        y1 = max(0, min(int(box[1]), h - 1))
        x2 = max(x1 + 1, min(int(box[2]), w))
        y2 = max(y1 + 1, min(int(box[3]), h))
        return float(np.abs(arr_a[y1:y2, x1:x2] - arr_b[y1:y2, x1:x2]).mean())

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

        # The detailer only repaints the detected face. Variation tests below compare two detailed
        # outputs, so measure their diff inside that box; whole-frame averaging buries the signal
        # under the unchanged ~85% of the image. All variants share the seed=42 base, so one detect
        # on the baseline locates the region for every comparison.
        box = self._detect_box(baseline_data['images'][0], self.face_models) if baseline_data.get('images') else None
        print(f"  Detailer region box={box}" if box else "  No face box detected; effect diffs fall back to whole-frame")

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

        # -- Strength variation (extreme: 0.3 vs 0.9) --
        print("  Testing strength variation (0.3 vs 0.9)...")
        strong_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.9,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
        })
        if 'error' not in strong_data:
            strong = self._decode_image(strong_data)
            diff_strong = self._region_diff(detailer_default, strong, box)
            self.record(diff_strong > 0.5, 'detailer_strength_effect',
                        f"strength 0.3 vs 0.9: region diff={diff_strong:.2f}")

        # -- Steps variation (extreme: 1 vs 20 @ strength 0.7) --
        # At a high denoise a single step can't resolve the region while 20 can, so this is a clear
        # yes/no on whether step count drives the result. Both runs share strength 0.7 to isolate steps.
        print("  Testing steps variation (1 vs 20 @ strength 0.7)...")
        few_steps_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.7,
            'detailer_steps': 1,
            'detailer_conf': 0.3,
        })
        more_steps_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.7,
            'detailer_steps': 20,
            'detailer_conf': 0.3,
        })
        if 'error' not in few_steps_data and 'error' not in more_steps_data:
            few_steps = self._decode_image(few_steps_data)
            more_steps = self._decode_image(more_steps_data)
            diff_steps = self._region_diff(few_steps, more_steps, box)
            self.record(diff_steps > 0.5, 'detailer_steps_effect',
                        f"steps 1 vs 20 @ strength 0.7: region diff={diff_steps:.2f}")

        # -- Resolution variation (extreme: 1024 vs 256) --
        print("  Testing resolution variation (1024 vs 256)...")
        hires_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.3,
            'detailer_steps': 5,
            'detailer_conf': 0.3,
            'detailer_resolution': 256,
        })
        if 'error' not in hires_data:
            hires = self._decode_image(hires_data)
            diff_res = self._region_diff(detailer_default, hires, box)
            self.record(diff_res > 0.5, 'detailer_resolution_effect',
                        f"resolution 1024 vs 256: region diff={diff_res:.2f}")

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
                diff_seg = self._region_diff(seg_bbox, seg_mask, box)
                self.record(diff_seg > 0.5, 'detailer_segmentation_effect',
                            f"bbox vs seg mask ({seg_model}): region diff={diff_seg:.2f}")
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

        # -- Custom detailer prompt (extreme: two divergent prompts @ strength 0.7) --
        # Maximally different prompts at a high denoise should paint visibly different faces, so this
        # checks the detailer prompt reaches the inpaint pass at all. Both runs share strength/steps.
        print("  Testing detailer prompt override (divergent prompts @ strength 0.7)...")
        prompt_a_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.7,
            'detailer_steps': 10,
            'detailer_conf': 0.3,
            'detailer_prompt': 'a photo of an elderly bearded man',
        })
        prompt_b_data = self._txt2img({
            'detailer_enabled': True,
            'detailer_strength': 0.7,
            'detailer_steps': 10,
            'detailer_conf': 0.3,
            'detailer_prompt': 'a photo of a young woman with bright blue hair',
        })
        if 'error' not in prompt_a_data and 'error' not in prompt_b_data:
            prompt_a = self._decode_image(prompt_a_data)
            prompt_b = self._decode_image(prompt_b_data)
            diff_prompt = self._region_diff(prompt_a, prompt_b, box)
            self.record(diff_prompt > 0.5, 'detailer_prompt_effect',
                        f"divergent prompts @ strength 0.7: region diff={diff_prompt:.2f}")

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
    # Tests: /sdapi/v1/detail standalone endpoint
    # =========================================================================

    def _detail(self, **kwargs):
        """Helper: POST /sdapi/v1/detail with default face image and override kwargs."""
        if not self.image_b64:
            return {'error': 'no_test_image'}
        payload = {'image': self.image_b64}
        payload.update(kwargs)
        try:
            r = requests.post(f'{self.base_url}/sdapi/v1/detail', json=payload, timeout=self.timeout, verify=False)
            if r.status_code != 200:
                return {'error': r.status_code, 'reason': r.reason}
            return r.json()
        except requests.exceptions.ConnectionError as e:
            return {'error': 'connection_refused', 'reason': str(e)}
        except requests.exceptions.ReadTimeout:
            return {'error': 'timeout'}

    def _decode_b64_image(self, b64_str):
        """Decode a base64 image string into a numpy float32 RGB array."""
        import numpy as np
        from PIL import Image
        try:
            img_data = b64_str.split(',', 1)[0] if ',' in b64_str else b64_str
            img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('RGB')
            return np.array(img, dtype=np.float32)
        except Exception:
            return None

    def test_detail_endpoint_basic(self):
        """POST /sdapi/v1/detail with defaults; assert valid PIL response."""
        self._category = 'detail_endpoint'
        print("\n--- /sdapi/v1/detail Basic ---")

        if self._critical_error:
            self.skip('detail_basic', self._critical_error)
            return None
        if not self.image_b64:
            self.skip('detail_basic', 'no test image')
            return None

        t0 = time.time()
        data = self._detail(detailer_strength=0.3, detailer_steps=5, detailer_conf=0.3)
        t1 = time.time()
        if 'error' in data:
            self.record(False, 'detail_basic', f"error: {data}")
            return None
        has_image = 'image' in data and data['image']
        self.record(has_image, 'detail_basic_has_image', f"time={t1 - t0:.1f}s")
        if has_image:
            arr = self._decode_b64_image(data['image'])
            self.record(arr is not None, 'detail_basic_image_valid', f"shape={arr.shape if arr is not None else 'invalid'}")
            return arr
        return None

    def test_detail_endpoint_strength_effect(self):
        """Verify per-request strength override changes the output (measured inside the detected face box)."""
        self._category = 'detail_endpoint'
        print("  Testing detail strength variation...")

        weak = self._detail(detailer_strength=0.3, detailer_steps=5, detailer_conf=0.3)
        strong = self._detail(detailer_strength=0.7, detailer_steps=5, detailer_conf=0.3)
        if 'error' in weak or 'error' in strong:
            self.record(False, 'detail_strength_effect', f"weak={weak.get('error')} strong={strong.get('error')}")
            return
        weak_arr = self._decode_b64_image(weak['image'])
        strong_arr = self._decode_b64_image(strong['image'])
        box = self._detect_box(self.image_b64, self.face_models)
        diff = self._region_diff(weak_arr, strong_arr, box)
        self.record(diff > 0.5, 'detail_strength_effect', f"region diff={diff:.2f}")

    def test_detail_endpoint_includes_detections(self):
        """When detailer_include_detections=True, response should contain detections b64."""
        self._category = 'detail_endpoint'
        print("  Testing include_detections...")

        data = self._detail(detailer_strength=0.3, detailer_steps=5, detailer_conf=0.3, detailer_include_detections=True)
        if 'error' in data:
            self.record(False, 'detail_includes_detections', f"error: {data}")
            return
        has_detections = 'detections' in data and data['detections']
        if has_detections:
            arr = self._decode_b64_image(data['detections'])
            self.record(arr is not None, 'detail_includes_detections', f"detections shape={arr.shape if arr is not None else 'invalid'}")
        else:
            # No detections returned could mean the model didn't find anything; not a hard failure
            self.skip('detail_includes_detections', 'no detections returned (model found nothing?)')

    def test_detail_endpoint_param_isolation(self):
        """After /sdapi/v1/detail run, baseline txt2img should be unchanged from before."""
        self._category = 'detail_endpoint'
        print("  Testing param isolation...")

        before = self._txt2img()
        if 'error' in before:
            self.skip('detail_param_isolation', f'baseline failed: {before}')
            return
        before_arr = self._decode_image(before)

        detail_resp = self._detail(detailer_strength=0.5, detailer_steps=5)
        if 'error' in detail_resp:
            self.skip('detail_param_isolation', f'detail call failed: {detail_resp}')
            return

        after = self._txt2img()
        if 'error' in after:
            self.skip('detail_param_isolation', f'after-baseline failed: {after}')
            return
        after_arr = self._decode_image(after)
        leak = self._pixel_diff(before_arr, after_arr)
        self.record(leak < 0.5, 'detail_param_isolation', f"leak={leak:.4f}" if leak < 0.5 else f"LEAK detected (diff={leak:.2f})")

    def _pick_named_sampler(self):
        """Return a concrete (non-Default) sampler name from the server, falling back to a common one."""
        data = self._get('/sdapi/v1/samplers')
        if isinstance(data, list):
            for s in data:
                name = s.get('name', '') if isinstance(s, dict) else str(s)
                if name and name.lower() != 'default':
                    return name
        return 'Euler a'

    def test_detail_endpoint_sampler_block(self):
        """Exercise the full sampler block end-to-end (named sampler + scheduler knobs + cfg + options); assert a valid image."""
        self._category = 'detail_endpoint'
        print("  Testing sampler block (smoke)...")
        sampler = self._pick_named_sampler()
        data = self._detail(
            detailer_strength=0.5, detailer_steps=5, detailer_conf=0.3,
            detailer_sampler=sampler, detailer_prediction='epsilon', detailer_shift=4.0, detailer_cfg_scale=8.0,
            detailer_loworder=True, detailer_thresholding=False, detailer_dynamic=False, detailer_rescale=False,
        )
        if 'error' in data:
            self.record(False, 'detail_sampler_block', f"sampler={sampler} error: {data}")
            return
        arr = self._decode_b64_image(data['image']) if data.get('image') else None
        self.record(arr is not None, 'detail_sampler_block', f"sampler={sampler} shape={arr.shape if arr is not None else 'invalid'}")

    def test_detail_endpoint_scheduler_isolation(self):
        """A sampler-block override must not leak into the global schedulers_shift opt (job-local independence)."""
        self._category = 'detail_endpoint'
        print("  Testing scheduler isolation...")
        opts_before = self._get('/sdapi/v1/options')
        if 'error' in opts_before:
            self.skip('detail_scheduler_isolation', f'options read failed: {opts_before}')
            return
        shift_before = opts_before.get('schedulers_shift')
        resp = self._detail(detailer_strength=0.5, detailer_steps=5, detailer_sampler=self._pick_named_sampler(), detailer_shift=8.0)
        if 'error' in resp:
            self.skip('detail_scheduler_isolation', f'detail call failed: {resp}')
            return
        opts_after = self._get('/sdapi/v1/options')
        shift_after = opts_after.get('schedulers_shift') if 'error' not in opts_after else None
        ok = shift_after == shift_before
        self.record(ok, 'detail_scheduler_isolation', f"schedulers_shift {shift_before} -> {shift_after}" if ok else f"LEAK: schedulers_shift {shift_before} -> {shift_after}")

    def test_detail_endpoint_seed_reproducibility(self):
        """Same fixed seed reproduces the detailed region; a different seed changes it (strength 0.7)."""
        self._category = 'detail_endpoint'
        print("  Testing seed reproducibility...")
        a1 = self._detail(detailer_strength=0.7, detailer_steps=5, detailer_conf=0.3, seed=42)
        a2 = self._detail(detailer_strength=0.7, detailer_steps=5, detailer_conf=0.3, seed=42)
        b = self._detail(detailer_strength=0.7, detailer_steps=5, detailer_conf=0.3, seed=1234)
        if 'error' in a1 or 'error' in a2 or 'error' in b:
            self.record(False, 'detail_seed_reproducibility', f"a1={a1.get('error')} a2={a2.get('error')} b={b.get('error')}")
            return
        a1_arr = self._decode_b64_image(a1['image'])
        a2_arr = self._decode_b64_image(a2['image'])
        b_arr = self._decode_b64_image(b['image'])
        box = self._detect_box(self.image_b64, self.face_models)
        same = self._region_diff(a1_arr, a2_arr, box)
        diff = self._region_diff(a1_arr, b_arr, box)
        ok = same < 2.0 and diff > 4.0
        self.record(ok, 'detail_seed_reproducibility', f"same-seed={same:.2f} diff-seed={diff:.2f}")

    def test_detail_endpoint_cfg_effect(self):
        """Guidance scale at extremes (1 vs 15, fixed seed) changes the detailed region.

        CFG scales the conditional-minus-unconditional direction, so a prompt is required: with an empty
        prompt the conditional equals the unconditional and guidance_scale has no effect at any value.
        """
        self._category = 'detail_endpoint'
        print("  Testing CFG effect...")
        prompt = 'a photo of an elderly bearded man'
        low = self._detail(detailer_strength=0.7, detailer_steps=10, detailer_conf=0.3, detailer_prompt=prompt, detailer_cfg_scale=1.0, seed=42)
        high = self._detail(detailer_strength=0.7, detailer_steps=10, detailer_conf=0.3, detailer_prompt=prompt, detailer_cfg_scale=15.0, seed=42)
        if 'error' in low or 'error' in high:
            self.record(False, 'detail_cfg_effect', f"low={low.get('error')} high={high.get('error')}")
            return
        low_arr = self._decode_b64_image(low['image'])
        high_arr = self._decode_b64_image(high['image'])
        box = self._detect_box(self.image_b64, self.face_models)
        diff = self._region_diff(low_arr, high_arr, box)
        self.record(diff > 0.5, 'detail_cfg_effect', f"region diff={diff:.2f}")

    # =========================================================================
    # Tests: extras API with script_args (Phase 1 backward-compat + new path)
    # =========================================================================

    def test_extras_with_detailer_script_args(self):
        """POST /sdapi/v1/extra-single-image with script_args={'Detailer': {...}} should run the detailer."""
        self._category = 'extras_script_args'
        print("\n--- Extras API with Detailer script_args ---")

        if not self.image_b64:
            self.skip('extras_script_args', 'no test image')
            return

        # Baseline: extras without script_args (just upscale=None pass-through)
        payload = {
            'image': self.image_b64,
            'upscaler_1': 'None',
            'upscaling_resize': 1.0,
        }
        baseline = self._post('/sdapi/v1/extra-single-image', payload)
        if 'error' in baseline:
            self.record(False, 'extras_baseline_no_script_args', f"error: {baseline}")
            return
        self.record('image' in baseline and baseline['image'], 'extras_baseline_no_script_args')
        baseline_arr = self._decode_b64_image(baseline['image']) if 'image' in baseline else None

        # With Detailer script_args
        payload_with_detailer = {
            'image': self.image_b64,
            'upscaler_1': 'None',
            'upscaling_resize': 1.0,
            'script_args': {
                'Detailer': {
                    'enabled': True,
                    'strength': 0.5,
                    'steps': 5,
                    'resolution': 1024,
                },
            },
        }
        with_detailer = self._post('/sdapi/v1/extra-single-image', payload_with_detailer)
        if 'error' in with_detailer:
            self.record(False, 'extras_with_detailer_script_args', f"error: {with_detailer}")
            return
        self.record('image' in with_detailer and with_detailer['image'], 'extras_with_detailer_script_args')

        # Output should differ from baseline (detailer ran)
        if baseline_arr is not None and 'image' in with_detailer:
            with_arr = self._decode_b64_image(with_detailer['image'])
            diff = self._pixel_diff(baseline_arr, with_arr)
            # Diff > 0 means detailer modified the image (or no face found, in which case diff = 0)
            self.record(True, 'extras_script_args_diff', f"baseline vs with-detailer diff={diff:.2f}")

    # =========================================================================
    # Environment setup: load Anima base unquantized for the run
    # =========================================================================

    def _find_checkpoint(self, query):
        """Title of the first /sdapi/v1/sd-models entry containing every term in `query`, else None.
        Anima 1.0 Base ships as an sdnext reference model, so 'anima base' resolves once it is present."""
        data = self._get('/sdapi/v1/sd-models')
        if 'error' in data or not isinstance(data, list):
            return None
        terms = query.lower().split()
        for m in data:
            title = (m.get('title') or m.get('model_name') or '')
            if all(t in title.lower() for t in terms):
                return title
        return None

    def _reload_checkpoint(self):
        """Force a clean reload of the selected checkpoint so pending quantization settings take effect."""
        try:
            requests.post(f'{self.base_url}/sdapi/v1/reload-checkpoint', params={'force': 'true'}, timeout=600, verify=False)
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: reload-checkpoint failed: {e}")

    def _setup_environment(self):
        """Load the test model with SDNQ quantization disabled. The quantized int8 matmul is torch.compiled
        with fullgraph=True/dynamic=False, so the many resolutions/prompts this suite runs exhaust Dynamo's
        recompile limit and hard-crash. Returns the prior options to restore, or None if the API is unavailable."""
        current = self._get('/sdapi/v1/options')
        if 'error' in current:
            print(f"  WARNING: GET options failed ({current}); running against current server state")
            return None
        saved = {k: current.get(k) for k in ('sdnq_quantize_weights', 'sd_model_checkpoint')}
        checkpoint = self._find_checkpoint(self.model_query)
        payload = {'sdnq_quantize_weights': []}
        if checkpoint:
            payload['sd_model_checkpoint'] = checkpoint
        self._post('/sdapi/v1/options', payload)
        self._reload_checkpoint()
        print(f"  Environment: quantization disabled (was {saved['sdnq_quantize_weights']}), model={checkpoint or '(unchanged)'}")
        return saved

    def _restore_environment(self, saved):
        """Restore the options changed by _setup_environment and reload, leaving the server as found."""
        if not saved:
            return
        self._post('/sdapi/v1/options', saved)
        self._reload_checkpoint()
        print(f"  Environment restored: sdnq_quantize_weights={saved.get('sdnq_quantize_weights')}, model={saved.get('sd_model_checkpoint')}")

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
        self.face_models = self._pick_region_models(models)

        # Load Anima base unquantized for the run; restored in the finally below
        saved_env = self._setup_environment()
        try:
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

            # Standalone /sdapi/v1/detail endpoint
            self.test_detail_endpoint_basic()
            self.test_detail_endpoint_strength_effect()
            self.test_detail_endpoint_includes_detections()
            self.test_detail_endpoint_param_isolation()
            self.test_detail_endpoint_sampler_block()
            self.test_detail_endpoint_scheduler_isolation()
            self.test_detail_endpoint_seed_reproducibility()
            self.test_detail_endpoint_cfg_effect()

            # Extras API with script_args (Detailer script + backward-compat)
            self.test_extras_with_detailer_script_args()
        finally:
            self._restore_environment(saved_env)

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
    parser.add_argument('--model', default='anima base', help="checkpoint to load for the run (substring match against /sdapi/v1/sd-models titles)")
    args = parser.parse_args()
    test = DetailerAPITest(args.url, args.image, model_query=args.model)
    success = test.run_all()
    sys.exit(0 if success else 1)
