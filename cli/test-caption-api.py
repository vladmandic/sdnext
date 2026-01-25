#!/usr/bin/env python
"""
Caption API Test Suite

Comprehensive tests for all Caption API endpoints and parameters:
- GET/POST /sdapi/v1/interrogate (OpenCLiP/DeepBooru)
- POST /sdapi/v1/vqa (VLM Captioning with annotated images)
- GET /sdapi/v1/vqa/models, /sdapi/v1/vqa/prompts
- POST /sdapi/v1/tagger
- GET /sdapi/v1/tagger/models

Usage:
    python cli/test-caption-api.py [--url URL] [--image PATH]

Examples:
    # Test against local server with default test image
    python cli/test-caption-api.py

    # Test against custom URL with specific image
    python cli/test-caption-api.py --url http://127.0.0.1:7860 --image html/sdnext-robot-2k.jpg
"""

import os
import sys
import time
import base64
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Default test images (in order of preference)
DEFAULT_TEST_IMAGES = [
    'html/sdnext-robot-2k.jpg',
    'html/favicon.png',
    'extensions-builtin/sdnext-modernui/html/logo.png',
]


class CaptionAPITest:
    """Test harness for Caption API endpoints."""

    def __init__(self, base_url, image_path=None, username=None, password=None):
        self.base_url = base_url.rstrip('/')
        self.image_path = image_path
        self.image_b64 = None
        self.results = {'passed': [], 'failed': [], 'skipped': []}
        self.auth = None
        if username and password:
            self.auth = (username, password)
        # Cache for model lists to avoid repeated calls
        self._interrogate_models = None
        self._vqa_models = None
        self._tagger_models = None

    def log_pass(self, msg):
        print(f"  [PASS] {msg}")
        self.results['passed'].append(msg)

    def log_fail(self, msg):
        print(f"  [FAIL] {msg}")
        self.results['failed'].append(msg)

    def log_skip(self, msg):
        print(f"  [SKIP] {msg}")
        self.results['skipped'].append(msg)

    def log_info(self, msg):
        print(f"  [INFO] {msg}")

    # =========================================================================
    # HTTP Helpers
    # =========================================================================
    def get(self, endpoint, params=None):
        """Make GET request and return JSON response."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, params=params, auth=self.auth, timeout=120, verify=False)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {'error': 'timeout', 'reason': 'Request timed out'}
        except requests.exceptions.HTTPError as e:
            try:
                return {'error': 'http', 'status': e.response.status_code, 'reason': e.response.json().get('detail', str(e))}
            except Exception:
                return {'error': 'http', 'status': e.response.status_code, 'reason': str(e)}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    def post(self, endpoint, json_data):
        """Make POST request and return JSON response."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.post(url, json=json_data, auth=self.auth, timeout=120, verify=False)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {'error': 'timeout', 'reason': 'Request timed out'}
        except requests.exceptions.HTTPError as e:
            try:
                return {'error': 'http', 'status': e.response.status_code, 'reason': e.response.json().get('detail', str(e))}
            except Exception:
                return {'error': 'http', 'status': e.response.status_code, 'reason': str(e)}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    # =========================================================================
    # Setup and Teardown
    # =========================================================================
    def setup(self):
        """Load test image and verify server connectivity."""
        print("=" * 70)
        print("CAPTION API TEST SUITE")
        print("=" * 70)
        print(f"\nServer: {self.base_url}")

        # Check server connectivity
        print("\nChecking server connectivity...")
        try:
            resp = requests.get(f"{self.base_url}/sdapi/v1/options", auth=self.auth, timeout=10, verify=False)
            if resp.status_code == 200:
                print("  Server is reachable")
            else:
                print(f"  Warning: Server returned status {resp.status_code}")
        except Exception as e:
            print(f"  ERROR: Cannot connect to server: {e}")
            print("  Make sure the server is running with --docs flag")
            return False

        # Find and load test image
        if self.image_path:
            if os.path.exists(self.image_path):
                print(f"\nUsing provided image: {self.image_path}")
            else:
                print(f"\nERROR: Provided image not found: {self.image_path}")
                return False
        else:
            # Find default test image
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for img in DEFAULT_TEST_IMAGES:
                full_path = os.path.join(script_dir, img)
                if os.path.exists(full_path):
                    self.image_path = full_path
                    print(f"\nUsing default test image: {img}")
                    break
            if not self.image_path:
                print("\nERROR: No test image found")
                return False

        # Load and encode image
        try:
            with open(self.image_path, 'rb') as f:
                image_data = f.read()
            self.image_b64 = base64.b64encode(image_data).decode('utf-8')
            print(f"  Image loaded: {len(image_data)} bytes")
        except Exception as e:
            print(f"  ERROR: Failed to load image: {e}")
            return False

        return True

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        print(f"\n  PASSED:  {len(self.results['passed'])}")
        for item in self.results['passed']:
            print(f"    - {item}")

        print(f"\n  FAILED:  {len(self.results['failed'])}")
        for item in self.results['failed']:
            print(f"    - {item}")

        print(f"\n  SKIPPED: {len(self.results['skipped'])}")
        for item in self.results['skipped']:
            print(f"    - {item}")

        total = len(self.results['passed']) + len(self.results['failed'])
        if total > 0:
            success_rate = len(self.results['passed']) / total * 100
            print(f"\n  SUCCESS RATE: {success_rate:.1f}% ({len(self.results['passed'])}/{total})")

        print("\n" + "=" * 70)

    # =========================================================================
    # TEST: GET /sdapi/v1/interrogate - List Models
    # =========================================================================
    def test_interrogate_list_models(self):
        """Test GET /sdapi/v1/interrogate returns model list."""
        print("\n" + "=" * 70)
        print("TEST: GET /sdapi/v1/interrogate")
        print("=" * 70)

        data = self.get('/sdapi/v1/interrogate')

        # Test 1: Returns list
        if 'error' in data:
            self.log_fail(f"Request failed: {data.get('reason', data)}")
            return

        if isinstance(data, list):
            self.log_pass(f"Returns list with {len(data)} models")
            self._interrogate_models = data
        else:
            self.log_fail(f"Expected list, got {type(data)}")
            return

        # Test 2: Contains deepdanbooru
        if 'deepdanbooru' in data:
            self.log_pass("Contains 'deepdanbooru'")
        else:
            self.log_fail("Missing 'deepdanbooru'")

        # Test 3: Contains OpenCLIP models (format: arch/dataset)
        clip_models = [m for m in data if '/' in m]
        if clip_models:
            self.log_pass(f"Contains {len(clip_models)} OpenCLIP models")
            self.log_info(f"Examples: {clip_models[:3]}")
        else:
            self.log_skip("No OpenCLIP models found (may need to download)")

    # =========================================================================
    # TEST: POST /sdapi/v1/interrogate - DeepBooru
    # =========================================================================
    def test_interrogate_post_deepbooru(self):
        """Test DeepBooru interrogation."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/interrogate (DeepBooru)")
        print("=" * 70)

        t0 = time.time()
        data = self.post('/sdapi/v1/interrogate', {
            'image': self.image_b64,
            'model': 'deepdanbooru'
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"DeepBooru: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        if data.get('caption'):
            caption_preview = data['caption'][:80] + '...' if len(data['caption']) > 80 else data['caption']
            self.log_pass(f"DeepBooru returns caption ({elapsed:.1f}s)")
            self.log_info(f"Caption: {caption_preview}")
        else:
            self.log_fail("DeepBooru returned empty caption")

    # =========================================================================
    # TEST: POST /sdapi/v1/interrogate - OpenCLIP Modes
    # =========================================================================
    def test_interrogate_post_clip_modes(self):
        """Test all 5 interrogation modes."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/interrogate (modes)")
        print("=" * 70)

        # Check if we have OpenCLIP models
        if not self._interrogate_models:
            self._interrogate_models = self.get('/sdapi/v1/interrogate')
        clip_models = [m for m in self._interrogate_models if '/' in m] if isinstance(self._interrogate_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]
        self.log_info(f"Using model: {model}")

        modes = ['best', 'fast', 'classic', 'caption', 'negative']
        for mode in modes:
            t0 = time.time()
            data = self.post('/sdapi/v1/interrogate', {
                'image': self.image_b64,
                'model': model,
                'mode': mode
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"mode='{mode}': {data.get('reason', 'failed')}")
            elif data.get('caption'):
                self.log_pass(f"mode='{mode}' returns caption ({len(data['caption'])} chars, {elapsed:.1f}s)")
            else:
                self.log_fail(f"mode='{mode}' returned empty caption")

    # =========================================================================
    # TEST: POST /sdapi/v1/interrogate - Analyze
    # =========================================================================
    def test_interrogate_analyze(self):
        """Test analyze=True returns breakdown fields."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/interrogate (analyze)")
        print("=" * 70)

        # Check if we have OpenCLIP models
        if not self._interrogate_models:
            self._interrogate_models = self.get('/sdapi/v1/interrogate')
        clip_models = [m for m in self._interrogate_models if '/' in m] if isinstance(self._interrogate_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        # Test without analyze
        data_no_analyze = self.post('/sdapi/v1/interrogate', {
            'image': self.image_b64,
            'model': model,
            'analyze': False
        })

        # Test with analyze
        data_analyze = self.post('/sdapi/v1/interrogate', {
            'image': self.image_b64,
            'model': model,
            'analyze': True
        })

        if 'error' in data_analyze:
            self.log_skip(f"Analyze test: {data_analyze.get('reason', 'failed')}")
            return

        # Verify analyze fields present
        analyze_fields = ['medium', 'artist', 'movement', 'trending', 'flavor']
        fields_found = 0
        for field in analyze_fields:
            if data_analyze.get(field):
                self.log_pass(f"analyze=True returns '{field}'")
                fields_found += 1
            else:
                self.log_skip(f"'{field}' empty or missing (may be image-dependent)")

        if fields_found == 0:
            self.log_fail("analyze=True returned no breakdown fields")

        # Verify fields absent without analyze
        if 'error' not in data_no_analyze:
            absent_in_no_analyze = all(
                data_no_analyze.get(field) is None
                for field in analyze_fields
            )
            if absent_in_no_analyze:
                self.log_pass("analyze=False omits breakdown fields")
            else:
                self.log_fail("analyze=False should not return breakdown fields")

    # =========================================================================
    # TEST: POST /sdapi/v1/interrogate - Invalid Inputs
    # =========================================================================
    def test_interrogate_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/interrogate (invalid inputs)")
        print("=" * 70)

        # Test missing image
        data = self.post('/sdapi/v1/interrogate', {
            'image': '',
            'model': 'ViT-L-14/openai'
        })
        if 'error' in data and data.get('status') == 404:
            self.log_pass("Missing image returns 404")
        else:
            self.log_fail(f"Missing image should return 404, got: {data}")

        # Test invalid model
        data = self.post('/sdapi/v1/interrogate', {
            'image': self.image_b64,
            'model': 'invalid-nonexistent-model'
        })
        if 'error' in data:
            self.log_pass(f"Invalid model returns error: {data.get('status', 'error')}")
        else:
            self.log_fail("Invalid model should return error")

    # =========================================================================
    # TEST: GET /sdapi/v1/vqa/models - VLM Models List
    # =========================================================================
    def test_vqa_models_list(self):
        """Test GET /sdapi/v1/vqa/models returns model details."""
        print("\n" + "=" * 70)
        print("TEST: GET /sdapi/v1/vqa/models")
        print("=" * 70)

        data = self.get('/sdapi/v1/vqa/models')

        if 'error' in data:
            self.log_fail(f"Request failed: {data.get('reason', data)}")
            return

        # Test 1: Returns list
        if isinstance(data, list) and len(data) > 0:
            self.log_pass(f"Returns list with {len(data)} models")
            self._vqa_models = data
        else:
            self.log_fail(f"Expected non-empty list, got {type(data)}")
            return

        # Test 2: Check model structure
        model = data[0]
        required_fields = ['name', 'repo', 'prompts', 'capabilities']
        for field in required_fields:
            if field in model:
                self.log_pass(f"Model has '{field}' field")
            else:
                self.log_fail(f"Model missing '{field}' field")

        # Test 3: Capabilities include expected values
        capabilities_found = set()
        for m in data:
            capabilities_found.update(m.get('capabilities', []))
        expected = ['caption', 'vqa', 'detection', 'ocr', 'thinking']
        for cap in expected:
            if cap in capabilities_found:
                self.log_pass(f"Capability '{cap}' found in models")

        # Log some model names
        model_names = [m['name'] for m in data[:5]]
        self.log_info(f"Sample models: {model_names}")

    # =========================================================================
    # TEST: GET /sdapi/v1/vqa/prompts - VLM Prompts List
    # =========================================================================
    def test_vqa_prompts_list(self):
        """Test GET /sdapi/v1/vqa/prompts returns prompt categories."""
        print("\n" + "=" * 70)
        print("TEST: GET /sdapi/v1/vqa/prompts")
        print("=" * 70)

        # Test without model filter
        data = self.get('/sdapi/v1/vqa/prompts')

        if 'error' in data:
            self.log_fail(f"Request failed: {data.get('reason', data)}")
            return

        # Verify categories
        expected_categories = ['common', 'florence', 'moondream']
        for cat in expected_categories:
            if cat in data and isinstance(data[cat], list):
                self.log_pass(f"Has '{cat}' category with {len(data[cat])} prompts")
            else:
                self.log_skip(f"Category '{cat}' missing or empty")

        # Test with model filter
        if self._vqa_models and len(self._vqa_models) > 0:
            model_name = self._vqa_models[0]['name']
            data_filtered = self.get('/sdapi/v1/vqa/prompts', params={'model': model_name})
            if 'available' in data_filtered:
                self.log_pass(f"Model filter returns 'available' prompts for '{model_name}'")
            else:
                self.log_fail("Model filter should return 'available' field")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Basic Caption
    # =========================================================================
    def test_vqa_caption_basic(self):
        """Test basic VQA captioning."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (basic)")
        print("=" * 70)

        t0 = time.time()
        data = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image'
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"VQA: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        if data.get('answer'):
            answer_preview = data['answer'][:100] + '...' if len(data['answer']) > 100 else data['answer']
            self.log_pass(f"VQA returns answer ({elapsed:.1f}s)")
            self.log_info(f"Answer: {answer_preview}")
        else:
            self.log_fail("VQA returned empty answer")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Different Prompts
    # =========================================================================
    def test_vqa_different_prompts(self):
        """Test different VQA prompts."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (prompts)")
        print("=" * 70)

        prompts = ['Short Caption', 'Normal Caption', 'Long Caption']
        for prompt in prompts:
            t0 = time.time()
            data = self.post('/sdapi/v1/vqa', {
                'image': self.image_b64,
                'question': prompt
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"prompt='{prompt}': {data.get('reason', 'failed')}")
            elif data.get('answer'):
                self.log_pass(f"prompt='{prompt}' returns answer ({len(data['answer'])} chars, {elapsed:.1f}s)")
            else:
                self.log_fail(f"prompt='{prompt}' returned empty answer")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Annotated Image
    # =========================================================================
    def test_vqa_annotated_image(self):
        """Test include_annotated=True returns annotated image for detection."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (annotated image)")
        print("=" * 70)

        # Find a Florence model for detection
        florence_model = None
        if self._vqa_models:
            for m in self._vqa_models:
                if 'florence' in m['name'].lower():
                    florence_model = m['name']
                    break

        if not florence_model:
            florence_model = 'Microsoft Florence 2 Base'  # Default
            self.log_info(f"Using default model: {florence_model}")

        # Test without annotation
        data_no_annot = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'model': florence_model,
            'question': '<OD>',
            'include_annotated': False
        })

        # Test with annotation
        t0 = time.time()
        data_annot = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'model': florence_model,
            'question': '<OD>',
            'include_annotated': True
        })
        elapsed = time.time() - t0

        if 'error' in data_annot:
            self.log_skip(f"Detection test: {data_annot.get('reason', 'failed')} (model may not be loaded)")
            return

        # Verify answer present
        if data_annot.get('answer'):
            self.log_pass(f"Detection task returns answer ({elapsed:.1f}s)")
        else:
            self.log_skip("Detection may not work without model loaded")
            return

        # Verify annotated_image field
        if data_annot.get('annotated_image'):
            # Verify it's valid base64
            try:
                img_data = base64.b64decode(data_annot['annotated_image'])
                if len(img_data) > 1000:  # Reasonable image size
                    self.log_pass(f"annotated_image returned ({len(img_data)} bytes)")
                else:
                    self.log_fail("annotated_image too small")
            except Exception as e:
                self.log_fail(f"annotated_image invalid base64: {e}")
        else:
            self.log_skip("annotated_image empty (may need detections in image)")

        # Verify absent when not requested
        if 'error' not in data_no_annot:
            if data_no_annot.get('annotated_image') is None:
                self.log_pass("include_annotated=False omits annotated_image")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - System Prompt
    # =========================================================================
    def test_vqa_system_prompt(self):
        """Test custom system prompt."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (system prompt)")
        print("=" * 70)

        # Test with custom system prompt
        custom_system = "You are a concise assistant. Reply with only 5 words maximum."
        t0 = time.time()
        data = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'system': custom_system
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"System prompt test: {data.get('reason', 'failed')}")
            return

        if data.get('answer'):
            self.log_pass(f"Custom system prompt accepted ({elapsed:.1f}s)")
            self.log_info(f"Answer: {data['answer'][:100]}")
        else:
            self.log_fail("Custom system prompt returned empty answer")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Invalid Inputs
    # =========================================================================
    def test_vqa_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (invalid inputs)")
        print("=" * 70)

        # Test missing image
        data = self.post('/sdapi/v1/vqa', {
            'image': '',
            'question': 'describe'
        })
        if 'error' in data and data.get('status') == 404:
            self.log_pass("Missing image returns 404")
        else:
            self.log_fail(f"Missing image should return 404, got: {data}")

    # =========================================================================
    # TEST: GET /sdapi/v1/tagger/models - Tagger Models List
    # =========================================================================
    def test_tagger_models_list(self):
        """Test GET /sdapi/v1/tagger/models returns model list."""
        print("\n" + "=" * 70)
        print("TEST: GET /sdapi/v1/tagger/models")
        print("=" * 70)

        data = self.get('/sdapi/v1/tagger/models')

        if 'error' in data:
            self.log_fail(f"Request failed: {data.get('reason', data)}")
            return

        # Test 1: Returns list
        if isinstance(data, list) and len(data) > 0:
            self.log_pass(f"Returns list with {len(data)} models")
            self._tagger_models = data
        else:
            self.log_fail(f"Expected non-empty list, got {type(data)}")
            return

        # Test 2: Check model structure
        model = data[0]
        if 'name' in model and 'type' in model:
            self.log_pass("Models have 'name' and 'type' fields")
        else:
            self.log_fail("Models missing required fields")

        # Test 3: Contains deepbooru
        has_deepbooru = any(m.get('name') == 'deepbooru' or m.get('type') == 'deepbooru' for m in data)
        if has_deepbooru:
            self.log_pass("Contains DeepBooru model")
        else:
            self.log_fail("Missing DeepBooru model")

        # Test 4: Contains WaifuDiffusion models
        wd_models = [m for m in data if m.get('type') == 'waifudiffusion']
        if wd_models:
            self.log_pass(f"Contains {len(wd_models)} WaifuDiffusion models")
            self.log_info(f"Models: {[m['name'] for m in wd_models[:3]]}")
        else:
            self.log_skip("No WaifuDiffusion models found")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Basic
    # =========================================================================
    def test_tagger_basic(self):
        """Test basic tagging."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (basic)")
        print("=" * 70)

        t0 = time.time()
        data = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"Tagger: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        if data.get('tags'):
            tags_preview = data['tags'][:80] + '...' if len(data['tags']) > 80 else data['tags']
            tag_count = len(data['tags'].split(', '))
            self.log_pass(f"Returns tags ({tag_count} tags, {elapsed:.1f}s)")
            self.log_info(f"Tags: {tags_preview}")
        else:
            self.log_fail("Tagger returned empty tags")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Threshold
    # =========================================================================
    def test_tagger_threshold(self):
        """Test threshold affects tag count."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (threshold)")
        print("=" * 70)

        data_high = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'threshold': 0.9
        })
        data_low = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'threshold': 0.1
        })

        if 'error' in data_high or 'error' in data_low:
            self.log_skip("Threshold test: model not loaded")
            return

        count_high = len(data_high.get('tags', '').split(', ')) if data_high.get('tags') else 0
        count_low = len(data_low.get('tags', '').split(', ')) if data_low.get('tags') else 0

        if count_low > count_high:
            self.log_pass(f"threshold effect: 0.9={count_high} tags, 0.1={count_low} tags")
        elif count_high == 0 and count_low == 0:
            self.log_skip("No tags returned (model may not be loaded)")
        else:
            self.log_fail(f"threshold no effect: 0.9={count_high}, 0.1={count_low}")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Max Tags
    # =========================================================================
    def test_tagger_max_tags(self):
        """Test max_tags limits output count."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (max_tags)")
        print("=" * 70)

        data_5 = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'max_tags': 5,
            'threshold': 0.1
        })
        data_50 = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'max_tags': 50,
            'threshold': 0.1
        })

        if 'error' in data_5 or 'error' in data_50:
            self.log_skip("max_tags test: model not loaded")
            return

        count_5 = len(data_5.get('tags', '').split(', ')) if data_5.get('tags') else 0
        count_50 = len(data_50.get('tags', '').split(', ')) if data_50.get('tags') else 0

        if count_5 <= 5:
            self.log_pass(f"max_tags=5 limits to {count_5} tags")
        else:
            self.log_fail(f"max_tags=5 returned {count_5} tags (expected <= 5)")

        if count_50 > count_5:
            self.log_pass(f"max_tags=50 returns more tags ({count_50} vs {count_5})")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Sort Alpha
    # =========================================================================
    def test_tagger_sort_alpha(self):
        """Test sort_alpha sorts tags alphabetically."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (sort_alpha)")
        print("=" * 70)

        data_conf = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'sort_alpha': False,
            'max_tags': 20,
            'threshold': 0.1
        })
        data_alpha = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'sort_alpha': True,
            'max_tags': 20,
            'threshold': 0.1
        })

        if 'error' in data_conf or 'error' in data_alpha:
            self.log_skip("sort_alpha test: model not loaded")
            return

        list_alpha = [t.strip() for t in data_alpha.get('tags', '').split(',') if t.strip()]

        if len(list_alpha) < 2:
            self.log_skip("Not enough tags to test sorting")
            return

        is_sorted = list_alpha == sorted(list_alpha, key=str.lower)
        if is_sorted:
            self.log_pass("sort_alpha=True returns alphabetically sorted tags")
        else:
            self.log_fail("sort_alpha=True did not sort tags alphabetically")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Use Spaces
    # =========================================================================
    def test_tagger_use_spaces(self):
        """Test use_spaces converts underscores to spaces."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (use_spaces)")
        print("=" * 70)

        data_under = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'use_spaces': False,
            'max_tags': 20,
            'threshold': 0.1
        })
        data_space = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'use_spaces': True,
            'max_tags': 20,
            'threshold': 0.1
        })

        if 'error' in data_under or 'error' in data_space:
            self.log_skip("use_spaces test: model not loaded")
            return

        tags_under = data_under.get('tags', '')
        tags_space = data_space.get('tags', '')

        self.log_info(f"use_spaces=False: {tags_under[:60]}...")
        self.log_info(f"use_spaces=True:  {tags_space[:60]}...")

        # Check if underscores are converted to spaces
        has_underscore_before = '_' in tags_under
        has_underscore_after = '_' in tags_space.replace(', ', ',')  # ignore comma-space

        if has_underscore_before and not has_underscore_after:
            self.log_pass("use_spaces=True converts underscores to spaces")
        elif not has_underscore_before:
            self.log_skip("No underscores in tags to convert")
        else:
            self.log_fail("use_spaces=True did not convert underscores")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Escape Brackets
    # =========================================================================
    def test_tagger_escape_brackets(self):
        """Test escape_brackets escapes parentheses."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (escape_brackets)")
        print("=" * 70)

        data_escaped = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'escape_brackets': True,
            'max_tags': 50,
            'threshold': 0.1
        })
        data_raw = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'escape_brackets': False,
            'max_tags': 50,
            'threshold': 0.1
        })

        if 'error' in data_escaped or 'error' in data_raw:
            self.log_skip("escape_brackets test: model not loaded")
            return

        tags_escaped = data_escaped.get('tags', '')
        tags_raw = data_raw.get('tags', '')

        self.log_info(f"escape=True:  {tags_escaped[:60]}...")
        self.log_info(f"escape=False: {tags_raw[:60]}...")

        # Check for escaped brackets (\\( or \\))
        has_escaped = '\\(' in tags_escaped or '\\)' in tags_escaped
        has_unescaped = '(' in tags_raw.replace('\\(', '') or ')' in tags_raw.replace('\\)', '')

        if has_escaped:
            self.log_pass("escape_brackets=True escapes parentheses")
        elif has_unescaped:
            self.log_fail("escape_brackets=True did not escape parentheses")
        else:
            self.log_skip("No brackets in tags to escape")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Exclude Tags
    # =========================================================================
    def test_tagger_exclude_tags(self):
        """Test exclude_tags removes specified tags."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (exclude_tags)")
        print("=" * 70)

        data_all = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'max_tags': 50,
            'threshold': 0.1,
            'exclude_tags': ''
        })

        if 'error' in data_all:
            self.log_skip("exclude_tags test: model not loaded")
            return

        tag_list = [t.strip().replace(' ', '_') for t in data_all.get('tags', '').split(',') if t.strip()]

        if len(tag_list) < 2:
            self.log_skip("Not enough tags to test exclusion")
            return

        # Exclude the first tag
        tag_to_exclude = tag_list[0]
        data_filtered = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'max_tags': 50,
            'threshold': 0.1,
            'exclude_tags': tag_to_exclude
        })

        if 'error' in data_filtered:
            self.log_skip("exclude_tags filtered request failed")
            return

        self.log_info(f"Excluding tag: '{tag_to_exclude}'")
        self.log_info(f"Before: {data_all.get('tags', '')[:60]}...")
        self.log_info(f"After:  {data_filtered.get('tags', '')[:60]}...")

        # Check if the tag was removed
        filtered_list = [t.strip().replace(' ', '_') for t in data_filtered.get('tags', '').split(',') if t.strip()]
        tag_space_variant = tag_to_exclude.replace('_', ' ')
        tag_present = tag_to_exclude in filtered_list or tag_space_variant in [t.strip() for t in data_filtered.get('tags', '').split(',')]

        if not tag_present:
            self.log_pass(f"exclude_tags removes '{tag_to_exclude}'")
        else:
            self.log_fail(f"exclude_tags did not remove '{tag_to_exclude}'")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Show Scores
    # =========================================================================
    def test_tagger_show_scores(self):
        """Test show_scores returns scores dict."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (show_scores)")
        print("=" * 70)

        data_no_scores = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'show_scores': False,
            'max_tags': 5
        })
        data_scores = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'show_scores': True,
            'max_tags': 5
        })

        if 'error' in data_no_scores or 'error' in data_scores:
            self.log_skip("show_scores test: model not loaded")
            return

        # Check scores dict is returned
        if 'scores' in data_scores and isinstance(data_scores['scores'], dict) and len(data_scores['scores']) > 0:
            self.log_pass(f"show_scores=True returns scores dict with {len(data_scores['scores'])} entries")

            # Verify scores are floats 0-1
            scores = list(data_scores['scores'].values())
            if all(isinstance(s, (int, float)) and 0 <= s <= 1 for s in scores):
                self.log_pass("All scores are floats in 0-1 range")
            else:
                self.log_fail(f"Some scores out of range: {scores}")
        else:
            self.log_fail("show_scores=True did not return scores dict")

        # Check tags contain scores
        tags_with_scores = data_scores.get('tags', '')
        if ':' in tags_with_scores:
            self.log_pass("Tags string includes score notation")
        else:
            self.log_skip("Tags string does not include inline scores")

        # Check scores absent without flag
        if data_no_scores.get('scores') is None:
            self.log_pass("show_scores=False omits scores dict")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Include Rating
    # =========================================================================
    def test_tagger_include_rating(self):
        """Test include_rating adds rating tags."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (include_rating)")
        print("=" * 70)

        data_no_rating = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'include_rating': False,
            'max_tags': 100,
            'threshold': 0.01
        })
        data_rating = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'include_rating': True,
            'max_tags': 100,
            'threshold': 0.01
        })

        if 'error' in data_no_rating or 'error' in data_rating:
            self.log_skip("include_rating test: model not loaded")
            return

        tags_no_rating = data_no_rating.get('tags', '').lower()
        tags_rating = data_rating.get('tags', '').lower()

        self.log_info(f"include_rating=False: {tags_no_rating[:60]}...")
        self.log_info(f"include_rating=True:  {tags_rating[:60]}...")

        # Rating tags typically are like "safe", "questionable", "explicit", "general", "sensitive"
        rating_keywords = ['rating:', 'safe', 'questionable', 'explicit', 'general', 'sensitive']

        has_rating_before = any(kw in tags_no_rating for kw in rating_keywords)
        has_rating_after = any(kw in tags_rating for kw in rating_keywords)

        if has_rating_after and not has_rating_before:
            self.log_pass("include_rating=True adds rating tags")
        elif has_rating_after and has_rating_before:
            self.log_skip("Rating tags appear in both (threshold may be very low)")
        elif not has_rating_after:
            self.log_skip("No rating tags detected")
        else:
            self.log_fail("include_rating did not work as expected")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Invalid Inputs
    # =========================================================================
    def test_tagger_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (invalid inputs)")
        print("=" * 70)

        # Test missing image
        data = self.post('/sdapi/v1/tagger', {
            'image': ''
        })
        if 'error' in data and data.get('status') == 404:
            self.log_pass("Missing image returns 404")
        else:
            self.log_fail(f"Missing image should return 404, got: {data}")

    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run_all_tests(self):
        """Run all tests."""
        if not self.setup():
            return False

        # Interrogate tests
        self.test_interrogate_list_models()
        self.test_interrogate_post_deepbooru()
        self.test_interrogate_post_clip_modes()
        self.test_interrogate_analyze()
        self.test_interrogate_invalid_inputs()

        # VQA tests
        self.test_vqa_models_list()
        self.test_vqa_prompts_list()
        self.test_vqa_caption_basic()
        self.test_vqa_different_prompts()
        self.test_vqa_annotated_image()
        self.test_vqa_system_prompt()
        self.test_vqa_invalid_inputs()

        # Tagger tests
        self.test_tagger_models_list()
        self.test_tagger_basic()
        self.test_tagger_threshold()
        self.test_tagger_max_tags()
        self.test_tagger_sort_alpha()
        self.test_tagger_use_spaces()
        self.test_tagger_escape_brackets()
        self.test_tagger_exclude_tags()
        self.test_tagger_show_scores()
        self.test_tagger_include_rating()
        self.test_tagger_invalid_inputs()

        self.print_summary()

        return len(self.results['failed']) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Caption API Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test against local server with default test image
    python cli/test-caption-api.py

    # Test against custom URL with specific image
    python cli/test-caption-api.py --url http://127.0.0.1:7860 --image html/sdnext-robot-2k.jpg

    # Test with authentication
    python cli/test-caption-api.py --username admin --password secret
        """
    )
    parser.add_argument('--url', default='http://127.0.0.1:7860', help='Server URL (default: http://127.0.0.1:7860)')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--username', help='HTTP Basic Auth username')
    parser.add_argument('--password', help='HTTP Basic Auth password')
    args = parser.parse_args()

    test = CaptionAPITest(
        base_url=args.url,
        image_path=args.image,
        username=args.username,
        password=args.password
    )
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
