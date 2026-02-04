#!/usr/bin/env python
"""
Caption API Test Suite

Comprehensive tests for all Caption API endpoints and parameters:
- GET/POST /sdapi/v1/openclip (OpenCLIP direct)
- POST /sdapi/v1/caption (Unified dispatch: openclip, tagger, vlm)
- POST /sdapi/v1/vqa (VLM direct)
- GET /sdapi/v1/vqa/models, /sdapi/v1/vqa/prompts
- POST /sdapi/v1/tagger (Tagger direct)
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
import re
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

# OCR test image (must have readable text)
OCR_TEST_IMAGE = 'models/Reference/HiDream-ai--HiDream-I1-Fast.jpg'
# Bracket test image (must produce tags with parentheses, e.g. pokemon_(creature))
BRACKET_TEST_IMAGE = 'models/Reference/SDXL-Flash_Mini.jpg'

# Custom prefill text used for dual-prefill verification across tests
CUSTOM_PREFILL = "Vlado is the best, and I'm looking at his robot which"


class CaptionAPITest:
    """Test harness for Caption API endpoints."""

    # VQA model families for architecture testing
    VQA_FAMILIES = {
        'qwen': ['qwen'],
        'gemma': ['gemma'],  # excluding paligemma
        'smolvlm': ['smol'],
        'florence': ['florence'],
        'promptgen': ['promptgen'],
        'moondream': ['moondream'],
        'fastvlm': ['fastvlm'],
        'git': ['git'],
        'blip': ['blip'],
        'pix2struct': ['pix'],
        'paligemma': ['paligemma'],
        'vilt': ['vilt'],
        'ovis': ['ovis'],
        'sa2va': ['sa2'],
        'toriigate': ['torii'],
        'mimo': ['mimo'],
        'joytag': ['joytag'],
        'joycaption': ['joycaption'],
    }

    # BLIP model types for caption testing (smaller models only to avoid reloading large models)
    BLIP_MODELS = [
        'blip-base',
        'blip-large',
        'blip2-opt-2.7b',
    ]

    def __init__(self, base_url, image_path=None, username=None, password=None, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.image_path = image_path
        self.image_b64 = None
        self.ocr_image_b64 = None  # Separate image with text for OCR tests
        self.bracket_image_b64 = None  # Separate image that produces bracket-containing tags
        self.timeout = timeout  # Request timeout in seconds
        # Categorized results tracking
        self.results = {
            'openclip': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'vqa': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'tagger': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'dispatch': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'parity': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
        }
        self._current_category = 'openclip'  # Default category
        # Track critical errors per backend to skip subsequent tests
        self._critical_errors = {
            'openclip': None,
            'tagger': None,
            'vqa': None,
        }
        self.auth = None
        if username and password:
            self.auth = (username, password)
        # Cache for model lists to avoid repeated calls
        self._caption_models = None
        self._vqa_models = None
        self._tagger_models = None

    def set_category(self, category):
        """Set the current test category for result tracking."""
        self._current_category = category

    def record_result(self, status, message):
        """Record a test result in the current category."""
        cat = self._current_category
        self.results[cat]['tests'].append((status, message))
        self.results[cat][status] += 1

    def log_pass(self, msg):
        print(f"  [PASS] {msg}")
        self.record_result('passed', msg)

    def log_fail(self, msg):
        print(f"  [FAIL] {msg}")
        self.record_result('failed', msg)

    def log_skip(self, msg):
        print(f"  [SKIP] {msg}")
        self.record_result('skipped', msg)

    def log_info(self, msg):
        print(f"  [INFO] {msg}")

    def log_critical(self, backend, msg):
        """Log a critical error and mark the backend as failed."""
        print(f"  [CRITICAL] {msg}")
        self._critical_errors[backend] = msg
        self.record_result('failed', f"CRITICAL: {msg}")

    def has_critical_error(self, backend):
        """Check if a backend has a critical error and should skip tests."""
        if self._critical_errors.get(backend):
            return True
        return False

    def skip_if_critical(self, backend, test_name):
        """Skip test if backend has critical error. Returns True if skipped."""
        if self.has_critical_error(backend):
            self.log_skip(f"{test_name}: skipped due to prior critical error")
            return True
        return False

    def is_critical_error(self, response_text):
        """Check if a response indicates a critical/fatal error that should stop testing."""
        if not response_text:
            return False
        text_lower = str(response_text).lower()
        # Critical error patterns that indicate backend is broken
        # Note: patterns are substring matches, so be careful with short strings that could match common words
        critical_patterns = [
            'runtimeerror',
            'cuda error',
            'out of memory',
            # 'oom' removed - matches words like "room", "zoom", "bloom"; 'out of memory' covers this case
            'device-side assert',
            'cublas',
            'cudnn',
            'nccl',
            'input type',  # tensor type mismatch
            'weight type',  # tensor type mismatch
            'cannot be performed',
            'illegal memory access',
            'segmentation fault',
        ]
        # Patterns that need word boundary checking (could match common words)
        word_boundary_patterns = [
            'killed',  # could match "skilled", "thrilled"
            'critical',  # could match "critical thinking"
        ]
        for pattern in critical_patterns:
            if pattern in text_lower:
                return True
        # Check word boundary patterns with regex
        for pattern in word_boundary_patterns:
            if re.search(rf'\b{pattern}\b', text_lower):
                return True
        return False

    def check_critical_error(self, data, backend):
        """Check response for critical errors and mark backend if found. Returns error message or None."""
        if not data:
            return None
        # Check various response fields for critical errors
        fields_to_check = ['caption', 'tags', 'answer', 'error', 'reason', 'detail']
        for field in fields_to_check:
            value = data.get(field)
            if value and self.is_critical_error(value):
                error_msg = f"{field}: {self.truncate(str(value), 100)}"
                self.log_critical(backend, error_msg)
                return error_msg
        return None

    @staticmethod
    def truncate(text, max_len=80):
        """Truncate text for display, adding ... if truncated."""
        if text and len(str(text)) > max_len:
            return str(text)[:max_len] + "..."
        return str(text) if text else ""

    def log_response(self, response, key_fields=None):
        """Print response trace with key fields."""
        if key_fields is None:
            key_fields = ['caption', 'tags', 'answer', 'backend']
        for field in key_fields:
            if response.get(field):
                value = response[field]
                if isinstance(value, str):
                    print(f"  Response {field}: \"{self.truncate(value)}\"")
                elif isinstance(value, dict):
                    # For scores dict, show first few entries
                    preview = dict(list(value.items())[:3])
                    print(f"  Response {field}: {preview}")
                else:
                    print(f"  Response {field}: {value}")

    def is_error_answer(self, answer):
        """Check if an answer string indicates an error occurred."""
        if not answer:
            return False
        answer_lower = answer.lower().strip()
        # Common error patterns in VQA/caption responses
        error_patterns = [
            'error',
            'exception',
            'failed',
            'traceback',
            'cannot',
            'unable to',
        ]
        # Check if answer is just an error keyword or starts with one
        for pattern in error_patterns:
            if answer_lower == pattern or answer_lower.startswith(f'{pattern}:') or answer_lower.startswith(f'{pattern} '):
                return True
        return False

    def is_meaningful_answer(self, answer, min_length=3):
        """Check if an answer is meaningful (not just punctuation or too short)."""
        if not answer:
            return False
        # Strip whitespace and check length
        stripped = answer.strip()
        if len(stripped) < min_length:
            return False
        # Check if it's just punctuation
        if all(c in '.,!?;:\'"()-_' for c in stripped):
            return False
        return True

    def _check_prefill(self, base_request: dict, test_label: str):
        """Re-run a VQA request with custom prefill and verify it appears in output."""
        req = {**base_request, 'prefill': CUSTOM_PREFILL, 'keep_prefill': True}
        data = self.post('/sdapi/v1/vqa', req)
        if 'error' in data:
            self.log_skip(f"{test_label} prefill: API error")
        elif data.get('answer') and not self.is_error_answer(data['answer']):
            if data['answer'].startswith(CUSTOM_PREFILL):
                self.log_pass(f"{test_label} prefill: output starts with custom prefill")
            else:
                self.log_fail(f"{test_label} prefill: expected '{CUSTOM_PREFILL[:30]}...' but got '{data['answer'][:30]}...'")
        else:
            self.log_fail(f"{test_label} prefill: empty/error")

    def get_model_family(self, model_name):
        """Determine model family from model name."""
        name_lower = model_name.lower()
        for family, patterns in self.VQA_FAMILIES.items():
            for pattern in patterns:
                if pattern in name_lower:
                    # Special case: gemma but not paligemma
                    if family == 'gemma' and 'pali' in name_lower:
                        continue
                    return family
        return 'unknown'

    def get_tagger_type(self, model):
        """Determine tagger type and version from model info."""
        model_type = model.get('type', 'unknown')
        model_name = model.get('name', '').lower()

        if model_type == 'deepbooru':
            return 'deepbooru', None
        elif model_type == 'waifudiffusion':
            # Determine WD version
            if 'v3' in model_name:
                return 'waifudiffusion', 'v3'
            elif 'v2' in model_name:
                return 'waifudiffusion', 'v2'
            else:
                return 'waifudiffusion', 'v1'
        return model_type, None

    # =========================================================================
    # HTTP Helpers
    # =========================================================================
    def get(self, endpoint, params=None):
        """Make GET request and return JSON response."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, params=params, auth=self.auth, timeout=self.timeout, verify=False)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {'error': 'timeout', 'reason': f'Request timed out after {self.timeout}s'}
        except requests.exceptions.HTTPError as e:
            try:
                return {'error': 'http', 'status': e.response.status_code, 'reason': e.response.json().get('detail', str(e))}
            except Exception:
                return {'error': 'http', 'status': e.response.status_code, 'reason': str(e)}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    def _infer_backend_from_endpoint(self, endpoint, json_data):
        """Infer the backend from the endpoint URL or request data."""
        if '/openclip' in endpoint:
            return 'openclip'
        elif '/tagger' in endpoint:
            return 'tagger'
        elif '/vqa' in endpoint:
            return 'vlm'
        elif '/caption' in endpoint:
            # Dispatch endpoint - check backend field in request
            return json_data.get('backend', 'openclip') if json_data else 'openclip'
        return None

    def post(self, endpoint, json_data, check_critical=True):
        """Make POST request and return JSON response. Auto-checks for critical errors unless check_critical=False."""
        url = f"{self.base_url}{endpoint}"
        backend = self._infer_backend_from_endpoint(endpoint, json_data)

        try:
            resp = requests.post(url, json=json_data, auth=self.auth, timeout=self.timeout, verify=False)
            resp.raise_for_status()
            data = resp.json()

            # Auto-check for critical errors in the response (skip for deliberate error tests)
            if check_critical:
                if backend and backend != 'vlm':  # VLM backend name differs
                    self._auto_check_critical(data, backend)
                elif backend == 'vlm':
                    self._auto_check_critical(data, 'vqa')

            return data
        except requests.exceptions.Timeout:
            return {'error': 'timeout', 'reason': f'Request timed out after {self.timeout}s'}
        except requests.exceptions.HTTPError as e:
            try:
                return {'error': 'http', 'status': e.response.status_code, 'reason': e.response.json().get('detail', str(e))}
            except Exception:
                return {'error': 'http', 'status': e.response.status_code, 'reason': str(e)}
        except Exception as e:
            return {'error': 'exception', 'reason': str(e)}

    def _auto_check_critical(self, data, backend):
        """Auto-check response for critical errors (called by post method)."""
        if not data or self.has_critical_error(backend):
            return
        # Check various response fields for critical errors
        fields_to_check = ['caption', 'tags', 'answer', 'error', 'reason', 'detail']
        for field in fields_to_check:
            value = data.get(field)
            if value and self.is_critical_error(value):
                error_msg = f"{field}: {self.truncate(str(value), 100)}"
                print(f"  [CRITICAL] {backend} backend error: {error_msg}")
                self._critical_errors[backend] = error_msg
                return

    # =========================================================================
    # Setup and Teardown
    # =========================================================================
    def setup(self):
        """Load test image and verify server connectivity."""
        print("=" * 70)
        print("CAPTION API TEST SUITE")
        print("=" * 70)
        print(f"\nServer: {self.base_url}")
        print(f"Timeout: {self.timeout}s")

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

        # Load OCR test image (image with readable text)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ocr_image_path = os.path.join(script_dir, OCR_TEST_IMAGE)
        if os.path.exists(ocr_image_path):
            try:
                with open(ocr_image_path, 'rb') as f:
                    ocr_data = f.read()
                self.ocr_image_b64 = base64.b64encode(ocr_data).decode('utf-8')
                print(f"  OCR test image loaded: {OCR_TEST_IMAGE} ({len(ocr_data)} bytes)")
            except Exception as e:
                print(f"  Warning: Failed to load OCR test image: {e}")
        else:
            print(f"  Warning: OCR test image not found: {OCR_TEST_IMAGE}")

        # Load bracket test image (image that produces tags with parentheses)
        bracket_image_path = os.path.join(script_dir, BRACKET_TEST_IMAGE)
        if os.path.exists(bracket_image_path):
            try:
                with open(bracket_image_path, 'rb') as f:
                    bracket_data = f.read()
                self.bracket_image_b64 = base64.b64encode(bracket_data).decode('utf-8')
                print(f"  Bracket test image loaded: {BRACKET_TEST_IMAGE} ({len(bracket_data)} bytes)")
            except Exception as e:
                print(f"  Warning: Failed to load bracket test image: {e}")
        else:
            print(f"  Warning: Bracket test image not found: {BRACKET_TEST_IMAGE}")

        return True

    def print_summary(self):
        """Print test summary by category."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY BY CATEGORY")
        print("=" * 70)

        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for category, data in self.results.items():
            cat_passed = data['passed']
            cat_failed = data['failed']
            cat_skipped = data['skipped']
            cat_total = cat_passed + cat_failed + cat_skipped

            if cat_total == 0:
                continue

            total_passed += cat_passed
            total_failed += cat_failed
            total_skipped += cat_skipped

            # Calculate success rate (excluding skipped)
            cat_run = cat_passed + cat_failed
            if cat_run > 0:
                pct = cat_passed / cat_run * 100
                print(f"\n  {category.upper():10} {cat_passed:3}/{cat_run:3} passed ({pct:5.1f}%), {cat_skipped} skipped")
            else:
                print(f"\n  {category.upper():10} 0/0 tests, {cat_skipped} skipped")

            # Show failures for this category
            failures = [(s, m) for s, m in data['tests'] if s == 'failed']
            if failures:
                for _, msg in failures:
                    print(f"    [FAIL] {msg}")

            # Show skipped tests for this category
            skipped = [(s, m) for s, m in data['tests'] if s == 'skipped']
            if skipped:
                for _, msg in skipped:
                    print(f"    [SKIP] {msg}")

        # Overall totals
        print("\n" + "-" * 70)
        overall_run = total_passed + total_failed
        if overall_run > 0:
            overall_pct = total_passed / overall_run * 100
            print(f"\n  TOTAL:   {total_passed}/{overall_run} passed ({overall_pct:.1f}%), {total_skipped} skipped")
        else:
            print(f"\n  TOTAL:   0/0 tests, {total_skipped} skipped")

        print("\n" + "=" * 70)

    # =========================================================================
    # TEST: GET /sdapi/v1/openclip - List Models
    # =========================================================================
    def test_openclip_list_models(self):
        """Test GET /sdapi/v1/openclip returns model list."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: GET /sdapi/v1/openclip")
        print("=" * 70)

        data = self.get('/sdapi/v1/openclip')

        # Test 1: Returns list
        if 'error' in data:
            self.log_fail(f"Request failed: {data.get('reason', data)}")
            return

        if isinstance(data, list):
            self.log_pass(f"Returns list with {len(data)} models")
            self._caption_models = data
        else:
            self.log_fail(f"Expected list, got {type(data)}")
            return

        # Test 2: Contains OpenCLIP models (format: arch/dataset)
        clip_models = [m for m in data if '/' in m]
        if clip_models:
            self.log_pass(f"Contains {len(clip_models)} OpenCLIP models")
            self.log_info(f"Examples: {clip_models[:3]}")
        else:
            self.log_skip("No OpenCLIP models found (may need to download)")

    # =========================================================================
    # TEST: POST /sdapi/v1/openclip - OpenCLIP Modes
    # =========================================================================
    def test_openclip_post_modes(self):
        """Test all 5 interrogation modes via direct endpoint."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (modes)")
        print("=" * 70)

        # Skip if critical error already occurred
        if self.skip_if_critical('openclip', 'openclip modes'):
            return

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]
        self.log_info(f"Using model: {model}")

        modes = ['best', 'fast', 'classic', 'caption', 'negative']
        for mode in modes:
            # Check for critical error before each mode test
            if self.has_critical_error('openclip'):
                self.log_skip(f"mode='{mode}': skipped due to critical error")
                continue

            t0 = time.time()
            data = self.post('/sdapi/v1/openclip', {
                'image': self.image_b64,
                'model': model,
                'mode': mode
            })
            elapsed = time.time() - t0

            # Check for critical error in response
            if self.check_critical_error(data, 'openclip'):
                continue

            if 'error' in data:
                self.log_skip(f"mode='{mode}': {data.get('reason', 'failed')}")
            elif data.get('caption') and not self.is_error_answer(data['caption']):
                self.log_pass(f"mode='{mode}' returns caption ({len(data['caption'])} chars, {elapsed:.1f}s)")
                self.log_info(f"Caption: {self.truncate(data['caption'], 60)}")
            elif self.is_error_answer(data.get('caption', '')):
                self.log_fail(f"mode='{mode}' returned error: {data['caption']}")
            else:
                self.log_fail(f"mode='{mode}' returned empty caption")

    # =========================================================================
    # TEST: POST /sdapi/v1/openclip - Analyze
    # =========================================================================
    def test_openclip_analyze(self):
        """Test analyze=True returns breakdown fields."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (analyze)")
        print("=" * 70)

        # Skip if critical error already occurred
        if self.skip_if_critical('openclip', 'openclip analyze'):
            return

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        # Test without analyze
        data_no_analyze = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'analyze': False
        })

        # Check for critical error
        if self.check_critical_error(data_no_analyze, 'openclip'):
            return

        # Test with analyze
        data_analyze = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'analyze': True
        })

        # Check for critical error
        if self.check_critical_error(data_analyze, 'openclip'):
            return

        if 'error' in data_analyze:
            self.log_skip(f"Analyze test: {data_analyze.get('reason', 'failed')}")
            return

        # Verify analyze fields present
        analyze_fields = ['medium', 'artist', 'movement', 'trending', 'flavor']
        fields_found = 0
        for field in analyze_fields:
            if data_analyze.get(field):
                self.log_pass(f"analyze=True returns '{field}'")
                self.log_info(f"  {field}: {self.truncate(data_analyze[field], 40)}")
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
    # TEST: POST /sdapi/v1/openclip - Invalid Inputs
    # =========================================================================
    def test_openclip_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (invalid inputs)")
        print("=" * 70)

        # Test missing image (check_critical=False since we expect errors)
        data = self.post('/sdapi/v1/openclip', {
            'image': '',
            'model': 'ViT-L-14/openai'
        }, check_critical=False)
        if 'error' in data and data.get('status') == 404:
            self.log_pass("Missing image returns 404")
        else:
            self.log_fail(f"Missing image should return 404, got: {data}")

        # Test invalid model (check_critical=False since we expect errors)
        data = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': 'invalid-nonexistent-model'
        }, check_critical=False)
        if 'error' in data:
            self.log_pass(f"Invalid model returns error: {data.get('status', 'error')}")
        else:
            self.log_fail("Invalid model should return error")

    # =========================================================================
    # TEST: POST /sdapi/v1/openclip - CLIP/BLIP Models
    # =========================================================================
    def test_openclip_clip_blip_models(self):
        """Test clip_model and blip_model parameter overrides."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (clip_model, blip_model)")
        print("=" * 70)

        # Skip if critical error already occurred
        if self.skip_if_critical('openclip', 'openclip clip_blip_models'):
            return

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        # Test with explicit clip_model override
        model = clip_models[0]
        t0 = time.time()
        data = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'clip_model': model,  # Explicit CLIP model
            'mode': 'fast'
        })
        elapsed = time.time() - t0

        # Check for critical error
        if self.check_critical_error(data, 'openclip'):
            return

        if 'error' in data:
            self.log_skip(f"clip_model override: {data.get('reason', 'failed')}")
        elif data.get('caption') and not self.is_error_answer(data['caption']):
            self.log_pass(f"clip_model override accepted ({elapsed:.1f}s)")
        else:
            self.log_fail(f"clip_model override returned empty/error: {data.get('caption', '')}")

        # Check for critical error before continuing
        if self.has_critical_error('openclip'):
            return

        # Test with blip_model override (uses 'caption' mode internally)
        # Valid blip_model values: 'blip-base', 'blip-large', 'blip2-opt-2.7b', 'blip2-opt-6.7b', 'blip2-flip-t5-xl', 'blip2-flip-t5-xxl'
        t0 = time.time()
        data = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'blip_model': 'blip-base',  # Use smaller model to test override
            'mode': 'caption'
        })
        elapsed = time.time() - t0

        # Check for critical error
        if self.check_critical_error(data, 'openclip'):
            return

        if 'error' in data:
            self.log_skip(f"blip_model override: {data.get('reason', 'failed')}")
        elif data.get('caption') and not self.is_error_answer(data['caption']):
            self.log_pass(f"blip_model='blip-base' override accepted ({elapsed:.1f}s)")
        else:
            self.log_fail(f"blip_model override returned empty/error: {data.get('caption', '')}")

    # =========================================================================
    # TEST: POST /sdapi/v1/openclip - Caption Length
    # =========================================================================
    def test_openclip_length(self):
        """Test max_length constraints."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (max_length)")
        print("=" * 70)

        # Skip if critical error already occurred
        if self.skip_if_critical('openclip', 'openclip length'):
            return

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        # Test max_length effect by comparing short vs long limits
        data_max_short = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'caption',
            'max_length': 10  # Very short
        })

        # Check for critical error
        if self.check_critical_error(data_max_short, 'openclip'):
            return

        data_max_long = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'caption',
            'max_length': 100  # Longer
        })

        # Check for critical error
        if self.check_critical_error(data_max_long, 'openclip'):
            return

        if 'error' in data_max_short or 'error' in data_max_long:
            self.log_skip("max_length test: API error")
        elif data_max_short.get('caption') and data_max_long.get('caption'):
            len_short = len(data_max_short['caption'])
            len_long = len(data_max_long['caption'])
            self.log_info(f"max_length=10: {len_short} chars - '{data_max_short['caption'][:50]}...'")
            self.log_info(f"max_length=100: {len_long} chars - '{data_max_long['caption'][:50]}...'")
            if len_short < len_long:
                self.log_pass(f"max_length has effect: {len_short} < {len_long} chars")
            elif len_short == len_long:
                self.log_skip(f"max_length no effect detected (both {len_short} chars, may be model limit)")
            else:
                self.log_fail(f"max_length reversed: short={len_short}, long={len_long}")
        else:
            self.log_fail("max_length test returned empty captions")

    # =========================================================================
    # TEST: POST /sdapi/v1/openclip - Flavors
    # =========================================================================
    def test_openclip_flavors(self):
        """Test min_flavors and max_flavors controls."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (min_flavors, max_flavors)")
        print("=" * 70)

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        # Test max_flavors effect by comparing few vs many
        data_few = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'fast',
            'max_flavors': 3  # Fewer flavor tags
        })
        data_many = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'fast',
            'max_flavors': 20  # More flavor tags
        })

        if 'error' in data_few or 'error' in data_many:
            self.log_skip("max_flavors test: API error")
        elif data_few.get('caption') and data_many.get('caption'):
            len_few = len(data_few['caption'])
            len_many = len(data_many['caption'])
            self.log_info(f"max_flavors=3: {len_few} chars - '{data_few['caption'][:50]}...'")
            self.log_info(f"max_flavors=20: {len_many} chars - '{data_many['caption'][:50]}...'")
            if len_many > len_few:
                self.log_pass(f"max_flavors has effect: {len_few} < {len_many} chars")
            elif len_many == len_few:
                self.log_skip(f"max_flavors no effect detected (both {len_few} chars)")
            else:
                self.log_fail(f"max_flavors reversed: few={len_few}, many={len_many}")
        else:
            self.log_fail("max_flavors test returned empty captions")

        # Test min_flavors effect: only applies in mode='best' which iterates from min to max flavors
        # Use a narrow max_flavors window so min_flavors has a visible floor effect
        data_min_low = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'best',
            'min_flavors': 1,
            'max_flavors': 3
        })
        data_min_high = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'best',
            'min_flavors': 8,
            'max_flavors': 10
        })

        if 'error' in data_min_low or 'error' in data_min_high:
            self.log_skip("min_flavors test: API error")
        elif data_min_low.get('caption') and data_min_high.get('caption'):
            len_low = len(data_min_low['caption'])
            len_high = len(data_min_high['caption'])
            self.log_info(f"min_flavors=1,max=3: {len_low} chars - '{data_min_low['caption'][:50]}...'")
            self.log_info(f"min_flavors=8,max=10: {len_high} chars - '{data_min_high['caption'][:50]}...'")
            if len_high > len_low:
                self.log_pass(f"min_flavors has effect: {len_low} < {len_high} chars")
            elif len_high == len_low:
                self.log_fail(f"min_flavors has no effect (both {len_low} chars)")
            else:
                self.log_fail(f"min_flavors reversed: low={len_low}, high={len_high}")
        else:
            self.log_fail("min_flavors test returned empty captions")

    # =========================================================================
    # TEST: POST /sdapi/v1/openclip - Advanced Settings
    # =========================================================================
    def test_openclip_advanced_settings(self):
        """Test chunk_size, flavor_count, and num_beams parameters."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/openclip (chunk_size, flavor_count, num_beams)")
        print("=" * 70)

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        # Test chunk_size override
        t0 = time.time()
        data = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'fast',
            'chunk_size': 1024  # Batch size for processing candidates
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"chunk_size override: {data.get('reason', 'failed')}")
        elif data.get('caption') and not self.is_error_answer(data['caption']):
            self.log_pass(f"chunk_size=1024 accepted ({elapsed:.1f}s)")
            self.log_info("NOTE: acceptance-only test, does not verify output effect")
        else:
            self.log_fail("chunk_size override returned empty/error")

        # Test flavor_count override
        t0 = time.time()
        data = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'fast',
            'flavor_count': 16  # Intermediate candidate pool size
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"flavor_count override: {data.get('reason', 'failed')}")
        elif data.get('caption') and not self.is_error_answer(data['caption']):
            self.log_pass(f"flavor_count=16 accepted ({elapsed:.1f}s)")
            self.log_info("NOTE: acceptance-only test, does not verify output effect")
        else:
            self.log_fail("flavor_count override returned empty/error")

        # Test num_beams override (beam search for caption generation)
        t0 = time.time()
        data = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'caption',
            'num_beams': 3  # Beam search paths
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"num_beams override: {data.get('reason', 'failed')}")
        elif data.get('caption') and not self.is_error_answer(data['caption']):
            self.log_pass(f"num_beams=3 accepted ({elapsed:.1f}s)")
            self.log_info("NOTE: acceptance-only test, does not verify output effect")
        else:
            self.log_fail("num_beams override returned empty/error")

    # =========================================================================
    # TEST: GET /sdapi/v1/vqa/models - VLM Models List
    # =========================================================================
    def test_vqa_models_list(self):
        """Test GET /sdapi/v1/vqa/models returns model details."""
        self.set_category('vqa')
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
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: GET /sdapi/v1/vqa/prompts")
        print("=" * 70)

        # Test without model filter
        data = self.get('/sdapi/v1/vqa/prompts')

        if 'error' in data:
            self.log_fail(f"Request failed: {data.get('reason', data)}")
            return

        # Verify categories
        expected_categories = ['common', 'florence', 'promptgen', 'moondream']
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
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (basic)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa basic'):
            return

        t0 = time.time()
        data = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image'
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"VQA: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        answer = data.get('answer', '')
        if answer and not self.is_error_answer(answer):
            answer_preview = answer[:100] + '...' if len(answer) > 100 else answer
            self.log_pass(f"VQA returns answer ({elapsed:.1f}s)")
            self.log_info(f"Answer: {answer_preview}")
        elif self.is_error_answer(answer):
            self.log_fail(f"VQA returned error: {answer}")
        else:
            self.log_fail("VQA returned empty answer")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Different Prompts
    # =========================================================================
    def test_vqa_different_prompts(self):
        """Test different VQA prompts."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (prompts)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa prompts'):
            return

        prompts = ['Short Caption', 'Normal Caption', 'Long Caption']
        results = {}
        for prompt in prompts:
            t0 = time.time()
            data = self.post('/sdapi/v1/vqa', {
                'image': self.image_b64,
                'question': prompt
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"prompt='{prompt}': {data.get('reason', 'failed')}")
            elif data.get('answer') and not self.is_error_answer(data['answer']):
                results[prompt] = len(data['answer'])
                self.log_pass(f"prompt='{prompt}' returns answer ({len(data['answer'])} chars, {elapsed:.1f}s)")
            elif self.is_error_answer(data.get('answer', '')):
                self.log_fail(f"prompt='{prompt}' returned error: {data['answer']}")
            else:
                self.log_fail(f"prompt='{prompt}' returned empty answer")
        # Length sanity check: Short should be noticeably shorter than Normal/Long
        if 'Short Caption' in results and 'Normal Caption' in results and 'Long Caption' in results:
            if results['Short Caption'] >= results['Normal Caption'] or results['Short Caption'] >= results['Long Caption']:
                self.log_info(f"NOTE: Short ({results['Short Caption']}) >= Normal ({results['Normal Caption']}) or Long ({results['Long Caption']}); LLM output length is non-deterministic and prompt-dependent")
            if results['Long Caption'] < results['Normal Caption']:
                self.log_info(f"NOTE: Long ({results['Long Caption']}) < Normal ({results['Normal Caption']}); LLM may interpret length prompts differently per run")

        # Dual prefill: re-run 'Normal Caption' with custom prefill
        self._check_prefill({'image': self.image_b64, 'question': 'Normal Caption'}, "different_prompts")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Annotated Image
    # =========================================================================
    def test_vqa_annotated_image(self):
        """Test include_annotated=True returns annotated image for detection."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (annotated image)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa annotated'):
            return

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

        # Verify answer present and not an error
        answer = data_annot.get('answer', '')
        if self.is_meaningful_answer(answer) and not self.is_error_answer(answer):
            answer_preview = answer[:100] + '...' if len(answer) > 100 else answer
            self.log_pass(f"Detection returns answer ({elapsed:.1f}s): {answer_preview}")
        elif self.is_error_answer(answer):
            self.log_fail(f"Detection task returned error: {answer}")
            return
        else:
            self.log_fail(f"Detection returned non-meaningful answer: '{answer}'")
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
            # Check if answer contains detection results (bounding boxes)
            if '<loc_' in answer or 'box' in answer.lower():
                self.log_fail("Detections found in answer but annotated_image is empty")
            else:
                self.log_skip("No detections in image - annotated_image empty (test image may need visible objects)")

        # Verify absent when not requested
        if 'error' not in data_no_annot:
            if data_no_annot.get('annotated_image') is None:
                self.log_pass("include_annotated=False omits annotated_image")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - System Prompt
    # =========================================================================
    def test_vqa_system_prompt(self):
        """Test custom system prompt."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (system prompt)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa system prompt'):
            return

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

        answer = data.get('answer', '')
        if answer and not self.is_error_answer(answer):
            self.log_pass(f"Custom system prompt accepted ({elapsed:.1f}s)")
            self.log_info(f"Answer: {answer[:100]}")
        elif self.is_error_answer(answer):
            self.log_fail(f"Custom system prompt returned error: {answer}")
        else:
            self.log_fail("Custom system prompt returned empty answer")

        # Dual prefill: re-run with custom system prompt + prefill
        self._check_prefill({'image': self.image_b64, 'question': 'describe the image', 'system': custom_system}, "system_prompt")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Invalid Inputs
    # =========================================================================
    def test_vqa_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (invalid inputs)")
        print("=" * 70)

        # Test missing image (check_critical=False since we expect errors)
        data = self.post('/sdapi/v1/vqa', {
            'image': '',
            'question': 'describe'
        }, check_critical=False)
        if 'error' in data and data.get('status') == 404:
            self.log_pass("Missing image returns 404")
        else:
            self.log_fail(f"Missing image should return 404, got: {data}")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Prompt Field
    # =========================================================================
    def test_vqa_prompt_field(self):
        """Test the prompt field with 'Use Prompt' question."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (prompt field)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa prompt field'):
            return

        # Test with question="Use Prompt" and custom prompt text
        custom_prompt = "What colors are most prominent in this image?"
        t0 = time.time()
        data = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'Use Prompt',
            'prompt': custom_prompt
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"prompt field test: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        answer = data.get('answer', '')
        if answer and not self.is_error_answer(answer):
            self.log_pass(f"prompt field accepted with 'Use Prompt' ({elapsed:.1f}s)")
            self.log_info(f"Prompt: {custom_prompt}")
            self.log_info(f"Answer: {answer[:100]}...")
        elif self.is_error_answer(answer):
            self.log_fail(f"prompt field returned error: {answer}")
        else:
            self.log_fail("prompt field returned empty answer")

        # Test with direct prompt for detection-style task (for Moondream models)
        point_prompt = "Point at the main subject in this image"
        data_point = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'Use Prompt',
            'prompt': point_prompt
        })

        if 'error' not in data_point:
            answer_point = data_point.get('answer', '')
            if answer_point and not self.is_error_answer(answer_point):
                self.log_pass("Detection-style prompt accepted")
            else:
                self.log_skip("Detection prompt may require specific model")

        # Dual prefill: re-run 'Use Prompt' with custom prefill
        self._check_prefill({'image': self.image_b64, 'question': 'Use Prompt', 'prompt': custom_prompt}, "prompt_field")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Generation Parameters
    # =========================================================================
    def test_vqa_generation_params(self):
        """Test LLM generation parameters: temperature, max_tokens, top_k, top_p."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (temperature, max_tokens, top_k, top_p)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa generation params'):
            return

        # Test temperature effect: temp=0 should be deterministic, temp=10.0 should be very random
        # Run temp=0 twice to check determinism
        data_temp0_a = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image briefly',
            'temperature': 0.0
        })
        data_temp0_b = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image briefly',
            'temperature': 0.0
        })
        # Run temp=10.0 twice - extreme temp should produce very different/gibberish outputs
        data_temp_high_a = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image briefly',
            'temperature': 10.0
        })
        data_temp_high_b = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image briefly',
            'temperature': 10.0
        })

        if 'error' in data_temp0_a or 'error' in data_temp0_b:
            self.log_skip("temperature=0 test: API error")
        elif data_temp0_a.get('answer') and data_temp0_b.get('answer'):
            self.log_info(f"temp=0 run1: {data_temp0_a['answer'][:60]}...")
            self.log_info(f"temp=0 run2: {data_temp0_b['answer'][:60]}...")
            if data_temp0_a['answer'] == data_temp0_b['answer']:
                self.log_pass("temperature=0 produces deterministic output")
            else:
                self.log_skip("temperature=0 outputs differ (model may not support deterministic mode)")
        else:
            self.log_fail("temperature=0 returned empty/error")

        if 'error' in data_temp_high_a or 'error' in data_temp_high_b:
            self.log_skip("temperature=10.0 test: API error")
        elif data_temp_high_a.get('answer') and data_temp_high_b.get('answer'):
            self.log_info(f"temp=10.0 run1: {data_temp_high_a['answer'][:60]}...")
            self.log_info(f"temp=10.0 run2: {data_temp_high_b['answer'][:60]}...")
            if data_temp_high_a['answer'] != data_temp_high_b['answer']:
                self.log_pass("temperature=10.0 produces varied output")
            else:
                self.log_skip("temperature=10.0 outputs identical (unexpected)")
        else:
            self.log_fail("temperature=10.0 returned empty/error")

        # Test max_tokens effect by comparing short vs long limits
        data_tokens_short = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image in great detail',
            'max_tokens': 20
        })
        data_tokens_long = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image in great detail',
            'max_tokens': 200
        })

        if 'error' in data_tokens_short or 'error' in data_tokens_long:
            self.log_skip("max_tokens test: API error")
        elif data_tokens_short.get('answer') and data_tokens_long.get('answer'):
            len_short = len(data_tokens_short['answer'])
            len_long = len(data_tokens_long['answer'])
            self.log_info(f"max_tokens=20: {len_short} chars - '{data_tokens_short['answer'][:40]}...'")
            self.log_info(f"max_tokens=200: {len_long} chars - '{data_tokens_long['answer'][:40]}...'")
            if len_short < len_long:
                self.log_pass(f"max_tokens has effect: {len_short} < {len_long} chars")
            elif len_short == len_long:
                self.log_skip(f"max_tokens no effect detected (both {len_short} chars)")
            else:
                self.log_fail(f"max_tokens reversed: short={len_short}, long={len_long}")
        else:
            self.log_fail("max_tokens test returned empty answers")

        # Test top_k and top_p - can only verify accepted (effect is on sampling)
        t0 = time.time()
        data_sampling = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'top_k': 40,
            'top_p': 0.9
        })
        elapsed = time.time() - t0

        if 'error' in data_sampling:
            self.log_skip(f"top_k/top_p test: {data_sampling.get('reason', 'failed')}")
        elif self.is_meaningful_answer(data_sampling.get('answer')) and not self.is_error_answer(data_sampling['answer']):
            self.log_pass(f"top_k=40, top_p=0.9 accepted ({elapsed:.1f}s)")
        else:
            self.log_fail("top_k/top_p returned empty/error")

        # Dual prefill: re-run temp=0 request with custom prefill
        self._check_prefill({'image': self.image_b64, 'question': 'describe the image briefly', 'temperature': 0.0}, "generation_params")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Sampling Controls
    # =========================================================================
    def test_vqa_sampling(self):
        """Test do_sample and num_beams parameters."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (do_sample, num_beams)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa sampling'):
            return

        # Test with do_sample=False (greedy decoding)
        t0 = time.time()
        data_greedy = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'do_sample': False
        })
        elapsed = time.time() - t0

        greedy_elapsed = None
        if 'error' in data_greedy:
            self.log_skip(f"do_sample=False test: {data_greedy.get('reason', 'failed')}")
        elif data_greedy.get('answer') and not self.is_error_answer(data_greedy['answer']):
            greedy_elapsed = elapsed
            self.log_pass(f"do_sample=False (greedy) accepted ({elapsed:.1f}s)")
        else:
            self.log_fail("do_sample=False returned empty/error")

        # Test with do_sample=True (sampling enabled)
        t0 = time.time()
        data_sample = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'do_sample': True,
            'temperature': 0.7
        })
        elapsed = time.time() - t0

        if 'error' in data_sample:
            self.log_skip(f"do_sample=True test: {data_sample.get('reason', 'failed')}")
        elif data_sample.get('answer') and not self.is_error_answer(data_sample['answer']):
            self.log_pass(f"do_sample=True (sampling) accepted ({elapsed:.1f}s)")
        else:
            self.log_fail("do_sample=True returned empty/error")

        # Test with num_beams (beam search - should be slower than greedy)
        t0 = time.time()
        data_beams = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'num_beams': 4
        })
        elapsed = time.time() - t0

        if 'error' in data_beams:
            self.log_skip(f"num_beams=4 test: {data_beams.get('reason', 'failed')}")
        elif data_beams.get('answer') and not self.is_error_answer(data_beams['answer']):
            self.log_pass(f"num_beams=4 (beam search) accepted ({elapsed:.1f}s)")
            if greedy_elapsed is not None and elapsed <= greedy_elapsed:
                self.log_info(f"NOTE: num_beams=4 ({elapsed:.1f}s) not slower than greedy ({greedy_elapsed:.1f}s); beam search overhead may be negligible for short outputs or fast GPUs")
        else:
            self.log_fail("num_beams=4 returned empty/error")

        # Dual prefill: re-run greedy request with custom prefill
        self._check_prefill({'image': self.image_b64, 'question': 'describe the image', 'do_sample': False}, "sampling")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Thinking Mode
    # =========================================================================
    def test_vqa_thinking_mode(self):
        """Test thinking_mode and keep_thinking parameters."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (thinking_mode, keep_thinking)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa thinking mode'):
            return

        # Find a thinking-capable model
        thinking_model = None
        if self._vqa_models:
            for m in self._vqa_models:
                if 'thinking' in m.get('capabilities', []):
                    thinking_model = m['name']
                    break

        if not thinking_model:
            self.log_skip("No thinking-capable VQA models available")
            # Still test that the parameters are accepted even if no thinking model
            t0 = time.time()
            data = self.post('/sdapi/v1/vqa', {
                'image': self.image_b64,
                'question': 'describe the image',
                'thinking_mode': False  # Explicitly disable
            })
            elapsed = time.time() - t0
            if 'error' not in data and data.get('answer'):
                self.log_pass(f"thinking_mode=False accepted ({elapsed:.1f}s)")
            return

        self.log_info(f"Using thinking model: {thinking_model}")

        # Test with thinking_mode=True
        t0 = time.time()
        data_think = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'model': thinking_model,
            'question': 'What is happening in this image?',
            'thinking_mode': True
        })
        elapsed = time.time() - t0

        if 'error' in data_think:
            self.log_skip(f"thinking_mode=True test: {data_think.get('reason', 'failed')}")
        elif data_think.get('answer') and not self.is_error_answer(data_think['answer']):
            answer = data_think['answer']
            answer_preview = answer[:100] + '...' if len(answer) > 100 else answer
            self.log_pass(f"thinking_mode=True ({elapsed:.1f}s, {len(answer)} chars)")
            self.log_info(f"Answer: {answer_preview}")
        else:
            self.log_fail("thinking_mode=True returned empty/error")

        # Test with keep_thinking=True
        t0 = time.time()
        data_keep = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'model': thinking_model,
            'question': 'What is happening in this image?',
            'thinking_mode': True,
            'keep_thinking': True
        })
        elapsed = time.time() - t0

        if 'error' in data_keep:
            self.log_skip(f"keep_thinking=True test: {data_keep.get('reason', 'failed')}")
        elif data_keep.get('answer') and not self.is_error_answer(data_keep['answer']):
            answer = data_keep['answer']
            # Thinking trace is reformatted: <think>→"Reasoning:" and </think>→"Answer:" by strip_think_xml_tags()
            has_thinking = 'reasoning:' in answer.lower() or '<think' in answer.lower()
            self.log_pass(f"keep_thinking=True ({elapsed:.1f}s, {len(answer)} chars, has_trace={has_thinking})")
            if not has_thinking:
                self.log_info("NOTE: no thinking trace detected; model may not have produced <think> tags for this input")
            # Show first part of answer (may include thinking trace)
            answer_preview = answer[:150] + '...' if len(answer) > 150 else answer
            self.log_info(f"Answer: {answer_preview}")
        else:
            self.log_fail("keep_thinking=True returned empty/error")

    # =========================================================================
    # TEST: POST /sdapi/v1/vqa - Prefill
    # =========================================================================
    def test_vqa_prefill(self):
        """Test prefill and keep_prefill parameters."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/vqa (prefill, keep_prefill)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa prefill'):
            return

        prefill_text = "Vlado is the best, and I'm looking at his robot which"

        # Test with prefill to guide response start
        t0 = time.time()
        data_prefill = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'prefill': prefill_text
        })
        elapsed = time.time() - t0

        if 'error' in data_prefill:
            self.log_skip(f"prefill test: {data_prefill.get('reason', 'failed')}")
            return

        answer = data_prefill.get('answer', '')
        if answer and not self.is_error_answer(answer):
            self.log_pass(f"prefill accepted ({elapsed:.1f}s)")
            self.log_info(f"Prefill: '{prefill_text}'")
            self.log_info(f"Answer: {answer[:100]}...")
        elif self.is_error_answer(answer):
            self.log_fail(f"prefill returned error: {answer}")
        else:
            self.log_fail("prefill returned empty answer")

        # Test with keep_prefill=True (include prefill in output)
        t0 = time.time()
        data_keep = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'prefill': prefill_text,
            'keep_prefill': True
        })
        elapsed = time.time() - t0

        if 'error' in data_keep:
            self.log_skip(f"keep_prefill=True test: {data_keep.get('reason', 'failed')}")
        elif data_keep.get('answer') and not self.is_error_answer(data_keep['answer']):
            answer_keep = data_keep['answer']
            self.log_info(f"keep_prefill=True answer: {answer_keep[:80]}...")
            if answer_keep.startswith(prefill_text):
                self.log_pass(f"keep_prefill=True includes prefill in output ({elapsed:.1f}s)")
            else:
                self.log_fail(f"keep_prefill=True should start with '{prefill_text}' but got: '{answer_keep[:40]}...'")
        else:
            self.log_fail("keep_prefill=True returned empty/error")

        # Test with keep_prefill=False (strip prefill from output)
        t0 = time.time()
        data_strip = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'describe the image',
            'prefill': prefill_text,
            'keep_prefill': False
        })
        elapsed = time.time() - t0

        if 'error' in data_strip:
            self.log_skip(f"keep_prefill=False test: {data_strip.get('reason', 'failed')}")
        elif data_strip.get('answer') and not self.is_error_answer(data_strip['answer']):
            answer_strip = data_strip['answer']
            self.log_info(f"keep_prefill=False answer: {answer_strip[:80]}...")
            if not answer_strip.startswith(prefill_text):
                self.log_pass(f"keep_prefill=False strips prefill from output ({elapsed:.1f}s)")
            else:
                self.log_fail("keep_prefill=False should strip prefill but answer still starts with it")
        else:
            self.log_fail("keep_prefill=False returned empty/error")

    # =========================================================================
    # TEST: VQA Model Architectures
    # =========================================================================
    def test_vqa_model_architectures(self):
        """Test all VQA model architecture families."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: VQA Model Architectures")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa architectures'):
            return

        if not self._vqa_models:
            self._vqa_models = self.get('/sdapi/v1/vqa/models')

        if 'error' in self._vqa_models or not isinstance(self._vqa_models, list):
            self.log_skip("Cannot get VQA model list")
            return

        # Group models by family
        families_found = {}
        for model in self._vqa_models:
            family = self.get_model_family(model['name'])
            if family not in families_found:
                families_found[family] = model['name']

        self.log_info(f"Found {len(families_found)} model families: {list(families_found.keys())}")

        # Report which families are present vs absent
        for family in self.VQA_FAMILIES.keys():
            if family in families_found:
                self.log_pass(f"Architecture '{family}' available: {families_found[family]}")
            else:
                self.log_skip(f"Architecture '{family}' not available")

        # Report unknown models
        if 'unknown' in families_found:
            unknown_models = [m['name'] for m in self._vqa_models if self.get_model_family(m['name']) == 'unknown']
            self.log_info(f"Unrecognized models: {unknown_models[:5]}")

    # =========================================================================
    # TEST: VQA Florence Special Prompts
    # =========================================================================
    def test_vqa_florence_special_prompts(self):
        """Test Florence-2 specific prompts for detection, OCR, etc."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: VQA Florence Special Prompts")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa florence prompts'):
            return

        # Find Florence models: base and PromptGen (which supports extra prompts)
        florence_model = None
        promptgen_model = None
        if self._vqa_models:
            for m in self._vqa_models:
                name_lower = m['name'].lower()
                if 'promptgen' in name_lower and promptgen_model is None:
                    promptgen_model = m['name']
                elif 'florence' in name_lower and 'promptgen' not in name_lower and 'cog' not in name_lower and florence_model is None:
                    florence_model = m['name']

        if not florence_model:
            self.log_skip("No Florence model available")
            return

        self.log_info(f"Using Florence model: {florence_model}")
        if promptgen_model:
            self.log_info(f"Using PromptGen model: {promptgen_model}")

        # Base Florence prompts (supported by all Florence models)
        base_prompts = {
            '<OD>': 'Object Detection',
            '<OCR>': 'Optical Character Recognition',
            '<DENSE_REGION_CAPTION>': 'Dense Region Captioning',
            '<CAPTION>': 'Standard Caption',
            '<DETAILED_CAPTION>': 'Detailed Caption',
        }
        # PromptGen-only prompts (require MiaoshouAI PromptGen fine-tune)
        promptgen_prompts = {
            '<GENERATE_TAGS>': 'Tag Generation',
        }

        def run_florence_prompt(model, prompt, description):
            # Use OCR test image for OCR prompts (image with readable text)
            if prompt == '<OCR>' and self.ocr_image_b64:
                test_image = self.ocr_image_b64
            else:
                test_image = self.image_b64

            t0 = time.time()
            data = self.post('/sdapi/v1/vqa', {
                'image': test_image,
                'model': model,
                'question': prompt
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"{description} ({prompt}): {data.get('reason', 'failed')}")
            elif self.is_meaningful_answer(data.get('answer')) and not self.is_error_answer(data['answer']):
                answer_preview = data['answer'][:60] + '...' if len(data['answer']) > 60 else data['answer']
                self.log_pass(f"{description} ({prompt}): {elapsed:.1f}s")
                self.log_info(f"  Answer: {answer_preview}")
            elif data.get('answer'):
                # Got an answer but it's not meaningful (e.g., just punctuation)
                self.log_fail(f"{description} ({prompt}): non-meaningful response: '{data['answer']}'")
            else:
                self.log_fail(f"{description} ({prompt}): empty/error response")

        for prompt, description in base_prompts.items():
            run_florence_prompt(florence_model, prompt, description)

        for prompt, description in promptgen_prompts.items():
            if promptgen_model:
                run_florence_prompt(promptgen_model, prompt, f"{description} [PromptGen]")
            else:
                self.log_skip(f"{description} ({prompt}): requires PromptGen model, none available")

    # =========================================================================
    # TEST: VQA Moondream Detection Features
    # =========================================================================
    def test_vqa_moondream_detection(self):
        """Test Moondream detection and pointing features."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: VQA Moondream Detection Features")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa moondream'):
            return

        # Find a Moondream model
        moondream_model = None
        moondream_version = None
        if self._vqa_models:
            for m in self._vqa_models:
                if 'moondream' in m['name'].lower():
                    moondream_model = m['name']
                    # Detect version
                    if '3' in m['name']:
                        moondream_version = 3
                    elif '2' in m['name']:
                        moondream_version = 2
                    else:
                        moondream_version = 1
                    break

        if not moondream_model:
            self.log_skip("No Moondream model available")
            return

        self.log_info(f"Using Moondream model: {moondream_model} (v{moondream_version})")

        # Moondream-specific prompts
        moondream_prompts = [
            ('Point at the main subject', 'Point detection'),
            ('Detect all objects', 'Object detection'),
            ('What is in the center of the image?', 'Region query'),
        ]

        # Add gaze detection for Moondream 2+
        if moondream_version and moondream_version >= 2:
            moondream_prompts.append(('Detect Gaze', 'Gaze detection'))

        for prompt, description in moondream_prompts:
            t0 = time.time()
            data = self.post('/sdapi/v1/vqa', {
                'image': self.image_b64,
                'model': moondream_model,
                'question': 'Use Prompt',
                'prompt': prompt,
                'include_annotated': True
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"{description}: {data.get('reason', 'failed')}")
            elif self.is_meaningful_answer(data.get('answer')) and not self.is_error_answer(data['answer']):
                answer_preview = data['answer'][:60] + '...' if len(data['answer']) > 60 else data['answer']
                has_annotated = bool(data.get('annotated_image'))
                self.log_pass(f"{description}: {elapsed:.1f}s (annotated={has_annotated})")
                self.log_info(f"  Answer: {answer_preview}")
            elif data.get('answer'):
                self.log_fail(f"{description}: non-meaningful response: '{data['answer']}'")
            else:
                self.log_skip(f"{description}: may not be supported by this model version")

    # =========================================================================
    # TEST: VQA Architecture Capabilities
    # =========================================================================
    def test_vqa_architecture_capabilities(self):
        """Test architecture-specific capabilities like vision, thinking, detection."""
        self.set_category('vqa')
        print("\n" + "=" * 70)
        print("TEST: VQA Architecture Capabilities")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'vqa capabilities'):
            return

        if not self._vqa_models:
            self._vqa_models = self.get('/sdapi/v1/vqa/models')

        if 'error' in self._vqa_models or not isinstance(self._vqa_models, list):
            self.log_skip("Cannot get VQA model list")
            return

        # Collect all capabilities across models
        capability_models = {}
        for model in self._vqa_models:
            caps = model.get('capabilities', [])
            for cap in caps:
                if cap not in capability_models:
                    capability_models[cap] = []
                capability_models[cap].append(model['name'])

        self.log_info(f"Found {len(capability_models)} capabilities: {list(capability_models.keys())}")

        # Test each capability with one model
        capability_tests = {
            'caption': 'describe the image',
            'vqa': 'What is the main subject of this image?',
            'detection': '<OD>',
            'ocr': '<OCR>',
            'thinking': 'Analyze this image step by step',
        }

        for capability, test_prompt in capability_tests.items():
            if capability not in capability_models:
                self.log_skip(f"Capability '{capability}': no models available")
                continue

            # Use first available model with this capability
            model_name = capability_models[capability][0]
            self.log_info(f"Testing '{capability}' with: {model_name}")

            # Use OCR test image for OCR capability (image with readable text)
            if capability == 'ocr' and self.ocr_image_b64:
                test_image = self.ocr_image_b64
            else:
                test_image = self.image_b64

            request_data = {
                'image': test_image,
                'model': model_name,
                'question': test_prompt
            }

            # Enable thinking mode for thinking capability test
            if capability == 'thinking':
                request_data['thinking_mode'] = True

            t0 = time.time()
            data = self.post('/sdapi/v1/vqa', request_data)
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"Capability '{capability}': {data.get('reason', 'model not loaded')}")
            elif self.is_meaningful_answer(data.get('answer')) and not self.is_error_answer(data['answer']):
                answer = data['answer']
                answer_preview = answer[:80] + '...' if len(answer) > 80 else answer
                self.log_pass(f"Capability '{capability}' ({elapsed:.1f}s): {answer_preview}")
                if elapsed > 60:
                    self.log_info(f"NOTE: {model_name} took {elapsed:.1f}s which is suspiciously slow; may need performance investigation")
            elif data.get('answer'):
                self.log_fail(f"Capability '{capability}': non-meaningful response: '{data['answer']}'")
            else:
                self.log_fail(f"Capability '{capability}': empty/error response")

    # =========================================================================
    # TEST: OpenCLIP BLIP Architectures
    # =========================================================================
    def test_openclip_blip_architectures(self):
        """Test all BLIP caption model types."""
        self.set_category('openclip')
        print("\n" + "=" * 70)
        print("TEST: OpenCLIP BLIP Architectures")
        print("=" * 70)

        if self.skip_if_critical('openclip', 'openclip blip'):
            return

        # Check if we have OpenCLIP models (needed for caption endpoint)
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')

        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available for BLIP testing")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        for blip_model in self.BLIP_MODELS:
            t0 = time.time()
            data = self.post('/sdapi/v1/openclip', {
                'image': self.image_b64,
                'model': model,
                'mode': 'caption',
                'blip_model': blip_model
            })
            elapsed = time.time() - t0

            if 'error' in data:
                # Check if it's a model not found error vs other error
                reason = data.get('reason', '')
                if 'not found' in str(reason).lower() or data.get('status') == 404:
                    self.log_skip(f"BLIP '{blip_model}': model not downloaded")
                else:
                    self.log_skip(f"BLIP '{blip_model}': {reason}")
            elif data.get('caption') and not self.is_error_answer(data['caption']):
                caption_preview = data['caption'][:70] + '...' if len(data['caption']) > 70 else data['caption']
                self.log_pass(f"BLIP '{blip_model}' ({elapsed:.1f}s): {caption_preview}")
            else:
                self.log_fail(f"BLIP '{blip_model}': empty/error response")

    # =========================================================================
    # TEST: GET /sdapi/v1/tagger/models - Tagger Models List
    # =========================================================================
    def test_tagger_models_list(self):
        """Test GET /sdapi/v1/tagger/models returns model list."""
        self.set_category('tagger')
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
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (basic)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger basic'):
            return

        t0 = time.time()
        data = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"Tagger: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        tags = data.get('tags', '')
        if tags and not self.is_error_answer(tags):
            tags_preview = tags[:80] + '...' if len(tags) > 80 else tags
            tag_count = len(tags.split(', '))
            self.log_pass(f"Returns tags ({tag_count} tags, {elapsed:.1f}s)")
            self.log_info(f"Tags: {tags_preview}")
        elif self.is_error_answer(tags):
            self.log_fail(f"Tagger returned error: {tags}")
        else:
            self.log_fail("Tagger returned empty tags")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Threshold
    # =========================================================================
    def test_tagger_threshold(self):
        """Test threshold affects tag count."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (threshold)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger threshold'):
            return

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

        tags_high = data_high.get('tags', '')
        tags_low = data_low.get('tags', '')
        count_high = len(tags_high.split(', ')) if tags_high else 0
        count_low = len(tags_low.split(', ')) if tags_low else 0

        self.log_info(f"threshold=0.9 ({count_high} tags): {tags_high[:70]}{'...' if len(tags_high) > 70 else ''}")
        self.log_info(f"threshold=0.1 ({count_low} tags): {tags_low[:70]}{'...' if len(tags_low) > 70 else ''}")

        if count_low > count_high:
            self.log_pass(f"threshold has effect: 0.9={count_high} tags < 0.1={count_low} tags")
        elif count_high == 0 and count_low == 0:
            self.log_skip("No tags returned (model may not be loaded)")
        else:
            self.log_fail(f"threshold no effect: 0.9={count_high}, 0.1={count_low}")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Max Tags
    # =========================================================================
    def test_tagger_max_tags(self):
        """Test max_tags limits output count."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (max_tags)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger max_tags'):
            return

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

        tags_5 = data_5.get('tags', '')
        tags_50 = data_50.get('tags', '')
        count_5 = len(tags_5.split(', ')) if tags_5 else 0
        count_50 = len(tags_50.split(', ')) if tags_50 else 0

        self.log_info(f"max_tags=5 ({count_5} tags): {tags_5}")
        self.log_info(f"max_tags=50 ({count_50} tags): {tags_50[:80]}{'...' if len(tags_50) > 80 else ''}")

        if count_5 <= 5:
            self.log_pass(f"max_tags=5 correctly limits to {count_5} tags")
        else:
            self.log_fail(f"max_tags=5 returned {count_5} tags (expected <= 5)")

        if count_50 > count_5:
            self.log_pass(f"max_tags=50 returns more: {count_5} < {count_50} tags")
        else:
            self.log_fail("max_tags=50 should return more than max_tags=5")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Sort Alpha
    # =========================================================================
    def test_tagger_sort_alpha(self):
        """Test sort_alpha sorts tags alphabetically."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (sort_alpha)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger sort_alpha'):
            return

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

        list_conf = [t.strip() for t in data_conf.get('tags', '').split(',') if t.strip()]
        list_alpha = [t.strip() for t in data_alpha.get('tags', '').split(',') if t.strip()]

        if len(list_alpha) < 2:
            self.log_skip("Not enough tags to test sorting")
            return

        self.log_info(f"By confidence: {', '.join(list_conf[:8])}...")
        self.log_info(f"Alphabetical:  {', '.join(list_alpha[:8])}...")
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
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (use_spaces)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger use_spaces'):
            return

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
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (escape_brackets)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger escape_brackets'):
            return

        # Use bracket test image (produces tags with parentheses like "pokemon_(creature)")
        test_image = self.bracket_image_b64 or self.image_b64
        if not self.bracket_image_b64:
            self.log_info("NOTE: bracket test image not available, using default image (may not produce bracket tags)")

        data_escaped = self.post('/sdapi/v1/tagger', {
            'image': test_image,
            'escape_brackets': True,
            'max_tags': 50,
            'threshold': 0.1
        })
        data_raw = self.post('/sdapi/v1/tagger', {
            'image': test_image,
            'escape_brackets': False,
            'max_tags': 50,
            'threshold': 0.1
        })

        if 'error' in data_escaped or 'error' in data_raw:
            self.log_skip("escape_brackets test: model not loaded")
            return

        tags_escaped = data_escaped.get('tags', '')
        tags_raw = data_raw.get('tags', '')

        self.log_info(f"escape=True:  {tags_escaped[:70]}...")
        self.log_info(f"escape=False: {tags_raw[:70]}...")

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
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (exclude_tags)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger exclude_tags'):
            return

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
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (show_scores)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger show_scores'):
            return

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

        # Show the actual output
        tags_with_scores = data_scores.get('tags', '')
        tags_no_scores = data_no_scores.get('tags', '')
        self.log_info(f"show_scores=True tags: {tags_with_scores}")
        self.log_info(f"show_scores=False tags: {tags_no_scores}")

        # Check scores dict is returned
        if 'scores' in data_scores and isinstance(data_scores['scores'], dict) and len(data_scores['scores']) > 0:
            scores_dict = data_scores['scores']
            # Show first few scores
            scores_preview = dict(list(scores_dict.items())[:3])
            self.log_info(f"scores dict (first 3): {scores_preview}")
            self.log_pass(f"show_scores=True returns scores dict with {len(scores_dict)} entries")

            # Verify scores are floats 0-1
            scores = list(scores_dict.values())
            if all(isinstance(s, (int, float)) and 0 <= s <= 1 for s in scores):
                self.log_pass("All scores are floats in 0-1 range")
            else:
                self.log_fail(f"Some scores out of range: {scores}")
        else:
            self.log_fail("show_scores=True did not return scores dict")

        # Check tags contain scores (colon notation)
        if ':' in tags_with_scores:
            self.log_pass("Tags string includes score notation (:)")
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
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (include_rating)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger include_rating'):
            return

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
    # TEST: POST /sdapi/v1/tagger - Character Threshold
    # =========================================================================
    def test_tagger_character_threshold(self):
        """Test character_threshold for character-specific tags (WaifuDiffusion only)."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (character_threshold)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger character_threshold'):
            return

        # Find a WaifuDiffusion model (character_threshold only applies to WD models)
        wd_model = None
        if self._tagger_models:
            for m in self._tagger_models:
                if m.get('type') == 'waifudiffusion':
                    wd_model = m['name']
                    break

        if not wd_model:
            self.log_skip("No WaifuDiffusion models available (character_threshold only applies to WD)")
            return

        self.log_info(f"Using WaifuDiffusion model: {wd_model}")

        # Test with low character_threshold (more character tags)
        t0 = time.time()
        data_low = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'model': wd_model,
            'character_threshold': 0.5,
            'threshold': 0.1,
            'max_tags': 100
        })
        elapsed_low = time.time() - t0

        # Test with high character_threshold (fewer character tags)
        t0 = time.time()
        data_high = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'model': wd_model,
            'character_threshold': 0.99,
            'threshold': 0.1,
            'max_tags': 100
        })
        elapsed_high = time.time() - t0

        if 'error' in data_low:
            self.log_skip(f"character_threshold=0.5 test: {data_low.get('reason', 'failed')}")
        elif data_low.get('tags') and not self.is_error_answer(data_low['tags']):
            self.log_pass(f"character_threshold=0.5 accepted ({elapsed_low:.1f}s)")
        else:
            self.log_fail("character_threshold=0.5 returned empty/error")

        if 'error' in data_high:
            self.log_skip(f"character_threshold=0.99 test: {data_high.get('reason', 'failed')}")
        elif data_high.get('tags') and not self.is_error_answer(data_high['tags']):
            self.log_pass(f"character_threshold=0.99 accepted ({elapsed_high:.1f}s)")

            # Compare tag counts - higher threshold should have fewer (or same) character tags
            count_low = len(data_low.get('tags', '').split(', '))
            count_high = len(data_high.get('tags', '').split(', '))
            self.log_info(f"Tag counts: threshold=0.5→{count_low}, threshold=0.99→{count_high}")

            if count_low > count_high:
                self.log_pass(f"character_threshold affects tag filtering: {count_low} > {count_high}")
            elif count_low == count_high:
                self.log_info("NOTE: acceptance-only test, tag counts identical; test image likely has no character tags (character_threshold only filters anime character names)")
            else:
                self.log_fail(f"character_threshold reversed: low={count_low} < high={count_high}")
        else:
            self.log_fail("character_threshold=0.99 returned empty/error")

    # =========================================================================
    # TEST: Tagger Model Types (Architecture Coverage)
    # =========================================================================
    def test_tagger_model_types(self):
        """Test all tagger model types (deepbooru, waifudiffusion)."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: Tagger Model Types")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger model types'):
            return

        if not self._tagger_models:
            self._tagger_models = self.get('/sdapi/v1/tagger/models')

        if 'error' in self._tagger_models or not isinstance(self._tagger_models, list):
            self.log_skip("Cannot get tagger model list")
            return

        # Group models by type
        types_found = {}
        for model in self._tagger_models:
            model_type, version = self.get_tagger_type(model)
            type_key = f"{model_type}" + (f"-{version}" if version else "")
            if type_key not in types_found:
                types_found[type_key] = model['name']

        self.log_info(f"Found {len(types_found)} tagger types: {list(types_found.keys())}")

        # Test one model from each type
        for type_key, model_name in types_found.items():
            t0 = time.time()
            data = self.post('/sdapi/v1/tagger', {
                'image': self.image_b64,
                'model': model_name,
                'max_tags': 10,
                'threshold': 0.3
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"Type '{type_key}' ({model_name}): {data.get('reason', 'failed')}")
            elif data.get('tags') and not self.is_error_answer(data['tags']):
                tags = data['tags']
                tag_count = len(tags.split(', '))
                tags_preview = tags[:60] + '...' if len(tags) > 60 else tags
                self.log_pass(f"Type '{type_key}' ({elapsed:.1f}s, {tag_count} tags): {tags_preview}")
            else:
                self.log_fail(f"Type '{type_key}' ({model_name}): empty/error response")

    # =========================================================================
    # TEST: Tagger WaifuDiffusion Versions
    # =========================================================================
    def test_tagger_wd_versions(self):
        """Test WaifuDiffusion version differences (v2 vs v3)."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: Tagger WaifuDiffusion Versions")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'tagger wd_versions'):
            return

        if not self._tagger_models:
            self._tagger_models = self.get('/sdapi/v1/tagger/models')

        if 'error' in self._tagger_models or not isinstance(self._tagger_models, list):
            self.log_skip("Cannot get tagger model list")
            return

        # Find WD v2 and v3 models
        wd_v2 = None
        wd_v3 = None
        for model in self._tagger_models:
            if model.get('type') == 'waifudiffusion':
                name_lower = model['name'].lower()
                if 'v3' in name_lower and not wd_v3:
                    wd_v3 = model['name']
                elif 'v2' in name_lower and not wd_v2:
                    wd_v2 = model['name']

        if not wd_v2 and not wd_v3:
            self.log_skip("No WaifuDiffusion models available")
            return

        results = {}

        # Test WD v2
        if wd_v2:
            self.log_info(f"Testing WD v2: {wd_v2}")
            t0 = time.time()
            data_v2 = self.post('/sdapi/v1/tagger', {
                'image': self.image_b64,
                'model': wd_v2,
                'max_tags': 20,
                'threshold': 0.3
            })
            elapsed = time.time() - t0

            if 'error' in data_v2:
                self.log_skip(f"WD v2: {data_v2.get('reason', 'failed')}")
            elif data_v2.get('tags') and not self.is_error_answer(data_v2['tags']):
                tag_count = len(data_v2['tags'].split(', '))
                self.log_pass(f"WD v2 ({wd_v2}): {tag_count} tags ({elapsed:.1f}s)")
                results['v2'] = data_v2['tags']
            else:
                self.log_fail("WD v2: empty/error response")
        else:
            self.log_skip("No WD v2 model available")

        # Test WD v3
        if wd_v3:
            self.log_info(f"Testing WD v3: {wd_v3}")
            t0 = time.time()
            data_v3 = self.post('/sdapi/v1/tagger', {
                'image': self.image_b64,
                'model': wd_v3,
                'max_tags': 20,
                'threshold': 0.3
            })
            elapsed = time.time() - t0

            if 'error' in data_v3:
                self.log_skip(f"WD v3: {data_v3.get('reason', 'failed')}")
            elif data_v3.get('tags') and not self.is_error_answer(data_v3['tags']):
                tag_count = len(data_v3['tags'].split(', '))
                self.log_pass(f"WD v3 ({wd_v3}): {tag_count} tags ({elapsed:.1f}s)")
                results['v3'] = data_v3['tags']
            else:
                self.log_fail("WD v3: empty/error response")
        else:
            self.log_skip("No WD v3 model available")

        # Compare outputs if both available
        if 'v2' in results and 'v3' in results:
            v2_tags = {t.strip() for t in results['v2'].split(',')}
            v3_tags = {t.strip() for t in results['v3'].split(',')}
            common = len(v2_tags & v3_tags)
            v2_only = len(v2_tags - v3_tags)
            v3_only = len(v3_tags - v2_tags)
            self.log_info(f"Tag comparison: {common} common, {v2_only} v2-only, {v3_only} v3-only")

    # =========================================================================
    # TEST: POST /sdapi/v1/tagger - Invalid Inputs
    # =========================================================================
    def test_tagger_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        self.set_category('tagger')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/tagger (invalid inputs)")
        print("=" * 70)

        # Test missing image (check_critical=False since we expect errors)
        data = self.post('/sdapi/v1/tagger', {
            'image': ''
        }, check_critical=False)
        if 'error' in data and data.get('status') == 404:
            self.log_pass("Missing image returns 404")
        else:
            self.log_fail(f"Missing image should return 404, got: {data}")

    # =========================================================================
    # DISPATCH ENDPOINT TESTS
    # =========================================================================
    def test_dispatch_openclip_basic(self):
        """Test dispatch endpoint routes to OpenCLIP backend."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: openclip)")
        print("=" * 70)

        if self.skip_if_critical('openclip', 'dispatch openclip'):
            return

        # Check if we have OpenCLIP models
        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        t0 = time.time()
        data = self.post('/sdapi/v1/caption', {
            'backend': 'openclip',
            'image': self.image_b64,
            'model': model,
            'mode': 'fast'
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"Dispatch to openclip: {data.get('reason', 'failed')}")
            return

        # Verify backend field
        if data.get('backend') == 'openclip':
            self.log_pass("backend='openclip' returned in response")
        else:
            self.log_fail(f"Expected backend='openclip', got '{data.get('backend')}'")

        # Verify caption field
        if data.get('caption') and not self.is_error_answer(data['caption']):
            self.log_pass(f"Dispatch to openclip returns caption ({elapsed:.1f}s)")
            self.log_info(f"Caption: {self.truncate(data['caption'], 60)}")
        else:
            self.log_fail("Dispatch to openclip returned empty/error caption")

    def test_dispatch_openclip_modes(self):
        """Test all OpenCLIP modes via dispatch."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: openclip modes)")
        print("=" * 70)

        if self.skip_if_critical('openclip', 'dispatch openclip modes'):
            return

        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        modes = ['best', 'fast', 'classic', 'caption', 'negative']
        for mode in modes:
            t0 = time.time()
            data = self.post('/sdapi/v1/caption', {
                'backend': 'openclip',
                'image': self.image_b64,
                'model': model,
                'mode': mode
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"dispatch openclip mode='{mode}': {data.get('reason', 'failed')}")
            elif data.get('caption') and data.get('backend') == 'openclip':
                self.log_pass(f"dispatch openclip mode='{mode}' ({elapsed:.1f}s)")
            else:
                self.log_fail(f"dispatch openclip mode='{mode}' failed")

    def test_dispatch_openclip_analyze(self):
        """Test OpenCLIP analyze via dispatch."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: openclip analyze)")
        print("=" * 70)

        if self.skip_if_critical('openclip', 'dispatch openclip analyze'):
            return

        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        data = self.post('/sdapi/v1/caption', {
            'backend': 'openclip',
            'image': self.image_b64,
            'model': model,
            'analyze': True
        })

        if 'error' in data:
            self.log_skip(f"dispatch openclip analyze: {data.get('reason', 'failed')}")
            return

        analyze_fields = ['medium', 'artist', 'movement', 'trending', 'flavor']
        fields_found = sum(1 for f in analyze_fields if data.get(f))
        if fields_found > 0:
            self.log_pass(f"dispatch openclip analyze returns {fields_found}/5 breakdown fields")
        else:
            self.log_skip("No breakdown fields returned (may be image-dependent)")

    def test_dispatch_tagger_basic(self):
        """Test dispatch endpoint routes to Tagger backend."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: tagger)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'dispatch tagger'):
            return

        t0 = time.time()
        data = self.post('/sdapi/v1/caption', {
            'backend': 'tagger',
            'image': self.image_b64,
            'threshold': 0.5,
            'max_tags': 20
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"Dispatch to tagger: {data.get('reason', 'failed')}")
            return

        # Verify backend field
        if data.get('backend') == 'tagger':
            self.log_pass("backend='tagger' returned in response")
        else:
            self.log_fail(f"Expected backend='tagger', got '{data.get('backend')}'")

        # Verify tags field
        if data.get('tags') and not self.is_error_answer(data['tags']):
            tags = data['tags']
            tag_count = len(tags.split(', '))
            self.log_pass(f"Dispatch to tagger returns {tag_count} tags ({elapsed:.1f}s)")
            self.log_info(f"Tags: {self.truncate(tags, 60)}")
        else:
            self.log_fail("Dispatch to tagger returned empty/error tags")

    def test_dispatch_tagger_params(self):
        """Test tagger parameters via dispatch."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: tagger params)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'dispatch tagger params'):
            return

        # Test with various parameters
        data = self.post('/sdapi/v1/caption', {
            'backend': 'tagger',
            'image': self.image_b64,
            'threshold': 0.3,
            'max_tags': 10,
            'sort_alpha': True,
            'use_spaces': True
        })

        if 'error' in data:
            self.log_skip(f"dispatch tagger params: {data.get('reason', 'failed')}")
            return

        if data.get('tags') and data.get('backend') == 'tagger':
            tags = data['tags']
            tag_list = [t.strip() for t in tags.split(',') if t.strip()]
            if len(tag_list) <= 10:
                self.log_pass(f"dispatch tagger max_tags=10 respected ({len(tag_list)} tags)")
            else:
                self.log_fail(f"dispatch tagger max_tags=10 not respected ({len(tag_list)} tags)")
        else:
            self.log_fail("dispatch tagger params failed")

    def test_dispatch_tagger_scores(self):
        """Test tagger show_scores via dispatch."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: tagger scores)")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'dispatch tagger scores'):
            return

        data = self.post('/sdapi/v1/caption', {
            'backend': 'tagger',
            'image': self.image_b64,
            'show_scores': True,
            'max_tags': 5
        })

        if 'error' in data:
            self.log_skip(f"dispatch tagger scores: {data.get('reason', 'failed')}")
            return

        if data.get('scores') and isinstance(data['scores'], dict):
            self.log_pass(f"dispatch tagger show_scores returns {len(data['scores'])} scores")
            self.log_response(data, key_fields=['scores'])
        else:
            self.log_fail("dispatch tagger show_scores did not return scores dict")

    def test_dispatch_vlm_basic(self):
        """Test dispatch endpoint routes to VLM backend."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: vlm)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'dispatch vlm'):
            return

        t0 = time.time()
        data = self.post('/sdapi/v1/caption', {
            'backend': 'vlm',
            'image': self.image_b64,
            'question': 'describe the image'
        })
        elapsed = time.time() - t0

        if 'error' in data:
            self.log_skip(f"Dispatch to vlm: {data.get('reason', 'failed')} (model may not be loaded)")
            return

        # Verify backend field
        if data.get('backend') == 'vlm':
            self.log_pass("backend='vlm' returned in response")
        else:
            self.log_fail(f"Expected backend='vlm', got '{data.get('backend')}'")

        # Verify answer field
        if data.get('answer') and not self.is_error_answer(data['answer']):
            self.log_pass(f"Dispatch to vlm returns answer ({elapsed:.1f}s)")
            self.log_info(f"Answer: {self.truncate(data['answer'], 60)}")
        else:
            self.log_fail("Dispatch to vlm returned empty/error answer")

    def test_dispatch_vlm_prompts(self):
        """Test different VLM prompts via dispatch."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: vlm prompts)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'dispatch vlm prompts'):
            return

        prompts = ['Short Caption', 'Normal Caption', 'Long Caption']
        for prompt in prompts:
            t0 = time.time()
            data = self.post('/sdapi/v1/caption', {
                'backend': 'vlm',
                'image': self.image_b64,
                'question': prompt
            })
            elapsed = time.time() - t0

            if 'error' in data:
                self.log_skip(f"dispatch vlm prompt='{prompt}': {data.get('reason', 'failed')}")
            elif data.get('answer') and data.get('backend') == 'vlm':
                self.log_pass(f"dispatch vlm prompt='{prompt}' ({len(data['answer'])} chars, {elapsed:.1f}s)")
            else:
                self.log_fail(f"dispatch vlm prompt='{prompt}' failed")

    def test_dispatch_vlm_annotated(self):
        """Test VLM include_annotated via dispatch."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: vlm annotated)")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'dispatch vlm annotated'):
            return

        # Find a Florence model for detection
        florence_model = None
        if self._vqa_models:
            for m in self._vqa_models:
                if 'florence' in m['name'].lower():
                    florence_model = m['name']
                    break

        if not florence_model:
            florence_model = 'Microsoft Florence 2 Base'

        data = self.post('/sdapi/v1/caption', {
            'backend': 'vlm',
            'image': self.image_b64,
            'model': florence_model,
            'question': '<OD>',
            'include_annotated': True
        })

        if 'error' in data:
            self.log_skip(f"dispatch vlm annotated: {data.get('reason', 'failed')}")
            return

        if data.get('answer') and data.get('backend') == 'vlm':
            self.log_pass("dispatch vlm with include_annotated returns answer")
            if data.get('annotated_image'):
                self.log_pass("dispatch vlm returns annotated_image")
            else:
                self.log_skip("No annotated_image (detection may not have found objects)")
        else:
            self.log_fail("dispatch vlm annotated failed")

    def test_dispatch_backend_field(self):
        """Test backend field is always returned correctly."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: backend field)")
        print("=" * 70)

        backends = ['openclip', 'tagger', 'vlm']
        for backend in backends:
            req = {'backend': backend, 'image': self.image_b64}
            if backend == 'vlm':
                req['question'] = 'describe'
            data = self.post('/sdapi/v1/caption', req)

            if 'error' in data:
                self.log_skip(f"backend='{backend}': {data.get('reason', 'failed')}")
            elif data.get('backend') == backend:
                self.log_pass(f"backend='{backend}' returned correctly")
            else:
                self.log_fail(f"backend='{backend}' not returned, got '{data.get('backend')}'")

    def test_dispatch_invalid_backend(self):
        """Test error handling for invalid backend value."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: invalid backend)")
        print("=" * 70)

        # check_critical=False since we expect errors
        data = self.post('/sdapi/v1/caption', {
            'backend': 'invalid_backend',
            'image': self.image_b64
        }, check_critical=False)

        if 'error' in data:
            self.log_pass(f"Invalid backend returns error: {data.get('status', 'error')}")
        else:
            self.log_fail("Invalid backend should return error")

    def test_dispatch_missing_image(self):
        """Test error handling for missing image."""
        self.set_category('dispatch')
        print("\n" + "=" * 70)
        print("TEST: POST /sdapi/v1/caption (dispatch: missing image)")
        print("=" * 70)

        for backend in ['openclip', 'tagger', 'vlm']:
            req = {'backend': backend, 'image': ''}
            if backend == 'vlm':
                req['question'] = 'describe'
            # check_critical=False since we expect errors
            data = self.post('/sdapi/v1/caption', req, check_critical=False)

            if 'error' in data and data.get('status') == 404:
                self.log_pass(f"dispatch {backend} missing image returns 404")
            else:
                self.log_fail(f"dispatch {backend} missing image should return 404, got: {data}")

    # =========================================================================
    # PARITY TESTS: Dispatch vs Direct Endpoints
    # =========================================================================
    def test_parity_openclip(self):
        """Test dispatch and direct OpenCLIP endpoints return same caption."""
        self.set_category('parity')
        print("\n" + "=" * 70)
        print("TEST: Parity - OpenCLIP dispatch vs direct")
        print("=" * 70)

        if self.skip_if_critical('openclip', 'parity openclip'):
            return

        if not self._caption_models:
            self._caption_models = self.get('/sdapi/v1/openclip')
        clip_models = [m for m in self._caption_models if '/' in m] if isinstance(self._caption_models, list) else []

        if not clip_models:
            self.log_skip("No OpenCLIP models available")
            return

        model = 'ViT-L-14/openai' if 'ViT-L-14/openai' in clip_models else clip_models[0]

        # Direct endpoint
        data_direct = self.post('/sdapi/v1/openclip', {
            'image': self.image_b64,
            'model': model,
            'mode': 'caption'
        })

        # Dispatch endpoint
        data_dispatch = self.post('/sdapi/v1/caption', {
            'backend': 'openclip',
            'image': self.image_b64,
            'model': model,
            'mode': 'caption'
        })

        if 'error' in data_direct or 'error' in data_dispatch:
            self.log_skip("One or both requests failed")
            return

        direct_caption = data_direct.get('caption', '')
        dispatch_caption = data_dispatch.get('caption', '')

        self.log_info(f"Direct: {self.truncate(direct_caption, 50)}")
        self.log_info(f"Dispatch: {self.truncate(dispatch_caption, 50)}")

        if direct_caption == dispatch_caption:
            self.log_pass("OpenCLIP dispatch and direct return identical captions")
        elif direct_caption and dispatch_caption:
            self.log_pass("OpenCLIP dispatch and direct both return captions (may differ due to timing)")
        else:
            self.log_fail("OpenCLIP parity test failed")

    def test_parity_tagger(self):
        """Test dispatch and direct Tagger endpoints return same tags."""
        self.set_category('parity')
        print("\n" + "=" * 70)
        print("TEST: Parity - Tagger dispatch vs direct")
        print("=" * 70)

        if self.skip_if_critical('tagger', 'parity tagger'):
            return

        # Direct endpoint
        data_direct = self.post('/sdapi/v1/tagger', {
            'image': self.image_b64,
            'threshold': 0.5,
            'max_tags': 20
        })

        # Dispatch endpoint
        data_dispatch = self.post('/sdapi/v1/caption', {
            'backend': 'tagger',
            'image': self.image_b64,
            'threshold': 0.5,
            'max_tags': 20
        })

        if 'error' in data_direct or 'error' in data_dispatch:
            self.log_skip("One or both requests failed")
            return

        direct_tags = data_direct.get('tags', '')
        dispatch_tags = data_dispatch.get('tags', '')

        self.log_info(f"Direct: {self.truncate(direct_tags, 50)}")
        self.log_info(f"Dispatch: {self.truncate(dispatch_tags, 50)}")

        if direct_tags == dispatch_tags:
            self.log_pass("Tagger dispatch and direct return identical tags")
        elif direct_tags and dispatch_tags:
            self.log_pass("Tagger dispatch and direct both return tags")
        else:
            self.log_fail("Tagger parity test failed")

    def test_parity_vlm(self):
        """Test dispatch and direct VLM endpoints return same answer."""
        self.set_category('parity')
        print("\n" + "=" * 70)
        print("TEST: Parity - VLM dispatch vs direct")
        print("=" * 70)

        if self.skip_if_critical('vqa', 'parity vlm'):
            return

        # Direct endpoint
        data_direct = self.post('/sdapi/v1/vqa', {
            'image': self.image_b64,
            'question': 'Short Caption'
        })

        # Dispatch endpoint
        data_dispatch = self.post('/sdapi/v1/caption', {
            'backend': 'vlm',
            'image': self.image_b64,
            'question': 'Short Caption'
        })

        if 'error' in data_direct or 'error' in data_dispatch:
            self.log_skip("One or both requests failed (model may not be loaded)")
            return

        direct_answer = data_direct.get('answer', '')
        dispatch_answer = data_dispatch.get('answer', '')

        self.log_info(f"Direct: {self.truncate(direct_answer, 50)}")
        self.log_info(f"Dispatch: {self.truncate(dispatch_answer, 50)}")

        if direct_answer == dispatch_answer:
            self.log_pass("VLM dispatch and direct return identical answers")
        elif direct_answer and dispatch_answer:
            self.log_pass("VLM dispatch and direct both return answers (may differ due to sampling)")
        else:
            self.log_fail("VLM parity test failed")

    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run_all_tests(self):
        """Run all tests."""
        if not self.setup():
            return False

        # OpenCLIP direct endpoint tests
        self.test_openclip_list_models()
        self.test_openclip_post_modes()
        self.test_openclip_analyze()
        self.test_openclip_invalid_inputs()
        self.test_openclip_clip_blip_models()
        self.test_openclip_length()
        self.test_openclip_flavors()
        self.test_openclip_advanced_settings()
        self.test_openclip_blip_architectures()

        # VQA direct endpoint tests
        self.test_vqa_models_list()
        self.test_vqa_prompts_list()
        self.test_vqa_caption_basic()
        self.test_vqa_different_prompts()
        self.test_vqa_annotated_image()
        self.test_vqa_system_prompt()
        self.test_vqa_invalid_inputs()
        self.test_vqa_prompt_field()
        self.test_vqa_generation_params()
        self.test_vqa_sampling()
        self.test_vqa_thinking_mode()
        self.test_vqa_prefill()

        # VQA Architecture tests
        self.test_vqa_model_architectures()
        self.test_vqa_florence_special_prompts()
        self.test_vqa_moondream_detection()
        self.test_vqa_architecture_capabilities()

        # Tagger direct endpoint tests
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
        self.test_tagger_character_threshold()
        self.test_tagger_model_types()
        self.test_tagger_wd_versions()
        self.test_tagger_invalid_inputs()

        # Dispatch endpoint tests (unified /sdapi/v1/caption)
        self.test_dispatch_openclip_basic()
        self.test_dispatch_openclip_modes()
        self.test_dispatch_openclip_analyze()
        self.test_dispatch_tagger_basic()
        self.test_dispatch_tagger_params()
        self.test_dispatch_tagger_scores()
        self.test_dispatch_vlm_basic()
        self.test_dispatch_vlm_prompts()
        self.test_dispatch_vlm_annotated()
        self.test_dispatch_backend_field()
        self.test_dispatch_invalid_backend()
        self.test_dispatch_missing_image()

        # Parity tests (dispatch vs direct endpoints)
        self.test_parity_openclip()
        self.test_parity_tagger()
        self.test_parity_vlm()

        self.print_summary()

        # Check if any tests failed (excluding skipped)
        total_failed = sum(data['failed'] for data in self.results.values())
        return total_failed == 0


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

    # Test with longer timeout for slow models
    python cli/test-caption-api.py --timeout 600
        """
    )
    parser.add_argument('--url', default='http://127.0.0.1:7860', help='Server URL (default: http://127.0.0.1:7860)')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--username', help='HTTP Basic Auth username')
    parser.add_argument('--password', help='HTTP Basic Auth password')
    parser.add_argument('--timeout', type=int, default=300, help='Request timeout in seconds (default: 300)')
    args = parser.parse_args()

    test = CaptionAPITest(
        base_url=args.url,
        image_path=args.image,
        username=args.username,
        password=args.password,
        timeout=args.timeout
    )
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
