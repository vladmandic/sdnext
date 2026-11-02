"""Offline OpenAI-compatible discovery, request translation, and consumer tests.

Run: python test/test-llm-openai.py
All HTTP is mocked; this script does not load models or contact external servers.
"""

import io
import json
from pathlib import Path
import sys
import types
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.llm import ModelRef, Request, resolve_model
from modules.llm.openai import DEFAULT_MAX_TOKENS, OpenAIBackend, generation_parameters
from modules.llm.catalog import Catalog, refresh_models


def model_entry(model_id='Qwen/vision-thinking', image=True, output='text'):
    return {
        'id': model_id, 'name': 'Friendly model',
        'architecture': {'input_modalities': ['text', 'image'] if image else ['text'], 'output_modalities': [output]},
        'context_length': 16384, 'status': {'value': 'unloaded'},
        # This is intentionally not an exhaustive parameter list.
        'supported_parameters': ['tools', 'tool_choice'],
    }


def json_response(value):
    return io.BytesIO(json.dumps(value).encode('utf-8'))


class OpenAITests(unittest.TestCase):
    def setUp(self):
        self.backend = OpenAIBackend('http://example.invalid/v1/', 'secret', 17)

    def discover(self, entries=None):
        with patch('modules.llm.openai.urlopen', return_value=json_response({'data': entries if entries is not None else [model_entry()]})) as http:
            models = self.backend.list_models()
        self.assertEqual(http.call_count, 1)
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, 'http://example.invalid/v1/models')
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
        self.assertEqual(http.call_args.kwargs, {'timeout': 17})
        return models

    def test_discovery_without_loading_or_name_heuristics(self):
        info = self.discover()[0]
        self.assertTrue(info.vision)
        self.assertTrue(info.text_generation)
        self.assertIsNone(info.thinking)
        self.assertEqual(info.context_length, 16384)
        self.assertEqual(info.selection, 'external/Qwen%2Fvision-thinking')
        self.assertEqual(resolve_model(info.selection), info.ref)
        self.assertEqual(info.ref, ModelRef('openai', 'Qwen/vision-thinking'))

    def test_missing_capabilities_have_no_fallback(self):
        entry = model_entry()
        del entry['architecture']
        with self.assertRaisesRegex(ValueError, 'missing architecture'):
            self.discover([entry])
        self.assertFalse(self.backend.discovered)

    def test_invalid_and_duplicate_discovery(self):
        for entries in [[model_entry(), model_entry()], [{'id': ''}], [None], [{**model_entry(), 'context_length': True}], [{**model_entry(), 'architecture': {'input_modalities': [], 'output_modalities': ['text']}}]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.discover(entries)

    def test_failed_refresh_clears_stale_models(self):
        self.discover()
        with patch('modules.llm.openai.urlopen', side_effect=URLError('secret')), self.assertRaisesRegex(RuntimeError, 'connection failed'):
            self.backend.list_models()
        self.assertEqual(self.backend.models, ())

    def test_empty_catalog_and_non_text_models(self):
        self.assertEqual(self.discover([]), ())
        info = self.discover([model_entry(output='audio')])[0]
        self.assertFalse(info.text_generation)
        with self.assertRaisesRegex(ValueError, 'text generation'):
            self.backend.generate(info.ref, Request('question'))

    def test_text_request_exact_model_and_reasoning_response(self):
        info = self.discover()[0]
        parameters = generation_parameters(200, 0, top_k=10, top_p=0.9, penalty=1.05, seed=4)
        completion = {'choices': [{'message': {'content': ' answer ', 'reasoning_content': ' trace '}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 3}}
        with patch('modules.llm.openai.urlopen', return_value=json_response(completion)) as http:
            response = self.backend.generate(info.ref, Request('question', system='system', parameters=parameters))
        request = http.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body, {
            'model': 'Qwen/vision-thinking', 'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}],
            'stream': False, 'max_tokens': 200, 'temperature': 0, 'top_k': 10, 'top_p': 0.9, 'repeat_penalty': 1.05, 'seed': 4,
        })
        self.assertEqual(response.text, 'answer')
        self.assertEqual(response.reasoning, 'trace')
        self.assertEqual(response.usage, {'completion_tokens': 3})

    def test_vision_and_prefill_message_payload(self):
        info = self.discover()[0]
        image = Mock()
        image.convert.return_value.save.side_effect = lambda stream, **_kwargs: stream.write(b'image bytes')
        with patch('modules.llm.openai.urlopen', return_value=json_response({'choices': [{'message': {'content': 'answer'}}]})) as http:
            self.backend.generate(info.ref, Request('question', image=image, prefill='prefix'))
        messages = json.loads(http.call_args.args[0].data)['messages']
        self.assertEqual(messages[-1], {'role': 'assistant', 'content': 'prefix'})
        self.assertEqual(messages[0]['content'][0], {'type': 'text', 'text': 'question'})
        self.assertEqual(messages[0]['content'][1]['image_url']['url'], 'data:image/jpeg;base64,aW1hZ2UgYnl0ZXM=')

    def test_text_only_model_rejects_image_before_inference(self):
        info = self.discover([model_entry(image=False)])[0]
        with patch('modules.llm.openai.urlopen') as http, self.assertRaisesRegex(ValueError, 'image input'):
            self.backend.generate(info.ref, Request('question', image=object()))
        http.assert_not_called()

    def test_unsupported_controls_and_unknown_identity(self):
        info = self.discover()[0]
        for model, request in [
            (ModelRef('inprocess', info.ref.model_id), Request('question')),
            (ModelRef('openai', 'missing'), Request('question')),
            (info.ref, Request('question', thinking=True)),
            (info.ref, Request('question', parameters={'num_beams': 4})),
            (info.ref, Request('question', parameters={'max_tokens': 16384})),
        ]:
            with self.subTest(model=model, request=request), patch('modules.llm.openai.urlopen') as http, self.assertRaises(ValueError):
                self.backend.generate(model, request)
            http.assert_not_called()

    def test_no_key_no_authorization_header(self):
        backend = OpenAIBackend('http://example.invalid/v1')
        with patch('modules.llm.openai.urlopen', return_value=json_response({'data': []})) as http:
            backend.list_models()
        self.assertIsNone(http.call_args.args[0].get_header('Authorization'))

    def test_invalid_urls_and_timeout(self):
        for url in ['', 'file:///etc/passwd', 'http://user:secret@host/v1', 'http://host/v1?key=secret', 'http://host/v1#fragment']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                OpenAIBackend(url)
        with self.assertRaises(ValueError):
            OpenAIBackend('http://host/v1', timeout=0)

    def test_malformed_completion_and_reasoning_only_are_errors(self):
        info = self.discover()[0]
        for body in [{}, {'choices': []}, {'choices': [{'message': {'content': ['not text']}}]}, {'choices': [{'message': {'content': None, 'reasoning_content': 'trace'}, 'finish_reason': 'length'}]}]:
            with self.subTest(body=body), patch('modules.llm.openai.urlopen', return_value=json_response(body)), self.assertRaises(ValueError):
                self.backend.generate(info.ref, Request('question'))

    def test_http_errors_do_not_echo_secrets(self):
        error = HTTPError('http://host/v1/models', 401, 'secret', {}, io.BytesIO(b'secret'))
        with patch('modules.llm.openai.urlopen', side_effect=error), self.assertRaisesRegex(RuntimeError, 'HTTP 401') as result:
            self.backend.list_models()
        self.assertNotIn('secret', str(result.exception))

    def test_parameter_translation_does_not_drop_local_only_features(self):
        self.assertEqual(generation_parameters(50, 0.8, sample=False), {'max_tokens': 50, 'temperature': 0})
        for options in [{'min_tokens': 1}, {'num_beams': 4}, {'custom_args': 'x=1'}, {'process_words': 'word'}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                generation_parameters(50, 0.8, **options)

    def test_shared_default_and_explicit_limits(self):
        self.assertEqual(generation_parameters(None, 0.8)['max_tokens'], DEFAULT_MAX_TOKENS)
        self.assertEqual(generation_parameters(None, 0.8, default_max_tokens=768)['max_tokens'], DEFAULT_MAX_TOKENS)
        self.assertEqual(generation_parameters(None, 0.8, default_max_tokens=16384)['max_tokens'], 16384)
        self.assertEqual(generation_parameters(128, 0.8, default_max_tokens=8192)['max_tokens'], 128)
        with self.assertRaises(ValueError):
            generation_parameters(0, 0.8)

    def test_backend_default_for_direct_requests(self):
        info = self.discover()[0]
        completion = {'choices': [{'message': {'content': 'answer'}}]}
        with patch('modules.llm.openai.urlopen', return_value=json_response(completion)) as http:
            self.backend.generate(info.ref, Request('question'))
        self.assertEqual(json.loads(http.call_args.args[0].data)['max_tokens'], DEFAULT_MAX_TOKENS)


class CatalogTests(unittest.TestCase):
    def test_domain_display_alias_never_shadows_existing_repository_ids(self):
        for host, remote_id, existing_id in [('google', 'gemma-3-4b-it', 'google/gemma-3-4b-it'), ('xai', 'grok-3-latest', 'xai/grok-3-latest')]:
            with self.subTest(host=host):
                shared = types.ModuleType('modules.shared')
                shared.opts = types.SimpleNamespace(llm_api_url=f'http://{host}/v1', llm_api_key='', llm_api_timeout=120)
                catalog = Catalog()
                expected = resolve_model(existing_id)
                with patch.dict(sys.modules, {'modules.shared': shared}), patch('modules.shared', shared, create=True), patch('modules.llm.openai.urlopen', return_value=json_response({'data': [model_entry(remote_id)]})):
                    info = catalog.backend(discover=True).models[0]
                    label = catalog.display_name(info.selection)
                    self.assertNotEqual(label, existing_id)
                    self.assertEqual(resolve_model(catalog.selection(existing_id)), expected)
                    self.assertEqual(resolve_model(catalog.selection(label)), info.ref)

    def test_domain_labels_do_not_change_selection_or_provider_identity(self):
        for url, prefix in [
            ('http://llm.example.com/v1', 'llm.example.com'),
            ('http://llm.example.com:80/custom/v1', 'llm.example.com'),
            ('https://llama.example:443/v1', 'llama.example'),
            ('http://localhost:8080/v1', 'localhost:8080'),
            ('http://[::1]:8080/v1', '[::1]:8080'),
            ('http://google/v1', 'google'),
        ]:
            with self.subTest(url=url):
                shared = types.ModuleType('modules.shared')
                shared.opts = types.SimpleNamespace(llm_api_url=url, llm_api_key='', llm_api_timeout=120)
                catalog = Catalog()
                with patch.dict(sys.modules, {'modules.shared': shared}), patch('modules.shared', shared, create=True), patch('modules.llm.openai.urlopen', return_value=json_response({'data': [model_entry()]})):
                    backend = catalog.backend(discover=True)
                    info = backend.models[0]
                    label = catalog.display_name(info.selection)
                    self.assertEqual(label, prefix + ' / Qwen%2Fvision-thinking')
                    self.assertEqual(catalog.selection(label), 'external/Qwen%2Fvision-thinking')
                    self.assertEqual(backend.model_info(resolve_model(catalog.selection(label))), info)
                    self.assertEqual(catalog.selection('unknown.example/model'), 'unknown.example/model')
                    self.assertEqual(catalog.selection('google/gemma-3-4b-it'), 'google/gemma-3-4b-it')
                    self.assertEqual(catalog.display_name('local/model'), 'local/model')
                    shared.opts.llm_api_url = 'http://changed.invalid/v1'
                    self.assertEqual(catalog.models(), ())
                    # A visible stale alias still resolves to an external ID,
                    # never into the local/HuggingFace model loading path.
                    self.assertEqual(catalog.selection(label), info.selection)

    def test_cache_is_invalidated_when_endpoint_or_key_changes(self):
        opts = types.SimpleNamespace(llm_api_url='http://first.invalid/v1', llm_api_key='', llm_api_timeout=120)
        shared = types.ModuleType('modules.shared')
        shared.opts = opts
        catalog = Catalog()
        with patch.dict(sys.modules, {'modules.shared': shared}), patch('modules.shared', shared, create=True), patch('modules.llm.openai.urlopen', return_value=json_response({'data': [model_entry()]})) as http:
            self.assertEqual(catalog.models(), ())
            http.assert_not_called()
            first = catalog.backend(discover=True)
            self.assertEqual(len(catalog.models()), 1)
            self.assertIs(catalog.backend(discover=True), first)
            self.assertEqual(http.call_count, 1)
            opts.llm_api_key = 'changed'
            self.assertEqual(catalog.models(), ())
            self.assertIsNot(catalog.backend(), first)
            opts.llm_api_url = ''
            self.assertEqual(catalog.models(), ())
            self.assertEqual(catalog.refresh(), ())

    def test_discovery_failure_can_preserve_existing_catalog_consumers(self):
        logger = types.ModuleType('modules.logger')
        logger.log = Mock()
        with patch('modules.llm.catalog.catalog') as catalog, patch.dict(sys.modules, {'modules.logger': logger}):
            catalog.refresh.side_effect = RuntimeError('unavailable')
            refresh_models(tolerate_errors=True)
            logger.log.error.assert_called_once()
            with self.assertRaises(RuntimeError):
                refresh_models()


if __name__ == '__main__':
    unittest.main()
