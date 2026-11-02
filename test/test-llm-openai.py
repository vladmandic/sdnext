"""Offline OpenAI-compatible discovery, request translation, and consumer tests.

Run: python test/test-llm-openai.py
All HTTP is mocked; this script does not load models or contact external servers.
"""

import importlib.util
import ast
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

from modules.llm import Backend, ModelRef, Request, Response, resolve_model
from modules.llm.openai import DEFAULT_MAX_TOKENS, OpenAIBackend, generation_parameters
from modules.llm.catalog import Catalog, refresh_models, is_external, model_info, display_name, resolve_display_name
from modules.llm.ui import bind_token_limit, bind_display_names
from pydantic import BaseModel, Field

spec = importlib.util.spec_from_file_location('llm_contract_tests', ROOT / 'test/test-llm-backends.py')
contract_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract_tests)


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


class DisplayLabelTests(unittest.TestCase):
    def test_enhancement_caption_and_saved_values_resolve_to_original_ids(self):
        shared = types.ModuleType('modules.shared')
        shared.opts = types.SimpleNamespace(llm_api_url='http://llm.example.com/v1', llm_api_key='', llm_api_timeout=120, save=Mock())
        catalog = Catalog()
        with patch.dict(sys.modules, {'modules.shared': shared}), patch('modules.shared', shared, create=True), patch('modules.llm.catalog.catalog', catalog), patch('modules.llm.openai.urlopen', return_value=json_response({'data': [model_entry()]})):
            info = catalog.backend(discover=True).models[0]
            label = catalog.display_name(info.selection)
            symbols = types.SimpleNamespace(vision='[vision]', reasoning='[thinking]', cloud='[cloud]')
            namespace = {
                'ui_symbols': symbols, 'Options': types.SimpleNamespace(img2img=[], cloud=[]),
                'is_external': is_external, 'external_display_name': display_name, 'resolve_display_name': resolve_display_name,
                'is_vision_model': lambda value: bool(model_info(value) and model_info(value).vision), 'is_thinking_model': lambda _value: False,
            }
            tree = ast.parse((ROOT / 'scripts/prompt_enhance/helpers.py').read_text(encoding='utf-8'))
            methods = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('get_model_display_name', 'get_model_repo_from_display', 'model_value_to_config', 'model_value_from_config')]
            exec(compile(ast.Module(body=methods, type_ignores=[]), 'helpers.py', 'exec'), namespace) # pylint: disable=exec-used
            decorated = namespace['get_model_display_name'](info.selection)
            self.assertEqual(decorated, label + ' [vision]')
            self.assertEqual(namespace['get_model_repo_from_display'](decorated), info.selection)
            self.assertEqual(namespace['model_value_to_config'](decorated), info.selection)
            self.assertEqual(namespace['model_value_from_config'](info.selection), decorated)
            self.assertEqual(namespace['model_value_to_config']('local/model'), 'local/model')
            for empty in (None, ''):
                self.assertIs(namespace['model_value_to_config'](empty), empty)
                self.assertIs(namespace['model_value_from_config'](empty), empty)

            tree = ast.parse((ROOT / 'modules/caption/models_def.py').read_text(encoding='utf-8'))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('get_vlm_repo', 'get_vlm_models', 'get_vlm_model_choices')]
            caption_namespace = {'vlm_models': {'Local': 'local/model'}}
            exec(compile(ast.Module(body=functions, type_ignores=[]), 'models_def.py', 'exec'), caption_namespace) # pylint: disable=exec-used
            self.assertIn(label, caption_namespace['get_vlm_model_choices']())
            self.assertEqual(caption_namespace['get_vlm_models']()[info.selection], info.selection)
            self.assertEqual(caption_namespace['get_vlm_repo'](label), info.selection)


class ExternalEnhanceTests(unittest.TestCase):
    # Reuse only the fixture, not the local/cloud test cases from PR1.
    setUp = contract_tests.EnhanceTests.setUp

    def test_external_dispatch_does_not_load_local_qwen(self):
        self.namespace['generation_parameters'] = generation_parameters

        class FakeExternal(Backend):
            def model_info(self, model):
                return types.SimpleNamespace(vision=False)

            def generate(self, model, request):
                self.model, self.request = model, request
                return Response('answer', reasoning='trace')

        backend = FakeExternal()
        self.namespace['get_backend'] = lambda *_args: backend
        response = self.script.enhance(model='external/Qwen%2Flocal-thinking', prompt='question', temperature=0, seed=1, return_thinking=True)
        self.assertEqual(response, ('answer', 'trace'))
        self.assertEqual(backend.model, ModelRef('openai', 'Qwen/local-thinking'))
        self.assertEqual(backend.request.parameters['temperature'], 0)
        self.assertEqual(backend.request.parameters['max_tokens'], DEFAULT_MAX_TOKENS)
        self.assertIsNone(backend.request.thinking)
        self.script.load.assert_not_called()
        self.script._generate_local.assert_not_called()
        self.assertFalse(self.script.busy)

    def test_explicit_enhancement_limit_is_not_increased(self):
        self.namespace['generation_parameters'] = generation_parameters

        class FakeExternal(Backend):
            def model_info(self, model):
                return types.SimpleNamespace(vision=False)

            def generate(self, model, request):
                self.request = request
                return Response('answer')

        backend = FakeExternal()
        self.namespace['get_backend'] = lambda *_args: backend
        result = self.script.enhance(model='external/Qwen', prompt='question', seed=1, max_tokens=128, default_max_tokens=8192)
        self.assertEqual(result, 'answer')
        self.assertEqual(backend.request.parameters['max_tokens'], 128)

    def test_enhancement_ui_changes_only_untouched_defaults(self):
        namespace = {'gr': types.SimpleNamespace(update=lambda **kwargs: kwargs), 'DEFAULT_MAX_TOKENS': DEFAULT_MAX_TOKENS, 'get_model_repo_from_display': lambda value: value}
        cls = contract_tests.load_consumer('scripts/prompt_enhance_ext.py', 'PromptEnhanceScript', ['update_token_limit'], namespace)
        script = cls()
        script.options = types.SimpleNamespace(max_tokens=512)
        self.assertEqual(script.update_token_limit('external/Qwen', 512, False), {'value': DEFAULT_MAX_TOKENS})
        self.assertEqual(script.update_token_limit('external/Qwen', 128, True), {'value': 128})
        self.assertEqual(script.update_token_limit('Qwen/local', DEFAULT_MAX_TOKENS, False), {'value': 512})

    def test_structured_remote_output_is_not_cleaned_as_prose(self):
        self.namespace['generation_parameters'] = generation_parameters

        class FakeExternal(Backend):
            def model_info(self, model):
                return types.SimpleNamespace(vision=False)

            def generate(self, model, request):
                return Response(self.answer)

        backend = FakeExternal()
        self.namespace['get_backend'] = lambda *_args: backend
        self.script.clean = Mock(side_effect=AssertionError('must not apply prose cleanup'))
        for style, answer in [('JSON', '{"prompt": "A red fox"}'), ('None', '["Scene 1: A red fox", "Scene 2: A blue lake"]'), ('Standard', 'Scene 1: A red fox')]:
            with self.subTest(style=style):
                backend.answer = answer
                response = self.script.enhance(model='external/Qwen', prompt='question', seed=1, prompt_style=style)
                self.assertEqual(response, answer)


class APIContractTests(unittest.TestCase):
    def test_response_models_preserve_remote_metadata(self):
        for path, name, extra in [
            ('modules/api/caption.py', 'ItemVLMModel', {'repo': 'external/Qwen', 'prompts': ['Normal Caption'], 'capabilities': ['caption']}),
            ('modules/api/models.py', 'ItemPromptEnhanceModel', {}),
        ]:
            with self.subTest(name=name):
                tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
                namespace = {'BaseModel': BaseModel, 'Field': Field}
                exec(compile(ast.Module(body=[cls], type_ignores=[]), path, 'exec'), namespace) # pylint: disable=exec-used
                data = {'name': 'external/Qwen', 'provider': 'openai', 'model_id': 'Qwen', 'display_name': 'Qwen friendly', 'vision': True, 'thinking': None, 'context_length': 8192, **extra}
                self.assertEqual(namespace[name](**data).model_dump(), data)

    def test_prompt_enhance_get_route_is_registered_with_response_model(self):
        tree = ast.parse((ROOT / 'modules/api/api.py').read_text(encoding='utf-8'))
        registrations = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_api_route' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '/sdapi/v1/prompt-enhance']
        get_route = next(node for node in registrations if any(keyword.arg == 'methods' and ast.literal_eval(keyword.value) == ['GET'] for keyword in node.keywords))
        self.assertEqual(ast.unparse(get_route.args[1]), 'self.process.get_prompt_enhance_models')
        self.assertTrue(any(keyword.arg == 'response_model' for keyword in get_route.keywords))

    def test_caption_tabs_reconcile_shared_backend_controls(self):
        tree = ast.parse((ROOT / 'modules/ui_caption.py').read_text(encoding='utf-8'))
        binding = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and ast.unparse(node.target) == '(model, tab, load_button, unload_button)')
        model, tab = Mock(), Mock()
        namespace = {
            'vlm_model': model, 'vlm_caption_tab': tab, 'vlm_load_btn': object(), 'vlm_unload_btn': object(),
            'analyze_model': Mock(), 'vlm_analyze_tab': Mock(), 'analyze_load_btn': object(), 'analyze_unload_btn': object(),
            **{key: object() for key in ('vlm_thinking_mode', 'vlm_num_beams', 'vlm_custom_args', 'vlm_max_tokens', 'token_limit_model', 'token_limit_explicit')},
            'update_vlm_backend_controls': object(),
        }
        exec(compile(ast.Module(body=[binding], type_ignores=[]), 'ui_caption.py', 'exec'), namespace) # pylint: disable=exec-used
        for selection, caption_tab, load_button, unload_button in [('vlm_model', 'vlm_caption_tab', 'vlm_load_btn', 'vlm_unload_btn'), ('analyze_model', 'vlm_analyze_tab', 'analyze_load_btn', 'analyze_unload_btn')]:
            for event in (namespace[selection].change, namespace[caption_tab].select):
                event.assert_called_once()
                events = event.call_args.kwargs
                self.assertEqual(events['inputs'], [namespace[selection], namespace['vlm_max_tokens'], namespace['token_limit_explicit']])
                self.assertEqual(events['outputs'][:2], [namespace[load_button], namespace[unload_button]])
                self.assertIs(events['fn'], namespace['update_vlm_backend_controls'])


class ExternalCaptionTests(unittest.TestCase):
    def test_caption_and_analysis_defaults_ignore_local_budget_but_honor_overrides(self):
        tree = ast.parse((ROOT / 'modules/caption/vqa.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_external_kwargs')
        overrides = {}
        namespace = {
            '_get_overrides': lambda: overrides, 'generation_parameters': generation_parameters,
            'shared': types.SimpleNamespace(opts=types.SimpleNamespace(
                caption_vlm_max_length=128, caption_vlm_temperature=0.75, caption_vlm_do_sample=True,
                caption_vlm_top_k=0, caption_vlm_top_p=0, caption_vlm_num_beams=1, caption_vlm_custom_args='',
            )),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'vqa.py', 'exec'), namespace) # pylint: disable=exec-used
        self.assertEqual(namespace['get_external_kwargs']()['max_tokens'], DEFAULT_MAX_TOKENS)
        overrides['max_tokens'] = 128
        self.assertEqual(namespace['get_external_kwargs']()['max_tokens'], 128)
        overrides['max_tokens'] = None
        self.assertEqual(namespace['get_external_kwargs']()['max_tokens'], DEFAULT_MAX_TOKENS)

    def test_caption_ui_defaults_explicit_limits_and_local_preferences(self):
        tree = ast.parse((ROOT / 'modules/ui_caption.py').read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('update_vlm_backend_controls', 'update_vlm_token_limit')]
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(caption_vlm_max_length=512, save=Mock()))
        namespace = {
            'gr': types.SimpleNamespace(update=lambda **kwargs: kwargs), 'shared': shared,
            'is_external': lambda value: value.startswith('external/'), 'DEFAULT_MAX_TOKENS': DEFAULT_MAX_TOKENS,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'ui_caption.py', 'exec'), namespace) # pylint: disable=exec-used
        update = namespace['update_vlm_backend_controls']
        self.assertEqual(update('external/Qwen', 512, False)[-2], {'value': DEFAULT_MAX_TOKENS})
        self.assertEqual(update('external/Qwen', 128, True)[-2], {'value': 128})
        self.assertEqual(update('Qwen/local', DEFAULT_MAX_TOKENS, False)[-2], {'value': 512})
        namespace['update_vlm_token_limit'](128, 'external/Qwen')
        self.assertEqual(shared.opts.caption_vlm_max_length, 512)
        shared.opts.save.assert_not_called()
        namespace['update_vlm_token_limit'](256, 'Qwen/local')
        self.assertEqual(shared.opts.caption_vlm_max_length, 256)
        shared.opts.save.assert_called_once()

    def test_external_caption_never_calls_local_handlers_or_offload(self):
        class Image:
            mode = 'RGB'
            width, height = 32, 32

        state = types.SimpleNamespace(begin=Mock(return_value='job'), end=Mock())
        shared = types.SimpleNamespace(state=state, opts=types.SimpleNamespace(caption_vlm_thinking_mode=False, caption_vlm_system='system'))
        offload, sd_models, devices = Mock(), Mock(), Mock()
        namespace = {
            'shared': shared, 'time': __import__('time'), 'vlm_prefill': '',
            'Image': types.SimpleNamespace(Image=Image), 'vlm_prompt_mapping': {'Normal Caption': '<DETAILED_CAPTION>'},
            'get_internal_prompt': lambda _question, _prompt: '<DETAILED_CAPTION>',
            'get_vlm_repo': lambda value: value, 'vlm_models': {}, 'uses_qwen_handler': lambda _value: False,
            'get_external_kwargs': lambda: {'max_tokens': 100, 'temperature': 0},
            'get_keep_thinking': lambda: True, 'clean': lambda value, *_args: value,
            'strip_think_xml_tags': lambda value, **_kwargs: value,
            'vlm_prompt_reverse_mapping': {'DETAILED CAPTION': 'Describe this image.'},
            'log': Mock(), 'debug': Mock(), 'errors': Mock(), 'offload_aux': offload,
            'sd_models': sd_models, 'devices': devices,
        }
        cls = contract_tests.load_consumer('modules/caption/vqa.py', 'VQA', ['caption'], namespace)

        class FakeExternal(Backend):
            def generate(self, model, request):
                self.model, self.request = model, request
                return Response('caption', reasoning='trace')

        backend = FakeExternal()
        namespace['get_backend'] = lambda *_args: backend
        instance = cls()
        instance._generate_local = Mock(side_effect=AssertionError('no local inference'))
        answer = instance.caption(question='Normal Caption', image=Image(), model_name='external/Qwen%2Fvision', prefill='', thinking_mode=False)
        self.assertEqual(answer, 'Reasoning:\ntrace\n\nAnswer:\ncaption')
        self.assertEqual(backend.model, ModelRef('openai', 'Qwen/vision'))
        self.assertEqual(backend.request.prompt, 'Describe this image.')
        self.assertEqual(backend.request.system, 'system')
        self.assertIsNone(backend.request.thinking)
        instance._generate_local.assert_not_called()
        offload.assert_not_called()
        sd_models.set_caption_load_options.assert_not_called()
        sd_models.set_huggingface_options.assert_not_called()
        devices.torch_gc.assert_not_called()
        state.end.assert_called_once_with('job')
        self.assertIsNone(instance._generation_overrides)


class StoryBudgetTests(unittest.TestCase):
    def test_every_external_story_stage_uses_shared_policy(self):
        for slide_count in (4, 99):
            with self.subTest(slide_count=slide_count):
                requests = []

                def enhance(requests=requests, **kwargs):
                    requests.append(generation_parameters(kwargs.get('max_tokens'), 0.75, default_max_tokens=kwargs.get('default_max_tokens')))
                    return 'answer'

                enhancer = types.SimpleNamespace(session=lambda _model: __import__('contextlib').nullcontext(object()), enhance=enhance)
                namespace = {
                    'ExitStack': __import__('contextlib').ExitStack, 'time': __import__('time'), 'log': Mock(), 'debug_log': Mock(),
                    'shared': types.SimpleNamespace(state=types.SimpleNamespace(begin=Mock(), end=Mock()), sd_model=object()),
                    'extra_networks': types.SimpleNamespace(re_extra_net=__import__('re').compile(r'<[^>]+>')),
                    'get_model_repo_from_display': lambda name: name, 'is_vision_model': lambda _model: True,
                    'processing': types.SimpleNamespace(get_fixed_seed=lambda value: value, process_init=Mock()),
                }
                cls = contract_tests.load_consumer('scripts/story.py', 'StoryScript', ['process'], namespace)
                script = cls()
                script.get_enhancer = lambda _p, enhancer=enhancer: enhancer
                script.get_control_image = lambda _p: object()
                script.parse_frame_prompts = lambda _response, _count: ['one', 'two']
                p = types.SimpleNamespace(all_prompts=['story'], all_negative_prompts=['negative'], seed=1, extra_generation_params={})
                script.process(p, True, slide_count, 'external/Qwen', True, True, False)
                self.assertIsNone(p.story_mode_error)
                self.assertEqual(len(requests), 4) # caption, story planning, two enhanced frames
                self.assertEqual([request['max_tokens'] for request in requests], [DEFAULT_MAX_TOKENS, max(DEFAULT_MAX_TOKENS, min(8192, 256 + slide_count * 128)), DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS])


class SavedTokenLimitTests(unittest.TestCase):
    def test_display_aliases_restore_and_save_only_canonical_ids(self):
        class Dropdown:
            visible = True
            label = 'LLM model'
            choices = ['local/model', 'llm.example.com / one', 'llm.example.com / two']
            value = 'local/model'

        gr = types.SimpleNamespace(Dropdown=Dropdown)
        for name in ('Slider', 'Radio', 'Checkbox', 'Textbox', 'Number', 'Button', 'Accordion', 'Tabs'):
            setattr(gr, name, type(name, (), {}))
        namespace = {'gr': gr, 'ToolButton': type('ToolButton', (), {}), 'debug_ui': False, 'log': Mock()}
        cls = contract_tests.load_consumer('modules/ui_loadsave.py', 'UiLoadsave', ['add_component', 'iter_all', 'iter_changes', 'ui_apply'], namespace)
        loader = cls()
        loader.filename = 'not-written.json'
        loader.ui_settings = {'enhancer/LLM model/value': 'external/one'}
        loader.ui_defaults, loader.component_mapping, loader.component_open = {}, {}, {}
        component = Dropdown()
        labels = {'external/one': 'llm.example.com / one', 'external/two': 'llm.example.com / two'}
        ids = {label: selection for selection, label in labels.items()}
        bind_display_names(component, lambda value: labels.get(value, value), lambda value: ids.get(value, value))
        loader.add_component('enhancer/LLM model', component)
        self.assertEqual(component.value, 'llm.example.com / one')
        self.assertEqual(loader.iter_all(['llm.example.com / one']), [None])
        changes = list(loader.iter_changes(['llm.example.com / two']))
        self.assertEqual(changes[0][2], 'external/two')
        loader.read_from_file = lambda: dict(loader.ui_settings)
        loader.write_to_file = Mock()
        loader.ui_apply('llm.example.com / two')
        loader.write_to_file.assert_called_once_with({'enhancer/LLM model/value': 'external/two'})

    def test_only_the_restored_active_tab_controls_caption_defaults(self):
        for selected, caption, analyze in [(0, 'external/Caption', 'Qwen/local'), (1, 'Qwen/local', 'external/Analyze')]:
            for saved_limit in (None, 128):
                with self.subTest(selected=selected, saved_limit=saved_limit):
                    modes = types.SimpleNamespace(selected=selected)
                    model = types.SimpleNamespace(value='Qwen/local')
                    analyze_model = types.SimpleNamespace(value='Qwen/local')
                    limit = types.SimpleNamespace(value=512)
                    explicit = types.SimpleNamespace(value=False)
                    active = types.SimpleNamespace(value='Qwen/local')
                    bind_token_limit(model, limit, explicit, 512, active_model=active, is_active=lambda modes=modes: modes.selected == 0)
                    bind_token_limit(analyze_model, limit, explicit, 512, active_model=active, is_active=lambda modes=modes: modes.selected == 1)
                    # UiLoadsave traverses caption model, shared slider, analyze model.
                    model.init_field(caption)
                    if saved_limit is not None:
                        limit.value = saved_limit
                        limit.init_field(saved_limit)
                    analyze_model.init_field(analyze)
                    self.assertEqual(active.value, caption if selected == 0 else analyze)
                    self.assertEqual(limit.value, DEFAULT_MAX_TOKENS if saved_limit is None else saved_limit)

    def test_saved_model_and_explicit_cap_restoration(self):
        """Exercise the real UiLoadsave path, which does not emit input events."""
        class Slider:
            visible = True
            label = 'Max tokens'
            minimum, maximum, step = 10, 32768, 1
            value = 512

        class Dropdown:
            visible = True
            label = 'LLM model'
            choices = ['Qwen/local', 'external/Qwen']
            value = 'Qwen/local'

        gr = types.SimpleNamespace(Slider=Slider, Dropdown=Dropdown)
        for name in ('Radio', 'Checkbox', 'Textbox', 'Number', 'Button', 'Accordion', 'Tabs'):
            setattr(gr, name, type(name, (), {}))
        namespace = {'gr': gr, 'ToolButton': type('ToolButton', (), {}), 'debug_ui': False, 'log': Mock()}
        cls = contract_tests.load_consumer('modules/ui_loadsave.py', 'UiLoadsave', ['add_component'], namespace)
        for saved_limit in (None, 128, 512):
            with self.subTest(saved_limit=saved_limit):
                loader = cls()
                loader.ui_settings = {'enhancer/LLM model/value': 'external/Qwen'}
                if saved_limit is not None:
                    loader.ui_settings['enhancer/Max tokens/value'] = saved_limit
                loader.ui_defaults, loader.component_mapping, loader.component_open = {}, {}, {}
                model, limit = Dropdown(), Slider()
                explicit = types.SimpleNamespace(value=False)
                active_model = types.SimpleNamespace(value='Qwen/local')
                bind_token_limit(model, limit, explicit, 512, active_model=active_model)
                loader.add_component('enhancer/LLM model', model)
                loader.add_component('enhancer/Max tokens', limit)
                self.assertEqual(model.value, 'external/Qwen')
                self.assertEqual(active_model.value, 'external/Qwen')
                self.assertEqual(limit.value, DEFAULT_MAX_TOKENS if saved_limit is None else saved_limit)
                self.assertEqual(explicit.value, saved_limit is not None)
                # A later model restoration must not overwrite an explicit cap.
                model.init_field('external/Qwen')
                self.assertEqual(limit.value, DEFAULT_MAX_TOKENS if saved_limit is None else saved_limit)


if __name__ == '__main__':
    unittest.main()
