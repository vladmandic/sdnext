"""Offline auxiliary-LLM contract and consumer regression tests.

Run: python test/test-llm-backends.py
Consumer methods are compiled from their source AST to exercise their actual
dispatch/postprocessing without importing the GPU runtime or loading models.
"""

import ast
import __future__
from contextlib import contextmanager, nullcontext, ExitStack
from pathlib import Path
import re
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.llm import Backend, InProcessBackend, ModelRef, Request, Response, get_backend, resolve_model


def load_consumer(path, class_name, methods, namespace):
    """Use real method bodies with only their heavyweight collaborators stubbed."""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.Module(body=[cls], type_ignores=[])
    namespace.update(Request=Request, Response=Response, get_backend=get_backend, resolve_model=resolve_model)
    namespace.setdefault('sd_hijack_te', types.SimpleNamespace(with_compile_stance=lambda _target: lambda function: function))
    exec(compile(module, str(ROOT / path), 'exec', flags=__future__.annotations.compiler_flag), namespace) # pylint: disable=exec-used
    return namespace[class_name]


class BackendTests(unittest.TestCase):
    def test_legacy_model_identity(self):
        for name, provider, model_id in [
            ('google/gemini-3.8-flash', 'gemini', 'gemini-3.8-flash'),
            ('gemini-3.8-flash', 'gemini', 'gemini-3.8-flash'),
            ('xai/grok-3-latest', 'grok', 'grok-3-latest'),
            ('grok-3-latest', 'grok', 'grok-3-latest'),
            ('google/gemma-3-4b-it', 'inprocess', 'google/gemma-3-4b-it'),
            ('Qwen/Qwen3.5-4B', 'inprocess', 'Qwen/Qwen3.5-4B'),
            ('someone/grok-finetune', 'inprocess', 'someone/grok-finetune'),
        ]:
            with self.subTest(name=name):
                self.assertEqual(resolve_model(name), ModelRef(provider, model_id))

    def test_provider_is_not_inferred_from_explicit_model_id(self):
        for model_id in ['Qwen/Qwen3.5-4B', 'google/gemma-3-4b-it', 'gemini-3.8-flash']:
            with self.assertRaisesRegex(ValueError, 'Unknown LLM provider'):
                get_backend(ModelRef('external', model_id), Mock())

    def test_local_passes_request_and_response_unchanged(self):
        model = ModelRef('inprocess', 'a/model')
        request = Request('prompt', image=object(), thinking=None, parameters={'temperature': 0})
        response = Response('answer', reasoning='trace', usage={'output_tokens': 3})
        callback = Mock(return_value=response)
        with get_backend(model, callback).session(model) as session:
            self.assertIs(session.generate(request), response)
        callback.assert_called_once_with('a/model', request)

    def test_backend_mismatch_never_calls_handler(self):
        callback = Mock()
        with self.assertRaises(ValueError):
            InProcessBackend(callback).generate(ModelRef('grok', 'Qwen'), Request('prompt'))
        callback.assert_not_called()
        with self.assertRaises(ValueError):
            get_backend(ModelRef('inprocess', 'Qwen'))

    def test_cloud_preserves_sdk_parameters_and_identity(self):
        for provider, model_id in [('gemini', 'gemini-3.8-flash'), ('grok', 'grok-3-latest')]:
            with self.subTest(provider=provider):
                predict = Mock(return_value='answer')
                module = types.ModuleType(f'modules.caption.{provider}')
                module.predict = predict
                image = object()
                parameters = {'temperature': 0, 'max_output_tokens': 123}
                request = Request('question', 'system', image, 'prefill', False, parameters)
                with patch.dict(sys.modules, {module.__name__: module}), patch(f'modules.caption.{provider}', module, create=True):
                    response = get_backend(ModelRef(provider, model_id)).generate(ModelRef(provider, model_id), request)
                self.assertEqual(response.text, 'answer')
                predict.assert_called_once_with('question', image, model_id, 'system', 'prefill', False, parameters)

    def test_request_options_are_not_shared(self):
        first, second = Request('one'), Request('two')
        first.parameters['temperature'] = 1
        self.assertEqual(second.parameters, {})

    def test_work_scope_can_group_calls_and_cleanup_on_failure(self):
        events = []

        class ScopedBackend(Backend):
            @contextmanager
            def session(self, model):
                events.append('enter')
                try:
                    with super().session(model) as session:
                        yield session
                finally:
                    events.append('exit')

            def generate(self, model, request):
                events.append((model, request.prompt))
                if request.prompt == 'fail':
                    raise RuntimeError('failure')
                return Response('answer')

        model = ModelRef('test', 'model')
        with self.assertRaises(RuntimeError), ScopedBackend().session(model) as session:
            session.generate(Request('caption'))
            session.generate(Request('story'))
            session.generate(Request('fail'))
        self.assertEqual(events, ['enter', (model, 'caption'), (model, 'story'), (model, 'fail'), 'exit'])


class EnhanceTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {
            'get_model_repo_from_display': lambda value: value,
            'is_cloud_model': lambda model: model.startswith(('google/gemini-', 'xai/grok-')),
            'is_vision_model': lambda _model: True,
            'get_system_prompt': lambda system, *_args, **_kwargs: system or 'default system',
            'time': time, 'random': Mock(), 'torch': Mock(), 'Image': types.SimpleNamespace(Image=type(None)),
            'log': Mock(), 'debug_log': Mock(), 're': re,
        }
        cls = load_consumer('scripts/prompt_enhance_ext.py', 'PromptEnhanceScript', ['session', 'enhance', 'extract_thinking'], self.namespace)
        self.script = cls()
        self.script.options = types.SimpleNamespace(default='Qwen/local', min_tokens=0, max_tokens=512, repetition_penalty=1.05, temperature=0.75, top_k=0, top_p=0, thinking_mode=False, do_sample=True)
        self.script.busy = False
        self.script.llm = None
        self.script.prompt = None
        self.script.image = None
        self.script.parent = 'control'
        self.script.load = Mock()
        self.script._generate_local = Mock(side_effect=AssertionError('cloud must not run local inference'))
        self.script.extract = lambda prompt: (prompt, [])
        self.script.get_image = lambda image: image
        self.script.tokenizer = types.SimpleNamespace(all_special_tokens=['<eos>'])
        self.script.clean = lambda response, **_kwargs: response
        self.script.post = lambda response, *_args: response
        self.script.censored = lambda _response: False

    def test_cloud_generation_and_thinking_parity(self):
        for provider, name, parameters in [
            ('gemini', 'google/gemini-3.8-flash', {'temperature': 0.75, 'min_output_tokens': 0, 'max_output_tokens': 123}),
            ('grok', 'xai/grok-3-latest', {'temperature': 0.75}),
        ]:
            with self.subTest(provider=provider):
                module = types.ModuleType(f'modules.caption.{provider}')
                module.predict = Mock(return_value='<think>trace</think>answer')
                with patch.dict(sys.modules, {module.__name__: module}), patch(f'modules.caption.{provider}', module, create=True):
                    result = self.script.enhance(model=name, prompt='question', system='system', max_tokens=123, seed=1, return_thinking=True)
                self.assertEqual(result, ('answer', 'trace'))
                self.assertEqual(module.predict.call_args.args[-1], parameters)
                self.script.load.assert_not_called()
                self.assertFalse(self.script.busy)

    def test_cloud_failure_releases_busy_flag(self):
        module = types.ModuleType('modules.caption.grok')
        module.predict = Mock(side_effect=RuntimeError('SDK initialization failure'))
        with patch.dict(sys.modules, {module.__name__: module}), patch('modules.caption.grok', module, create=True), self.assertRaises(RuntimeError):
            self.script.enhance(model='xai/grok-3-latest', prompt='question', seed=1)
        self.assertFalse(self.script.busy)

    def test_local_dispatch_and_postprocessing(self):
        self.script.llm = object()
        self.script._generate_local = Mock(return_value=Response('decoded', raw_text='<think>trace</think>answer<eos>', usage={'input_tokens': 5, 'output_tokens': 2}, metrics={'generate_time': 1, 'offload_time': 0}))
        self.assertEqual(self.script.enhance(model='Qwen/local', prompt='question', seed=1, return_thinking=True), ('answer', 'trace'))
        self.script.load.assert_called_once_with('Qwen/local', use_openai=False)
        self.assertEqual(self.script._generate_local.call_args.args[0], 'Qwen/local')
        self.assertFalse(self.script.busy)

    def test_local_tokenize_failure_preserves_fallback(self):
        self.script.llm = object()
        self.script._generate_local = Mock(return_value=Response('question', metrics={'tokenize_error': True}))
        self.assertEqual(self.script.enhance(model='Qwen/local', prompt='question', seed=1), 'question')
        self.assertFalse(self.script.busy)

    def test_reuses_work_session_and_rejects_wrong_model(self):
        session = types.SimpleNamespace(model=ModelRef('grok', 'grok-3-latest'), generate=Mock(return_value=Response('answer')))
        self.script.session = Mock(side_effect=AssertionError('must reuse session'))
        self.assertEqual(self.script.enhance(model='xai/grok-3-latest', prompt='question', seed=1, llm_session=session), 'answer')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.script.enhance(model='Qwen/local', prompt='question', llm_session=session)


class CaptionTests(unittest.TestCase):
    def test_specialized_local_dispatch_preserves_arguments(self):
        namespace = {'uses_qwen_handler': lambda name: 'qwen' in name.lower(), 'debug': Mock()}
        cls = load_consumer('modules/caption/vqa.py', 'VQA', ['_generate_local'], namespace)
        instance = cls()
        for model, handler in [('Qwen/Qwen3.5-4B', 'qwen'), ('google/gemma-3-4b-it', 'gemma'), ('mistral/model', 'mistral'), ('org/fastvlm', 'fastvlm')]:
            with self.subTest(model=model):
                callback = Mock(return_value='caption')
                setattr(instance, f'_{handler}', callback)
                image = object()
                request = Request('question', 'system', image, 'prefill', False, {'display_name': 'display'})
                response = get_backend(ModelRef('inprocess', model), instance._generate_local).generate(ModelRef('inprocess', model), request)
                self.assertEqual(response.text, 'caption')
                self.assertEqual(response.metrics['handler'], handler)
                if handler != 'fastvlm':
                    callback.assert_called_once_with('question', image, model, 'system', 'display', 'prefill', False)
                else:
                    callback.assert_called_once_with('question', image, model, 'display')


class LocalInferenceTests(unittest.TestCase):
    def setUp(self):
        class Tensor:
            shape = (1, 9)

            def __getitem__(self, key):
                self.last_slice = key
                return self

        class Inputs(dict):
            def to(self, _device):
                return self

        inputs = Inputs(input_ids=types.SimpleNamespace(shape=(1, 5)))
        self.tokenizer = Mock(return_value=inputs)
        self.tokenizer.apply_chat_template.return_value = 'chat template'
        self.tokenizer.batch_decode.side_effect = [['raw answer'], ['decoded answer']]
        self.devices = types.SimpleNamespace(device=types.SimpleNamespace(type='cpu'), dtype='dtype', llm_context=nullcontext, torch_gc=Mock())
        self.offload = Mock(return_value=0.25)
        self.helpers = types.SimpleNamespace(get_custom_args=Mock(return_value={'top_p': 0.9}), get_default_args=Mock(return_value={}))
        namespace = {
            'set_template': Mock(return_value=[]), 'is_thinking_model': lambda _model: False,
            'debug_log': Mock(), 'log': Mock(), 'errors': Mock(), 'debug_enabled': False,
            'devices': self.devices, 'time': time, 'torch': types.SimpleNamespace(Tensor=Tensor),
            'move_aux_to_gpu': Mock(return_value=0.1), 'offload_aux': self.offload,
            'helpers': self.helpers,
        }
        cls = load_consumer('scripts/prompt_enhance_ext.py', 'PromptEnhanceScript', ['_generate_local'], namespace)
        self.script = cls()
        self.script.tokenizer = self.tokenizer
        self.script.processor = None
        self.script.options = object()
        self.script.parent = 'control'
        self.outputs = Tensor()
        self.script.llm = types.SimpleNamespace(generate=Mock(return_value=self.outputs))
        self.request = Request('question', 'system', prefill='prefill', thinking=False, parameters={
            'sample': True, 'min_tokens': 2, 'max_tokens': 20, 'temperature': 0.5,
            'penalty': 1.1, 'top_k': 4, 'top_p': 0.8, 'nsfw': True, 'prompt_style': 'Standard',
            'process_words': '', 'semantic_threshold': 0, 'embedding_similarity': 0, 'custom_args': '',
        })

    def test_tokenization_generation_decode_and_offload(self):
        response = self.script._generate_local('Qwen/local', self.request)
        self.assertEqual(response.text, 'decoded answer')
        self.assertEqual(response.raw_text, 'raw answer')
        self.assertEqual(response.usage, {'input_tokens': 5, 'output_tokens': 4})
        self.assertAlmostEqual(response.metrics['offload_time'], 0.35)
        self.tokenizer.assert_called_once_with('chat templateprefill', return_tensors='pt')
        self.assertEqual(self.outputs.last_slice, (slice(None), slice(5, None)))
        self.assertEqual(self.tokenizer.batch_decode.call_args_list[0].args, (self.outputs,))
        self.assertEqual(self.tokenizer.batch_decode.call_args_list[0].kwargs, {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False})
        self.assertEqual(self.tokenizer.batch_decode.call_args_list[1].kwargs, {'skip_special_tokens': True, 'clean_up_tokenization_spaces': True})
        self.assertEqual(self.script.llm.generate.call_args.kwargs, {
            'input_ids': self.tokenizer.return_value['input_ids'], 'do_sample': True, 'temperature': 0.5,
            'max_new_tokens': 20, 'repetition_penalty': 1.1, 'min_new_tokens': 2, 'top_k': 4, 'top_p': 0.9,
        })
        self.offload.assert_called_once_with('prompt_enhance')
        self.devices.torch_gc.assert_called_once_with(force=False, reason='prompt-enhance')

    def test_generation_failure_still_offloads(self):
        self.script.llm.generate.side_effect = RuntimeError('generate failed')
        response = self.script._generate_local('Qwen/local', self.request)
        self.assertEqual(response.text, 'Error: generate failed')
        self.assertEqual(response.usage['output_tokens'], 0)
        self.offload.assert_called_once_with('prompt_enhance')

    def test_tokenization_failure_does_not_enter_inference(self):
        self.tokenizer.side_effect = RuntimeError('tokenize failed')
        response = self.script._generate_local('Qwen/local', self.request)
        self.assertTrue(response.metrics['tokenize_error'])
        self.assertEqual(response.text, 'question')
        self.script.llm.generate.assert_not_called()
        self.offload.assert_not_called()


class StoryTests(unittest.TestCase):
    def test_story_uses_one_scope_and_closes_before_image_setup(self):
        for fail in (False, True, 'exception'):
            with self.subTest(fail=fail):
                events = []
                work_session = object()

                @contextmanager
                def session(_model, events=events, work_session=work_session):
                    events.append('enter')
                    try:
                        yield work_session
                    finally:
                        events.append('exit')

                def enhance(events=events, work_session=work_session, fail=fail, **kwargs):
                    self.assertIs(kwargs['llm_session'], work_session)
                    if 'Analyze the attached image' in kwargs['prompt']:
                        events.append('caption')
                        return 'reference caption'
                    if kwargs.get('max_tokens'):
                        events.append('story')
                        if fail == 'exception':
                            raise RuntimeError('inference failed')
                        return 'Error: failed' if fail else '["frame one", "frame two"]'
                    events.append('enhance')
                    return 'enhanced frame'

                enhancer = types.SimpleNamespace(session=session, enhance=enhance)
                state = types.SimpleNamespace(begin=Mock(return_value='job'), end=Mock())
                namespace = {
                    'ExitStack': ExitStack, 'time': time, 'log': Mock(), 'debug_log': Mock(),
                    'shared': types.SimpleNamespace(state=state, sd_model=object()),
                    'extra_networks': types.SimpleNamespace(re_extra_net=re.compile(r'<[^>]+>')),
                    'get_model_repo_from_display': lambda name: name, 'is_vision_model': lambda _model: True,
                    'processing': types.SimpleNamespace(get_fixed_seed=lambda value: value, process_init=lambda _p, events=events: events.append('image setup')),
                }
                cls = load_consumer('scripts/story.py', 'StoryScript', ['process'], namespace)
                script = cls()
                script.get_enhancer = lambda _p, enhancer=enhancer: enhancer
                script.get_control_image = lambda _p: object()
                script.parse_frame_prompts = lambda _response, _count: ['frame one', 'frame two']
                p = types.SimpleNamespace(all_prompts=['story'], all_negative_prompts=['negative'], seed=1, extra_generation_params={})
                script.process(p, True, 2, 'Qwen/vision', True, True, False)
                expected = ['enter', 'caption', 'story', 'exit'] if fail else ['enter', 'caption', 'story', 'enhance', 'enhance', 'exit', 'image setup']
                self.assertEqual(events, expected)
                self.assertEqual(p.story_mode_error is not None, bool(fail))
                state.end.assert_called_once_with('job')

    def test_compile_decorators_remain_on_public_consumers(self):
        for path, class_name, method in [
            ('scripts/prompt_enhance_ext.py', 'PromptEnhanceScript', 'enhance'),
            ('modules/caption/vqa.py', 'VQA', 'caption'),
        ]:
            tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
            self.assertEqual([ast.unparse(decorator) for decorator in function.decorator_list], ["sd_hijack_te.with_compile_stance('LLM')"])


if __name__ == '__main__':
    unittest.main()
