"""Offline regression test for caption and analyze UI argument binding."""

import ast
import inspect
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


root = Path(__file__).resolve().parent.parent


def load_function(filename, name, namespace):
    tree = ast.parse((root / filename).read_text(encoding='utf8'))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace) # pylint: disable=exec-used
    return namespace[name]


class CaptionBindingTests(unittest.TestCase):
    def test_caption_and_analyze_bind_optional_arguments_by_name(self):
        vqa = SimpleNamespace(caption=Mock(return_value='caption'), analyze=Mock(return_value='analysis'), get_last_annotated_image=lambda: None)
        caption = ModuleType('modules.caption')
        caption.vqa = vqa
        namespace = {'log': Mock(), 'gr': SimpleNamespace(update=lambda **kwargs: kwargs)}
        function = load_function('modules/ui_caption.py', 'caption_wrapper', namespace)
        fields = list(inspect.signature(function).parameters)
        values = dict.fromkeys(fields, None)
        values.update(tab=0, vlm_prefill='prefill', vlm_thinking_mode=True, vlm_custom_args='custom')
        packages = {'modules': ModuleType('modules'), 'modules.caption': caption}
        with patch.dict(sys.modules, packages):
            self.assertEqual(function(**values)[0], 'caption')
            self.assertEqual(len(vqa.caption.call_args.args), 5)
            self.assertEqual(vqa.caption.call_args.kwargs, {'prefill': 'prefill', 'thinking_mode': True, 'custom_args': 'custom'})
            values['tab'] = 1
            self.assertEqual(function(**values)[0], 'analysis')
            self.assertEqual(len(vqa.analyze.call_args.args), 5)
            self.assertEqual(vqa.analyze.call_args.kwargs, {'thinking_mode': True, 'custom_args': 'custom'})


if __name__ == '__main__':
    unittest.main()
