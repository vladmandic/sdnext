"""CPU-only coverage for refiner prompt restoration in Standard UI.

Read the actual UI field bindings without constructing the model-dependent UI,
then run the real metadata parser and paste callback with lightweight UI stubs.
Run with: python test/test-refiner-prompt-restore.py
"""

import ast
import importlib.util
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def refiner_fields(tab):
    tree = ast.parse((root / 'modules' / f'ui_{tab}.py').read_text(encoding='utf-8'))
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            component, key = node.elts
            if isinstance(component, ast.Name) and component.id in ('refiner_prompt', 'refiner_negative') and isinstance(key, ast.Constant):
                fields.append((SimpleNamespace(value='', elem_id=component.id), key.value))
    return fields


class TestRefinerPromptRestore(unittest.TestCase):
    def test_restore_refiner_prompts(self):
        infotext = ('princess, white dress\nNegative prompt: blurry\n'
                    'Steps: 20, Seed: 42, Refiner prompt: "princess, red dress", Refiner negative: "blue, green"')
        with tempfile.TemporaryDirectory() as directory:
            params_path = pathlib.Path(directory) / 'params.txt'
            params_path.write_text(infotext, encoding='utf-8')
            stubs = {
                'PIL': SimpleNamespace(Image=Mock()),
                'gradio': SimpleNamespace(update=lambda **kwargs: kwargs),
                'modules.shared': SimpleNamespace(opts=SimpleNamespace(disable_apply_params='', clip_skip_enabled=True)),
                'modules.gr_tempdir': Mock(),
                'modules.script_callbacks': Mock(),
                'modules.images': Mock(),
                'modules.logger': SimpleNamespace(log=Mock()),
                'modules.paths': SimpleNamespace(params_path=str(params_path)),
            }
            with patch.dict(sys.modules, stubs):
                spec = importlib.util.spec_from_file_location('paste_under_test', root / 'modules' / 'generation_parameters_copypaste.py')
                paste = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(paste)
                for tab in ('txt2img', 'img2img', 'control'):
                    fields = refiner_fields(tab)
                    self.assertEqual(len(fields), 2)
                    for source in (infotext, ''):
                        with self.subTest(tab=tab, source='paste' if source else 'params.txt'):
                            button = Mock()
                            paste.connect_paste(button, fields, None, None, tab)
                            callback = button.click.call_args_list[0].kwargs['fn']
                            updates = callback(source)
                            values = {field.elem_id: update.get('value') for (field, _), update in zip(fields, updates)}
                            self.assertEqual(values['refiner_prompt'], 'princess, red dress')
                            self.assertEqual(values['refiner_negative'], 'blue, green')


if __name__ == '__main__':
    unittest.main()
