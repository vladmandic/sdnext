"""CPU regression for the effective hires strength saved in infotext.

The Standard UI case invokes the real txt2img handler with source-derived UI
argument ordering. Model construction/inference are replaced by a capture stub;
processing defaults, metadata serialization, and parsing come from production.
Run with: python test/test-hires-strength-metadata.py
"""

import ast
import importlib.util
import inspect
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from modules.infotext import parse


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, root / 'modules' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def processing_defaults():
    tree = ast.parse((root / 'modules' / 'processing_class.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'StableDiffusionProcessing')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    return {arg.arg: ast.literal_eval(value) for arg, value in zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}


class ConstructionCaptured(Exception):
    pass


class TestHiresStrengthMetadata(unittest.TestCase):
    def setUp(self):
        self.defaults = processing_defaults()
        shared = SimpleNamespace(
            sd_model=SimpleNamespace(sd_checkpoint_info=None),
            opts=SimpleNamespace(prompt_attention='native', sd_text_encoder='Default', sd_unet='Default',
                                 add_model_name_to_info=False, add_model_hash_to_info=False,
                                 outdir_samples='', outdir_txt2img_samples='', outdir_grids='', outdir_txt2img_grids=''),
        )
        self.constructor = Mock(side_effect=ConstructionCaptured)
        stubs = {
            'installer': SimpleNamespace(git_commit='test'),
            'modules.shared': shared,
            'modules.sd_samplers_common': Mock(),
            'modules.sd_vae': SimpleNamespace(loaded_vae_file=None),
            'modules.logger': SimpleNamespace(log=Mock()),
            'modules.processing_class': SimpleNamespace(StableDiffusionProcessing=SimpleNamespace),
            'modules.processing': SimpleNamespace(StableDiffusionProcessingTxt2Img=self.constructor, get_sampler_name=lambda _: 'Default'),
            'modules.scripts_manager': Mock(),
            'modules.generation_parameters_copypaste': SimpleNamespace(create_override_settings_dict=lambda _: {}),
            'modules.ui_common': SimpleNamespace(plaintext_to_html=lambda text: text),
            'modules.paths': SimpleNamespace(resolve_output_path=lambda *args: ''),
        }
        self.patches = patch.dict(sys.modules, stubs)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.info = load_module('info_under_test', 'processing_info.py')
        self.txt = load_module('txt_under_test', 'txt2img.py')

    def metadata(self, **values):
        defaults = self.defaults.copy()
        defaults.update(prompt='a landscape', negative_prompt='', seed=42, subseed=42,
                        sampler_name='Default', hr_sampler_name='Default', ops=['txt2img', 'upscale', 'hires'],
                        all_prompts=[], all_negative_prompts=[], all_seeds=[], all_subseeds=[],
                        all_templates=None, all_negative_templates=None, extra_generation_params={},
                        hr_upscaler='ESRGAN', hr_resize_mode=1, hr_force=True,
                        hr_upscale_to_x=2048, hr_upscale_to_y=2048)
        defaults.update(values)
        return parse(self.info.create_infotext(SimpleNamespace(**defaults)))

    def test_standard_txt2img_slider_is_saved(self):
        tree = ast.parse((root / 'modules' / 'ui_txt2img.py').read_text(encoding='utf-8'))
        ui_args = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'txt2img_args' for target in node.targets))
        parameters = list(inspect.signature(self.txt.txt2img).parameters)[:-1]
        self.assertEqual(len(ui_args.elts), len(parameters))
        for strength in (0.3, 0.65):
            with self.subTest(strength=strength):
                values = [self.defaults.get(name) for name in parameters]
                values[parameters.index('grading_lut_file')] = None # empty Gradio file input
                for name, value in {'enable_hr': True, 'hr_force': True, 'hr_upscaler': 'ESRGAN', 'hr_resize_mode': 1}.items():
                    values[parameters.index(name)] = value
                for index, node in enumerate(ui_args.elts):
                    if isinstance(node, ast.Name) and node.id == 'hr_denoising_strength':
                        values[index] = strength
                with self.assertRaises(ConstructionCaptured):
                    self.txt.txt2img(*values)
                kwargs = self.constructor.call_args.kwargs
                self.assertEqual(kwargs['denoising_strength'], strength)
                self.assertEqual(kwargs.get('hr_denoising_strength', self.defaults['hr_denoising_strength']), 0.0)
                metadata = self.metadata(**kwargs)
                self.assertEqual(metadata['Hires strength'], strength)

    def test_explicit_hires_strength_takes_precedence(self):
        self.assertEqual(self.metadata(denoising_strength=0.65, hr_denoising_strength=0.2)['Hires strength'], 0.2)

    def test_zero_strength_is_preserved(self):
        self.assertEqual(self.metadata(denoising_strength=0.0, hr_denoising_strength=0.0, ops=['upscale'])['Hires strength'], 0.0)


if __name__ == '__main__':
    unittest.main()
