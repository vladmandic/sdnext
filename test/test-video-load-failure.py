"""CPU loader regressions using the real lazy shared.sd_model property.

Downloads, pipeline construction, and device configuration are stubbed. Model
state, cache decisions, and video loader control flow execute production code.
Run with: python test/test-video-load-failure.py
"""

import importlib.util
import pathlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VideoPipeline:
    from_pretrained = Mock()


class TestVideoLoadFailure(unittest.TestCase):
    def setUp(self):
        shared = ModuleType('modules.shared')
        shared.opts = SimpleNamespace(sd_model_checkpoint='selected-image-model', offline_mode=True, hfcache_dir='unused')
        shared.state = Mock()
        shared.state.begin.return_value = 'load-job'
        sd_models = Mock()
        sd_models.reload_model_weights.return_value = SimpleNamespace()
        stubs = {
            'torch': SimpleNamespace(nn=SimpleNamespace(Module=type('TensorModule', (), {}))),
            'transformers': Mock(),
            'diffusers': Mock(),
            'sdnq': Mock(),
            'modules.shared': shared,
            'modules.errors': Mock(),
            'modules.logger': SimpleNamespace(log=Mock()),
            'modules.sd_models': sd_models,
            'modules.sd_checkpoint': Mock(),
            'modules.model_quant': SimpleNamespace(do_post_load_quant=lambda pipe, **kwargs: pipe),
            'modules.devices': SimpleNamespace(dtype='test'),
            'modules.modular_load': SimpleNamespace(is_modular=lambda _: False),
            'modules.sd_hijack_te': Mock(),
            'modules.sd_hijack_vae': Mock(),
            'modules.sd_hijack_modular': Mock(),
            'modules.video_models.models_def': SimpleNamespace(Model=SimpleNamespace),
            'modules.video_models.video_utils': Mock(),
            'modules.video_models.video_overrides': SimpleNamespace(load_override=lambda *args, **kwargs: {}),
            'modules.video_models.video_cache': Mock(),
            'pipelines.generic': Mock(),
        }
        self.patches = patch.dict(sys.modules, stubs)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.env = patch.dict('os.environ', {})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.modeldata = load_module('modeldata_under_test', 'modules/modeldata.py')
        shared.__class__ = self.modeldata.Shared
        self.modeldata.model_data.locked = False # normal post-startup state in webui.py
        sd_models.unload_model_weights.side_effect = lambda: setattr(self.modeldata.model_data, 'sd_model', None)
        self.shared = shared
        self.sd_models = sd_models
        self.loader = load_module('video_load_under_test', 'modules/video_models/video_load.py')
        self.selected = SimpleNamespace(name='video-model', repo='test/video', repo_cls=VideoPipeline,
                                        repo_revision=None, dit_cls=None, te_cls=None, workflow=None,
                                        vae_hijack=False, te_hijack=False, image_hijack=False)
        VideoPipeline.from_pretrained.reset_mock(return_value=True, side_effect=True)

    def assert_failed_load(self):
        message = self.loader.load_model(self.selected)
        self.sd_models.reload_model_weights.assert_not_called()
        self.assertFalse(self.shared.sd_loaded)
        self.assertIsNone(self.loader.loaded_model)
        self.assertIn('failed', message)
        self.shared.state.end.assert_called_once_with('load-job')
        self.sd_models.set_diffuser_options.assert_not_called()

    def test_pipeline_exception_does_not_load_default_image_model(self):
        VideoPipeline.from_pretrained.side_effect = OSError('component download failed')
        self.assert_failed_load()

    def test_missing_pipeline_does_not_load_default_image_model(self):
        VideoPipeline.from_pretrained.return_value = None
        self.assert_failed_load()

    def test_failed_switch_invalidates_cached_model_name(self):
        self.shared.sd_model = VideoPipeline()
        self.loader.loaded_model = 'previous-video-model'
        VideoPipeline.from_pretrained.side_effect = OSError('component download failed')
        self.assert_failed_load()
        self.sd_models.unload_model_weights.assert_called_once()

    def test_retry_succeeds_and_cache_keeps_loaded_pipeline(self):
        VideoPipeline.from_pretrained.side_effect = OSError('component download failed')
        self.assert_failed_load()
        pipe = VideoPipeline()
        VideoPipeline.from_pretrained.side_effect = None
        VideoPipeline.from_pretrained.return_value = pipe
        self.loader.load_model(self.selected)
        self.assertIs(self.shared.sd_model, pipe)
        self.assertEqual(self.loader.loaded_model, self.selected.name)
        self.assertEqual(VideoPipeline.from_pretrained.call_count, 2)
        self.assertEqual(self.loader.load_model(self.selected), '')
        self.assertEqual(VideoPipeline.from_pretrained.call_count, 2)
        self.sd_models.reload_model_weights.assert_not_called()


if __name__ == '__main__':
    unittest.main()
