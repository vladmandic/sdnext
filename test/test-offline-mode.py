"""Process-wide Hugging Face offline regressions, with real libraries and synthetic caches.

Run: python test/test-offline-mode.py
No server, downloaded model weights, or network access required. Each probe runs in a
fresh interpreter because the libraries intentionally cache offline state at import time.
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')


class TestOfflineMode(unittest.TestCase):
    def probe(self, code, inherited=None, offline=True):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='offline-test-', dir=ROOT / 'tmp') as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({'offline_mode': offline}), encoding='utf8')
            env = os.environ.copy()
            for key in FLAGS:
                env.pop(key, None)
            env.update(inherited or {})
            env['HF_HOME'] = str(Path(directory) / 'hf-home')
            env['HF_HUB_DISABLE_TELEMETRY'] = '1'
            bootstrap = f"""
import os
import sys
from pathlib import Path
import installer
assert 'huggingface_hub.constants' not in sys.modules
assert 'transformers' not in sys.modules
assert 'diffusers' not in sys.modules
installer.args.config = {str(config)!r}
installer.args.secrets = {str(Path(directory) / 'no-secrets.json')!r}
installer.read_options()
installer.set_offline_mode()
work = Path({directory!r})
"""
            result = subprocess.run([sys.executable, '-c', bootstrap + textwrap.dedent(code)],
                                    cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_startup_order(self):
        tree = ast.parse((ROOT / 'launch.py').read_text(encoding='utf8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node.value.func.attr for node in main.body
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)]
        self.assertEqual(calls[calls.index('read_options') + 1], 'set_offline_mode')
        self.assertLess(calls.index('set_offline_mode'), calls.index('check_torch'))

    def test_offline_config_overrides_false_environment(self):
        self.probe("""
            from huggingface_hub import constants
            from transformers.utils.hub import is_offline_mode
            from diffusers.utils import hub_utils
            assert constants.HF_HUB_OFFLINE
            assert is_offline_mode()
            assert hub_utils.HF_HUB_OFFLINE
            assert os.environ['HF_HUB_OFFLINE'] == '1'
            assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
        """, inherited={key: '0' for key in FLAGS})

    def test_online_and_external_environment(self):
        for inherited, expected in (({}, False), ({'HF_HUB_OFFLINE': '1'}, True), ({'TRANSFORMERS_OFFLINE': '1'}, True)):
            with self.subTest(inherited=inherited):
                self.probe(f"""
                    from huggingface_hub import constants
                    assert constants.HF_HUB_OFFLINE is {expected!r}
                    for key in {FLAGS!r}:
                        assert os.environ.get(key) == {inherited!r}.get(key)
                """, inherited=inherited, offline=False)

    def test_restart_restores_original_environment(self):
        for inherited in ({}, {'HF_HUB_OFFLINE': '0'}, {'HF_HUB_OFFLINE': '1'}, {'TRANSFORMERS_OFFLINE': '1'}):
            with self.subTest(inherited=inherited):
                next_process = f"""
import os
import installer
for key in {FLAGS!r}:
    assert os.environ.get(key) == {inherited!r}.get(key)
installer.opts = {{'offline_mode': False}}
installer.set_offline_mode()
from huggingface_hub import constants
assert constants.HF_HUB_OFFLINE is {('1' in inherited.values())!r}
"""
                self.probe(f"""
                    assert os.environ['HF_HUB_OFFLINE'] == '1'
                    assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
                    installer.restart(['-c', {next_process!r}])
                """, inherited=inherited)

    def test_cached_and_missing_models_without_loader_flags(self):
        self.probe("""
            from unittest.mock import patch
            from contextlib import ExitStack

            attempts = []
            def reject_network(*args, **kwargs):
                attempts.append((args, kwargs))
                raise AssertionError('unexpected network access')

            with ExitStack() as stack:
                for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.getaddrinfo', 'socket.create_connection'):
                    stack.enter_context(patch(target, side_effect=reject_network))

                import torch
                from huggingface_hub import HfApi, hf_hub_download, snapshot_download
                from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
                from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizer
                from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

                cache = work / 'cache'
                def snapshot(repo):
                    base = cache / ('models--' + repo.replace('/', '--'))
                    folder = base / 'snapshots' / ('a' * 40)
                    folder.mkdir(parents=True)
                    (base / 'refs').mkdir()
                    (base / 'refs' / 'main').write_text('a' * 40, encoding='utf8')
                    return folder

                text_repo = 'offline-test/text'
                text_cache = snapshot(text_repo)
                model = BertModel(BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
                model.save_pretrained(text_cache)
                tokenizer = BertTokenizer(vocab={'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'hello': 5})
                tokenizer.save_pretrained(text_cache)

                image_repo = 'offline-test/image'
                image_cache = snapshot(image_repo)
                unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3, block_out_channels=(8,),
                                   down_block_types=('DownBlock2D',), up_block_types=('UpBlock2D',), norm_num_groups=4, layers_per_block=1)
                DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10)).save_pretrained(image_cache)

                # Deliberately omit local_files_only everywhere, as a new model integration might.
                assert hf_hub_download(text_repo, 'config.json', cache_dir=cache) == str(text_cache / 'config.json')
                assert snapshot_download(text_repo, cache_dir=cache) == str(text_cache)
                assert AutoConfig.from_pretrained(text_repo, cache_dir=cache).hidden_size == 8
                loaded_text = AutoModel.from_pretrained(text_repo, cache_dir=cache)
                torch.testing.assert_close(next(loaded_text.parameters()), next(model.parameters()))
                assert AutoTokenizer.from_pretrained(text_repo, cache_dir=cache)('hello')['input_ids'] == [2, 5, 3]
                loaded_image = DDPMPipeline.from_pretrained(image_repo, cache_dir=cache)
                torch.testing.assert_close(next(loaded_image.unet.parameters()), next(unet.parameters()))
                assert DDPMScheduler.from_pretrained(image_repo, subfolder='scheduler', cache_dir=cache).config.num_train_timesteps == 10

                failures = (
                    (lambda: hf_hub_download(text_repo, 'missing.json', cache_dir=cache), LocalEntryNotFoundError),
                    (lambda: snapshot_download('offline-test/missing', cache_dir=cache), LocalEntryNotFoundError),
                    (lambda: AutoConfig.from_pretrained('offline-test/missing', cache_dir=cache), OSError),
                    (lambda: AutoModel.from_pretrained('offline-test/missing', cache_dir=cache), OSError),
                    (lambda: AutoTokenizer.from_pretrained('offline-test/missing', cache_dir=cache), OSError),
                    (lambda: DDPMPipeline.from_pretrained('offline-test/missing', cache_dir=cache), OSError),
                    (lambda: HfApi().model_info(text_repo), OfflineModeIsEnabled),
                )
                for load, expected in failures:
                    try:
                        load()
                    except expected:
                        pass
                    else:
                        raise AssertionError('uncached load did not fail')
                assert not attempts, attempts
        """)

    def test_network_guard_detects_online_requests(self):
        self.probe("""
            from unittest.mock import patch
            from huggingface_hub import hf_hub_download
            attempts = []
            def reject_network(*args, **kwargs):
                attempts.append(args)
                raise AssertionError('network guard')
            with patch('socket.getaddrinfo', side_effect=reject_network):
                try:
                    hf_hub_download('offline-test/missing', 'config.json', cache_dir=work / 'cache')
                except AssertionError:
                    pass
            assert attempts, 'network guard did not observe the online request'
        """, offline=False)


if __name__ == '__main__':
    unittest.main()
