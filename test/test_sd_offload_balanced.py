import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
import accelerate.hooks

from modules import shared, devices, memstats
from modules import sd_offload_balanced


class TestBalancedOffloadDispatch(unittest.TestCase):
    def setUp(self):
        self.offload_mode = shared.opts.diffusers_offload_mode
        shared.opts.diffusers_offload_mode = 'balanced'

    def tearDown(self):
        shared.opts.diffusers_offload_mode = self.offload_mode

    def test_single_device_uses_root_alignment_hook(self):
        module = torch.nn.Sequential(torch.nn.Linear(4, 2))
        inputs = torch.randn(3, 4)
        expected = module(inputs).detach()

        with patch.object(sd_offload_balanced.accelerate, 'dispatch_model', return_value=module) as dispatch:
            dispatched, align_hook = sd_offload_balanced._dispatch_balanced_module(
                module,
                {'': 'cpu'},
                torch.device(devices.cpu),
                '/tmp/offload-test',
            )

        self.assertFalse(dispatch.call_args.kwargs['force_hooks'])
        self.assertIsInstance(align_hook, accelerate.hooks.AlignDevicesHook)
        self.assertTrue(torch.equal(dispatched(inputs), expected))

    def test_split_device_map_keeps_accelerate_hooks(self):
        module = torch.nn.Linear(4, 2)

        with patch.object(sd_offload_balanced.accelerate, 'dispatch_model', return_value=module) as dispatch:
            dispatched, align_hook = sd_offload_balanced._dispatch_balanced_module(
                module,
                {'': 'cuda:0', 'linear': 'cpu'},
                torch.device('cuda:0'),
                '/tmp/offload-test',
            )

        self.assertTrue(dispatch.call_args.kwargs['force_hooks'])
        self.assertIsNone(align_hook)
        self.assertIs(dispatched, module)

    def test_bnb_modules_keep_forced_accelerate_hooks(self):
        module = torch.nn.Linear(4, 2)
        module.is_loaded_in_4bit = True

        with patch.object(sd_offload_balanced.accelerate, 'dispatch_model', return_value=module) as dispatch:
            _dispatched, align_hook = sd_offload_balanced._dispatch_balanced_module(
                module,
                {'': 'cuda:0'},
                torch.device('cuda:0'),
                '/tmp/offload-test',
            )

        self.assertTrue(dispatch.call_args.kwargs['force_hooks'])
        self.assertIsNone(align_hook)


class TestBalancedOffloadMoves(unittest.TestCase):
    def test_streamed_move_uses_non_blocking_transfer(self):
        module = Mock()
        module.module_name = 'test'
        module.to.return_value = module
        hook = SimpleNamespace(
            offload_map={'test': 1},
            offload_never=[],
            offload_always=[],
            model_size=Mock(return_value=1),
            matches=Mock(return_value=False),
        )
        stream = MagicMock()

        with (
            patch.dict(shared.opts.data, {'diffusers_offload_streams': True}),
            patch.object(shared, 'gpu_memory', 1),
            patch.object(sd_offload_balanced.s, 'move_stream', None),
            patch.object(sd_offload_balanced.s, 'offload_hook_instance', hook),
            patch.object(devices, 'create_stream', return_value=stream) as create_stream,
            patch.object(memstats, 'memory_stats', return_value={'gpu': {'used': 0}, 'ram': {'used': 0}}),
            patch.object(devices, 'torch_gc') as torch_gc,
        ):
            moved = sd_offload_balanced.move_module_to_cpu(module, force=True)

        self.assertTrue(moved)
        create_stream.assert_called_once_with(devices.device)
        module.to.assert_called_once_with(devices.cpu, non_blocking=True)
        torch_gc.assert_not_called()

    def test_offload_gc_runs_only_after_move(self):
        for moved, expected_gc_calls in ((False, 0), (True, 1)):
            with self.subTest(moved=moved):
                module = Mock()
                module.module_name = 'test'
                module._hf_hook = SimpleNamespace(execution_device=None)
                hook = Mock()
                gc = Mock()
                with (
                    patch.object(sd_offload_balanced.accelerate.hooks, 'remove_hook_from_module', return_value=module),
                    patch.object(sd_offload_balanced.accelerate.hooks, 'add_hook_to_module', return_value=module),
                    patch.object(sd_offload_balanced, 'move_module_to_cpu', return_value=moved),
                    patch.object(devices, 'torch_gc', gc),
                    patch.object(sd_offload_balanced.s, 'offload_hook_instance', hook),
                    patch.object(sd_offload_balanced.s, 'offload_post', set()),
                    patch.dict(shared.opts.data, {'layerwise_quantization': False}),
                ):
                    sd_offload_balanced.apply_balanced_offload_to_module(module)

                self.assertEqual(gc.call_count, expected_gc_calls)


if __name__ == '__main__':
    unittest.main()
