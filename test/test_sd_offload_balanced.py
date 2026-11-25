import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
import accelerate.hooks

from modules import shared, devices, memstats
from modules import sd_offload_balanced
from modules import sd_offload_balanced_pin


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
    def test_module_state_reports_untrackable_tensor(self):
        module = torch.nn.Linear(2, 2, device='meta')

        state, reason = sd_offload_balanced_pin.module_state(module)

        self.assertIsNone(state)
        self.assertEqual(reason, 'parameter=weight is meta')

    def test_inference_buffer_can_restore_when_unchanged(self):
        module = torch.nn.Module()
        with patch.dict(shared.opts.data, {'diffusers_offload_pin': True, 'layerwise_quantization': False}), patch.object(
            sd_offload_balanced_pin, 'pin_cpu_tensor', side_effect=lambda tensor: tensor.detach().clone()
        ):
            with torch.inference_mode():
                module.register_buffer('inv_freq', torch.arange(4, dtype=torch.float32))
                self.assertTrue(sd_offload_balanced_pin.pin_cpu_master(module))
                module.inv_freq = module.inv_freq.clone()
                sd_offload_balanced_pin.record_onload(module, eligible=True)

                self.assertTrue(sd_offload_balanced_pin.restore_cpu_master(module))

        self.assertTrue(module.inv_freq.is_inference())

    def test_inference_buffer_mutation_invalidates_master(self):
        module = torch.nn.Module()
        with patch.dict(shared.opts.data, {'diffusers_offload_pin': True, 'layerwise_quantization': False}), patch.object(
            sd_offload_balanced_pin, 'pin_cpu_tensor', side_effect=lambda tensor: tensor.detach().clone()
        ):
            with torch.inference_mode():
                module.register_buffer('inv_freq', torch.arange(4, dtype=torch.float32))
                self.assertTrue(sd_offload_balanced_pin.pin_cpu_master(module))
                module.inv_freq = module.inv_freq.clone()
                sd_offload_balanced_pin.record_onload(module, eligible=True)
                module.inv_freq.add_(1)

                self.assertFalse(sd_offload_balanced_pin.restore_cpu_master(module))

        self.assertFalse(hasattr(module, 'sdnext_balanced_cpu_master'))

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
            patch.dict(shared.opts.data, {'diffusers_offload_streams': True, 'diffusers_offload_nonblocking': True, 'diffusers_offload_pin': False}),
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

    def test_pinned_master_is_restored_without_copy_back(self):
        module = torch.nn.Linear(2, 2)
        module.register_buffer('scale', torch.ones(2))
        module.module_name = 'test'
        hook = SimpleNamespace(
            offload_map={'test': 1},
            offload_never=[],
            offload_always=[],
            model_size=Mock(return_value=1),
            matches=Mock(return_value=False),
        )
        expected = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}
        expected.update({name: tensor.detach().clone() for name, tensor in module.named_buffers()})

        with patch.dict(shared.opts.data, {'diffusers_offload_pin': True, 'layerwise_quantization': False}), patch.object(
            sd_offload_balanced_pin, 'pin_cpu_tensor', side_effect=lambda tensor: tensor.detach().clone()
        ):
            self.assertTrue(sd_offload_balanced_pin.pin_cpu_master(module))
            for parameter in module.parameters():
                parameter.data = parameter.detach().clone()
            for name, tensor in module.named_buffers():
                sd_offload_balanced_pin.set_buffer(module, name, tensor.detach().clone())
            sd_offload_balanced_pin.record_onload(module, eligible=True)

            with (
                patch.object(shared, 'gpu_memory', 1),
                patch.object(sd_offload_balanced.s, 'offload_hook_instance', hook),
                patch.object(memstats, 'memory_stats', return_value={'gpu': {'used': 0}, 'ram': {'used': 0}}),
                patch.object(devices, 'torch_gc') as torch_gc,
                patch.object(module, 'to', wraps=module.to) as move,
            ):
                moved = sd_offload_balanced.move_module_to_cpu(module, force=True)

        self.assertTrue(moved)
        move.assert_not_called()
        torch_gc.assert_not_called()
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            self.assertTrue(torch.equal(tensor, expected[name]))

    def test_changed_weights_invalidate_pinned_master(self):
        module = torch.nn.Linear(2, 2)
        with patch.dict(shared.opts.data, {'diffusers_offload_pin': True, 'layerwise_quantization': False}), patch.object(
            sd_offload_balanced_pin, 'pin_cpu_tensor', side_effect=lambda tensor: tensor.detach().clone()
        ):
            self.assertTrue(sd_offload_balanced_pin.pin_cpu_master(module))
            for parameter in module.parameters():
                parameter.data = parameter.detach().clone()
            sd_offload_balanced_pin.record_onload(module, eligible=True)
            with torch.no_grad():
                module.weight.add_(1)

            self.assertFalse(sd_offload_balanced_pin.restore_cpu_master(module))
            self.assertFalse(hasattr(module, 'sdnext_balanced_cpu_master'))

    def test_nested_weight_update_invalidates_component_master(self):
        module = torch.nn.Sequential(torch.nn.Linear(2, 2))
        with patch.dict(shared.opts.data, {'diffusers_offload_pin': True, 'layerwise_quantization': False}), patch.object(
            sd_offload_balanced_pin, 'pin_cpu_tensor', side_effect=lambda tensor: tensor.detach().clone()
        ):
            self.assertTrue(sd_offload_balanced_pin.pin_cpu_master(module))
            sd_offload_balanced_pin.invalidate_cpu_master(module[0])

        self.assertFalse(hasattr(module, 'sdnext_balanced_cpu_master'))

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
                    patch.dict(shared.opts.data, {'diffusers_offload_pin': False, 'layerwise_quantization': False}),
                ):
                    sd_offload_balanced.apply_balanced_offload_to_module(module)

                self.assertEqual(gc.call_count, expected_gc_calls)


if __name__ == '__main__':
    unittest.main()
