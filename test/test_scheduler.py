import sys
import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from modules import scheduler
from modules.scheduler import _decode_value, _encode_value


class TestSchedulerSerialization(unittest.TestCase):
    def model_options(self, **overrides):
        values = {key: f'original-{key}' for key in scheduler._model_option_keys}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_image_round_trip_uses_blob_reference(self):
        image = Image.new('RGB', (3, 2), (10, 20, 30))
        blobs = []
        encoded = _encode_value({'inputs': [image]}, blobs)

        self.assertEqual(len(blobs), 1)
        decoded = _decode_value(encoded, blobs)
        self.assertEqual(decoded['inputs'][0].size, (3, 2))
        self.assertEqual(decoded['inputs'][0].getpixel((0, 0)), (10, 20, 30))

    def test_control_unit_round_trip_excludes_runtime_objects(self):
        with patch.object(sys, 'argv', [sys.argv[0]]):
            from modules.control.unit import Unit

        source = Unit(index=1, enabled=True, strength=0.75, unit_type='reference', start=0.2, end=0.8)
        source.model_name = 'reference-model'
        source.process_name = 'none'
        source.fidelity = 0.6
        source.override = Image.new('RGB', (1, 1), 'blue')
        blobs = []
        encoded = _encode_value([source], blobs)
        restored = _decode_value(encoded, blobs)[0]

        self.assertEqual(restored.index, 1)
        self.assertEqual(restored.model_name, 'reference-model')
        self.assertEqual(restored.fidelity, 0.6)
        self.assertEqual(restored.override.getpixel((0, 0)), (0, 0, 255))

    def test_enqueue_captures_model_name_with_checkpoint_fallback(self):
        for model, expected in (
            (SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name='Display Name'), sd_model_checkpoint='model/path'), 'Display Name'),
            (SimpleNamespace(sd_checkpoint_info=SimpleNamespace(name=''), sd_model_checkpoint='model/path'), 'model/path'),
        ):
            parameters = {'control': {}}
            options = self.model_options()
            shared = SimpleNamespace(sd_model=model, opts=options)
            with self.subTest(expected=expected), \
                    patch.object(sys.modules['modules'], 'shared', shared, create=True), \
                    patch.dict(sys.modules, {'modules.shared': shared}), \
                    patch.object(scheduler, '_capture_request', return_value=(parameters, [])), \
                    patch.object(scheduler.scheduler_db, 'create_task', return_value=1) as create_task:
                list(scheduler.enqueue_control_request())
            self.assertEqual(parameters['model'], expected)
            self.assertNotIn('model_options', parameters)
            for key, value in vars(options).items():
                self.assertEqual(parameters[key], value)
            create_task.assert_called_once_with(parameters, image_blobs=[])

    def test_task_model_options_restore_after_execution_error(self):
        original = self.model_options()
        shared = SimpleNamespace(opts=original)
        task_options = {key: f'task-{key}' for key in scheduler._model_option_keys}

        def fail_during_execution(_task):
            self.assertEqual(vars(original), task_options)
            raise RuntimeError('task failed')

        with patch.object(sys.modules['modules'], 'shared', shared, create=True), \
                patch.dict(sys.modules, {'modules.shared': shared}), \
                patch.object(scheduler, '_execute_task', side_effect=fail_during_execution) as execute:
            with self.assertRaisesRegex(RuntimeError, 'task failed'):
                scheduler._run_task({'parameters': task_options})
        execute.assert_called_once()
        self.assertEqual(vars(original), {key: f'original-{key}' for key in scheduler._model_option_keys})

    def test_execute_task_persists_process_stats_and_infotexts(self):
        from modules import timer

        result = ([SimpleNamespace(already_saved_as='/tmp/output.png')], None, '', None, ['prompt: test, steps: 20'])
        stats = {'total': 5.2, 'sample': 4.1}

        def control_run():
            yield result

        task = {'id': 7, 'image_blobs': [], 'parameters': {'control': {}}}
        with patch.dict(sys.modules, {'modules.control.run': SimpleNamespace(control_run=control_run)}), \
                patch.object(scheduler, '_decode_value', return_value={}), \
                patch.object(timer.process, 'dct', return_value=stats), \
                patch.object(scheduler.scheduler_db, 'update_results', return_value=True) as update_results:
            scheduler._execute_task(task)

        update_results.assert_called_once_with(
            7,
            ['/tmp/output.png'],
            process_stats=stats,
            infotexts=['prompt: test, steps: 20'],
        )

    def test_update_task_parameters_accepts_control_and_model_fields(self):
        task = {
            'parameters': {
                'control': {'prompt': 'old prompt', 'negative_prompt': '', 'steps': 20, 'width_before': 512, 'height_before': 512},
                'sd_model_checkpoint': 'old-model',
                'sd_model_refiner': 'None',
                'sd_unet': 'Default',
                'sd_vae': 'Automatic',
                'sd_text_encoder': 'Default',
            },
        }
        updates = {
            'prompt': 'new prompt',
            'negative_prompt': 'avoid this',
            'steps': 30,
            'width_before': 768,
            'height_before': 1024,
            'sd_model_checkpoint': 'new-model',
            'sd_model_refiner': 'new-refiner',
            'sd_unet': 'new-unet',
            'sd_vae': 'new-vae',
            'sd_text_encoder': 'new-text-encoder',
        }
        with patch.object(scheduler.scheduler_db, 'get_task', return_value=task), \
                patch.object(scheduler.scheduler_db, 'update_parameters', return_value=True) as update:
            self.assertTrue(scheduler.update_task_parameters(12, updates))
        self.assertEqual(task['parameters']['control'], {key: value for key, value in updates.items() if key in task['parameters']['control']})
        self.assertEqual({key: task['parameters'][key] for key in scheduler._model_option_keys if key in updates}, {key: value for key, value in updates.items() if key not in task['parameters']['control']})
        update.assert_called_once_with(12, task['parameters'], db_path=None)

    def test_update_task_parameters_rejects_invalid_common_values(self):
        task = {'parameters': {'control': {'steps': 20}, 'sd_model_checkpoint': 'old-model'}}
        with patch.object(scheduler.scheduler_db, 'get_task', return_value=task):
            self.assertFalse(scheduler.update_task_parameters(12, {'steps': True}))
            self.assertFalse(scheduler.update_task_parameters(12, {'sd_model_checkpoint': 42}))
            self.assertFalse(scheduler.update_task_parameters(12, {'unknown': 'value'}))


class TestSchedulerPolicy(unittest.TestCase):
    def patch_options(self, options):
        with patch.object(sys, 'argv', [sys.argv[0]]), patch.object(scheduler, 'start_worker'):
            from modules import shared
        return patch.object(shared, 'opts', options)

    def test_auto_window_boundaries_and_overnight_window(self):
        options = SimpleNamespace(scheduler_state='Auto', scheduler_start='10:00', scheduler_end='18:00')
        with self.patch_options(options):
            now = datetime.datetime.now().astimezone()
            at_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
            at_end = now.replace(hour=18, minute=0, second=0, microsecond=0)
            self.assertTrue(scheduler.scheduler_enabled(at_start))
            self.assertFalse(scheduler.scheduler_enabled(at_end))

            options.scheduler_start = '22:00'
            options.scheduler_end = '06:00'
            self.assertTrue(scheduler.scheduler_enabled(now.replace(hour=23, minute=0)))
            self.assertTrue(scheduler.scheduler_enabled(now.replace(hour=5, minute=59)))
            self.assertFalse(scheduler.scheduler_enabled(now.replace(hour=12, minute=0)))

            options.scheduler_start = '06:00'
            options.scheduler_end = '06:00'
            self.assertTrue(scheduler.scheduler_enabled(at_end))

    def test_active_paused_and_persisted_settings(self):
        class FakeOptions:
            scheduler_state = 'Active'
            scheduler_start = '10:00'
            scheduler_end = '18:00'

            def __init__(self):
                self.saved = False

            def set(self, key, value):
                setattr(self, key, value)
                return value

            def save(self):
                self.saved = True

        options = FakeOptions()
        with self.patch_options(options):
            self.assertTrue(scheduler.scheduler_enabled())
            scheduler.set_scheduler_state('Paused', '22:30', '05:15')
            self.assertFalse(scheduler.scheduler_enabled())
            self.assertEqual(options.scheduler_state, 'Paused')
            self.assertEqual(options.scheduler_start, '22:30')
            self.assertEqual(options.scheduler_end, '05:15')
            self.assertTrue(options.saved)
            scheduler.set_scheduler_state('Active')
            self.assertTrue(scheduler.scheduler_enabled())

    def test_run_once_claims_manual_execution_while_scheduler_disabled(self):
        from modules import call_queue

        task = {'id': 12, 'attempts': 1}
        with patch.object(scheduler, 'scheduler_enabled', return_value=False), \
                patch.object(call_queue, 'is_locked', return_value=False), \
                patch.object(scheduler.scheduler_db, 'claim_next_task', return_value=task) as claim, \
                patch.object(scheduler.scheduler_db, 'get_task', return_value={'image_blobs': []}), \
                patch.object(scheduler, '_run_task') as run_task:
            self.assertTrue(scheduler.run_once())
        claim.assert_called_once_with(force_only=True)
        run_task.assert_called_once_with({'id': 12, 'attempts': 1, 'image_blobs': []})

    def test_worker_start_initializes_database_before_starting_thread(self):
        worker = unittest.mock.Mock()
        events = []
        shared = SimpleNamespace(opts=SimpleNamespace(scheduler_dir='models/scheduler'))
        with patch.dict(sys.modules, {'modules.shared': shared}), \
            patch.object(sys.modules['modules'], 'shared', shared, create=True), \
            patch.object(scheduler, '_worker', None), \
                patch.object(scheduler.scheduler_db, 'configure', side_effect=lambda *_args: events.append('configure')), \
                patch.object(scheduler.scheduler_db, 'initialize', side_effect=lambda: events.append('initialize')) as initialize, \
                patch.object(scheduler.threading, 'Thread', return_value=worker), \
                patch.object(scheduler.atexit, 'register'):
            worker.start.side_effect = lambda: events.append('start')
            self.assertIs(scheduler.start_worker(), worker)
        self.assertEqual(events, ['configure', 'initialize', 'start'])
        initialize.assert_called_once_with()
        worker.start.assert_called_once_with()

    def test_worker_start_runs_diagnostics_before_starting_thread(self):
        worker = unittest.mock.Mock()
        events = []
        shared = SimpleNamespace(opts=SimpleNamespace(scheduler_dir='models/scheduler'))
        with patch.dict(sys.modules, {'modules.shared': shared}), \
            patch.object(sys.modules['modules'], 'shared', shared, create=True), \
            patch.object(scheduler, '_worker', None), \
                patch.object(scheduler.scheduler_db, 'configure', side_effect=lambda *_args: events.append('configure')), \
                patch.object(scheduler.scheduler_db, 'initialize', side_effect=lambda: events.append('initialize')), \
                patch.object(scheduler.scheduler_db, 'diagnostics', side_effect=lambda: events.append('diagnostics') or {'ok': True}), \
                patch.object(scheduler.threading, 'Thread', return_value=worker), \
                patch.object(scheduler.atexit, 'register'):
            worker.start.side_effect = lambda: events.append('start')
            self.assertIs(scheduler.start_worker(diag=True), worker)
        self.assertEqual(events, ['configure', 'initialize', 'diagnostics', 'start'])
        worker.start.assert_called_once_with()

    def test_worker_start_disables_scheduler_after_initialization_failure(self):
        shared = SimpleNamespace(opts=SimpleNamespace(scheduler_dir='models/scheduler'))
        with patch.dict(sys.modules, {'modules.shared': shared}), \
                patch.object(sys.modules['modules'], 'shared', shared, create=True), \
                patch.object(scheduler, '_worker', None), \
                patch.object(scheduler, '_worker_start_failed', False), \
                patch.object(scheduler.scheduler_db, 'configure'), \
                patch.object(scheduler.scheduler_db, 'initialize', side_effect=RuntimeError('database unavailable')) as initialize, \
                patch.object(scheduler.threading, 'Thread') as thread, \
                patch.object(scheduler.atexit, 'register'), \
                patch.object(scheduler.log, 'error') as log_error:
            self.assertIsNone(scheduler.start_worker())
            self.assertIsNone(scheduler.start_worker())
        initialize.assert_called_once_with()
        thread.assert_not_called()
        log_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
