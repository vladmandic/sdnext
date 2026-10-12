import sys
import datetime as dt
import os
import tempfile
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

    def test_video_file_inputs_round_trip_from_temporary_paths(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as task_dir:
            source_path = os.path.join(source_dir, 'condition.mp4')
            with open(source_path, 'wb') as source:
                source.write(b'video-bytes')
            blobs = []

            encoded = _encode_value({'condition_video': source_path}, blobs)
            restored = _decode_value(encoded, blobs, media_dir=task_dir)

            self.assertEqual(restored['condition_video'], os.path.join(task_dir, '0-condition.mp4'))
            with open(restored['condition_video'], 'rb') as restored_file:
                self.assertEqual(restored_file.read(), b'video-bytes')
            self.assertEqual(encoded['condition_video']['media_type'], 'video/mp4')

    def test_enqueue_video_request_captures_handler_arguments_and_media(self):
        from modules import scheduler_db

        def handler(task_id, _ui_state, model, condition_video):
            return task_id, model, condition_video

        options = SimpleNamespace(
            sd_model_checkpoint='checkpoint',
            sd_model_refiner='None',
            sd_unet='Default',
            sd_unet_secondary='Default',
            sd_vae='Automatic',
            sd_text_encoder='Default',
        )
        shared = SimpleNamespace(sd_loaded=False, opts=options)
        with tempfile.TemporaryDirectory() as temp_dir:
            media_path = os.path.join(temp_dir, 'condition.mp4')
            with open(media_path, 'wb') as media:
                media.write(b'video-bytes')
            with patch.object(sys.modules['modules'], 'shared', shared, create=True), \
                    patch.dict(sys.modules, {'modules.shared': shared}), \
                    patch.object(scheduler, '_video_handler', return_value=handler), \
                    patch.object(scheduler_db, 'create_task', return_value=23) as create_task:
                response = scheduler.enqueue_video_request('ltx', 'client-id', 'ui-state', 'LTX model', media_path)

        self.assertEqual(response, [None, 'Queued scheduler task 23'])
        parameters = create_task.call_args.args[0]
        self.assertEqual(parameters['video']['surface'], 'ltx')
        self.assertEqual(parameters['video']['arguments']['model'], 'LTX model')
        self.assertEqual(parameters['video']['arguments']['condition_video']['media_type'], 'video/mp4')
        self.assertEqual(create_task.call_args.kwargs['image_blobs'], [b'video-bytes'])
        self.assertEqual(create_task.call_args.kwargs['max_attempts'], scheduler.MAX_ATTEMPTS)
        self.assertEqual(create_task.call_args.kwargs['mode'], 'video')

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
            parameters = {'control': {'units': []}}
            options = self.model_options()
            shared = SimpleNamespace(sd_loaded=True, sd_model=model, opts=options)
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
            create_task.assert_called_once()
            self.assertEqual(create_task.call_args.kwargs['max_attempts'], scheduler.MAX_ATTEMPTS)

    def test_capture_model_options_does_not_access_unloaded_model(self):
        class UnloadedShared(SimpleNamespace):
            @property
            def sd_model(self):
                raise AssertionError('sd_model accessed while sd_loaded is false')

        options = self.model_options()
        shared = UnloadedShared(sd_loaded=False, opts=options)
        parameters = {}
        with patch.object(sys.modules['modules'], 'shared', shared, create=True), \
                patch.dict(sys.modules, {'modules.shared': shared}):
            scheduler._capture_model_options(parameters)

        self.assertEqual(parameters['model'], options.sd_model_checkpoint)

    def test_task_model_options_restore_after_execution_error(self):
        original = self.model_options()
        original.samples_save = True
        shared = SimpleNamespace(opts=original)
        task_options = {key: f'task-{key}' for key in scheduler._model_option_keys}

        def fail_during_execution(_task):
            self.assertEqual({key: getattr(original, key) for key in task_options}, task_options)
            raise RuntimeError('task failed')

        with patch.object(sys.modules['modules'], 'shared', shared, create=True), \
                patch.dict(sys.modules, {'modules.shared': shared}), \
                patch.object(scheduler, '_execute_task', side_effect=fail_during_execution) as execute:
            with self.assertRaisesRegex(RuntimeError, 'task failed'):
                scheduler._run_task({'parameters': task_options})
        execute.assert_called_once()
        self.assertEqual({key: getattr(original, key) for key in scheduler._model_option_keys}, {key: f'original-{key}' for key in scheduler._model_option_keys})
        self.assertTrue(original.samples_save)

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

    def test_execute_video_task_consumes_generator_and_persists_video_path(self):
        from modules import timer

        def ltx_handler(task_id, prompt):
            self.assertEqual(task_id, 'task(scheduler-19)')
            self.assertEqual(prompt, 'video prompt')
            yield '/tmp/output.mp4', 'Video | File /tmp/output.mp4'

        task = {
            'id': 19,
            'mode': 'video',
            'image_blobs': [],
            'parameters': {
                'video': {
                    'surface': 'ltx',
                    'arguments': {'task_id': 'client-task', 'prompt': 'video prompt'},
                },
            },
        }
        stats = {'total': 12.5}
        with patch.object(scheduler, '_video_handler', return_value=ltx_handler), \
                patch.object(timer.process, 'dct', return_value=stats), \
                patch.object(scheduler.scheduler_db, 'update_results', return_value=True) as update_results:
            scheduler._execute_task(task)

        update_results.assert_called_once_with(
            19,
            ['/tmp/output.mp4'],
            process_stats=stats,
            infotexts=['Video | File /tmp/output.mp4'],
        )

    def test_execute_framepack_task_preserves_streamed_result_path(self):
        from modules import timer

        def framepack_handler(task_id, prompt):
            self.assertEqual(task_id, 'task(scheduler-20)')
            self.assertEqual(prompt, 'framepack prompt')
            yield '/tmp/framepack.mp4', None, 'saved'
            yield {'__type__': 'update'}, {'__type__': 'update'}, 'Generate finished'

        task = {
            'id': 20,
            'mode': 'video',
            'image_blobs': [],
            'parameters': {
                'video': {
                    'surface': 'framepack',
                    'arguments': {'task_id': 'client-task', 'prompt': 'framepack prompt'},
                },
            },
        }
        with patch.object(scheduler, '_video_handler', return_value=framepack_handler), \
                patch.object(timer.process, 'dct', return_value={}), \
                patch.object(scheduler.scheduler_db, 'update_results', return_value=True) as update_results:
            scheduler._execute_task(task)

        update_results.assert_called_once_with(20, ['/tmp/framepack.mp4'], process_stats={}, infotexts=['Generate finished'])

    def test_video_error_detection_uses_status_fields_only(self):
        self.assertIsNone(scheduler._video_result_error('video', ([], None, '{"prompt":"not found"}', '', '')))
        self.assertEqual(scheduler._video_result_error('video', (None, 'Model not loaded', '', '', '')), 'Model not loaded')
        self.assertIsNone(scheduler._video_result_error('minimax', (None, 'Generate finished')))

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

    def test_update_video_parameters_and_media_atomically(self):
        media = {'__sd_scheduler_type__': 'file', 'id': 0, 'filename': 'condition.mp4', 'media_type': 'video/mp4'}
        task = {
            'mode': 'video',
            'parameters': {'video': {'surface': 'ltx', 'arguments': {'prompt': 'old', 'model': 'old-model', 'condition_video': media}}, 'model': 'old-model'},
            'image_blobs': [b'old-media'],
        }
        with patch.object(scheduler.scheduler_db, 'get_task', return_value=task) as get_task, \
                patch.object(scheduler.scheduler_db, 'update_parameters', return_value=True) as update:
            self.assertTrue(scheduler.update_task_parameters_with_media(
                15,
                {'prompt': 'new', 'model': 'new-model'},
                media_replacements={0: b'new-media'},
            ))

        get_task.assert_called_once_with(15, include_images=True, db_path=None)
        self.assertEqual(task['parameters']['video']['arguments']['prompt'], 'new')
        self.assertEqual(task['parameters']['model'], 'new-model')
        update.assert_called_once_with(15, task['parameters'], db_path=None, image_blobs=[b'new-media'])


class TestSchedulerPolicy(unittest.TestCase):
    def patch_options(self, options):
        with patch.object(sys, 'argv', [sys.argv[0]]), patch.object(scheduler, 'start_worker'):
            from modules import shared
        return patch.object(shared, 'opts', options)

    def test_worker_wait_interval_keeps_busy_queue_polling_fast(self):
        call_queue = SimpleNamespace(is_locked=lambda: True)

        with patch.dict(sys.modules, {'modules.call_queue': call_queue}):
            self.assertEqual(scheduler._worker_wait_interval(), 1.0)
            call_queue.is_locked = lambda: False
            self.assertEqual(scheduler._worker_wait_interval(), 5.0)

    def test_auto_window_boundaries_and_overnight_window(self):
        options = SimpleNamespace(scheduler_state='Auto', scheduler_start='10:00', scheduler_end='18:00')
        with self.patch_options(options):
            now = dt.datetime.now().astimezone()
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

    def test_run_once_uses_configured_retry_delay(self):
        from modules import call_queue

        task = {'id': 14, 'attempts': 1}
        with patch.object(call_queue, 'is_locked', return_value=False), \
                patch.object(scheduler.scheduler_db, 'claim_next_task', return_value=task), \
                patch.object(scheduler.scheduler_db, 'get_task', return_value={'image_blobs': []}), \
                patch.object(scheduler, '_run_task', side_effect=RuntimeError('generation failed')), \
                patch.object(scheduler.scheduler_db, 'retry_or_fail') as retry_or_fail:
            self.assertTrue(scheduler.run_once())

        retry_or_fail.assert_called_once()
        self.assertEqual(retry_or_fail.call_args.args[0], 14)
        self.assertEqual(str(retry_or_fail.call_args.args[1]), 'generation failed')
        self.assertEqual(retry_or_fail.call_args.kwargs['retry_delay'], scheduler.RETRY_DELAY)

    def test_run_once_locks_standard_video_replay(self):
        from modules import call_queue

        task = {
            'id': 13,
            'attempts': 1,
            'mode': 'video',
            'parameters': {'video': {'surface': 'video'}},
        }
        lock_state = {'held': False}

        class QueueLock:
            def __enter__(self):
                lock_state['held'] = True

            def __exit__(self, _exc_type, _exc_value, _traceback):
                lock_state['held'] = False

        def execute(_task):
            self.assertTrue(lock_state['held'])

        with patch.object(scheduler, 'scheduler_enabled', return_value=False), \
                patch.object(call_queue, 'is_locked', return_value=False), \
                patch.object(call_queue, 'get_lock', return_value=QueueLock()) as get_lock, \
                patch.object(scheduler.scheduler_db, 'claim_next_task', return_value=task), \
                patch.object(scheduler.scheduler_db, 'get_task', return_value={'image_blobs': []}), \
                patch.object(scheduler, '_run_task', side_effect=execute):
            self.assertTrue(scheduler.run_once())

        get_lock.assert_called_once_with()

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
