import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from modules.api import scheduler as scheduler_api


class FakeApi:
    def __init__(self):
        self.routes = []

    def add_api_route(self, path, endpoint, **kwargs):
        self.routes.append((path, endpoint, kwargs))


class TestSchedulerApiRegistration(unittest.TestCase):
    def test_routes_are_disabled_when_scheduler_worker_start_failed(self):
        api = FakeApi()
        scheduler_api.register_api(api)
        availability_check = next(
            dependency.dependency
            for path, _endpoint, options in api.routes
            if path == '/sdapi/v1/scheduler/tasks'
            for dependency in options['dependencies']
        )

        for _path, _endpoint, options in api.routes:
            if _endpoint in (scheduler_api.get_scheduler_status, scheduler_api.update_scheduler_settings):
                continue
            self.assertTrue(any(
                dependency.dependency is availability_check
                for dependency in options['dependencies']
            ))

        with patch.object(scheduler_api.scheduler, '_worker_start_failed', True):
            with self.assertRaises(HTTPException) as caught:
                availability_check()
        self.assertEqual(caught.exception.status_code, 503)

        with patch.object(scheduler_api.scheduler, '_worker_start_failed', False), \
            patch.object(scheduler_api.scheduler, 'scheduler_stopped', return_value=True):
            with self.assertRaises(HTTPException) as caught:
                availability_check()
        self.assertEqual(caught.exception.status_code, 503)

    def test_management_routes_are_registered_without_submission_or_result_write(self):
        api = FakeApi()
        scheduler_api.register_api(api)
        registered = {(path, method) for path, _endpoint, options in api.routes for method in options['methods']}

        self.assertEqual(len(api.routes), 17)
        self.assertTrue(all(path.startswith('/sdapi/v1/scheduler') for path, _endpoint, _options in api.routes))
        self.assertIn(('/sdapi/v1/scheduler', 'GET'), registered)
        self.assertIn(('/sdapi/v1/scheduler/tasks', 'GET'), registered)
        self.assertIn(('/sdapi/v1/scheduler/tasks/{task_id}/execute', 'POST'), registered)
        self.assertIn(('/sdapi/v1/scheduler/tasks/{task_id}/move', 'POST'), registered)
        self.assertIn(('/sdapi/v1/scheduler/state', 'POST'), registered)
        self.assertIn(('/sdapi/v1/scheduler/tasks/{task_id}/images/{image_index}', 'GET'), registered)
        self.assertIn(('/sdapi/v1/scheduler/tasks/{task_id}/media/{media_index}', 'GET'), registered)
        self.assertIn(('/sdapi/v1/scheduler/tasks/{task_id}/parameters/media', 'PATCH'), registered)
        self.assertFalse(any(path == '/sdapi/v1/scheduler/tasks' and method == 'POST' for path, method in registered))
        self.assertFalse(any(path.endswith('/results') for path, _endpoint, _options in api.routes))

    def test_input_image_endpoint_returns_only_image_references(self):
        task = {
            'parameters': {
                'control': {
                    'inputs': [
                        {'__sd_scheduler_type__': 'image', 'id': 0},
                        {'__sd_scheduler_type__': 'blob', 'id': 1},
                    ],
                },
            },
        }
        with patch.object(scheduler_api.scheduler, 'get_task', return_value=task), patch.object(scheduler_api.scheduler, 'get_task_image', return_value=b'png-bytes'):
            response = scheduler_api.get_task_image(5, 0)
            self.assertEqual(response.body, b'png-bytes')
            self.assertEqual(response.media_type, 'image/png')
            with self.assertRaises(HTTPException) as caught:
                scheduler_api.get_task_image(5, 1)
            self.assertEqual(caught.exception.status_code, 404)
            with self.assertRaises(HTTPException) as caught:
                scheduler_api.get_task_image(5, -1)
            self.assertEqual(caught.exception.status_code, 404)

    def test_input_media_endpoint_uses_stored_mime_type(self):
        task = {
            'parameters': {
                'video': {
                    'arguments': {
                        'condition_video': {'__sd_scheduler_type__': 'file', 'id': 2, 'media_type': 'video/mp4'},
                    },
                },
            },
        }
        with patch.object(scheduler_api.scheduler, 'get_task', return_value=task), patch.object(scheduler_api.scheduler, 'get_task_image', return_value=b'mp4-bytes'):
            response = scheduler_api.get_task_media(5, 2)
        self.assertEqual(response.body, b'mp4-bytes')
        self.assertEqual(response.media_type, 'video/mp4')

    def test_multipart_parameter_update_forwards_media_replacements(self):
        task = {'id': 5}
        upload = UploadFile(filename='replacement.mp4', file=io.BytesIO(b'video-bytes'))
        with patch.object(scheduler_api.scheduler, 'get_task', return_value=task), \
                patch.object(scheduler_api.scheduler, 'update_task_parameters_with_media', return_value=True) as update:
            result = asyncio.run(scheduler_api.update_task_parameters_media(5, '{"prompt":"edited"}', '[3]', [upload]))
        self.assertEqual(result, task)
        update.assert_called_once_with(5, {'prompt': 'edited'}, media_replacements={3: b'video-bytes'})

    def test_execute_endpoint_requests_immediate_task_execution(self):
        with patch.object(scheduler_api.scheduler, 'get_task', return_value={'id': 12}), \
                patch.object(scheduler_api.scheduler, 'queue_existing_task', return_value=True) as queue_task:
            result = scheduler_api.execute_task(12)
        self.assertEqual(result, {'task_id': 12, 'status': 'scheduled'})
        queue_task.assert_called_once_with(12, immediate=True)

    def test_scheduler_state_endpoint_persists_requested_mode_and_window(self):
        expected = {'state': 'Auto', 'start': '21:00', 'end': '06:00'}
        with patch.object(scheduler_api.scheduler, 'set_scheduler_state') as set_state, \
                patch.object(scheduler_api.scheduler, 'get_status', return_value=expected), \
                patch.object(scheduler_api.scheduler, 'start_worker') as start_worker:
            result = scheduler_api.update_scheduler_settings(scheduler_api.SchedulerSettingsRequest(**expected))
        self.assertEqual(result, expected)
        set_state.assert_called_once_with('Auto', '21:00', '06:00')
        start_worker.assert_called_once_with()

        with patch.object(scheduler_api.scheduler, 'set_scheduler_state', side_effect=ValueError('Invalid time')):
            with self.assertRaises(HTTPException) as caught:
                scheduler_api.update_scheduler_settings(scheduler_api.SchedulerSettingsRequest(start='invalid'))
            self.assertEqual(caught.exception.status_code, 422)

    def test_global_pause_and_resume_map_to_explicit_modes(self):
        with patch.object(scheduler_api.scheduler, 'pause_all') as pause, patch.object(scheduler_api.scheduler, 'resume_all') as resume:
            self.assertEqual(scheduler_api.pause_scheduler(), {'state': 'Paused', 'paused': True})
            self.assertEqual(scheduler_api.resume_scheduler(), {'state': 'Active', 'paused': False})
        pause.assert_called_once_with()
        resume.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
