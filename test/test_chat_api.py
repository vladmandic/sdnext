import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI

from modules.chat import api as chat_api
from modules.chat import db
from modules import shared


class TestChatApiRegistration(unittest.TestCase):
    def test_registers_only_session_routes_and_initializes_database(self):
        app = FastAPI()
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                chat_api.register_api(app)
            self.assertTrue(os.path.isfile(db_path))

        registered = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, 'methods', set())
            if route.path.startswith('/sdapi/v1/chat/')
        }
        self.assertEqual(registered, {
            ('/sdapi/v1/chat/sessions', 'GET'),
            ('/sdapi/v1/chat/sessions', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}', 'GET'),
            ('/sdapi/v1/chat/sessions/{session_id}', 'PATCH'),
            ('/sdapi/v1/chat/sessions/{session_id}', 'DELETE'),
            ('/sdapi/v1/chat/sessions/{session_id}/restore', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/messages', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/messages/{message_id}', 'PATCH'),
        })

    def test_message_submission_persists_command_exchange_and_restore_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='set steps=24'),
                )
                detail = chat_api.get_session(session['id'])
                restored = chat_api.restore_session(session['id'])

                self.assertEqual(submission.session.title, 'set steps=24')
                self.assertEqual(submission.user_message.content, 'set steps=24')
                self.assertEqual(submission.assistant_message.content, 'steps set to 24')
                self.assertEqual(submission.result.kind, 'text')
                self.assertEqual(detail.session.drafts['image']['steps'], 24)
                self.assertEqual([message.role for message in detail.messages], ['user', 'assistant'])
                self.assertEqual(restored.restore_errors, [])
                self.assertEqual(restored.session.id, session['id'])

                invalid = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='set steps=999'),
                )
                self.assertEqual(invalid.result.kind, 'error')
                self.assertEqual(invalid.assistant_message.metadata['kind'], 'error')

                endpoint = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='version'))
                self.assertEqual(endpoint.result.kind, 'endpoint')
                self.assertEqual(endpoint.result.data['endpoints'][0]['path'], '/sdapi/v1/version')
                updated = chat_api.patch_message(
                    session['id'],
                    endpoint.assistant_message.id,
                    chat_api.ReqChatMessageUpdate(
                        content='{"version": "test"}',
                        metadata={'kind': 'endpoint', 'alias': 'version', 'status': 200},
                    ),
                )
                self.assertEqual(updated['content'], '{"version": "test"}')
                self.assertEqual(updated['metadata']['status'], 200)

                self.assertTrue(chat_api.delete_session(session['id']).status_code == 204)
                with self.assertRaises(Exception) as context:
                    chat_api.get_session(session['id'])
                self.assertEqual(getattr(context.exception, 'status_code', None), 404)

    def test_title_command_reads_default_name_and_updates_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                current = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='title'))
                renamed = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='title A useful name'),
                )

        self.assertEqual(current.assistant_message.content, 'Current session name: New chat')
        self.assertEqual(current.session.title, 'New chat')
        self.assertEqual(renamed.assistant_message.content, 'Session name set to A useful name')
        self.assertEqual(renamed.session.title, 'A useful name')

    def test_image_generation_uses_control_api_and_persists_no_image_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a landscape'))
                response = SimpleNamespace(
                    images=['base64-image'],
                    info='generated <b>info</b>',
                    params={'prompt': 'a landscape'},
                )
                control_api = Mock(post_control=Mock(return_value=response))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(control=control_api),
                )
                operations = db.list_operations(session['id'])
                messages = db.list_messages(session['id'])
                self.assertEqual(submission.session.last_generation_snapshot['prompt'], 'a landscape')
                replay = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='again another landscape'),
                    SimpleNamespace(control=control_api),
                )
                operations = db.list_operations(session['id'])

        self.assertEqual(control_api.post_control.call_count, 2)
        request = control_api.post_control.call_args_list[0].args[0]
        self.assertIsInstance(request, chat_api.api_control.ReqControl)
        self.assertEqual(request.prompt, 'a landscape')
        self.assertEqual(request.inputs, [])
        self.assertTrue(request.send_images)
        self.assertFalse(request.save_images)
        self.assertEqual(submission.result.kind, 'operation')
        self.assertEqual(submission.result.data['images'], ['base64-image'])
        self.assertEqual(submission.assistant_message.content, 'Generated 1 image(s)\ngenerated <b>info</b>')
        self.assertEqual(submission.assistant_message.metadata['kind'], 'operation')
        self.assertEqual(submission.assistant_message.metadata['format'], 'html')
        self.assertEqual(operations[0]['status'], 'completed')
        self.assertEqual(operations[0]['result']['image_count'], 1)
        self.assertNotIn('images', operations[0]['result'])
        self.assertNotIn('base64-image', str(messages))
        self.assertEqual(control_api.post_control.call_args.args[0].prompt, 'another landscape')
        self.assertIn('again', [operation['command'] for operation in operations])
        self.assertEqual(replay.result.kind, 'operation')

    def test_image_generation_records_failure_from_control_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a landscape'))
                control_api = Mock(post_control=Mock(side_effect=RuntimeError('control failed')))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(control=control_api),
                )
                operation = db.list_operations(session['id'])[0]

        self.assertEqual(submission.result.kind, 'operation')
        self.assertEqual(operation['status'], 'failed')
        self.assertEqual(operation['error'], 'control failed')
        self.assertEqual(submission.assistant_message.metadata['status'], 'failed')

    def test_image_generation_reports_saved_output_paths(self):
        saved_path = '/outputs/generated image.png'
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                response = SimpleNamespace(
                    images=['base64-image'], saved_images=[saved_path], info='', params={},
                )
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(control=Mock(post_control=Mock(return_value=response))),
                )

        self.assertIn(f'<code>{saved_path}</code>', submission.assistant_message.content)
        serialized = chat_api.api_control.ResControl(saved_images=[saved_path]).model_dump()
        self.assertNotIn('saved_images', serialized)

    def test_video_generation_uses_video_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='video'))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a test clip'))
                response = SimpleNamespace(
                    video='video-base64', video_path=None, thumbnail='thumbnail-base64',
                    frames=[], frames_count=8, info='video info', params={'prompt': 'a test clip'},
                )
                video_api = Mock(post_video=Mock(return_value=response))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(video=video_api),
                )
                operation = db.list_operations(session['id'])[0]

        request = video_api.post_video.call_args.args[0]
        self.assertEqual(request.prompt, 'a test clip')
        self.assertEqual(submission.result.data['video'], 'video-base64')
        self.assertEqual(submission.result.data['thumbnail'], 'thumbnail-base64')
        self.assertEqual(submission.assistant_message.content, 'Generated video\nvideo info')
        self.assertEqual(operation['context'], 'video')

    def test_process_generation_uses_transient_image_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='process'))
                process_api = Mock(extras_single_image_api=Mock(return_value=SimpleNamespace(
                    image='processed-base64', html_info='processed <b>info</b>',
                )))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate', image='input-image-base64'),
                    SimpleNamespace(process=process_api),
                )
                replay = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='reprocess', image='latest-image-base64'),
                    SimpleNamespace(process=process_api),
                )
                operations = db.list_operations(session['id'])
                messages = db.list_messages(session['id'])

        first_request = process_api.extras_single_image_api.call_args_list[0].args[0]
        replay_request = process_api.extras_single_image_api.call_args_list[1].args[0]
        self.assertEqual(first_request.image, 'input-image-base64')
        self.assertEqual(replay_request.image, 'latest-image-base64')
        self.assertCountEqual([operation['command'] for operation in operations], ['again', 'generate'])
        self.assertTrue(all('image' not in operation['snapshot'] for operation in operations))
        self.assertNotIn('input-image-base64', str(messages))
        self.assertEqual(submission.result.data['images'], ['processed-base64'])
        self.assertEqual(submission.assistant_message.content, 'Image processed\nprocessed <b>info</b>')
        self.assertEqual(replay.result.kind, 'operation')

    def test_caption_generation_uses_transient_image_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.api.caption.post_vqa', return_value=SimpleNamespace(
                        answer='A blue bird', annotated_image=None,
                    )) as post_vqa:
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='caption'))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate', image='caption-input-base64'),
                )
                operation = db.list_operations(session['id'])[0]

        request = post_vqa.call_args.args[0]
        self.assertEqual(request.image, 'caption-input-base64')
        self.assertNotIn('image', operation['snapshot'])
        self.assertEqual(submission.result.data['caption'], 'A blue bird')
        self.assertEqual(submission.assistant_message.content, 'A blue bird')

    def test_schedule_dispatches_image_request_to_scheduler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.scheduler.enqueue_control_request', return_value=iter(
                        [(None, None, 'Queued scheduler task task-1', None)]
                    )) as enqueue:
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='image'))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a landscape'))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='schedule'),
                )

        self.assertEqual(submission.result.kind, 'schedule')
        self.assertEqual(submission.assistant_message.content, 'Queued scheduler task task-1')
        self.assertEqual(enqueue.call_args.kwargs['prompt'], 'a landscape')
        self.assertNotIn('send_images', enqueue.call_args.kwargs)
        self.assertNotIn('save_images', enqueue.call_args.kwargs)

    def test_schedule_dispatches_video_request_to_scheduler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.scheduler.enqueue_video_request', return_value=[
                        None, None, None, None, 'Queued scheduler task task-2',
                    ]) as enqueue:
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='video'))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a test clip'))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='schedule'),
                )

        self.assertEqual(submission.result.kind, 'schedule')
        self.assertEqual(submission.assistant_message.content, 'Queued scheduler task task-2')
        self.assertEqual(enqueue.call_args.args[0], 'video')
        self.assertEqual(enqueue.call_args.kwargs['prompt'], 'a test clip')

    def test_version_two_database_migrates_operation_contexts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with sqlite3.connect(db_path) as connection:
                connection.execute('''
                    CREATE TABLE operations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                        command TEXT NOT NULL,
                        context TEXT NOT NULL CHECK (context IN ('image', 'video')),
                        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
                        snapshot TEXT NOT NULL DEFAULT '{}',
                        result TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT
                    )
                ''')
                connection.execute('CREATE INDEX chat_operations_session_idx ON operations(session_id, created_at DESC, id)')
                connection.execute('PRAGMA user_version = 2')

            with patch.object(db, '_db_path', return_value=db_path):
                db.initialize()
                session = db.create_session(context='caption')
                operation = db.create_operation(session['id'], 'generate', 'caption', {})
                diagnostics = db.diagnostics()

        self.assertEqual(operation['context'], 'caption')
        self.assertEqual(diagnostics['user_version'], 3)
        self.assertTrue(diagnostics['ok'])

    def test_database_initialization_failure_disables_chat_without_routes(self):
        app = FastAPI()
        with patch.dict(shared.opts.data, {'chat_enabled': True}), \
                patch.object(db, 'initialize', side_effect=sqlite3.OperationalError('database is locked')), \
                patch.object(chat_api.log, 'error'):
            self.assertFalse(chat_api.register_api(app))
            self.assertFalse(shared.opts.data['chat_enabled'])

        self.assertFalse(any(route.path.startswith('/sdapi/v1/chat/') for route in app.routes))

    def test_help_sections_are_persisted_in_assistant_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='help'),
                )

        expected_titles = ['Process', 'Params', 'Special', 'Global', 'Server']
        self.assertEqual([section['title'] for section in submission.result.sections], expected_titles)
        self.assertEqual(
            [section['title'] for section in submission.assistant_message.metadata['sections']],
            expected_titles,
        )
