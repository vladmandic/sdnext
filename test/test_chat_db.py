from importlib import reload
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from modules.chat import db


class TestChatDatabase(unittest.TestCase):
    temp_dir: str
    db_path: pathlib.Path

    def setUp(self):
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.db_path = pathlib.Path(self.temp_dir) / 'chat.db'

    def test_session_message_operation_crud_and_reopen(self):
        session = db.create_session('Test session', db_path=self.db_path)
        session = db.update_session(
            session['id'],
            db_path=self.db_path,
            context='video',
            drafts={'image': {'steps': 20}, 'video': {'frames': 17}},
            last_generation_prompt='a test prompt',
            last_generation_snapshot={'prompt': 'a test prompt', 'steps': 20},
        )
        self.assertEqual(session['context'], 'video')
        self.assertEqual(session['drafts']['video']['frames'], 17)

        message = db.create_message(session['id'], 'user', 'generate', {'source': 'test'}, db_path=self.db_path)
        operation = db.create_operation(
            session['id'], 'generate', 'video', {'frames': 17}, message_id=message['id'], db_path=self.db_path,
        )
        self.assertEqual(operation['status'], 'running')
        self.assertEqual(operation['snapshot'], {'frames': 17})
        operation = db.update_operation(operation['id'], 'completed', {'video': 'inline'}, db_path=self.db_path)
        self.assertEqual(operation['result'], {'video': 'inline'})

        reload(db)
        db.initialize(self.db_path)
        self.assertEqual(db.get_session(session['id'], db_path=self.db_path)['last_generation_snapshot']['steps'], 20)
        self.assertEqual(db.list_messages(session['id'], db_path=self.db_path)[0]['metadata'], {'source': 'test'})
        self.assertEqual(db.list_operations(session['id'], db_path=self.db_path)[0]['status'], 'completed')
        self.assertTrue(db.diagnostics(self.db_path)['ok'])

    def test_initialization_marks_running_operations_failed(self):
        session = db.create_session(db_path=self.db_path)
        operation = db.create_operation(session['id'], 'generate', 'image', {}, db_path=self.db_path)

        with patch.object(db.log, 'info') as log_info:
            reload(db)
            db.initialize(self.db_path)

        message = log_info.call_args[0][0]
        self.assertIn('integrity=ok recovered=1', message)
        fields = ['db=', 'path=', 'schema=', 'user=', 'size=', 'journal=', 'stat=', 'mode=', 'time=', 'integrity=', 'recovered=']
        positions = [message.index(field) for field in fields]
        self.assertEqual(positions, sorted(positions))

        recovered = db.get_operation(operation['id'], db_path=self.db_path)
        self.assertEqual(recovered['status'], 'failed')
        self.assertEqual(recovered['error'], 'Server restarted before operation completed')
        self.assertIsNotNone(recovered['completed_at'])

    def test_context_schema_migration_preserves_chat_rows(self):
        session = db.create_session('Legacy session', db_path=self.db_path)
        message = db.create_message(session['id'], 'user', 'hello', db_path=self.db_path)
        operation = db.create_operation(session['id'], 'generate', 'image', {}, db_path=self.db_path)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute('PRAGMA foreign_keys = OFF')
            connection.execute('''
                CREATE TABLE sessions_old (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    context TEXT NOT NULL CHECK (context IN ('image', 'video')),
                    drafts TEXT NOT NULL DEFAULT '{}',
                    last_generation_prompt TEXT,
                    last_generation_snapshot TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            connection.execute('''
                INSERT INTO sessions_old SELECT id, title, context, drafts, last_generation_prompt,
                    last_generation_snapshot, created_at, updated_at FROM sessions
            ''')
            connection.execute('DROP TABLE sessions')
            connection.execute('ALTER TABLE sessions_old RENAME TO sessions')
            connection.execute('PRAGMA user_version = 1')

        reload(db)
        db.initialize(self.db_path)
        updated = db.update_session(session['id'], db_path=self.db_path, context='caption')

        self.assertEqual(updated['context'], 'caption')
        self.assertEqual(db.list_messages(session['id'], db_path=self.db_path)[0]['id'], message['id'])
        recovered = db.get_operation(operation['id'], db_path=self.db_path)
        self.assertEqual(recovered['status'], 'failed')
        self.assertEqual(recovered['error'], 'Server restarted before operation completed')
        self.assertTrue(db.diagnostics(self.db_path)['ok'])

    def test_selected_image_list_migration_preserves_legacy_selection(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute('''
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    context TEXT NOT NULL,
                    drafts TEXT NOT NULL DEFAULT '{}',
                    last_generation_prompt TEXT,
                    last_generation_snapshot TEXT,
                    selected_image_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            connection.execute(
                'INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                ('legacy', 'Legacy session', 'image', '{}', None, None, 'artifact-1', '2025-01-01', '2025-01-01'),
            )
            connection.execute('PRAGMA user_version = 4')

        reload(db)
        migrated = db.get_session('legacy', db_path=self.db_path)

        self.assertEqual(migrated['selected_image_id'], 'artifact-1')
        self.assertEqual(migrated['selected_image_ids'], ['artifact-1'])
        self.assertTrue(db.diagnostics(self.db_path)['ok'])

    def test_initialization_retries_transient_database_lock(self):
        connect = db.connect
        attempts = 0

        def lock_twice(db_path=None, timeout=5):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError('database is locked')
            return connect(db_path, timeout=timeout)

        with patch.object(db, 'connect', side_effect=lock_twice), patch.object(db.time, 'sleep'):
            db.initialize(self.db_path)

        self.assertEqual(attempts, 3)

    def test_session_deletion_cascades_messages_and_operations(self):
        session = db.create_session(db_path=self.db_path)
        message = db.create_message(session['id'], 'user', 'hello', db_path=self.db_path)
        operation = db.create_operation(session['id'], 'generate', 'image', {}, message_id=message['id'], db_path=self.db_path)

        self.assertTrue(db.delete_session(session['id'], db_path=self.db_path))
        self.assertIsNone(db.get_session(session['id'], db_path=self.db_path))
        self.assertEqual(db.list_messages(session['id'], db_path=self.db_path), [])
        self.assertEqual(db.list_operations(session['id'], db_path=self.db_path), [])
        self.assertIsNone(db.get_operation(operation['id'], db_path=self.db_path))
        self.assertFalse(db.delete_session(session['id'], db_path=self.db_path))

    def test_artifact_deletion_preserves_files_and_marks_lineage(self):
        session = db.create_session(db_path=self.db_path)
        message = db.create_message(session['id'], 'user', 'image upload', db_path=self.db_path)
        source_path = pathlib.Path(self.temp_dir) / 'source.png'
        result_path = pathlib.Path(self.temp_dir) / 'result.png'
        source_path.write_bytes(b'source')
        result_path.write_bytes(b'result')
        source = db.create_artifact(
            session['id'], 'image/png', 'source.png', 'input', str(source_path),
            message_id=message['id'], db_path=self.db_path,
        )
        result = db.create_artifact(
            session['id'], 'image/png', 'result.png', 'result', str(result_path),
            source_artifact_id=source['id'], db_path=self.db_path,
        )
        db.set_selected_image(session['id'], source['id'], db_path=self.db_path)

        removed = db.delete_artifact(session['id'], source['id'], db_path=self.db_path)
        descendant = db.get_artifact(session['id'], result['id'], db_path=self.db_path)
        updated_session = db.get_session(session['id'], db_path=self.db_path)

        self.assertEqual(removed['id'], source['id'])
        self.assertEqual(descendant['source_artifact_id'], source['id'])
        self.assertEqual(descendant['source_deleted'], 1)
        self.assertIsNone(updated_session['selected_image_id'])
        self.assertTrue(source_path.is_file())
        self.assertTrue(result_path.is_file())

        self.assertTrue(db.delete_session(session['id'], db_path=self.db_path))
        self.assertTrue(result_path.is_file())
        self.assertEqual(db.list_artifacts(session['id'], db_path=self.db_path), [])

    def test_invalid_context_and_role_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid chat context'):
            db.create_session(context='audio', db_path=self.db_path)
        session = db.create_session(db_path=self.db_path)
        with self.assertRaisesRegex(ValueError, 'Invalid chat message role'):
            db.create_message(session['id'], 'tool', 'hello', db_path=self.db_path)
