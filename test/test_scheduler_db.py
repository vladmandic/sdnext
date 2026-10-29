import datetime as dt
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from modules import scheduler_db
from modules import scheduler


class TestSchedulerDatabase(unittest.TestCase):
    temp_dir: str
    db_path: pathlib.Path

    def setUp(self):
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.db_path = pathlib.Path(self.temp_dir) / 'scheduler.db'
        self.addCleanup(scheduler_db.configure, None)

    def test_connection_signals_database_changes(self):
        class FakeConnection:
            total_changes = 0

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        connection = FakeConnection()
        scheduler_db.change_event.clear()
        try:
            with patch.object(scheduler_db, 'connect', return_value=connection), \
                    patch.object(scheduler_db.log, 'trace', create=True):
                with scheduler_db._connection():
                    connection.total_changes += 1
            self.assertTrue(scheduler_db.change_event.is_set())
        finally:
            scheduler_db.change_event.clear()

    def test_configured_database_directory_and_overrides(self):
        scheduler_dir = pathlib.Path(self.temp_dir) / 'custom'
        scheduler_db.configure(scheduler_dir)
        self.assertEqual(scheduler_db._db_path(), os.fspath(scheduler_dir / 'scheduler.db'))
        explicit_path = self.db_path / 'explicit.db'
        self.assertEqual(scheduler_db._db_path(explicit_path), os.fspath(explicit_path))
        env_path = pathlib.Path(self.temp_dir) / 'environment.db'
        with patch.dict(os.environ, {'SD_SCHEDULER_DB': os.fspath(env_path)}):
            self.assertEqual(scheduler_db._db_path(), os.fspath(env_path))

    def test_default_database_path_tracks_models_directory(self):
        from types import SimpleNamespace
        from modules import paths

        shared = SimpleNamespace(opts=SimpleNamespace(scheduler_dir=None))
        local_data = pathlib.Path(self.temp_dir) / 'data'
        with patch.dict(scheduler_db.sys.modules, {'modules.shared': shared}), \
                patch.dict(os.environ, {'SD_SCHEDULER_DB': ''}), \
                patch.object(scheduler_db, '_configured_db_path', None), \
                patch.object(paths, 'data_path', os.fspath(local_data)), \
                patch.object(paths, 'models_path', '/mnt/models'), \
                patch.object(paths, 'config', {}):
            self.assertEqual(
                scheduler_db._db_path(),
                os.fspath(pathlib.Path('/mnt/models') / 'scheduler' / 'scheduler.db'),
            )

    def test_rollback_journal_is_not_rewritten_when_already_enabled(self):
        class FakeCursor:
            def fetchone(self):
                return ('delete',)

        class FakeConnection:
            def __init__(self):
                self.statements = []

            def execute(self, statement):
                self.statements.append(statement)
                return FakeCursor()

        connection = FakeConnection()
        scheduler_db._ensure_rollback_journal(connection)
        self.assertEqual(connection.statements, ['PRAGMA journal_mode'])

    def test_wal_database_is_switched_to_rollback_journal(self):
        class FakeCursor:
            def __init__(self, mode):
                self.mode = mode

            def fetchone(self):
                return (self.mode,)

        class FakeConnection:
            def __init__(self):
                self.statements = []

            def execute(self, statement):
                self.statements.append(statement)
                return FakeCursor('wal' if statement == 'PRAGMA journal_mode' else 'delete')

        connection = FakeConnection()
        scheduler_db._ensure_rollback_journal(connection)
        self.assertEqual(connection.statements, ['PRAGMA journal_mode', 'PRAGMA journal_mode = DELETE'])

    def test_initialization_retries_external_exclusive_lock(self):
        lock_connection = scheduler_db.connect(self.db_path)
        try:
            lock_connection.execute('BEGIN EXCLUSIVE')

            def release_lock(_duration):
                lock_connection.rollback()

            with patch.object(scheduler_db.time, 'sleep', side_effect=release_lock):
                scheduler_db.initialize(self.db_path)
        finally:
            lock_connection.close()

        connection = scheduler_db.connect(self.db_path)
        try:
            task_table = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'tasks'").fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(task_table)

    def test_failed_initialization_is_not_retried(self):
        with patch.object(scheduler_db, '_ensure_rollback_journal', side_effect=sqlite3.OperationalError('database is locked')) as ensure_journal, \
                patch.object(scheduler_db.time, 'sleep'), \
                patch.object(scheduler_db.log, 'warning'):
            with self.assertRaises(sqlite3.OperationalError):
                scheduler_db.initialize(self.db_path)
            failed_attempts = ensure_journal.call_count
            self.assertEqual(failed_attempts, 10)

            with self.assertRaisesRegex(sqlite3.OperationalError, 'previously failed'):
                scheduler_db.initialize(self.db_path)
            self.assertEqual(ensure_journal.call_count, failed_attempts)

    def test_create_claim_complete_and_retrieve(self):
        task_id = scheduler_db.create_task({'prompt': 'test'}, [b'image-bytes'], db_path=self.db_path)

        claimed = scheduler_db.claim_next_task(db_path=self.db_path)
        self.assertEqual(claimed['id'], task_id)
        self.assertEqual(claimed['status'], 'running')
        self.assertEqual(claimed['attempts'], 1)
        self.assertEqual(scheduler_db.get_task(task_id, include_images=True, db_path=self.db_path)['image_blobs'], [b'image-bytes'])
        self.assertEqual(scheduler_db.get_task_image(task_id, 0, db_path=self.db_path), b'image-bytes')
        self.assertIsNone(scheduler_db.get_task_image(task_id, 1, db_path=self.db_path))
        self.assertIsNone(scheduler_db.get_task_image(task_id, -1, db_path=self.db_path))
        process_stats = {'total': 5.2, 'sample': 4.1}
        infotexts = ['prompt: test, steps: 20']
        self.assertTrue(scheduler_db.update_results(
            task_id,
            ['/tmp/output.png'],
            db_path=self.db_path,
            process_stats=process_stats,
            infotexts=infotexts,
        ))
        task = scheduler_db.get_task(task_id, db_path=self.db_path)
        self.assertEqual(task['status'], 'completed')
        self.assertEqual(task['results'], ['/tmp/output.png'])
        self.assertEqual(task['process_stats'], process_stats)
        self.assertEqual(task['infotexts'], infotexts)
        self.assertTrue(scheduler_db.queue_task(task_id, db_path=self.db_path))
        task = scheduler_db.get_task(task_id, db_path=self.db_path)
        self.assertEqual(task['process_stats'], {})
        self.assertEqual(task['infotexts'], [])

    def test_task_mode_is_persisted_and_validated(self):
        task_id = scheduler_db.create_task({'video': {'surface': 'video'}}, mode='video', db_path=self.db_path)

        self.assertEqual(scheduler_db.get_task(task_id, db_path=self.db_path)['mode'], 'video')
        with self.assertRaisesRegex(ValueError, 'Invalid scheduler task mode'):
            scheduler_db.create_task({}, mode='invalid', db_path=self.db_path)

    def test_foreign_keys_are_enabled_on_each_connection(self):
        task_id = scheduler_db.create_task({'prompt': 'test'}, [b'image-bytes'], db_path=self.db_path)

        connection = scheduler_db.connect(self.db_path)
        try:
            self.assertEqual(connection.execute('PRAGMA foreign_keys').fetchone()[0], 1)
        finally:
            connection.close()

        self.assertTrue(scheduler_db.delete_task(task_id, db_path=self.db_path))
        stats, _ = scheduler_db.diagnostics(self.db_path)
        self.assertEqual(stats['task_images'], 0)
        self.assertEqual(stats['foreign_key_errors'], [])

    def test_diagnostics_reports_database_health_and_task_statistics(self):
        scheduler_db.create_task({'prompt': 'test'}, [b'image-bytes'], db_path=self.db_path)

        stats = scheduler_db.diagnostics(self.db_path)

        self.assertTrue(stats['ok'])
        self.assertEqual(stats['integrity'], ['ok'])
        self.assertEqual(stats['foreign_key_errors'], [])
        self.assertEqual(stats['tasks'], 1)
        self.assertEqual(stats['statuses'], {'scheduled': 1})
        self.assertEqual(stats['task_images'], 1)
        self.assertEqual(stats['image_bytes'], len(b'image-bytes'))
        self.assertEqual(stats['journal_mode'], 'delete')
        self.assertGreater(stats['file_bytes'], 0)

    def test_diagnostics_reports_invalid_task_json(self):
        task_id = scheduler_db.create_task({'prompt': 'test'}, db_path=self.db_path)
        with scheduler_db._connection(self.db_path) as connection:
            connection.execute('UPDATE tasks SET parameters = ? WHERE id = ?', ('invalid-json', task_id))

        stats = scheduler_db.diagnostics(self.db_path)

        self.assertFalse(stats['ok'])
        self.assertEqual(stats['json_errors'][0]['task_id'], task_id)
        self.assertEqual(stats['json_errors'][0]['field'], 'parameters')

    def test_prune_compacts_and_analyzes_after_deleting_tasks(self):
        task_id = scheduler_db.create_task({'prompt': 'old'}, db_path=self.db_path)
        with scheduler_db._connection(self.db_path) as connection:
            connection.execute("UPDATE tasks SET created_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (task_id,))

        statements = []
        connect = scheduler_db.connect

        def traced_connect(db_path=None, timeout=5):
            connection = connect(db_path, timeout)
            connection.set_trace_callback(statements.append)
            return connection

        with patch.object(scheduler_db, 'connect', side_effect=traced_connect):
            deleted = scheduler_db.prune_tasks('2021-01-01T00:00:00+00:00', db_path=self.db_path)

        self.assertEqual(deleted, 1)
        statements = [statement.strip().upper() for statement in statements]
        delete_index = next(index for index, statement in enumerate(statements) if statement.startswith('DELETE FROM TASKS'))
        vacuum_index = statements.index('VACUUM')
        analyze_index = statements.index('ANALYZE;')
        self.assertLess(delete_index, vacuum_index)
        self.assertLess(vacuum_index, analyze_index)

    def test_fifo_reorder_and_scheduled_head_blocking(self):
        future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        first = scheduler_db.create_task({'prompt': 'future'}, scheduled_at=future, db_path=self.db_path)
        second = scheduler_db.create_task({'prompt': 'second'}, db_path=self.db_path)
        third = scheduler_db.create_task({'prompt': 'third'}, db_path=self.db_path)

        self.assertTrue(scheduler_db.move_task(third, first, 'before', db_path=self.db_path))
        self.assertEqual([task['id'] for task in scheduler_db.list_tasks(db_path=self.db_path)], [third, first, second])
        self.assertEqual(scheduler_db.claim_next_task(db_path=self.db_path)['id'], third)
        scheduler_db.update_results(third, [], db_path=self.db_path)
        self.assertIsNone(scheduler_db.claim_next_task(db_path=self.db_path))

        task = scheduler_db.get_task(first, db_path=self.db_path)
        self.assertEqual(task['scheduled_by_user'], 1)
        self.assertEqual(scheduler_db.get_task(second, db_path=self.db_path)['scheduled_by_user'], 0)
        self.assertTrue(scheduler_db.queue_task(third, db_path=self.db_path))
        self.assertEqual([task['id'] for task in scheduler_db.list_tasks(db_path=self.db_path)], [first, second, third])

    def test_claim_skips_paused_queue_head(self):
        paused = scheduler_db.create_task({'prompt': 'paused'}, db_path=self.db_path)
        next_task = scheduler_db.create_task({'prompt': 'next'}, db_path=self.db_path)
        self.assertTrue(scheduler_db.pause_task(paused, db_path=self.db_path))

        claimed = scheduler_db.claim_next_task(db_path=self.db_path)

        self.assertEqual(claimed['id'], next_task)
        self.assertEqual(scheduler_db.get_task(paused, db_path=self.db_path)['status'], 'paused')

    def test_empty_claim_skips_write_transaction(self):
        scheduler_db.initialize(self.db_path)
        statements = []
        connect = scheduler_db.connect

        def traced_connect(db_path=None, timeout=5):
            connection = connect(db_path, timeout=timeout)
            connection.set_trace_callback(statements.append)
            return connection

        with patch.object(scheduler_db, 'connect', side_effect=traced_connect):
            self.assertIsNone(scheduler_db.claim_next_task(db_path=self.db_path))
        self.assertFalse(any(statement.upper().startswith('BEGIN IMMEDIATE') for statement in statements))

    def test_existing_database_migrates_fifo_and_schedule_markers(self):
        future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).replace(microsecond=0).isoformat()
        connection = scheduler_db.connect(self.db_path)
        try:
            connection.executescript('''
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    results TEXT NOT NULL DEFAULT '[]',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    last_error TEXT
                );
            ''')
            connection.execute(
                "INSERT INTO tasks(created_at, scheduled_at, status, parameters) VALUES (?, ?, 'scheduled', '{}')",
                ('2026-01-01T00:00:00+00:00', future,),
            )
            connection.execute(
                "INSERT INTO tasks(created_at, scheduled_at, status, parameters) VALUES (?, ?, 'scheduled', '{}')",
                ('2026-01-02T00:00:00+00:00', (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).replace(microsecond=0).isoformat(),),
            )
        finally:
            connection.close()

        scheduler_db.initialize(self.db_path)
        tasks = scheduler_db.list_tasks(db_path=self.db_path)
        self.assertEqual([task['id'] for task in tasks], [1, 2])
        self.assertEqual([task['queue_position'] for task in tasks], [0, 1])
        self.assertEqual([task['mode'] for task in tasks], ['control', 'control'])
        self.assertEqual([task['scheduled_by_user'] for task in tasks], [1, 0])
        self.assertTrue(all('execute_now' not in task for task in tasks))

    def test_immediate_queue_preempts_future_and_paused_tasks(self):
        future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        scheduled = scheduler_db.create_task({'prompt': 'future'}, scheduled_at=future, db_path=self.db_path)
        manual = scheduler_db.create_task({'prompt': 'manual'}, db_path=self.db_path)
        trailing = scheduler_db.create_task({'prompt': 'trailing'}, db_path=self.db_path)
        self.assertTrue(scheduler_db.pause_task(manual, db_path=self.db_path))
        self.assertIsNone(scheduler_db.claim_next_task(force_only=True, db_path=self.db_path))

        self.assertTrue(scheduler_db.queue_task(manual, db_path=self.db_path, immediate=True))
        tasks = scheduler_db.list_tasks(db_path=self.db_path)
        self.assertEqual([task['id'] for task in tasks], [manual, scheduled, trailing])
        self.assertEqual([task['queue_position'] for task in tasks], [0, 1, 2])
        connection = scheduler_db.connect(self.db_path)
        try:
            execute_now = connection.execute('SELECT execute_now FROM tasks WHERE id = ?', (manual,)).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(execute_now, 1)
        self.assertNotIn('execute_now', tasks[0])
        self.assertEqual(tasks[0]['status'], 'scheduled')
        self.assertEqual(tasks[0]['scheduled_by_user'], 0)

        claimed = scheduler_db.claim_next_task(force_only=True, db_path=self.db_path)
        self.assertEqual(claimed['id'], manual)
        self.assertIsNone(scheduler_db.claim_next_task(force_only=True, db_path=self.db_path))
        self.assertIsNone(scheduler_db.claim_next_task(db_path=self.db_path))

    def test_search_updates_pause_resume_and_prune(self):
        first = scheduler_db.create_task({'prompt': 'one'}, db_path=self.db_path)
        second = scheduler_db.create_task(
            {'control': {'prompt': 'two', 'steps': 20, 'state': '', 'scheduler': '', 'units': [], 'input_script_args': []}},
            [b'input-image'],
            db_path=self.db_path,
        )
        old_time = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).isoformat()
        connection = scheduler_db.connect(self.db_path)
        try:
            connection.execute('UPDATE tasks SET created_at = ? WHERE id = ?', (old_time, first))
        finally:
            connection.close()

        self.assertEqual([task['id'] for task in scheduler_db.search_tasks(parameters={'prompt': 'two'}, db_path=self.db_path)], [second])
        self.assertTrue(scheduler.update_task_parameters(second, {'prompt': 'changed'}, db_path=self.db_path))
        self.assertTrue(scheduler_db.pause_task(second, db_path=self.db_path))
        self.assertTrue(scheduler_db.resume_task(second, db_path=self.db_path))
        task = scheduler_db.get_task(second, include_images=True, db_path=self.db_path)
        self.assertEqual(task['parameters']['control']['prompt'], 'changed')
        self.assertEqual(task['image_blobs'], [b'input-image'])
        for key in ('state', 'scheduler', 'units', 'input_script_args'):
            self.assertFalse(scheduler.update_task_parameters(second, {key: None}, db_path=self.db_path))
        self.assertEqual(scheduler_db.prune_tasks('2025-01-01T00:00:00+00:00', db_path=self.db_path), 1)
        self.assertIsNone(scheduler_db.get_task(first, db_path=self.db_path))

    def test_retry_limit(self):
        task_id = scheduler_db.create_task({'prompt': 'retry'}, max_attempts=1, db_path=self.db_path)

        self.assertEqual(scheduler_db.claim_next_task(db_path=self.db_path)['id'], task_id)
        self.assertTrue(scheduler_db.retry_or_fail(task_id, 'generation failed', db_path=self.db_path))
        task = scheduler_db.get_task(task_id, db_path=self.db_path)
        self.assertEqual(task['status'], 'failed')
        self.assertEqual(task['last_error'], 'generation failed')
        self.assertTrue(scheduler_db.reschedule_task(task_id, dt.datetime.now(dt.timezone.utc), db_path=self.db_path))
        task = scheduler_db.get_task(task_id, db_path=self.db_path)
        self.assertEqual(task['status'], 'scheduled')
        self.assertEqual(task['attempts'], 0)
        self.assertIsNone(task['last_error'])

    def test_manual_execution_remains_force_eligible_after_retry(self):
        task_id = scheduler_db.create_task({'prompt': 'manual retry'}, db_path=self.db_path)
        self.assertTrue(scheduler_db.queue_task(task_id, immediate=True, db_path=self.db_path))

        first_attempt = scheduler_db.claim_next_task(force_only=True, db_path=self.db_path)
        self.assertEqual(first_attempt['id'], task_id)
        self.assertTrue(scheduler_db.retry_or_fail(task_id, 'generation failed', retry_delay=0, db_path=self.db_path))

        second_attempt = scheduler_db.claim_next_task(force_only=True, db_path=self.db_path)
        self.assertEqual(second_attempt['id'], task_id)
        self.assertEqual(second_attempt['attempts'], 2)
        self.assertTrue(scheduler_db.update_results(task_id, [], db_path=self.db_path))
        self.assertIsNone(scheduler_db.claim_next_task(force_only=True, db_path=self.db_path))


if __name__ == '__main__':
    unittest.main()
