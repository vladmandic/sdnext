"""SQLite persistence for scheduled generation tasks."""

import datetime as dt
import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from urllib.parse import quote

from modules.logger import log
from modules import timer


_initialized_paths = set()
_initialization_failures = {}
_init_lock = threading.Lock()
_configured_db_path = None
change_event = threading.Event()
_task_modes = {'txt2img', 'img2img', 'control', 'video'}
debug = os.environ.get('SD_SCHEDULER_DEBUG', None) is not None


def _timestamp(value=None):
    if value is None:
        value = dt.datetime.now(dt.timezone.utc)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).isoformat(timespec='seconds')
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc).isoformat(timespec='seconds')
        except ValueError:
            pass
    return str(value)


def _db_path(db_path=None):
    if db_path:
        return os.fspath(db_path)
    configured = os.environ.get('SD_SCHEDULER_DB')
    if configured:
        return configured
    from modules import paths
    if _configured_db_path:
        return _configured_db_path
    loaded_shared = sys.modules.get('modules.shared')
    opts = getattr(loaded_shared, 'opts', None)
    scheduler_dir = getattr(opts, 'scheduler_dir', None) or paths.config.get('scheduler_dir')
    if not scheduler_dir:
        scheduler_dir = os.path.join(paths.models_path, 'scheduler')
    if not os.path.isabs(scheduler_dir):
        scheduler_dir = os.path.join(paths.data_path, scheduler_dir)
    return os.path.join(scheduler_dir, 'scheduler.db')


def configure(directory=None):
    global _configured_db_path # pylint: disable=global-statement
    if not directory:
        _configured_db_path = None
        return
    directory = os.fspath(directory)
    if not os.path.isabs(directory):
        from modules import paths
        directory = os.path.join(paths.data_path, directory)
    _configured_db_path = os.path.join(directory, 'scheduler.db')


def connect(db_path=None, timeout=5):
    path = _db_path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.name == 'nt':
        connection = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    else:
        uri = f'file:{quote(os.path.abspath(path), safe="/")}?vfs=unix-dotfile'
        connection = sqlite3.connect(uri, uri=True, timeout=timeout, isolation_level=None)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection(db_path=None, timeout=5):
    if debug:
        fn = f'{sys._getframe(3).f_code.co_name}:{sys._getframe(2).f_code.co_name}' # pylint: disable=protected-access
        log.trace(f'DB: {fn}')
    connection = connect(db_path, timeout=timeout)
    changes = connection.total_changes
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        changed = connection.total_changes > changes
        connection.close()
        if changed:
            change_event.set()


def _get_sqlite_stat1(connection):
    connection.execute('PRAGMA optimize')
    cursor = connection.cursor()
    cursor.execute("ANALYZE;")
    stat = cursor.execute("SELECT stat FROM sqlite_stat1 WHERE tbl = 'tasks';").fetchone()
    if stat:
        return list(stat)
    return ['empty']


def _fail_on_restart(connection):
    connection.execute("UPDATE tasks SET status = 'failed' WHERE status NOT IN ('scheduled', 'paused', 'completed', 'failed')")

def _initialize_db(connection):
    t0 = time.time()
    connection.executescript('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            scheduled_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            status TEXT NOT NULL CHECK (status IN ('scheduled', 'running', 'paused', 'completed', 'failed')),
            mode TEXT NOT NULL DEFAULT 'control' CHECK (mode IN ('txt2img', 'img2img', 'control', 'video')),
            queue_position INTEGER NOT NULL DEFAULT 0,
            scheduled_by_user INTEGER NOT NULL DEFAULT 0,
            execute_now INTEGER NOT NULL DEFAULT 0,
            retry_at TEXT,
            parameters TEXT NOT NULL,
            results TEXT NOT NULL DEFAULT '[]',
            process_stats TEXT NOT NULL DEFAULT '{}',
            infotexts TEXT NOT NULL DEFAULT '[]',
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            last_error TEXT
        );
        CREATE INDEX IF NOT EXISTS tasks_due_idx ON tasks(status, scheduled_at, id);
        CREATE INDEX IF NOT EXISTS tasks_created_idx ON tasks(created_at);
        CREATE TABLE IF NOT EXISTS task_images (
            task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            data BLOB NOT NULL,
            PRIMARY KEY (task_id, position)
        );
    ''')
    index_list = connection.execute("PRAGMA index_list('tasks')").fetchall()
    exists = any(index[1] == 'tasks_queue_idx' for index in index_list)
    connection.execute('CREATE INDEX IF NOT EXISTS tasks_queue_idx ON tasks(status, queue_position, id)')
    connection.execute('PRAGMA user_version = 1')
    connection.execute('PRAGMA foreign_keys = ON')
    connection.execute('PRAGMA auto_vacuum = 1')
    journal = connection.execute('PRAGMA journal_mode = TRUNCATE').fetchone()[0]
    db = connection.execute('PRAGMA database_list').fetchone()
    user_ver = connection.execute('PRAGMA user_version').fetchone()[0]
    schema_ver = connection.execute('PRAGMA schema_version').fetchone()[0]
    pg_count = connection.execute('PRAGMA page_count').fetchone()[0]
    pg_size = connection.execute('PRAGMA page_size').fetchone()[0]
    stat = _get_sqlite_stat1(connection)
    _fail_on_restart(connection)
    t1 = time.perf_counter()
    log.info(f'Scheduler: db={db[1]} path="{db[2]}" schema={schema_ver} user={user_ver} size={pg_count*pg_size} journal={journal} stat={stat} mode={"open" if exists else "create"} time={t1 - t0:.3f}')


def initialize(db_path=None):
    path = os.path.abspath(_db_path(db_path))
    t0 = time.perf_counter()
    with _init_lock:
        if path in _initialized_paths:
            return
        if path in _initialization_failures:
            failure = _initialization_failures[path]
            raise sqlite3.OperationalError(f'Scheduler database initialization previously failed: {failure}') from failure
        try:
            for attempt in range(10):
                try:
                    with _connection(path, timeout=1) as connection:
                        _initialize_db(connection)
                    break
                except sqlite3.OperationalError as exc:
                    log.warning(f'Scheduler: attempt={attempt} {str(exc)}')
                    if 'locked' not in str(exc).lower() or attempt == 9:
                        raise
                    time.sleep(0.1)
            _initialized_paths.add(path)
        except Exception as exc:
            _initialization_failures[path] = exc
            raise
    t1 = time.perf_counter()
    timer.startup.add('scheduler', t1 - t0)


def diagnostics(db_path=None):
    initialize(db_path)
    path = os.path.abspath(_db_path(db_path))
    required_tables = {'tasks', 'task_images'}
    required_columns = {
        'tasks': {'id', 'status', 'mode', 'parameters', 'results', 'process_stats', 'infotexts'},
        'task_images': {'task_id', 'position', 'data'},
    }
    with _connection(path) as connection:
        integrity = [row[0] for row in connection.execute('PRAGMA integrity_check')]
        foreign_key_errors = [dict(row) for row in connection.execute('PRAGMA foreign_key_check')]
        tables = {row['name'] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing_tables = sorted(required_tables - tables)
        missing_columns = {}
        for table, expected in required_columns.items():
            if table not in tables:
                continue
            columns = {row['name'] for row in connection.execute(f'PRAGMA table_info({table})')}
            missing = sorted(expected - columns)
            if missing:
                missing_columns[table] = missing
        task_count = connection.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
        statuses = {row['status']: row['count'] for row in connection.execute('SELECT status, COUNT(*) AS count FROM tasks GROUP BY status')}
        image_stats = connection.execute('SELECT COUNT(*) AS count, COALESCE(SUM(length(data)), 0) AS bytes FROM task_images').fetchone()
        json_errors = []
        for row in connection.execute('SELECT id, parameters, results, process_stats, infotexts FROM tasks'):
            for field in ('parameters', 'results', 'process_stats', 'infotexts'):
                try:
                    json.loads(row[field])
                except (TypeError, json.JSONDecodeError) as exc:
                    json_errors.append({'task_id': row['id'], 'field': field, 'error': str(exc)})
        stats = {
            'path': path,
            'integrity': integrity,
            'foreign_key_errors': foreign_key_errors,
            'missing_tables': missing_tables,
            'missing_columns': missing_columns,
            'json_errors': json_errors,
            'tasks': task_count,
            'statuses': statuses,
            'task_images': image_stats['count'],
            'image_bytes': image_stats['bytes'],
            'file_bytes': os.path.getsize(path),
            'page_size': connection.execute('PRAGMA page_size').fetchone()[0],
            'page_count': connection.execute('PRAGMA page_count').fetchone()[0],
            'freelist_count': connection.execute('PRAGMA freelist_count').fetchone()[0],
            'journal_mode': connection.execute('PRAGMA journal_mode').fetchone()[0],
            'user_version': connection.execute('PRAGMA user_version').fetchone()[0],
        }
        compile_options = connection.execute('PRAGMA compile_options').fetchall()
        opts = [opt[0] for opt in compile_options]
    stats['ok'] = (
        stats['integrity'] == ['ok']
        and not stats['foreign_key_errors']
        and not stats['missing_tables']
        and not stats['missing_columns']
        and not stats['json_errors']
    )
    return stats, opts


def _task(row):
    if row is None:
        return None
    task = dict(row)
    task.pop('execute_now', None)
    task['parameters'] = json.loads(task['parameters'])
    task['results'] = json.loads(task['results'])
    task['process_stats'] = json.loads(task['process_stats'])
    task['infotexts'] = json.loads(task['infotexts'])
    return task


def create_task(parameters, image_blobs=None, scheduled_at=None, max_attempts=3, db_path=None, mode='control'):
    initialize(db_path)
    if mode not in _task_modes:
        raise ValueError(f'Invalid scheduler task mode: {mode}')
    now = _timestamp()
    due = _timestamp(scheduled_at) if scheduled_at is not None else now
    images = image_blobs or []
    with _connection(db_path) as connection:
        connection.execute('BEGIN IMMEDIATE')
        position = connection.execute('SELECT COALESCE(MAX(queue_position), -1) + 1 FROM tasks').fetchone()[0]
        cursor = connection.execute(
            'INSERT INTO tasks(created_at, scheduled_at, status, mode, queue_position, scheduled_by_user, parameters, max_attempts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (now, due, 'scheduled', mode, position, int(scheduled_at is not None), json.dumps(parameters, separators=(',', ':')), max_attempts),
        )
        task_id = cursor.lastrowid
        connection.executemany(
            'INSERT INTO task_images(task_id, position, data) VALUES (?, ?, ?)',
            [(task_id, index, sqlite3.Binary(blob)) for index, blob in enumerate(images)],
        )
        return task_id


def get_task(task_id, include_images=False, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        task = _task(connection.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone())
        if task is not None and include_images:
            task['image_blobs'] = [bytes(row['data']) for row in connection.execute(
                'SELECT data FROM task_images WHERE task_id = ? ORDER BY position', (task_id,)
            )]
        return task


def get_task_image(task_id, image_index, db_path=None):
    initialize(db_path)
    if image_index < 0:
        return None
    with _connection(db_path) as connection:
        row = connection.execute(
            'SELECT data FROM task_images WHERE task_id = ? AND position = ?',
            (task_id, image_index),
        ).fetchone()
        return bytes(row['data']) if row is not None else None


def list_tasks(status=None, limit=None, offset=0, db_path=None):
    initialize(db_path)
    query = 'SELECT * FROM tasks'
    params = []
    if status:
        query += ' WHERE status = ?'
        params.append(status)
    query += ' ORDER BY queue_position, id'
    if limit is None:
        query += ' LIMIT -1 OFFSET ?'
        params.append(max(0, int(offset)))
    else:
        query += ' LIMIT ? OFFSET ?'
        params.extend((max(0, int(limit)), max(0, int(offset))))
    with _connection(db_path) as connection:
        return [_task(row) for row in connection.execute(query, params)]


def search_tasks(start=None, end=None, status=None, parameters=None, limit=None, offset=0, db_path=None):
    initialize(db_path)
    clauses = []
    values = []
    if start is not None:
        clauses.append('created_at >= ?')
        values.append(_timestamp(start))
    if end is not None:
        clauses.append('created_at <= ?')
        values.append(_timestamp(end))
    if status is not None:
        clauses.append('status = ?')
        values.append(status)
    query = 'SELECT * FROM tasks'
    if clauses:
        query += ' WHERE ' + ' AND '.join(clauses)
    query += ' ORDER BY queue_position, id'
    with _connection(db_path) as connection:
        tasks = [_task(row) for row in connection.execute(query, values)]
    if parameters:
        tasks = [
            task for task in tasks
            if all(task['parameters'].get('control', task['parameters']).get(key) == value for key, value in parameters.items())
        ]
    if limit is not None:
        tasks = tasks[max(0, int(offset)):max(0, int(offset)) + max(0, int(limit))]
    elif offset:
        tasks = tasks[max(0, int(offset)):]
    return tasks


def update_parameters(task_id, parameters, image_blobs=None, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        connection.execute('BEGIN IMMEDIATE')
        cursor = connection.execute(
            "UPDATE tasks SET parameters = ? WHERE id = ? AND status != 'running'",
            (json.dumps(parameters, separators=(',', ':')), task_id),
        )
        if cursor.rowcount == 0:
            return False
        if image_blobs is not None:
            connection.execute('DELETE FROM task_images WHERE task_id = ?', (task_id,))
            connection.executemany(
                'INSERT INTO task_images(task_id, position, data) VALUES (?, ?, ?)',
                [(task_id, index, sqlite3.Binary(blob)) for index, blob in enumerate(image_blobs)],
            )
        return True


def update_results(task_id, results, db_path=None, process_stats=None, infotexts=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE tasks SET results = ?, process_stats = ?, infotexts = ?, status = 'completed', completed_at = ?, last_error = NULL, execute_now = 0 WHERE id = ? AND status = 'running'",
            (
                json.dumps(results, separators=(',', ':')),
                json.dumps(process_stats or {}, separators=(',', ':')),
                json.dumps(infotexts or [], separators=(',', ':')),
                _timestamp(),
                task_id,
            ),
        )
        return cursor.rowcount > 0


def claim_next_task(db_path=None, force_only=False):
    initialize(db_path)
    now = _timestamp()
    with _connection(db_path) as connection:
        query = "SELECT id, status, scheduled_at, scheduled_by_user, retry_at, execute_now FROM tasks WHERE status = 'scheduled'"
        if force_only:
            query += ' AND execute_now = 1'
        query += ' ORDER BY queue_position, id LIMIT 1'

        def is_ready(row):
            if row is None:
                return False
            if row['execute_now']:
                return True
            if row['scheduled_by_user'] and row['scheduled_at'] > now:
                return False
            if row['retry_at'] is not None and row['retry_at'] > now:
                return False
            return True

        row = connection.execute(query).fetchone()
        if not is_ready(row):
            return None

        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute(query).fetchone()
        if not is_ready(row):
            return None
        task_id = row['id']
        cursor = connection.execute(
            "UPDATE tasks SET status = 'running', started_at = ?, attempts = attempts + 1, retry_at = NULL WHERE id = ? AND status = 'scheduled'",
            (now, task_id),
        )
        if cursor.rowcount == 0:
            return None
        return _task(connection.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone())


def retry_or_fail(task_id, error, retry_delay=60, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute('SELECT attempts, max_attempts FROM tasks WHERE id = ? AND status = ?', (task_id, 'running')).fetchone()
        if row is None:
            return False
        if row['attempts'] < row['max_attempts']:
            due = _timestamp(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=retry_delay))
            connection.execute(
                "UPDATE tasks SET status = 'scheduled', retry_at = ?, last_error = ? WHERE id = ?",
                (due, str(error), task_id),
            )
        else:
            connection.execute(
                "UPDATE tasks SET status = 'failed', completed_at = ?, retry_at = NULL, last_error = ?, execute_now = 0 WHERE id = ?",
                (_timestamp(), str(error), task_id),
            )
        return True


def pause_task(task_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute("UPDATE tasks SET status = 'paused' WHERE id = ? AND status = 'scheduled'", (task_id,))
        return cursor.rowcount > 0


def resume_task(task_id, scheduled_at=None, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        if scheduled_at is None:
            cursor = connection.execute(
                "UPDATE tasks SET status = 'scheduled' WHERE id = ? AND status = 'paused'",
                (task_id,),
            )
        else:
            cursor = connection.execute(
                "UPDATE tasks SET status = 'scheduled', scheduled_at = ?, scheduled_by_user = 1, retry_at = NULL WHERE id = ? AND status = 'paused'",
                (_timestamp(scheduled_at), task_id),
            )
        return cursor.rowcount > 0


def reschedule_task(task_id, scheduled_at, db_path=None):
    initialize(db_path)
    due = _timestamp(scheduled_at) if scheduled_at is not None else _timestamp()
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE tasks SET status = 'scheduled', scheduled_at = ?, scheduled_by_user = ?, retry_at = NULL, attempts = 0, started_at = NULL, completed_at = NULL, results = '[]', process_stats = '{}', infotexts = '[]', last_error = NULL WHERE id = ? AND status IN ('scheduled', 'paused', 'failed')",
            (due, int(scheduled_at is not None), task_id),
        )
        return cursor.rowcount > 0


def queue_task(task_id, db_path=None, immediate=False):
    initialize(db_path)
    with _connection(db_path) as connection:
        connection.execute('BEGIN IMMEDIATE')
        if immediate:
            cursor = connection.execute(
                "UPDATE tasks SET status = 'scheduled', scheduled_at = ?, scheduled_by_user = 0, retry_at = NULL, attempts = 0, started_at = NULL, completed_at = NULL, results = '[]', process_stats = '{}', infotexts = '[]', last_error = NULL, execute_now = 1 WHERE id = ? AND status IN ('scheduled', 'paused', 'completed', 'failed')",
                (_timestamp(), task_id),
            )
            if cursor.rowcount:
                rows = connection.execute(
                    'SELECT id FROM tasks WHERE id != ? ORDER BY queue_position, id', (task_id,)
                ).fetchall()
                connection.executemany(
                    'UPDATE tasks SET queue_position = ? WHERE id = ?',
                    [(position + 1, row['id']) for position, row in enumerate(rows)],
                )
                connection.execute('UPDATE tasks SET queue_position = 0 WHERE id = ?', (task_id,))
        else:
            cursor = connection.execute(
                "UPDATE tasks SET status = 'scheduled', scheduled_at = ?, scheduled_by_user = 0, retry_at = NULL, queue_position = (SELECT COALESCE(MAX(queue_position), -1) + 1 FROM tasks WHERE id != ?), attempts = 0, started_at = NULL, completed_at = NULL, results = '[]', process_stats = '{}', infotexts = '[]', last_error = NULL, execute_now = 0 WHERE id = ? AND status IN ('scheduled', 'paused', 'completed', 'failed')",
                (_timestamp(), task_id, task_id),
            )
        return cursor.rowcount > 0


def move_task(task_id, target_id, position='before', db_path=None):
    initialize(db_path)
    if position not in ('before', 'after') or task_id == target_id:
        return False
    with _connection(db_path) as connection:
        connection.execute('BEGIN IMMEDIATE')
        rows = connection.execute('SELECT id FROM tasks ORDER BY queue_position, id').fetchall()
        ordered = [row['id'] for row in rows]
        if task_id not in ordered or target_id not in ordered:
            return False
        ordered.remove(task_id)
        target_index = ordered.index(target_id) + (position == 'after')
        ordered.insert(target_index, task_id)
        connection.executemany(
            'UPDATE tasks SET queue_position = ? WHERE id = ?',
            list(enumerate(ordered)),
        )
        return True


def delete_task(task_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM tasks WHERE id = ? AND status != 'running'", (task_id,))
        return cursor.rowcount > 0


def prune_tasks(before, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM tasks WHERE created_at < ? AND status != 'running'", (_timestamp(before),))
        connection.execute('VACUUM')
        _get_sqlite_stat1(connection)
        return cursor.rowcount
