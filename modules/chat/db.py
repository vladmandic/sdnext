"""SQLite persistence for optional chat sessions and operations."""

import datetime as dt
import json
import os
import sqlite3
import threading
import time
import uuid
from urllib.parse import quote

from modules.logger import log


SCHEMA_VERSION = 6
CONTEXTS = {'image', 'video', 'process', 'caption'}
MESSAGE_ROLES = {'system', 'user', 'assistant'}
OPERATION_STATUSES = {'running', 'completed', 'failed', 'cancelled'}
_initialized_paths = set()
_initialization_failures = {}
_init_lock = threading.Lock()


def _timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def _db_path(db_path=None):
    if db_path is not None:
        return os.fspath(db_path)
    from modules import paths, shared
    directory = shared.opts.chat_dir or os.path.join(paths.models_path, 'chat')
    if not os.path.isabs(directory):
        directory = os.path.join(paths.data_path, directory)
    return os.path.join(directory, 'chat.db')


def connect(db_path=None, timeout=5):
    path = os.path.abspath(_db_path(db_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.name == 'nt':
        connection = sqlite3.connect(path, timeout=timeout)
    else:
        uri = f'file:{quote(path, safe="/")}?vfs=unix-dotfile'
        connection = sqlite3.connect(uri, uri=True, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    connection.execute('PRAGMA busy_timeout = 5000')
    return connection


class _Connection:
    def __init__(self, db_path=None, timeout=5):
        self.db_path = db_path
        self.timeout = timeout
        self.connection = None

    def __enter__(self):
        self.connection = connect(self.db_path, timeout=self.timeout)
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection is None:
            return False
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()
        return False


def _connection(db_path=None, timeout=5):
    return _Connection(db_path, timeout)


def _reset_schema(connection):
    connection.execute('PRAGMA foreign_keys = OFF')
    try:
        connection.execute('BEGIN')
        objects = connection.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for item in objects:
            name = item['name'].replace('"', '""')
            connection.execute(f'DROP {item["type"].upper()} IF EXISTS "{name}"')
        connection.execute('PRAGMA user_version = 0')
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute('PRAGMA foreign_keys = ON')


def _dump(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def _decode_row(row, json_fields):
    if row is None:
        return None
    item = dict(row)
    for field, default in json_fields.items():
        value = item[field]
        item[field] = json.loads(value) if value is not None else default
    return item


def _session(row):
    session = _decode_row(row, {
        'drafts': {}, 'last_generation_snapshot': None, 'selected_image_ids': [],
    })
    if session is None:
        return None
    selected_ids = session.get('selected_image_ids')
    if not isinstance(selected_ids, list):
        selected_ids = []
    selected_ids = [artifact_id for artifact_id in selected_ids if isinstance(artifact_id, str) and artifact_id]
    if not selected_ids and session.get('selected_image_id'):
        selected_ids = [session['selected_image_id']]
    session['selected_image_ids'] = selected_ids
    session['selected_image_id'] = selected_ids[0] if selected_ids else None
    return session


def _message(row):
    return _decode_row(row, {'metadata': {}})


def _operation(row):
    return _decode_row(row, {'snapshot': {}, 'result': None})


def initialize(db_path=None):
    path = os.path.abspath(_db_path(db_path))
    with _init_lock:
        if path in _initialized_paths:
            return
        if path in _initialization_failures:
            failure = _initialization_failures[path]
            raise sqlite3.OperationalError(f'Chat database initialization previously failed: {failure}') from failure
        try:
            for attempt in range(10):
                try:
                    started = time.perf_counter()
                    with _connection(path, timeout=1) as connection:
                        has_schema = connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' LIMIT 1",
                        ).fetchone() is not None
                        version = connection.execute('PRAGMA user_version').fetchone()[0]
                        reset = version != SCHEMA_VERSION and (has_schema or version != 0)
                        if reset:
                            _reset_schema(connection)
                        connection.executescript('''
                            CREATE TABLE IF NOT EXISTS sessions (
                                id TEXT PRIMARY KEY,
                                title TEXT NOT NULL,
                                context TEXT NOT NULL CHECK (context IN ('image', 'video', 'process', 'caption')),
                                drafts TEXT NOT NULL DEFAULT '{}',
                                last_generation_prompt TEXT,
                                last_generation_snapshot TEXT,
                                active_settings_revision_id INTEGER,
                                selected_image_id TEXT,
                                selected_image_ids TEXT NOT NULL DEFAULT '[]',
                                created_at TEXT NOT NULL,
                                updated_at TEXT NOT NULL
                            );
                            CREATE INDEX IF NOT EXISTS chat_sessions_updated_idx ON sessions(updated_at DESC, id);
                            CREATE TABLE IF NOT EXISTS messages (
                                id TEXT PRIMARY KEY,
                                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                                content TEXT NOT NULL,
                                metadata TEXT NOT NULL DEFAULT '{}',
                                created_at TEXT NOT NULL
                            );
                            CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON messages(session_id, created_at, id);
                            CREATE TABLE IF NOT EXISTS operations (
                                id TEXT PRIMARY KEY,
                                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                                command TEXT NOT NULL,
                                context TEXT NOT NULL CHECK (context IN ('image', 'video', 'process', 'caption')),
                                status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
                                snapshot TEXT NOT NULL DEFAULT '{}',
                                result TEXT,
                                error TEXT,
                                created_at TEXT NOT NULL,
                                started_at TEXT NOT NULL,
                                completed_at TEXT
                            );
                            CREATE INDEX IF NOT EXISTS chat_operations_session_idx ON operations(session_id, created_at DESC, id);
                            CREATE TABLE IF NOT EXISTS settings_revisions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                context TEXT NOT NULL CHECK (context = 'image'),
                                snapshot TEXT NOT NULL,
                                created_at TEXT NOT NULL
                            );
                            CREATE INDEX IF NOT EXISTS chat_settings_revisions_session_idx ON settings_revisions(session_id, id DESC);
                            CREATE TABLE IF NOT EXISTS artifacts (
                                id TEXT PRIMARY KEY,
                                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                                media_type TEXT NOT NULL,
                                display_name TEXT NOT NULL,
                                role TEXT NOT NULL CHECK (role IN ('input', 'result')),
                                storage_ref TEXT NOT NULL,
                                source_artifact_id TEXT,
                                source_deleted INTEGER NOT NULL DEFAULT 0,
                                created_at TEXT NOT NULL
                            );
                            CREATE INDEX IF NOT EXISTS chat_artifacts_session_idx ON artifacts(session_id, created_at);
                        ''')
                        connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
                        restarted = connection.execute(
                            "UPDATE operations SET status = 'failed', error = ?, completed_at = ? WHERE status = 'running'",
                            ('Server restarted before operation completed', _timestamp()),
                        ).rowcount
                        connection.execute('PRAGMA optimize')
                        connection.execute('ANALYZE')
                        database = connection.execute('PRAGMA database_list').fetchone()
                        schema_version = connection.execute('PRAGMA schema_version').fetchone()[0]
                        version = connection.execute('PRAGMA user_version').fetchone()[0]
                        page_count = connection.execute('PRAGMA page_count').fetchone()[0]
                        page_size = connection.execute('PRAGMA page_size').fetchone()[0]
                        journal = connection.execute('PRAGMA journal_mode').fetchone()[0]
                        stats_table = connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'",
                        ).fetchone()
                        stats = connection.execute("SELECT stat FROM sqlite_stat1 WHERE tbl = 'sessions'").fetchone() if stats_table else None
                        stat = [stats[0]] if stats else ['empty']
                        integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
                        elapsed = time.perf_counter() - started
                        mode = 'reset' if reset else 'open' if has_schema else 'create'
                        log.info(f'Chat: db={database[1]} path="{database[2]}" schema={schema_version} user={version} size={page_count*page_size} journal={journal} stat={stat} mode={mode} time={elapsed:.3f} integrity={integrity} recovered={restarted}')
                    break
                except sqlite3.OperationalError as exc:
                    log.warning(f'Chat: attempt={attempt} path="{path}" {exc}')
                    if 'locked' not in str(exc).lower() or attempt == 9:
                        raise
                    time.sleep(0.1)
            _initialized_paths.add(path)
        except Exception as exc:
            _initialization_failures[path] = exc
            raise


def compact_and_stats(db_path=None):
    initialize(db_path)
    path = os.path.abspath(_db_path(db_path))
    with _connection(path) as connection:
        connection.execute('VACUUM')
        page_count = connection.execute('PRAGMA page_count').fetchone()[0]
        page_size = connection.execute('PRAGMA page_size').fetchone()[0]
        free_pages = connection.execute('PRAGMA freelist_count').fetchone()[0]
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in ('sessions', 'messages', 'operations', 'settings_revisions', 'artifacts')
        }
    return {
        'path': path,
        'size_bytes': os.path.getsize(path),
        'page_count': page_count,
        'page_size': page_size,
        'free_pages': free_pages,
        **counts,
    }


def create_session(title='New chat', context='image', session_id=None, db_path=None):
    initialize(db_path)
    if context not in CONTEXTS:
        raise ValueError(f'Invalid chat context: {context}')
    now = _timestamp()
    session_id = session_id or uuid.uuid4().hex
    title = str(title or 'New chat').strip() or 'New chat'
    with _connection(db_path) as connection:
        connection.execute(
            'INSERT INTO sessions(id, title, context, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
            (session_id, title, context, now, now),
        )
    return get_session(session_id, db_path=db_path)


def get_session(session_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        return _session(connection.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone())


def list_sessions(limit=50, offset=0, db_path=None):
    initialize(db_path)
    limit = min(100, max(1, int(limit)))
    offset = max(0, int(offset))
    with _connection(db_path) as connection:
        rows = connection.execute(
            'SELECT * FROM sessions ORDER BY updated_at DESC, id LIMIT ? OFFSET ?',
            (limit, offset),
        ).fetchall()
    return [_session(row) for row in rows]


def update_session(session_id, db_path=None, **fields):
    initialize(db_path)
    allowed = {'title', 'context', 'drafts', 'last_generation_prompt', 'last_generation_snapshot', 'selected_image_id', 'selected_image_ids'}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f'Unsupported chat session fields: {", ".join(sorted(unknown))}')
    if 'context' in fields and fields['context'] not in CONTEXTS:
        raise ValueError(f'Invalid chat context: {fields["context"]}')
    if 'title' in fields:
        fields['title'] = str(fields['title'] or '').strip()
        if not fields['title']:
            raise ValueError('Chat session title cannot be empty')
    if 'selected_image_id' in fields and 'selected_image_ids' not in fields:
        fields['selected_image_ids'] = [fields['selected_image_id']] if fields['selected_image_id'] else []
    if 'selected_image_ids' in fields:
        selected_ids = fields['selected_image_ids'] or []
        if not isinstance(selected_ids, list) or any(not isinstance(item, str) or not item for item in selected_ids):
            raise ValueError('selected_image_ids must be a list of image artifact IDs')
        selected_ids = list(dict.fromkeys(selected_ids))
        fields['selected_image_ids'] = _dump(selected_ids)
        fields['selected_image_id'] = selected_ids[0] if selected_ids else None
    for field in ('drafts', 'last_generation_snapshot'):
        if field in fields and fields[field] is not None:
            if not isinstance(fields[field], dict):
                raise ValueError(f'{field} must be an object')
            fields[field] = _dump(fields[field])
    if not fields:
        return get_session(session_id, db_path=db_path)
    fields['updated_at'] = _timestamp()
    assignments = ', '.join(f'{key} = ?' for key in fields)
    with _connection(db_path) as connection:
        connection.execute(
            f'UPDATE sessions SET {assignments} WHERE id = ?',
            (*fields.values(), session_id),
        )
        row = connection.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    return _session(row)


def create_settings_revision(session_id, snapshot, drafts, db_path=None):
    initialize(db_path)
    if not isinstance(snapshot, dict):
        raise ValueError('Chat settings revision snapshot must be an object')
    if snapshot.get('context') != 'image':
        raise ValueError('Chat settings snapshots are supported only for image context')
    if not isinstance(drafts, dict):
        raise ValueError('Chat drafts must be an object')
    serialized_snapshot = _dump(snapshot)
    serialized_drafts = _dump(drafts)
    now = _timestamp()
    with _connection(db_path) as connection:
        cursor = connection.execute(
            'INSERT INTO settings_revisions(session_id, context, snapshot, created_at) VALUES (?, ?, ?, ?)',
            (session_id, 'image', serialized_snapshot, now),
        )
        revision_id = cursor.lastrowid
        updated = connection.execute(
            'UPDATE sessions SET drafts = ?, active_settings_revision_id = ?, updated_at = ? WHERE id = ? AND context = ?',
            (serialized_drafts, revision_id, now, session_id, 'image'),
        )
        if not updated.rowcount:
            raise ValueError('Chat session not found')
        revision_row = connection.execute(
            'SELECT * FROM settings_revisions WHERE id = ? AND session_id = ?',
            (revision_id, session_id),
        ).fetchone()
        session_row = connection.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    return {'revision': _settings_revision(revision_row), 'session': _session(session_row)}


def _settings_revision(row):
    return _decode_row(row, {'snapshot': {}})


def list_settings_revisions(session_id, limit=200, db_path=None):
    initialize(db_path)
    limit = min(1000, max(1, int(limit)))
    with _connection(db_path) as connection:
        rows = connection.execute(
            'SELECT * FROM settings_revisions WHERE session_id = ? ORDER BY id DESC LIMIT ?',
            (session_id, limit),
        ).fetchall()
    return [_settings_revision(row) for row in rows]


def get_settings_revision(session_id, revision_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            'SELECT * FROM settings_revisions WHERE session_id = ? AND id = ?',
            (session_id, revision_id),
        ).fetchone()
    return _settings_revision(row)


def activate_settings_revision(session_id, revision_id, drafts, db_path=None):
    initialize(db_path)
    if not isinstance(drafts, dict):
        raise ValueError('Chat drafts must be an object')
    serialized_drafts = _dump(drafts)
    now = _timestamp()
    with _connection(db_path) as connection:
        revision_row = connection.execute(
            'SELECT * FROM settings_revisions WHERE session_id = ? AND id = ? AND context = ?',
            (session_id, revision_id, 'image'),
        ).fetchone()
        if revision_row is None:
            return None
        updated = connection.execute(
            'UPDATE sessions SET drafts = ?, active_settings_revision_id = ?, updated_at = ? WHERE id = ? AND context = ?',
            (serialized_drafts, revision_id, now, session_id, 'image'),
        )
        if not updated.rowcount:
            return None
        session_row = connection.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    return {'revision': _settings_revision(revision_row), 'session': _session(session_row)}


def delete_session(session_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        return cursor.rowcount > 0


def create_artifact(session_id, media_type, display_name, role, storage_ref, message_id=None, source_artifact_id=None, db_path=None):
    initialize(db_path)
    if role not in {'input', 'result'}:
        raise ValueError(f'Invalid chat artifact role: {role}')
    artifact_id = uuid.uuid4().hex
    created_at = _timestamp()
    with _connection(db_path) as connection:
        connection.execute(
            'INSERT INTO artifacts(id, session_id, message_id, media_type, display_name, role, storage_ref, source_artifact_id, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (artifact_id, session_id, message_id, media_type, display_name, role, storage_ref, source_artifact_id, created_at),
        )
        row = connection.execute('SELECT * FROM artifacts WHERE id = ?', (artifact_id,)).fetchone()
        connection.execute('UPDATE sessions SET updated_at = ? WHERE id = ?', (created_at, session_id))
    return dict(row)


def get_artifact(session_id, artifact_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            'SELECT * FROM artifacts WHERE id = ? AND session_id = ?',
            (artifact_id, session_id),
        ).fetchone()
    return dict(row) if row is not None else None


def list_artifacts(session_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute(
            'SELECT * FROM artifacts WHERE session_id = ? ORDER BY created_at, rowid',
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_artifact(session_id, artifact_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            'SELECT * FROM artifacts WHERE id = ? AND session_id = ?',
            (artifact_id, session_id),
        ).fetchone()
        if row is None:
            return None
        artifact = dict(row)
        connection.execute(
            'UPDATE artifacts SET source_deleted = 1 WHERE session_id = ? AND source_artifact_id = ?',
            (session_id, artifact_id),
        )
        session = connection.execute(
            'SELECT selected_image_ids FROM sessions WHERE id = ?', (session_id,),
        ).fetchone()
        selected_ids = json.loads(session['selected_image_ids'] or '[]') if session is not None else []
        if artifact_id in selected_ids:
            selected_ids = [item for item in selected_ids if item != artifact_id]
            connection.execute(
                'UPDATE sessions SET selected_image_id = ?, selected_image_ids = ?, updated_at = ? WHERE id = ?',
                (selected_ids[0] if selected_ids else None, _dump(selected_ids), _timestamp(), session_id),
            )
        connection.execute('DELETE FROM artifacts WHERE id = ? AND session_id = ?', (artifact_id, session_id))
    return artifact


def set_selected_images(session_id, artifact_ids, db_path=None):
    initialize(db_path)
    if not isinstance(artifact_ids, list) or any(not isinstance(item, str) or not item for item in artifact_ids):
        raise ValueError('Selected images must be a list of artifact IDs')
    artifact_ids = list(dict.fromkeys(artifact_ids))
    with _connection(db_path) as connection:
        for artifact_id in artifact_ids:
            artifact = connection.execute(
                'SELECT 1 FROM artifacts WHERE id = ? AND session_id = ? AND media_type LIKE \'image/%\'',
                (artifact_id, session_id),
            ).fetchone()
            if artifact is None:
                raise ValueError('Image artifact not found in this session')
        connection.execute(
            'UPDATE sessions SET selected_image_id = ?, selected_image_ids = ?, updated_at = ? WHERE id = ?',
            (artifact_ids[0] if artifact_ids else None, _dump(artifact_ids), _timestamp(), session_id),
        )
        row = connection.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    return _session(row)


def set_selected_image(session_id, artifact_id, db_path=None):
    return set_selected_images(session_id, [artifact_id] if artifact_id is not None else [], db_path=db_path)


def create_message(session_id, role, content, metadata=None, message_id=None, db_path=None):
    initialize(db_path)
    if role not in MESSAGE_ROLES:
        raise ValueError(f'Invalid chat message role: {role}')
    if not isinstance(content, str):
        raise ValueError('Chat message content must be text')
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError('Chat message metadata must be an object')
    message_id = message_id or uuid.uuid4().hex
    created_at = _timestamp()
    with _connection(db_path) as connection:
        connection.execute(
            'INSERT INTO messages(id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            (message_id, session_id, role, content, _dump(metadata), created_at),
        )
        row = connection.execute('SELECT * FROM messages WHERE id = ?', (message_id,)).fetchone()
        connection.execute('UPDATE sessions SET updated_at = ? WHERE id = ?', (created_at, session_id))
    return _message(row)


def list_messages(session_id, limit=500, offset=0, db_path=None):
    initialize(db_path)
    limit = min(1000, max(1, int(limit)))
    offset = max(0, int(offset))
    with _connection(db_path) as connection:
        rows = connection.execute(
            'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, rowid LIMIT ? OFFSET ?',
            (session_id, limit, offset),
        ).fetchall()
    return [_message(row) for row in rows]


def update_message(session_id, message_id, content, metadata, db_path=None):
    initialize(db_path)
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Chat message content must be non-empty text')
    if not isinstance(metadata, dict):
        raise ValueError('Chat message metadata must be an object')
    updated_at = _timestamp()
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE messages SET content = ?, metadata = ? WHERE id = ? AND session_id = ? AND role = 'assistant'",
            (content, _dump(metadata), message_id, session_id),
        )
        row = connection.execute(
            'SELECT * FROM messages WHERE id = ? AND session_id = ? AND role = \'assistant\'',
            (message_id, session_id),
        ).fetchone() if cursor.rowcount else None
        if row is not None:
            connection.execute('UPDATE sessions SET updated_at = ? WHERE id = ?', (updated_at, session_id))
    return _message(row)


def create_operation(session_id, command, context, snapshot, message_id=None, operation_id=None, db_path=None):
    initialize(db_path)
    if context not in CONTEXTS:
        raise ValueError(f'Invalid chat context: {context}')
    if not isinstance(snapshot, dict):
        raise ValueError('Chat operation snapshot must be an object')
    now = _timestamp()
    operation_id = operation_id or uuid.uuid4().hex
    with _connection(db_path) as connection:
        connection.execute(
            'INSERT INTO operations(id, session_id, message_id, command, context, status, snapshot, created_at, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (operation_id, session_id, message_id, str(command), context, 'running', _dump(snapshot), now, now),
        )
        row = connection.execute('SELECT * FROM operations WHERE id = ?', (operation_id,)).fetchone()
        connection.execute('UPDATE sessions SET updated_at = ? WHERE id = ?', (now, session_id))
    return _operation(row)


def get_operation(operation_id, db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        return _operation(connection.execute('SELECT * FROM operations WHERE id = ?', (operation_id,)).fetchone())


def list_operations(session_id, limit=50, offset=0, db_path=None):
    initialize(db_path)
    limit = min(200, max(1, int(limit)))
    offset = max(0, int(offset))
    with _connection(db_path) as connection:
        rows = connection.execute(
            'SELECT * FROM operations WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?',
            (session_id, limit, offset),
        ).fetchall()
    return [_operation(row) for row in rows]


def update_operation(operation_id, status, result=None, error=None, db_path=None):
    initialize(db_path)
    if status not in OPERATION_STATUSES:
        raise ValueError(f'Invalid chat operation status: {status}')
    completed_at = _timestamp() if status != 'running' else None
    serialized_result = _dump(result) if result is not None else None
    with _connection(db_path) as connection:
        connection.execute(
            'UPDATE operations SET status = ?, result = ?, error = ?, completed_at = ? WHERE id = ?',
            (status, serialized_result, error, completed_at, operation_id),
        )
        row = connection.execute('SELECT * FROM operations WHERE id = ?', (operation_id,)).fetchone()
    return _operation(row)


def diagnostics(db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        integrity = [row[0] for row in connection.execute('PRAGMA integrity_check')]
        foreign_key_errors = [dict(row) for row in connection.execute('PRAGMA foreign_key_check')]
        tables = {row['name'] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        required_tables = {'sessions', 'messages', 'operations', 'settings_revisions', 'artifacts'}
        required_columns = {
            'sessions': {'id', 'title', 'context', 'drafts', 'last_generation_prompt', 'last_generation_snapshot', 'active_settings_revision_id', 'selected_image_id', 'selected_image_ids', 'created_at', 'updated_at'},
            'messages': {'id', 'session_id', 'role', 'content', 'metadata', 'created_at'},
            'operations': {'id', 'session_id', 'message_id', 'command', 'context', 'status', 'snapshot', 'result', 'error', 'created_at', 'started_at', 'completed_at'},
            'settings_revisions': {'id', 'session_id', 'context', 'snapshot', 'created_at'},
            'artifacts': {'id', 'session_id', 'message_id', 'media_type', 'display_name', 'role', 'storage_ref', 'source_artifact_id', 'source_deleted', 'created_at'},
        }
        table_columns = {}
        missing_columns = {}
        for table, expected in required_columns.items():
            if table not in tables:
                continue
            table_columns[table] = {row['name'] for row in connection.execute(f'PRAGMA table_info({table})')}
            missing = sorted(expected - table_columns[table])
            if missing:
                missing_columns[table] = missing
        json_errors = []
        json_columns = {'sessions': ('drafts', 'last_generation_snapshot'), 'messages': ('metadata',), 'operations': ('snapshot', 'result'), 'settings_revisions': ('snapshot',)}
        for table, fields in json_columns.items():
            if table not in tables or not {'id', *fields} <= table_columns.get(table, set()):
                continue
            for row in connection.execute(f'SELECT id, {", ".join(fields)} FROM {table}'):
                for column in fields:
                    try:
                        if row[column] is not None:
                            json.loads(row[column])
                    except (TypeError, json.JSONDecodeError) as exc:
                        json_errors.append({'id': row['id'], 'table': table, 'column': column, 'error': str(exc)})
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        table_counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in required_tables if table in tables
        }
        compile_options = [row[0] for row in connection.execute('PRAGMA compile_options')]
        page_size = connection.execute('PRAGMA page_size').fetchone()[0]
        page_count = connection.execute('PRAGMA page_count').fetchone()[0]
        freelist_count = connection.execute('PRAGMA freelist_count').fetchone()[0]
        journal_mode = connection.execute('PRAGMA journal_mode').fetchone()[0]
        schema_version = connection.execute('PRAGMA schema_version').fetchone()[0]
    return {
        'path': os.path.abspath(_db_path(db_path)),
        'integrity': integrity,
        'foreign_key_errors': foreign_key_errors,
        'missing_tables': sorted(required_tables - tables),
        'missing_columns': missing_columns,
        'json_errors': json_errors,
        'tables': table_counts,
        'file_bytes': os.path.getsize(_db_path(db_path)),
        'page_size': page_size,
        'page_count': page_count,
        'freelist_count': freelist_count,
        'journal_mode': journal_mode,
        'schema_version': schema_version,
        'user_version': version,
        'compile_options': compile_options,
        'ok': integrity == ['ok'] and not foreign_key_errors and required_tables <= tables and not missing_columns and not json_errors and version == SCHEMA_VERSION,
    }
