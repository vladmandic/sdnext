import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone


class JobStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._write_lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                type         TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                priority     INTEGER NOT NULL DEFAULT 0,
                params       TEXT NOT NULL,
                result       TEXT,
                error        TEXT,
                progress     REAL NOT NULL DEFAULT 0,
                step         INTEGER NOT NULL DEFAULT 0,
                steps        INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT NOT NULL,
                started_at   TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
        """)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get('result') and isinstance(d['result'], str):
            try:
                d['result'] = json.loads(d['result'])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def create(self, job_type: str, params: dict, priority: int = 0) -> dict:
        job_id = uuid.uuid4().hex[:16]
        now = self._now()
        params_json = json.dumps(params, default=str)
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO jobs (id, type, status, priority, params, created_at) VALUES (?, ?, 'pending', ?, ?, ?)",
                (job_id, job_type, priority, params_json, now),
            )
            self._conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, status: str | None = None, job_type: str | None = None, limit: int = 20, offset: int = 0) -> tuple[list[dict], int]:
        where_parts: list[str] = []
        binds: list = []
        if status:
            where_parts.append("status = ?")
            binds.append(status)
        if job_type:
            where_parts.append("type = ?")
            binds.append(job_type)
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        count_row = self._conn.execute(f"SELECT COUNT(*) FROM jobs {where_clause}", binds).fetchone()
        total = count_row[0] if count_row else 0
        rows = self._conn.execute(f"SELECT * FROM jobs {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?", [*binds, limit, offset]).fetchall()
        return [self._row_to_dict(r) for r in rows], total

    def update_status(self, job_id: str, status: str, **kwargs) -> None:
        sets = ["status = ?"]
        binds: list = [status]
        for key in ('started_at', 'completed_at', 'error'):
            if key in kwargs:
                sets.append(f"{key} = ?")
                binds.append(kwargs[key])
        if 'result' in kwargs:
            sets.append("result = ?")
            val = kwargs['result']
            binds.append(json.dumps(val, default=str) if not isinstance(val, str) else val)
        binds.append(job_id)
        with self._write_lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", binds)
            self._conn.commit()

    def update_progress(self, job_id: str, progress: float, step: int, steps: int) -> None:
        with self._write_lock:
            self._conn.execute("UPDATE jobs SET progress = ?, step = ?, steps = ? WHERE id = ?", (progress, step, steps, job_id))
            self._conn.commit()

    def cancel(self, job_id: str) -> bool:
        with self._write_lock:
            cur = self._conn.execute("UPDATE jobs SET status = 'cancelled', completed_at = ? WHERE id = ? AND status IN ('pending', 'running')", (self._now(), job_id))
            self._conn.commit()
            return cur.rowcount > 0

    def cleanup(self, max_age_hours: int = 168) -> int:
        cutoff = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'cancelled') AND completed_at IS NOT NULL AND completed_at < datetime(?, '-' || ? || ' hours')",
                (cutoff, max_age_hours),
            )
            self._conn.commit()
            return cur.rowcount

    def next_pending(self) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY priority DESC, created_at ASC LIMIT 1").fetchone()
        return self._row_to_dict(row) if row else None

    def close(self):
        self._conn.close()
