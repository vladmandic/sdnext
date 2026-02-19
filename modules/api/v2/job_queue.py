import asyncio
import io
import json
import os
import threading
import time
from modules.api.v2.job_store import JobStore


class JobQueue:
    def __init__(self):
        self.store: JobStore | None = None
        self._worker_thread: threading.Thread | None = None
        self._job_event = threading.Event()
        self._cancel_ids: set[str] = set()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._sub_lock = threading.Lock()
        self._current_job_id: str | None = None
        self._initialized = False

    def init(self, data_path: str) -> None:
        if self._initialized:
            return
        db_path = os.path.join(data_path, 'jobs.db')
        self.store = JobStore(db_path)
        # Mark any jobs left running from a previous session as failed
        self._recover_stale_jobs()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="v2-job-worker")
        self._worker_thread.start()
        self._initialized = True

    def _recover_stale_jobs(self):
        if self.store is None:
            return
        jobs, _ = self.store.list(status='running', limit=100)
        for job in jobs:
            self.store.update_status(job['id'], 'failed', error='Server restarted', completed_at=JobStore._now())

    def submit(self, job_type: str, params: dict, priority: int = 0) -> dict:
        job = self.store.create(job_type=job_type, params=params, priority=priority)
        self._job_event.set()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None:
            return False
        if job['status'] == 'running':
            self._cancel_ids.add(job_id)
            # Interrupt the running generation
            try:
                from modules import shared
                shared.state.interrupt()
            except Exception:
                pass
            return True
        return self.store.cancel(job_id)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._sub_lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            self._subscribers[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            subs = self._subscribers.get(job_id)
            if subs and queue in subs:
                subs.remove(queue)
                if not subs:
                    del self._subscribers[job_id]

    def _push_progress(self, job_id: str, data: dict) -> None:
        with self._sub_lock:
            subs = self._subscribers.get(job_id, [])
            for q in subs:
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass

    def _push_binary(self, job_id: str, data: bytes) -> None:
        with self._sub_lock:
            subs = self._subscribers.get(job_id, [])
            for q in subs:
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass

    def _worker_loop(self) -> None:
        from installer import log
        log.debug('Job queue: worker started')
        while True:
            self._job_event.wait(timeout=1.0)
            self._job_event.clear()
            if self.store is None:
                continue
            job = self.store.next_pending()
            if job is None:
                continue
            self._execute_job(job)

    def _execute_job(self, job: dict) -> None:
        from installer import log
        job_id = job['id']
        job_type = job['type']
        self._current_job_id = job_id
        log.info(f'Job queue: executing id={job_id} type={job_type}')
        self.store.update_status(job_id, 'running', started_at=JobStore._now())
        self._push_progress(job_id, {'type': 'status', 'status': 'running'})

        # Start progress poller thread
        poller_stop = threading.Event()
        poller = threading.Thread(target=self._progress_poller, args=(job_id, poller_stop), daemon=True, name=f"v2-progress-{job_id[:8]}")
        poller.start()

        try:
            from modules.call_queue import queue_lock
            with queue_lock:
                if job_id in self._cancel_ids:
                    self._cancel_ids.discard(job_id)
                    self.store.update_status(job_id, 'cancelled', completed_at=JobStore._now())
                    self._push_progress(job_id, {'type': 'status', 'status': 'cancelled'})
                    return
                result = self._dispatch(job_type, job.get('params', {}), job_id)
            result_json = json.dumps(result, default=str)
            self.store.update_status(job_id, 'completed', completed_at=JobStore._now(), result=result_json)
            self._push_progress(job_id, {'type': 'completed', 'result': result})
            log.info(f'Job queue: completed id={job_id}')
        except Exception as e:
            from modules import errors
            errors.display(e, f'Job queue: {job_type}')
            error_msg = f'{type(e).__name__}: {e}'
            self.store.update_status(job_id, 'failed', completed_at=JobStore._now(), error=error_msg)
            self._push_progress(job_id, {'type': 'error', 'error': error_msg})
            # Handle cancellation via interrupt
            if job_id in self._cancel_ids:
                self._cancel_ids.discard(job_id)
                self.store.update_status(job_id, 'cancelled', completed_at=JobStore._now())
                self._push_progress(job_id, {'type': 'status', 'status': 'cancelled'})
        finally:
            poller_stop.set()
            poller.join(timeout=2.0)
            self._current_job_id = None
            # Signal in case more jobs are pending
            if self.store.next_pending():
                self._job_event.set()

    def _dispatch(self, job_type: str, params, job_id: str) -> dict:
        if isinstance(params, str):
            params = json.loads(params)
        from modules.api.v2.executors import EXECUTORS
        executor = EXECUTORS.get(job_type)
        if executor is None:
            raise ValueError(f"Unknown job type: {job_type}")
        return executor(params, job_id)

    def _progress_poller(self, job_id: str, stop_event: threading.Event) -> None:
        from modules import shared
        last_step = -1
        last_preview_id = -1
        while not stop_event.is_set():
            try:
                if self._current_job_id != job_id:
                    break
                state = shared.state
                if state.sampling_step != last_step:
                    last_step = state.sampling_step
                    status = state.status()
                    step = status.step if hasattr(status, 'step') else getattr(status, 'sampling_step', 0)
                    steps = status.steps if hasattr(status, 'steps') else getattr(status, 'sampling_steps', 0)
                    progress_val = status.progress if hasattr(status, 'progress') else 0
                    eta_val = status.eta if hasattr(status, 'eta') else None
                    progress_data = {
                        'type': 'progress',
                        'step': step,
                        'steps': steps,
                        'progress': progress_val,
                        'eta': eta_val,
                    }
                    if self.store is not None:
                        self.store.update_progress(job_id, progress_val, step, steps)
                    self._push_progress(job_id, progress_data)
                    # Send preview image as binary if available
                    if state.id_live_preview != last_preview_id and state.current_image is not None:
                        last_preview_id = state.id_live_preview
                        try:
                            buf = io.BytesIO()
                            state.current_image.save(buf, format="JPEG", quality=75)
                            self._push_binary(job_id, buf.getvalue())
                        except Exception:
                            pass
            except Exception:
                pass
            stop_event.wait(timeout=0.1)


job_queue = JobQueue()
