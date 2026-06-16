import base64
import os
import io
import time
import asyncio
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from starlette.websockets import WebSocket, WebSocketState
import modules.shared as shared
from modules.logger import log


current_task = None
pending_tasks = {}
finished_tasks = []
recorded_results = []
recorded_results_limit = 2
debug = os.environ.get('SD_PREVIEW_DEBUG', None) is not None
debug_log = log.trace if debug else lambda *args, **kwargs: None


class PreviewManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self._loop = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._loop = asyncio.get_running_loop()
        self.active.append(ws)
        debug_log(f'Preview WS connect: client={ws.client.host} total={len(self.active)}')

    def disconnect(self, ws: WebSocket):
        try:
            self.active.remove(ws)
        except ValueError:
            pass
        debug_log(f'Preview WS disconnect: client={ws.client.host} total={len(self.active)}')

    async def broadcast(self, data: dict):
        disconnected = []
        for ws in list(self.active):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(data)
                else:
                    disconnected.append(ws)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            try:
                self.active.remove(ws)
            except ValueError:
                pass

    def push(self, data: dict):
        if self._loop is None or not self.active:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)
        except Exception:
            pass


preview_manager = PreviewManager()


def push_progress(step: int, steps: int, progress: float, active: bool = True):
    try:
        data = {
            'type': 'progress',
            'step': step,
            'steps': steps,
            'progress': progress,
            'active': active,
            'job': shared.state.job,
            'paused': shared.state.paused,
        }
        preview_manager.push(data)
    except Exception as e:
        debug_log(f'Progress push error: {e}')


def push_live_preview(image, step: int, steps: int, progress: float):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format='jpeg', quality=60)
        b64 = base64.b64encode(buffered.getvalue()).decode('ascii')
        data = {
            'type': 'preview',
            'live_preview': f'data:image/jpeg;base64,{b64}',
            'id_live_preview': shared.state.id_live_preview,
            'step': step,
            'steps': steps,
            'progress': progress,
            'job': shared.state.job,
        }
        preview_manager.push(data)
    except Exception as e:
        debug_log(f'Preview push error: {e}')


def push_complete():
    try:
        data = {
            'type': 'complete',
            'active': False,
            'job': shared.state.job,
        }
        preview_manager.push(data)
    except Exception as e:
        debug_log(f'Complete push error: {e}')


def start_task(id_task):
    global current_task # pylint: disable=global-statement
    current_task = id_task
    pending_tasks.pop(id_task, None)


def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def finish_task(id_task):
    global current_task # pylint: disable=global-statement
    if current_task == id_task:
        current_task = None
    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)
    push_complete()


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()


class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")


class InternalProgressResponse(BaseModel):
    job: str = Field(default=None, title="Job name", description="Internal job name")
    textinfo: str|None = Field(default=None, title="Info text", description="Info text used by WebUI.")
    # status fields
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    paused: bool = Field(title="Whether the task is paused")
    completed: bool = Field(title="Whether the task has already finished")
    debug: bool = Field(title="Debug logging level")
    # raw fields
    step: int = Field(default=None, title="Current step", description="Current step of the task")
    steps: int = Field(default=None, title="Total steps", description="Total number of steps")
    batch_no: int = Field(default=None, title="Current batch", description="Current batch")
    batch_count: int = Field(default=None, title="Total batches", description="Total number of batches")
    # calculated fields
    progress: float = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float|None = Field(default=None, title="ETA in secs")
    # image fields
    live_preview: str|None = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int|None = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")


def api_progress(req: ProgressRequest):
    active = (req.id_task == current_task) or (req.id_task == 'undefined' and len(shared.state.job) > 0)
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks
    paused = shared.state.paused
    step = max(shared.state.sampling_step, 0)
    steps = max(shared.state.sampling_steps, 1)
    batch_no = max(shared.state.batch_no, 0)
    batch_count = max(shared.state.batch_count, 0)

    current = step / steps if step > 0 and steps > 0 else 0
    batch = batch_no / batch_count if batch_no > 0 and batch_count > 0 else 1
    progress = round(min(1, current * batch), 2)

    elapsed = time.time() - shared.state.time_start if shared.state.time_start is not None else 0
    predicted = elapsed / progress if progress > 0 else None
    eta = predicted - elapsed if predicted is not None else None
    id_live_preview = req.id_live_preview
    live_preview = None
    textinfo = shared.state.textinfo
    if not active:
        id_live_preview = -1
        textinfo = "Queued..." if queued else "Waiting..."

    debug_log(f'Progress: job="{shared.state.job}" active={active} progress={step}/{steps}/{progress} image={shared.state.current_image_sampling_step} request={id_live_preview} last={shared.state.id_live_preview} job={shared.state.preview_job} elapsed={elapsed:.3f}')

    if active:
        shared.state.set_current_image()
    if shared.state.current_image is not None and req.id_live_preview != shared.state.id_live_preview:
        live_preview = shared.state.live_preview_b64

    id_live_preview = shared.state.id_live_preview

    res = InternalProgressResponse(
        job=shared.state.job,
        textinfo=textinfo,
        active=active,
        queued=queued,
        paused=paused,
        completed=completed,
        debug=debug,
        progress=progress,
        step=step,
        steps=steps,
        batch_no=batch_no,
        batch_count=batch_count,
        job_timestamp=shared.state.time_start,
        eta=eta,
        live_preview=live_preview,
        id_live_preview=id_live_preview,
    )
    return res


def setup_progress_api():
    shared.api.add_api_route("/internal/progress", api_progress, methods=["POST"], response_model=InternalProgressResponse)

    @shared.api.app.websocket("/ws/preview")
    async def ws_preview(ws: WebSocket):
        await preview_manager.connect(ws)
        try:
            while True:
                data = await ws.receive_text()
                if data == 'ping':
                    await ws.send_text('pong')
                elif data == 'end':
                    break
        except Exception:
            pass
        preview_manager.disconnect(ws)
