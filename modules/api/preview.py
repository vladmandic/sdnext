import io
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from modules.logger import log


class PreviewManager:
    """Manages WebSocket connections for step-based live preview streaming.

    Each denoising step, diffusers_callback calls push_step() from the
    generation thread.  Image encoding happens there (non‑blocking for
    TAESD, the default preview method) and the binary JPEG + JSON progress
    are dispatched to the event loop via run_coroutine_threadsafe().
    """

    def __init__(self):
        self.connections: set[WebSocket] = set()
        self._client_hidden: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Lifecycle (called from the event-loop thread)
    # ------------------------------------------------------------------

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        log.debug(f'Preview WS connect: client={ws.client.host} total={len(self.connections)}')

    async def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)
        log.debug(f'Preview WS disconnect: client={ws.client.host} total={len(self.connections)}')

    async def handle_message(self, data: dict):
        """Receive visibility-change messages from the client."""
        if data.get('type') == 'visibility':
            self._client_hidden = not data.get('visible', True)

    # ------------------------------------------------------------------
    # Push from generation thread (thread‑safe)
    # ------------------------------------------------------------------

    def push_step(self, step: int, steps: int, state):
        """Called from the generation thread inside diffusers_callback."""
        if not self.connections or self._client_hidden or self._loop is None:
            return

        # --- lightweight metadata (no GPU ops) ---
        progress = round(step / steps, 2) if steps > 0 else 0
        data = {
            "active": True,
            "step": step,
            "steps": steps,
            "progress": progress,
            "job": state.job,
            "paused": state.paused,
        }

        # --- encode preview image ---
        img_bytes = None
        try:
            if state.set_current_image() and state.current_image is not None:
                buf = io.BytesIO()
                state.current_image.save(buf, format='jpeg', quality=60)
                img_bytes = buf.getvalue()
        except Exception:
            pass

        data["has_image"] = img_bytes is not None

        # --- dispatch to event loop ---
        asyncio.run_coroutine_threadsafe(
            self._broadcast(data, img_bytes),
            self._loop
        )

    def push_complete(self):
        """Called when a generation job finishes."""
        if not self.connections or self._loop is None:
            return
        data = {
            "active": False,
            "step": 0,
            "steps": 0,
            "progress": 1.0,
            "job": "",
            "paused": False,
            "completed": True,
            "has_image": False,
        }
        asyncio.run_coroutine_threadsafe(
            self._broadcast(data, None),
            self._loop
        )

    # ------------------------------------------------------------------
    # Async broadcast (runs on event loop)
    # ------------------------------------------------------------------

    async def _broadcast(self, data: dict, img_bytes: bytes | None):
        dead: set[WebSocket] = set()
        for ws in self.connections:
            try:
                await ws.send_json(data)
                if img_bytes:
                    await ws.send_bytes(img_bytes)
            except Exception:
                dead.add(ws)
        self.connections -= dead


preview_manager = PreviewManager()


async def ws_preview(websocket: WebSocket):
    await preview_manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                import json
                msg = json.loads(raw)
                await preview_manager.handle_message(msg)
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await preview_manager.disconnect(websocket)
