import io
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from installer import log


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.debug(f'WebSocket: connected clients={len(self.active)}')

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        log.debug(f'WebSocket: disconnected clients={len(self.active)}')

    async def send_json(self, ws: WebSocket, data: dict):
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)

    async def send_bytes(self, ws: WebSocket, data: bytes):
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_bytes(data)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


async def push_progress(ws: WebSocket):
    from modules import shared
    from modules.api import helpers
    last_step = -1
    last_preview_id = -1
    last_download_snapshot = None
    while ws.client_state == WebSocketState.CONNECTED:
        try:
            state = shared.state
            if state.job_count > 0 and state.sampling_step != last_step:
                last_step = state.sampling_step
                status = state.status()
                await manager.send_json(ws, {"type": "progress", "data": status.dict() if hasattr(status, 'dict') else status.model_dump()})
                if state.id_live_preview != last_preview_id and state.current_image is not None:
                    last_preview_id = state.id_live_preview
                    buf = io.BytesIO()
                    state.current_image.save(buf, format="JPEG", quality=75)
                    await manager.send_bytes(ws, buf.getvalue())
            elif state.job_count == 0 and last_step != -1:
                last_step = -1
                status = state.status()
                await manager.send_json(ws, {"type": "status", "data": status.dict() if hasattr(status, 'dict') else status.model_dump()})
            # Push download progress when downloads are active
            try:
                from modules.civitai.download_civitai import download_manager
                active = download_manager.get_active_items()
                if active or last_download_snapshot:
                    snapshot = str(active)
                    if snapshot != last_download_snapshot:
                        last_download_snapshot = snapshot if active else None
                        await manager.send_json(ws, {"type": "download", "data": active})
            except ImportError:
                pass
        except Exception as e:
            log.debug(f'WebSocket push error: {e}')
            break
        await asyncio.sleep(0.1)


async def handle_command(ws: WebSocket, data: dict):
    from modules import shared
    msg_type = data.get("type", "")
    if msg_type == "interrupt":
        shared.state.interrupt()
        await manager.send_json(ws, {"type": "ack", "data": {"command": "interrupt"}})
    elif msg_type == "skip":
        shared.state.skip()
        await manager.send_json(ws, {"type": "ack", "data": {"command": "skip"}})
    elif msg_type == "download_cancel":
        download_id = data.get("id", "")
        if download_id:
            try:
                from modules.civitai.download_civitai import download_manager
                result = download_manager.cancel(download_id)
                await manager.send_json(ws, {"type": "ack", "data": {"command": "download_cancel", "id": download_id, "success": result}})
            except ImportError:
                await manager.send_json(ws, {"type": "ack", "data": {"command": "download_cancel", "id": download_id, "success": False}})
    elif msg_type == "ping":
        await manager.send_json(ws, {"type": "pong"})


async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    push_task = None
    try:
        push_task = asyncio.create_task(push_progress(ws))
        while True:
            data = await ws.receive_json()
            await handle_command(ws, data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug(f'WebSocket error: {e}')
    finally:
        manager.disconnect(ws)
        if push_task:
            push_task.cancel()


def register_ws(app):
    app.add_api_websocket_route("/sdapi/v1/ws", ws_endpoint)
    log.debug('WebSocket: registered /sdapi/v1/ws')
