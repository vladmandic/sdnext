import io
import os
import time
import base64
from urllib.parse import quote, unquote
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketState
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from PIL import Image
from modules import shared, images, files_cache, modelstats
from modules.logger import log
from modules.paths import resolve_output_path


debug = log.debug if os.environ.get('SD_BROWSER_DEBUG', None) is not None else lambda *args, **kwargs: None


OPTS_FOLDERS = [
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_control_samples",
    "outdir_extras_samples",
    "outdir_save",
    "outdir_video",
    "outdir_init_images",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_img2img_grids",
    "outdir_control_grids",
]

### class definitions

class ReqFiles(BaseModel):
    folder: str = Field(title="Folder")

### ws connection manager

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        agent = ws._headers.get("user-agent", "") # pylint: disable=protected-access
        debug(f'Browser WS connect: client={ws.client.host} agent="{agent}"')
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        debug(f'Browser WS disconnect: client={ws.client.host}')
        self.active.remove(ws)

    async def send(self, ws: WebSocket, data: str | dict | bytes):
        # debug(f'Browser WS send: client={ws.client.host} data={type(data)}')
        if ws.client_state != WebSocketState.CONNECTED:
            return
        if isinstance(data, bytes):
            await ws.send_bytes(data)
        elif isinstance(data, dict):
            await ws.send_json(data)
        elif isinstance(data, str):
            await ws.send_text(data)
        else:
            debug(f'Browser WS send: client={ws.client.host} data={type(data)} unknown')

    async def broadcast(self, data: str | dict | bytes):
        for ws in self.active:
            await self.send(ws, data)

### api definitions

def register_api(app: FastAPI): # register api
    manager = ConnectionManager()

    def get_video_thumbnail(filepath):
        from modules.video import get_video_params
        try:
            stat_size, stat_mtime = modelstats.stat(filepath)
            frames, fps, duration, width, height, codec, frame = get_video_params(filepath, capture=True)
            h = shared.opts.extra_networks_card_size
            w = shared.opts.extra_networks_card_size if shared.opts.browser_fixed_width else width * h // height
            frame = frame.convert('RGB')
            frame.thumbnail((w, h), Image.Resampling.HAMMING)
            buffered = io.BytesIO()
            frame.save(buffered, format='jpeg')
            data_url = f'data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode("ascii")}'
            frame.close()
            content = {
                'exif': f'Codec: {codec}, Frames: {frames}, Duration: {duration:.2f} sec, FPS: {fps:.2f}',
                'data': data_url,
                'width': width,
                'height': height,
                'size': stat_size,
                'mtime': stat_mtime.timestamp() * 1000, # JS timestamps use milliseconds
            }
            return content
        except Exception as e:
            log.error(f'Gallery video: file="{filepath}" {e}')
            return {}

    def get_image_thumbnail(filepath):
        try:
            stat_size, stat_mtime = modelstats.stat(filepath)
            image = Image.open(filepath)
            geninfo, _items = images.read_info_from_image(image)
            h = shared.opts.extra_networks_card_size
            w = shared.opts.extra_networks_card_size if shared.opts.browser_fixed_width else image.width * h // image.height
            width, height = image.width, image.height
            image = image.convert('RGB')
            image.thumbnail((w, h), Image.Resampling.HAMMING)
            buffered = io.BytesIO()
            image.save(buffered, format='jpeg')
            data_url = f'data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode("ascii")}'
            image.close()
            content = {
                'exif': geninfo,
                'data': data_url,
                'width': width,
                'height': height,
                'size': stat_size,
                'mtime': stat_mtime.timestamp() * 1000, # JS timestamps use milliseconds
            }
            return content
        except Exception as e:
            log.error(f'Gallery image: file="{filepath}" {e}')
            return {}

    # @app.get('/sdapi/v1/browser/folders', response_model=List[str])
    def get_folders():
        def make_folder(path, label=None):
            """Create folder entry with path and display label."""
            if label is None:
                label = os.path.basename(path) or path
            return {"path": path, "label": label}

        reference_dir = os.path.join('models', 'Reference')
        base_samples = shared.opts.outdir_samples
        base_grids = shared.opts.outdir_grids
        # Build list of resolved output paths with labels
        folders = []
        if base_samples:
            folders.append(make_folder(base_samples, os.path.basename(base_samples.rstrip('/\\'))))
        if base_grids and base_grids != base_samples:
            folders.append(make_folder(base_grids, os.path.basename(base_grids.rstrip('/\\'))))
        # Use the specific folder setting values as labels (e.g., "outputs/text" -> "outputs/text")
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_txt2img_samples), shared.opts.outdir_txt2img_samples))
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_img2img_samples), shared.opts.outdir_img2img_samples))
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_control_samples), shared.opts.outdir_control_samples))
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_extras_samples), shared.opts.outdir_extras_samples))
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_save), shared.opts.outdir_save))
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_video), shared.opts.outdir_video))
        folders.append(make_folder(resolve_output_path(base_samples, shared.opts.outdir_init_images), shared.opts.outdir_init_images))
        folders.append(make_folder(resolve_output_path(base_grids, shared.opts.outdir_txt2img_grids), shared.opts.outdir_txt2img_grids))
        folders.append(make_folder(resolve_output_path(base_grids, shared.opts.outdir_img2img_grids), shared.opts.outdir_img2img_grids))
        folders.append(make_folder(resolve_output_path(base_grids, shared.opts.outdir_control_grids), shared.opts.outdir_control_grids))
        # Custom browser folders and reference dir
        for f in shared.opts.browser_folders.split(','):
            f = f.strip()
            if f:
                folders.append(make_folder(f))
        folders.append(make_folder(reference_dir, 'Reference'))
        # Filter empty and duplicates (by path)
        seen_paths = set()
        unique_folders = []
        for f in folders:
            path = f["path"].strip()
            if path and path not in seen_paths and os.path.isdir(path):
                seen_paths.add(path)
                unique_folders.append(f)
                if shared.demo is not None and path not in shared.demo.allowed_paths:
                    debug(f'Browser folders allow: {path}')
                    shared.demo.allowed_paths.append(quote(path))
        debug(f'Browser folders: {unique_folders}')
        return JSONResponse(content=unique_folders)

    # @app.get("/sdapi/v1/browser/thumb", response_model=dict)
    async def get_thumb(file: str):
        try:
            decoded = unquote(file).replace('%3A', ':')
            if decoded.lower().endswith('.mp4'):
                return JSONResponse(content=get_video_thumbnail(decoded))
            else:
                return JSONResponse(content=get_image_thumbnail(decoded))
        except Exception as e:
            log.error(f'Gallery: {file} {e}')
            content = { 'error': str(e) }
            return JSONResponse(content=content)

    # @app.get("/sdapi/v1/browser/files", response_model=list)
    async def ht_files(folder: str):
        try:
            t0 = time.time()
            files = files_cache.directory_files(folder, recursive=True)
            lines = []
            for f in files:
                file = os.path.relpath(f, folder)
                msg = quote(folder) + '##F##' + quote(file)
                msg = msg[:1] + ":" + msg[4:] if msg[1:4] == "%3A" else msg
                lines.append(msg)
            t1 = time.time()
            log.debug(f'Gallery: type=ht folder="{folder}" files={len(lines)} time={t1-t0:.3f}')
            return lines
        except Exception as e:
            log.error(f'Gallery: {folder} {e}')
            return []

    shared.api.add_api_route("/sdapi/v1/browser/folders", get_folders, methods=["GET"], response_model=list[str])
    shared.api.add_api_route("/sdapi/v1/browser/thumb", get_thumb, methods=["GET"], response_model=dict)
    shared.api.add_api_route("/sdapi/v1/browser/files", ht_files, methods=["GET"], response_model=list)

    @app.websocket("/sdapi/v1/browser/files")
    async def ws_files(ws: WebSocket):
        try:
            await manager.connect(ws)
            folder = await ws.receive_text()
            folder = unquote(folder).replace('%3A', ':')
            t0 = time.time()
            numFiles = 0
            files = files_cache.list_files(folder, recursive=True)
            # files = list(files_cache.directory_files(folder, recursive=True))
            # files.sort(key=os.path.getmtime)
            for f in files:
                numFiles += 1
                file = os.path.relpath(f, folder)
                msg = quote(folder) + '##F##' + quote(file)
                msg = msg[:1] + ":" + msg[4:] if msg[1:4] == "%3A" else msg
                await manager.send(ws, msg)
            await manager.send(ws, '#END#')
            t1 = time.time()
            log.debug(f'Gallery: type=ws folder="{folder}" files={numFiles} time={t1-t0:.3f}')
        except Exception as e:
            debug(f'Browser WS error: {e}')
        manager.disconnect(ws)
