import io
import os
import shutil
import time
import base64
import zipfile
from urllib.parse import quote, unquote
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
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

class ReqDelete(BaseModel):
    files: list[str] = Field(title="File paths to delete")

class ReqMove(BaseModel):
    files: list[str] = Field(title="Source file paths")
    destination: str = Field(title="Target folder path")

class ReqDownload(BaseModel):
    files: list[str] = Field(title="File paths to download")


### security helpers

def _get_allowed_roots():
    """Collect all resolved gallery root folders."""
    roots = set()
    base_samples = shared.opts.outdir_samples
    base_grids = shared.opts.outdir_grids
    if base_samples:
        roots.add(os.path.abspath(os.path.realpath(base_samples)))
    if base_grids:
        roots.add(os.path.abspath(os.path.realpath(base_grids)))
    for opt_key in OPTS_FOLDERS:
        val = getattr(shared.opts, opt_key, None)
        if val:
            base = base_grids if 'grid' in opt_key else base_samples
            resolved = resolve_output_path(base, val) if base else val
            roots.add(os.path.abspath(os.path.realpath(resolved)))
    for f in shared.opts.browser_folders.split(','):
        f = f.strip()
        if f:
            roots.add(os.path.abspath(os.path.realpath(f)))
    reference_dir = os.path.join('models', 'Reference')
    roots.add(os.path.abspath(os.path.realpath(reference_dir)))
    return roots


def is_allowed_path(filepath: str) -> bool:
    """Check if a file path is within allowed gallery folders."""
    resolved = os.path.abspath(os.path.realpath(filepath))
    for root in _get_allowed_roots():
        if resolved.startswith(root + os.sep) or resolved == root:
            return True
    return False


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
        """List configured output folders for the image browser with path and display label."""
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
    def get_thumb(file: str):
        """Return a thumbnail and metadata (EXIF, dimensions, size, mtime) for a single image or video file."""
        try:
            decoded = unquote(file).replace('%3A', ':')
            video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.m4v'}
            ext = os.path.splitext(decoded)[1].lower()
            if ext in video_extensions:
                return JSONResponse(content=get_video_thumbnail(decoded))
            else:
                return JSONResponse(content=get_image_thumbnail(decoded))
        except Exception as e:
            log.error(f'Gallery: {file} {e}')
            content = { 'error': str(e) }
            return JSONResponse(content=content)

    # @app.get("/sdapi/v1/browser/files", response_model=list)
    async def ht_files(folder: str):
        """List all files in a gallery folder recursively. Returns encoded path strings for client-side rendering."""
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

    def get_folder_info(folder: str):
        """
        Get directory modification time for cache invalidation.

        Returns the latest mtime across all files in the folder (recursive).
        Clients can poll this to detect when new images have been saved.
        """
        try:
            decoded = unquote(folder).replace('%3A', ':')
            mtime = files_cache.directory_mtime(decoded, recursive=True)
            return JSONResponse(content={'mtime': mtime})
        except Exception as e:
            shared.log.error(f'Gallery folder-info: folder="{folder}" {e}')
            return JSONResponse(content={'mtime': 0})

    def get_subdirs(folder: str):
        """
        List subdirectories of a gallery folder.

        Returns immediate child directories with their path and display label,
        sorted alphabetically. Used for building folder tree navigation.
        """
        decoded = unquote(folder).replace('%3A', ':')
        directory = files_cache.get_directory(decoded, fetch=True)
        if not directory or not directory.directories:
            return JSONResponse(content=[])
        subdirs = []
        for d in sorted(directory.directories):
            label = os.path.basename(d)
            if label:
                subdirs.append({'path': d, 'label': label})
        return JSONResponse(content=subdirs)

    def delete_files(req: ReqDelete):
        """Delete gallery files after validating paths are within allowed folders."""
        deleted = []
        errors = []
        for filepath in req.files:
            if not is_allowed_path(filepath):
                errors.append({'file': filepath, 'error': 'Path not allowed'})
                continue
            resolved = os.path.abspath(os.path.realpath(filepath))
            try:
                if os.path.isfile(resolved):
                    os.remove(resolved)
                    deleted.append(filepath)
                    folder = os.path.dirname(resolved)
                    files_cache.delete_cached_directory(folder)
                else:
                    errors.append({'file': filepath, 'error': 'File not found'})
            except Exception as e:
                errors.append({'file': filepath, 'error': str(e)})
        log.debug(f'Gallery delete: deleted={len(deleted)} errors={len(errors)}')
        return JSONResponse(content={'deleted': deleted, 'errors': errors})

    def move_files(req: ReqMove):
        """Move gallery files to a destination folder after path validation."""
        if not is_allowed_path(req.destination):
            return JSONResponse(content={'moved': [], 'errors': [{'file': req.destination, 'error': 'Destination not allowed'}]}, status_code=403)
        dest_resolved = os.path.abspath(os.path.realpath(req.destination))
        if not os.path.isdir(dest_resolved):
            return JSONResponse(content={'moved': [], 'errors': [{'file': req.destination, 'error': 'Destination is not a directory'}]}, status_code=400)
        moved = []
        errors = []
        source_folders = set()
        for filepath in req.files:
            if not is_allowed_path(filepath):
                errors.append({'file': filepath, 'error': 'Path not allowed'})
                continue
            resolved = os.path.abspath(os.path.realpath(filepath))
            try:
                if os.path.isfile(resolved):
                    source_folders.add(os.path.dirname(resolved))
                    dest_path = os.path.join(dest_resolved, os.path.basename(resolved))
                    shutil.move(resolved, dest_path)
                    moved.append(filepath)
                else:
                    errors.append({'file': filepath, 'error': 'File not found'})
            except Exception as e:
                errors.append({'file': filepath, 'error': str(e)})
        for folder in source_folders:
            files_cache.delete_cached_directory(folder)
        files_cache.delete_cached_directory(dest_resolved)
        log.debug(f'Gallery move: moved={len(moved)} errors={len(errors)} dest="{req.destination}"')
        return JSONResponse(content={'moved': moved, 'errors': errors})

    def download_files(req: ReqDownload):
        """Create a zip archive of gallery files and stream it to the client."""
        for filepath in req.files:
            if not is_allowed_path(filepath):
                return JSONResponse(content={'error': f'Path not allowed: {filepath}'}, status_code=403)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
            seen_names = set()
            for filepath in req.files:
                resolved = os.path.abspath(os.path.realpath(filepath))
                if not os.path.isfile(resolved):
                    continue
                name = os.path.basename(resolved)
                # Deduplicate filenames in zip
                if name in seen_names:
                    base, ext = os.path.splitext(name)
                    counter = 1
                    while f'{base}_{counter}{ext}' in seen_names:
                        counter += 1
                    name = f'{base}_{counter}{ext}'
                seen_names.add(name)
                zf.write(resolved, name)
        buf.seek(0)
        return StreamingResponse(buf, media_type='application/zip', headers={'Content-Disposition': 'attachment; filename="gallery.zip"'})

    shared.api.add_api_route("/sdapi/v2/browser/folders", get_folders, methods=["GET"], response_model=list[str], tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/thumb", get_thumb, methods=["GET"], response_model=dict, tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/files", ht_files, methods=["GET"], response_model=list, tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/folder-info", get_folder_info, methods=["GET"], tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/subdirs", get_subdirs, methods=["GET"], tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/delete", delete_files, methods=["POST"], tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/move", move_files, methods=["POST"], tags=["Gallery"])
    shared.api.add_api_route("/sdapi/v2/browser/download", download_files, methods=["POST"], tags=["Gallery"])

    @app.websocket("/sdapi/v2/browser/files")
    async def ws_files(ws: WebSocket):
        if shared.cmd_opts.auth or shared.cmd_opts.auth_file:
            from modules.api.security import ws_tickets
            ticket = ws.query_params.get("ticket")
            if not ticket or not ws_tickets.validate(ticket):
                await ws.close(code=1008, reason="Invalid or expired ticket")
                return
        try:
            await manager.connect(ws)
            folder = await ws.receive_text()
            folder = unquote(folder).replace('%3A', ':')
            if not is_allowed_path(folder):
                await ws.close(code=1008, reason="Path not allowed")
                return
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
