import os
import uuid
import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import rich.progress as p
from installer import log
from modules import shared, paths
from modules.logger import console


@dataclass
class DownloadItem:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = ""
    filename: str = ""
    folder: str = ""
    model_type: str = ""
    expected_hash: str = ""
    token: str | None = None
    model_id: int = 0
    version_id: int = 0
    status: str = "queued"  # queued | downloading | verifying | completed | failed | cancelled
    progress: float = 0.0
    bytes_downloaded: int = 0
    bytes_total: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "filename": self.filename,
            "folder": self.folder,
            "model_type": self.model_type,
            "status": self.status,
            "progress": round(self.progress, 4),
            "bytes_downloaded": self.bytes_downloaded,
            "bytes_total": self.bytes_total,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DownloadManager:
    def __init__(self, max_workers: int = 2):
        self._queue: deque[DownloadItem] = deque()
        self._active: dict[str, DownloadItem] = {}
        self._completed: deque[DownloadItem] = deque(maxlen=50)
        self._cancel_ids: set[str] = set()
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._worker_count = 0

    def enqueue(self, url: str, folder: str, filename: str, model_type: str = "",
                expected_hash: str = "", token: str | None = None,
                model_id: int = 0, version_id: int = 0) -> DownloadItem:
        item = DownloadItem(
            url=url,
            filename=filename,
            folder=folder,
            model_type=model_type,
            expected_hash=expected_hash,
            token=token,
            model_id=model_id,
            version_id=version_id,
        )
        with self._lock:
            self._queue.append(item)
        log.info(f'CivitAI download queued: id={item.id} file="{filename}" url="{url}"')
        self._try_start_worker()
        return item

    def cancel(self, download_id: str) -> bool:
        with self._lock:
            # Check if in queue (not yet started)
            for item in self._queue:
                if item.id == download_id:
                    item.status = "cancelled"
                    item.completed_at = datetime.now()
                    self._queue.remove(item)
                    self._completed.append(item)
                    log.info(f'CivitAI download cancelled (queued): id={download_id}')
                    return True
            # Check if active
            if download_id in self._active:
                self._cancel_ids.add(download_id)
                log.info(f'CivitAI download cancel requested: id={download_id}')
                return True
        return False

    def status(self) -> dict:
        with self._lock:
            return {
                "active": [item.to_dict() for item in self._active.values()],
                "queued": [item.to_dict() for item in self._queue],
                "completed": [item.to_dict() for item in self._completed],
            }

    def get_active_items(self) -> list[dict]:
        with self._lock:
            return [item.to_dict() for item in self._active.values()]

    def _try_start_worker(self):
        with self._lock:
            if self._worker_count >= self._max_workers:
                return
            if not self._queue:
                return
            self._worker_count += 1
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _worker(self):
        try:
            while True:
                item = None
                with self._lock:
                    if not self._queue:
                        break
                    item = self._queue.popleft()
                    self._active[item.id] = item
                if item:
                    self._download(item)
                    with self._lock:
                        self._active.pop(item.id, None)
                        self._completed.append(item)
        finally:
            with self._lock:
                self._worker_count -= 1
            # Start more workers if items still queued
            self._try_start_worker()

    def _download(self, item: DownloadItem):
        # Create temp file name from URL hash
        url_hash = hashlib.sha256(item.url.encode('utf-8')).hexdigest()[:8]
        temp_file = os.path.join(item.folder, f'{url_hash}.tmp')
        final_file = os.path.join(item.folder, item.filename)

        # Check if already exists
        if os.path.isfile(final_file):
            item.status = "completed"
            item.progress = 1.0
            item.completed_at = datetime.now()
            item.error = "already exists"
            log.info(f'CivitAI download: id={item.id} file="{final_file}" already exists')
            return

        # Ensure folder exists
        os.makedirs(item.folder, exist_ok=True)

        # Resume support
        headers = {}
        starting_pos = 0
        if os.path.isfile(temp_file):
            starting_pos = os.path.getsize(temp_file)
            headers['Range'] = f'bytes={starting_pos}-'

        # Auth
        token = item.token or self._get_token()
        if token and 'civit' in item.url.lower():
            headers['Authorization'] = f'Bearer {token}'

        item.status = "downloading"
        item.bytes_downloaded = starting_pos

        try:
            r = shared.req(item.url, headers=headers if headers else None, stream=True)
            if r.status_code not in (200, 206):
                item.status = "failed"
                item.error = f'HTTP {r.status_code}'
                item.completed_at = datetime.now()
                return

            total_size = int(r.headers.get('content-length', 0))
            item.bytes_total = starting_pos + total_size

            # Resolve filename from Content-Disposition if not set
            if not item.filename or item.filename == "Unknown":
                cn = r.headers.get('content-disposition', '')
                if 'filename=' in cn:
                    item.filename = cn.split('filename=')[-1].strip('"')
                    final_file = os.path.join(item.folder, item.filename)

            block_size = 65536  # 64KB blocks
            written = starting_pos
            log.info(f'CivitAI download: id={item.id} file="{item.filename}" size={round((starting_pos + total_size) / 1024 / 1024, 1)}MB')
            pbar = p.Progress(p.TextColumn('[cyan]{task.description}'), p.DownloadColumn(), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TransferSpeedColumn(), p.TextColumn('[cyan]{task.fields[name]}'), console=console)
            with pbar:
                task = pbar.add_task(description="Download", total=starting_pos + total_size, name=item.filename)
                with open(temp_file, 'ab') as f:
                    for chunk in r.iter_content(block_size):
                        # Check cancellation
                        if item.id in self._cancel_ids:
                            with self._lock:
                                self._cancel_ids.discard(item.id)
                            item.status = "cancelled"
                            item.completed_at = datetime.now()
                            log.info(f'CivitAI download cancelled: id={item.id}')
                            try:
                                os.remove(temp_file)
                            except OSError:
                                pass
                            return

                        f.write(chunk)
                        written += len(chunk)
                        item.bytes_downloaded = written
                        if item.bytes_total > 0:
                            item.progress = written / item.bytes_total
                        pbar.update(task, completed=written)

            # Validate minimum size
            if written < 1024:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                item.status = "failed"
                item.error = f'download too small: {written} bytes'
                item.completed_at = datetime.now()
                return

            # Check for incomplete download
            if starting_pos + total_size != written:
                item.status = "failed"
                item.error = f'incomplete: expected={starting_pos + total_size} got={written}'
                item.completed_at = datetime.now()
                return

        except Exception as e:
            item.status = "failed"
            item.error = str(e)
            item.completed_at = datetime.now()
            log.error(f'CivitAI download error: id={item.id} {e}')
            return

        # Hash verification
        if item.expected_hash:
            item.status = "verifying"
            try:
                from modules import hashes
                computed = hashes.calculate_sha256(temp_file, quiet=True)
                if computed.upper() != item.expected_hash.upper():
                    discard = getattr(shared.opts, 'civitai_discard_hash_mismatch', True)
                    if discard:
                        try:
                            os.remove(temp_file)
                        except OSError:
                            pass
                        item.status = "failed"
                        item.error = f'hash mismatch: expected={item.expected_hash[:16]}... got={computed[:16]}...'
                        item.completed_at = datetime.now()
                        log.error(f'CivitAI download hash mismatch: id={item.id} expected={item.expected_hash[:16]} got={computed[:16]}')
                        return
                    log.warning(f'CivitAI download hash mismatch (kept): id={item.id} expected={item.expected_hash[:16]} got={computed[:16]}')
            except Exception as e:
                log.warning(f'CivitAI download hash check failed: id={item.id} {e}')

        # Move temp to final
        try:
            os.rename(temp_file, final_file)
        except OSError as e:
            item.status = "failed"
            item.error = f'rename failed: {e}'
            item.completed_at = datetime.now()
            return

        item.status = "completed"
        item.progress = 1.0
        item.completed_at = datetime.now()
        log.info(f'CivitAI download complete: id={item.id} file="{final_file}" size={item.bytes_downloaded}')

        # Write verified hash to cache so check-local finds it immediately
        if item.expected_hash:
            try:
                from modules import hashes
                model_type_map = {'Checkpoint': 'checkpoint', 'LORA': 'lora', 'TextualInversion': 'embedding', 'VAE': 'vae'}
                prefix = model_type_map.get(item.model_type, item.model_type.lower())
                name = os.path.splitext(item.filename)[0]
                title = f"{prefix}/{name}"
                hashes.cache().add_hash(title, os.path.getmtime(final_file), item.expected_hash.lower())
                hashes.save_cache()
            except Exception:
                pass

        # Download metadata and preview
        self._fetch_sidecar(item, final_file)

        # Refresh model list and extra-networks cache
        try:
            from modules.sd_models import list_models
            list_models()
        except Exception:
            pass
        try:
            from modules.api.loras import _invalidate_extra_networks
            _invalidate_extra_networks()
        except Exception:
            pass

    def _fetch_sidecar(self, item: DownloadItem, final_file: str):
        """Download metadata JSON and preview image for a completed download."""
        if not item.model_id:
            return
        try:
            code, _size, _note = download_civit_meta(final_file, item.model_id)
            if code == 200:
                log.info(f'CivitAI metadata saved: id={item.id} model_id={item.model_id}')
        except Exception as e:
            log.warning(f'CivitAI metadata fetch failed: id={item.id} {e}')
        if not item.version_id:
            return
        try:
            from modules.civitai.client_civitai import client
            version = client.get_version(item.version_id)
            if version and version.images:
                for img in version.images:
                    if img.url:
                        code, _size, _note = download_civit_preview(final_file, img.url)
                        if code == 200:
                            log.info(f'CivitAI preview saved: id={item.id}')
                            break
        except Exception as e:
            log.warning(f'CivitAI preview fetch failed: id={item.id} {e}')

    def _get_token(self) -> str | None:
        tok = getattr(shared.opts, 'civitai_token', '') or ''
        if tok:
            return tok
        return os.environ.get('CIVITAI_TOKEN', None)


download_manager = DownloadManager()


# ---- Legacy compatibility functions ----

def download_civit_meta(model_path: str, model_id):
    fn = os.path.splitext(model_path)[0] + '.json'
    url = f'https://civitai.com/api/v1/models/{model_id}'
    r = shared.req(url)
    if r.status_code == 200:
        try:
            data = r.json()
            from modules.json_helpers import writefile
            writefile(data, filename=fn, mode='w', silent=True)
            log.info(f'CivitAI download: id={model_id} url={url} file="{fn}"')
            return r.status_code, len(data), ''
        except Exception as e:
            from modules import errors
            errors.display(e, 'civitai meta')
            log.error(f'CivitAI meta: id={model_id} url={url} file="{fn}" {e}')
            return r.status_code, '', str(e)
    return r.status_code, '', ''


def download_civit_preview(model_path: str, preview_url: str):
    if model_path is None:
        return 500, '', ''
    ext = os.path.splitext(preview_url)[1]
    preview_file = os.path.splitext(model_path)[0] + ext
    is_video = preview_file.lower().endswith('.mp4')
    is_json = preview_file.lower().endswith('.json')
    if is_json:
        log.warning(f'CivitAI download: url="{preview_url}" skip json')
        return 500, '', 'expected preview image got json'
    if os.path.exists(preview_file):
        return 304, '', 'already exists'
    r = shared.req(preview_url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 16384
    written = 0
    jobid = shared.state.begin('Download CivitAI')
    try:
        with open(preview_file, 'wb') as f:
            for data in r.iter_content(block_size):
                written += len(data)
                f.write(data)
        if written < 1024:
            os.remove(preview_file)
            return 400, '', 'removed invalid download'
        if is_video:
            from modules.civitai.video_helper import save_video_frame
            save_video_frame(preview_file)
        else:
            from PIL import Image
            img = Image.open(preview_file)
            log.info(f'CivitAI download: url={preview_url} file="{preview_file}" size={total_size} image={img.size}')
            img.close()
    except Exception as e:
        log.error(f'CivitAI download error: url={preview_url} file="{preview_file}" written={written} {e}')
        shared.state.end(jobid)
        return 500, '', str(e)
    shared.state.end(jobid)
    return 200, str(total_size), ''


def download_civit_model(model_url: str, model_name: str = '', model_path: str = '', model_type: str = '', token: str | None = None,
                         base_model: str = '', model_id: int = 0, version_id: int = 0):
    """Legacy function — delegates to DownloadManager for non-blocking downloads."""
    if not model_url:
        log.error('Model download: no url provided')
        return None
    if not version_id:
        import re
        match = re.search(r'/api/download/models/(\d+)', model_url)
        if match:
            version_id = int(match.group(1))
    from modules.civitai.filemanage_civitai import get_type_folder
    if not model_path:
        if getattr(shared.opts, 'civitai_save_subfolder_enabled', False):
            from modules.civitai.filemanage_civitai import resolve_save_path
            folder = str(resolve_save_path(model_type or 'Checkpoint', model_name=model_name, base_model=base_model))
        else:
            folder = str(get_type_folder(model_type or 'Checkpoint'))
    elif os.path.isabs(model_path):
        folder = model_path
    else:
        folder = os.path.join(paths.models_path, model_path)
    item = download_manager.enqueue(
        url=model_url,
        folder=folder,
        filename=model_name or "Unknown",
        model_type=model_type,
        token=token,
        model_id=model_id,
        version_id=version_id,
    )
    # Wait for completion (legacy blocking behavior)
    while item.status in ("queued", "downloading", "verifying"):
        time.sleep(0.5)
    if item.status == "completed" and not item.error:
        from modules.sd_models import list_models
        list_models()
        return os.path.join(item.folder, item.filename)
    return None
