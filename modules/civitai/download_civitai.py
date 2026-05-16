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
                        code, _size, _note = download_civit_preview(final_file, img.url, meta=img.meta)
                        if code == 200:
                            log.info(f'CivitAI preview saved: id={item.id}')
                            break
                        if code == 304 and backfill_preview_parameters(final_file, img.url, img.meta):
                            log.info(f'CivitAI preview backfilled: id={item.id}')
                            break
        except Exception as e:
            log.warning(f'CivitAI preview fetch failed: id={item.id} {e}')

    def _get_token(self) -> str | None:
        tok = getattr(shared.opts, 'civitai_token', '') or ''
        if tok:
            return tok
        return os.environ.get('CIVITAI_TOKEN', None)


download_manager = DownloadManager()


# ---- Preview metadata helpers ----

NOISE_META_KEYS = frozenset({'hashes', 'comfy', 'comfyui', 'workflow', 'extrametadata'})
PASSTHROUGH_VALUE_LIMIT = 512  # chars; pass-through values longer than this are dropped


def civitai_meta_to_parameters(meta: dict | None) -> str:
    """Convert Civitai version-image meta dict to sdnext parameters string.

    Output matches sdnext's standard `parameters` channel (`modules/image/save.py:65-72`):
    positive prompt on line one, optional `Negative prompt:` line two,
    comma-joined `Key: Value` pairs on line three. Round-trippable through
    `modules.infotext.parse`. Generator-specific noise keys (full ComfyUI
    workflows, etc.) are dropped and any pass-through value exceeding
    `PASSTHROUGH_VALUE_LIMIT` chars is omitted to keep the embedded chunk
    compact.
    """
    if not meta or not isinstance(meta, dict):
        return ''
    import json
    from modules.infotext import quote
    lower = {k.lower(): (k, v) for k, v in meta.items()}

    def lookup(*keys):
        for k in keys:
            if k.lower() in lower:
                return lower[k.lower()][1]
        return None

    prompt = lookup('prompt') or ''
    negative = lookup('negativePrompt', 'negative_prompt', 'Negative prompt') or ''
    pairs = []
    mapping = [
        (('steps',), 'Steps'),
        (('sampler',), 'Sampler'),
        (('cfgScale', 'cfg_scale', 'CFG scale'), 'CFG scale'),
        (('seed',), 'Seed'),
        (('Size',), 'Size'),
        (('Model',), 'Model'),
        (('Model hash', 'modelHash'), 'Model hash'),
        (('clipSkip', 'clip_skip', 'CLiP-skip'), 'CLiP-skip'),
        (('denoisingStrength', 'Denoising strength'), 'Denoising strength'),
    ]
    consumed = {'prompt', 'negativeprompt', 'negative_prompt'}
    for src_keys, out_key in mapping:
        v = lookup(*src_keys)
        if v is None or v == '':
            continue
        pairs.append(f'{out_key}: {quote(v)}')
        for k in src_keys:
            consumed.add(k.lower())
    for k, v in meta.items():
        if k.lower() in consumed:
            continue
        if k.lower() in NOISE_META_KEYS:
            continue
        if k in ('resources', 'civitaiResources'):
            try:
                pairs.append(f'Civitai resources: {quote(json.dumps(v, separators=(",", ":")))}')
            except Exception:
                pass
            continue
        if v is None or v == '':
            continue
        quoted = quote(v)
        if len(str(quoted)) > PASSTHROUGH_VALUE_LIMIT:
            continue
        pairs.append(f'{k}: {quoted}')
    lines = [str(prompt).strip()]
    if negative:
        lines.append(f'Negative prompt: {negative}')
    if pairs:
        lines.append(', '.join(pairs))
    return '\n'.join(lines)


def fit_parameters_for_exif(parameters: str, limit: int = 30000) -> str:
    """Trim parameters string to fit JPEG/WEBP EXIF UserComment.

    JPEG's APP1 segment caps at 64KB; UserComment is UTF-16-LE encoded so
    each char takes 2 bytes. A 30000-char limit keeps the encoded payload
    around 60KB with headroom for piexif overhead. Drops the
    `Civitai resources` field first (typical bloat source) and truncates as
    last resort. PNG callers don't need this since `tEXt` chunks are
    unbounded.
    """
    if len(parameters) <= limit:
        return parameters
    lines = parameters.split('\n')
    if len(lines) >= 3:
        parts = [p for p in lines[2].split(', ') if not p.startswith('Civitai resources:')]
        lines[2] = ', '.join(parts)
        parameters = '\n'.join(lines)
        if len(parameters) <= limit:
            return parameters
    return parameters[:max(0, limit - 32)].rstrip() + '\n[truncated]'


def embed_preview_parameters(preview_file: str, parameters: str) -> bool:
    """Embed parameters string into preview image at `preview_file`.

    PNG -> `tEXt` chunk with key `parameters`. JPEG/WEBP -> EXIF
    `UserComment` via piexif. RGBA is converted to RGB before JPEG save.
    Other extensions: no-op. Returns success.

    Writes are atomic (temp file + os.replace). On success, any co-located
    `<base>.thumb.jpg` is removed so the lazy thumb generator re-emits it
    carrying the new params. Embed failures are swallowed and logged at
    debug; the source file is removed only when PIL cannot read it back,
    so the caller can re-download a fresh copy.
    """
    if not preview_file or not parameters:
        return False
    ext = os.path.splitext(preview_file)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.webp'):
        return False
    tmp_file = preview_file + '.embed.tmp'
    try:
        from PIL import Image
        img = Image.open(preview_file)
        img.load()
        try:
            if ext == '.png':
                from PIL import PngImagePlugin
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text('parameters', parameters)
                img.save(tmp_file, format='PNG', pnginfo=pnginfo)
            else:
                import piexif
                import piexif.helper
                payload = fit_parameters_for_exif(parameters)
                exif_bytes = piexif.dump({'Exif': {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(payload, encoding='unicode')}})
                if ext in ('.jpg', '.jpeg'):
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(tmp_file, format='JPEG', quality=95, exif=exif_bytes)
                else:
                    img.save(tmp_file, format='WEBP', quality=95, exif=exif_bytes)
        finally:
            img.close()
        os.replace(tmp_file, preview_file)
        thumb_base = os.path.splitext(preview_file)[0]
        if thumb_base.endswith('.preview'):
            thumb_base = thumb_base[:-len('.preview')]
        thumb_file = thumb_base + '.thumb.jpg'
        if os.path.exists(thumb_file):
            try:
                os.remove(thumb_file)
                log.debug(f'CivitAI thumb invalidated: file="{thumb_file}"')
            except Exception:
                pass
        return True
    except Exception as e:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        try:
            from PIL import Image as PILImage
            with PILImage.open(preview_file) as probe:
                probe.verify()
        except Exception:
            try:
                os.remove(preview_file)
                log.warning(f'CivitAI preview removing invalid: image={preview_file}')
            except Exception:
                pass
        log.debug(f'CivitAI preview embed failed: file="{preview_file}" {e}')
        return False


def resolve_preview_file(item: dict) -> str | None:
    """Resolve the actual on-disk preview file for a network item.

    `item['local_preview']` is the aspirational save path (e.g.
    `<base>.<samples_format>`), not necessarily the existing file. This
    helper extracts the real path from `item['preview']` URL (set by
    `ExtraNetworksPage.link_preview`) and falls back to scanning common
    extensions at the model base. Returns None when nothing on-disk
    matches.
    """
    preview_url = item.get('preview') or ''
    if preview_url and 'missing.png' not in preview_url:
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(preview_url)
            fn = urllib.parse.parse_qs(parsed.query).get('filename', [None])[0]
            if fn:
                fn = urllib.parse.unquote(fn)
                if os.path.isfile(fn):
                    return fn
        except Exception:
            pass
    filename = item.get('filename')
    if not filename:
        return None
    return find_ui_preview_file(filename)


def preview_has_parameters(preview_file: str) -> bool:
    """Check whether `preview_file` carries a non-empty embedded parameters string.

    Mirrors `modules/image/metadata.py:read_info_from_image`: PNG
    `image.info["parameters"]` or JPEG/WEBP EXIF `UserComment`. EXIF
    UserComment is decoded via piexif.helper to verify a non-empty body;
    the 8-byte `b"UNICODE\\x00"` header is present in any UserComment
    even when its body is empty.
    """
    if not preview_file or not os.path.exists(preview_file):
        return False
    try:
        from PIL import Image
        img = Image.open(preview_file)
        try:
            info = img.info or {}
            for key in ('parameters', 'UserComment'):
                value = info.get(key)
                if value and str(value).strip():
                    return True
            exif = info.get('exif')
            if exif:
                import piexif
                import piexif.helper
                try:
                    parsed = piexif.load(exif)
                    raw = parsed.get('Exif', {}).get(piexif.ExifIFD.UserComment)
                    if raw:
                        try:
                            decoded = piexif.helper.UserComment.load(raw)
                        except Exception:
                            decoded = ''
                        if decoded and decoded.strip():
                            return True
                except Exception:
                    pass
        finally:
            img.close()
    except Exception:
        pass
    return False


VIDEO_PREVIEW_EXTENSIONS = ('.mp4', '.webm')
UI_PREVIEW_EXTS = ('jpg', 'jpeg', 'png', 'webp')
UI_PREVIEW_MIDS = ('.thumb.', '.', '.preview.')


def find_ui_preview_file(model_path: str) -> str | None:
    """Return the preview file the modernUI surfaces for `model_path`.

    Iteration mirrors `ExtraNetworksPage.find_preview` at
    `modules/ui_extra_networks.py:488` so that backfill embeds into the
    same file the UI reads.
    """
    base = os.path.splitext(model_path)[0]
    for ext in UI_PREVIEW_EXTS:
        for mid in UI_PREVIEW_MIDS:
            candidate = f'{base}{mid}{ext}'
            if os.path.isfile(candidate):
                return candidate
    return None


def backfill_preview_parameters(model_path: str, preview_url: str, meta: dict | None) -> bool:
    """Embed Civitai meta into an existing preview file when it lacks parameters.

    Used by the rescan path to retroactively populate preview metadata
    without re-downloading bytes. The embed target is the file
    `find_ui_preview_file` surfaces, which matches what the modernUI
    displays. For video previews the embed target is the extracted
    `<base>.thumb.jpg` frame instead of the unembeddable video file.
    Returns True only if a new chunk was written; False for no-op (file
    missing, no meta, already populated, embed failed).
    """
    if not meta or not model_path or not preview_url:
        return False
    ext = os.path.splitext(preview_url)[1].lower()
    base = os.path.splitext(model_path)[0]
    if ext in VIDEO_PREVIEW_EXTENSIONS:
        if not os.path.exists(base + ext):
            return False
        preview_file = base + '.thumb.jpg'
        if not os.path.exists(preview_file):
            return False
    else:
        preview_file = find_ui_preview_file(model_path)
        if not preview_file:
            return False
    if preview_has_parameters(preview_file):
        return False
    parameters = civitai_meta_to_parameters(meta)
    if not parameters:
        return False
    if embed_preview_parameters(preview_file, parameters):
        log.info(f'CivitAI preview backfill: file="{preview_file}"')
        return True
    return False


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


def download_civit_preview(model_path: str, preview_url: str, meta: dict | None = None):
    if model_path is None:
        return 500, '', ''
    ext = os.path.splitext(preview_url)[1]
    preview_file = os.path.splitext(model_path)[0] + ext
    is_video = preview_file.lower().endswith(VIDEO_PREVIEW_EXTENSIONS)
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
            if meta:
                thumb_file = os.path.splitext(preview_file)[0] + '.thumb.jpg'
                if os.path.exists(thumb_file):
                    try:
                        parameters = civitai_meta_to_parameters(meta)
                        if parameters and embed_preview_parameters(thumb_file, parameters):
                            log.debug(f'CivitAI preview embed: file="{thumb_file}"')
                    except Exception as e:
                        log.debug(f'CivitAI preview embed skipped: file="{thumb_file}" {e}')
        else:
            from PIL import Image
            img = Image.open(preview_file)
            log.info(f'CivitAI download: url={preview_url} file="{preview_file}" size={total_size} image={img.size}')
            img.close()
            if meta:
                try:
                    parameters = civitai_meta_to_parameters(meta)
                    if parameters and embed_preview_parameters(preview_file, parameters):
                        log.debug(f'CivitAI preview embed: file="{preview_file}"')
                except Exception as e:
                    log.debug(f'CivitAI preview embed skipped: file="{preview_file}" {e}')
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
