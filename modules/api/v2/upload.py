import os
import time
import uuid
import threading
from typing import Optional
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from modules import shared
from modules.logger import log


class UploadEntry:
    __slots__ = ('content_type', 'created', 'name', 'path', 'ref_id', 'size')

    def __init__(self, ref_id: str, path: str, name: str, size: int, content_type: str):
        self.ref_id = ref_id
        self.path = path
        self.name = name
        self.size = size
        self.content_type = content_type
        self.created = time.time()


class UploadStore:
    def __init__(self, staging_dir: str, ttl: int = 1800):
        self.staging_dir = staging_dir
        self.ttl = ttl
        self._entries: dict[str, UploadEntry] = {}
        self._lock = threading.Lock()
        os.makedirs(staging_dir, exist_ok=True)

    def store(self, data: bytes, original_name: str, content_type: str) -> UploadEntry:
        ref_id = uuid.uuid4().hex[:16]
        ext = os.path.splitext(original_name)[1] or '.png'
        path = os.path.join(self.staging_dir, f"{ref_id}{ext}")
        with open(path, 'wb') as f:
            f.write(data)
        entry = UploadEntry(ref_id=ref_id, path=path, name=original_name, size=len(data), content_type=content_type)
        with self._lock:
            self._entries[ref_id] = entry
        return entry

    def get(self, ref_id: str) -> Optional[UploadEntry]:
        with self._lock:
            entry = self._entries.get(ref_id)
        if entry and os.path.isfile(entry.path):
            return entry
        return None

    def resolve_to_image(self, ref_id: str) -> Optional[Image.Image]:
        entry = self.get(ref_id)
        if entry is None:
            return None
        return Image.open(entry.path)

    def remove(self, ref_id: str):
        with self._lock:
            entry = self._entries.pop(ref_id, None)
        if entry:
            try:
                os.remove(entry.path)
            except OSError:
                pass

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = []
        with self._lock:
            for ref_id, entry in self._entries.items():
                if now - entry.created > self.ttl:
                    expired.append(ref_id)
            for ref_id in expired:
                self._entries.pop(ref_id, None)
        removed = 0
        for ref_id in expired:
            # entry was already popped, find path by convention
            for fname in os.listdir(self.staging_dir):
                if fname.startswith(ref_id):
                    try:
                        os.remove(os.path.join(self.staging_dir, fname))
                        removed += 1
                    except OSError:
                        pass
        return removed


# Module-level singleton
upload_store: Optional[UploadStore] = None


def init_upload_store(staging_dir: str, ttl: int = 1800):
    global upload_store
    upload_store = UploadStore(staging_dir, ttl)
    log.debug(f'Upload store: dir={staging_dir} ttl={ttl}s')


def get_upload_store() -> UploadStore:
    if upload_store is None:
        raise RuntimeError("Upload store not initialized")
    return upload_store


# Pydantic models

class UploadRef(BaseModel):
    ref: str
    name: str
    size: int
    url: str


class UploadResponse(BaseModel):
    uploads: list[UploadRef]


# Router

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILES_PER_REQUEST = 20

upload_router = APIRouter(prefix="/sdapi/v2", tags=["v2-upload"])


@upload_router.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile]):
    store = get_upload_store()
    store.cleanup_expired()
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES_PER_REQUEST} files per request")
    refs: list[UploadRef] = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
        entry = store.store(data, f.filename or "upload.png", f.content_type or "image/png")
        refs.append(UploadRef(
            ref=f"upload:{entry.ref_id}",
            name=entry.name,
            size=entry.size,
            url=f"/sdapi/v2/uploads/{entry.ref_id}",
        ))
    return UploadResponse(uploads=refs)


@upload_router.get("/uploads/{ref_id}")
async def get_upload(ref_id: str):
    store = get_upload_store()
    entry = store.get(ref_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    return FileResponse(entry.path, media_type=entry.content_type, filename=entry.name)


@upload_router.delete("/uploads/{ref_id}")
async def delete_upload(ref_id: str):
    store = get_upload_store()
    entry = store.get(ref_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    store.remove(ref_id)
    return {"status": "ok"}
