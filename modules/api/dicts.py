"""V1 dictionary / tag autocomplete endpoints.

Serves pre-built tag dictionaries (Danbooru, e621, natural language, artists)
from JSON files in the configured dicts directory. Remote dicts are hosted
on HuggingFace and downloaded on demand.
"""

import asyncio
import json
import os

from fastapi.exceptions import HTTPException

from modules.api.models import ItemDict, ItemDictContent, ItemDictRemote
from modules.logger import log

dicts_dir: str = ""
cache: dict[str, dict] = {}

HF_REPO = "CalamitousFelicitousness/prompt-vocab"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
MANIFEST_CACHE_SEC = 300  # re-fetch manifest every 5 minutes
manifest_cache: dict = {}  # {"data": [...], "fetched_at": float}


def init(path: str) -> None:
    """Set the dicts directory path. Called once during API registration."""
    global dicts_dir  # noqa: PLW0603
    dicts_dir = path


def get_cached(name: str) -> dict:
    """Load a dict file, returning cached version if file hasn't changed."""
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail="Invalid dict name")
    path = os.path.join(dicts_dir, f"{name}.json")
    if not os.path.isfile(path):
        cache.pop(name, None)
        raise HTTPException(status_code=404, detail=f"Dict not found: {name}")
    stat = os.stat(path)
    entry = cache.get(name)
    if entry and entry['mtime'] == stat.st_mtime:
        return entry
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    entry = {
        'mtime': stat.st_mtime,
        'size': stat.st_size,
        'meta': {
            'name': data.get('name', name),
            'version': data.get('version', ''),
            'tag_count': len(data.get('tags', [])),
            'categories': {
                str(k): v.get('name', str(k)) if isinstance(v, dict) else str(v)
                for k, v in data.get('categories', {}).items()
            },
        },
        'content': data,
    }
    cache[name] = entry
    return entry


def list_dicts_sync() -> list[ItemDict]:
    """Scan dicts directory and return metadata for each dict file."""
    if not dicts_dir or not os.path.isdir(dicts_dir):
        return []
    items = []
    for filename in sorted(os.listdir(dicts_dir)):
        if not filename.endswith('.json') or filename.startswith('.') or filename == 'manifest.json':
            continue
        name = filename.rsplit('.', 1)[0]
        try:
            entry = get_cached(name)
            meta = entry['meta']
            items.append(ItemDict(
                name=meta['name'],
                version=meta['version'],
                tag_count=meta['tag_count'],
                categories=meta['categories'],
                size=entry['size'],
            ))
        except Exception:
            pass
    return items


async def list_dicts() -> list[ItemDict]:
    """List available tag dictionaries."""
    return await asyncio.to_thread(list_dicts_sync)


async def get_dict(name: str) -> ItemDictContent:
    """Get full dict content by name."""
    def _load():
        return get_cached(name)
    try:
        entry = await asyncio.to_thread(_load)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    content = entry['content']
    return ItemDictContent(
        name=content.get('name', name),
        version=content.get('version', ''),
        categories=content.get('categories', {}),
        tags=content.get('tags', []),
    )


# ── Remote dict management ──

def fetch_manifest_sync() -> list[dict]:
    """Fetch manifest.json from HuggingFace, with caching."""
    import time
    import requests
    now = time.time()
    if manifest_cache.get('data') and now - manifest_cache.get('fetched_at', 0) < MANIFEST_CACHE_SEC:
        return manifest_cache['data']
    url = f"{HF_BASE}/manifest.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get('dicts', data) if isinstance(data, dict) else data
        manifest_cache['data'] = entries
        manifest_cache['fetched_at'] = now
        return entries
    except Exception as e:
        log.warning(f"Failed to fetch dict manifest: {e}")
        return manifest_cache.get('data', [])


def local_dict_names() -> set[str]:
    """Return set of locally available dict names."""
    if not dicts_dir or not os.path.isdir(dicts_dir):
        return set()
    return {
        f.rsplit('.', 1)[0]
        for f in os.listdir(dicts_dir)
        if f.endswith('.json') and not f.startswith('.') and f != 'manifest.json'
    }


def local_dict_version(name: str) -> str:
    """Return the version string of a locally downloaded dict, or empty string."""
    path = os.path.join(dicts_dir, f"{name}.json")
    if not os.path.isfile(path):
        return ""
    try:
        entry = cache.get(name)
        if entry:
            return entry['meta'].get('version', '')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('version', '')
    except Exception:
        return ""


async def list_remote() -> list[ItemDictRemote]:
    """List dicts available for download from HuggingFace."""
    entries = await asyncio.to_thread(fetch_manifest_sync)
    local = await asyncio.to_thread(local_dict_names)
    results = []
    for e in entries:
        name = e['name']
        is_local = name in local
        remote_version = e.get('version', '')
        update = False
        if is_local and remote_version:
            local_version = await asyncio.to_thread(local_dict_version, name)
            update = bool(local_version and local_version != remote_version)
        results.append(ItemDictRemote(
            name=name,
            description=e.get('description', ''),
            version=remote_version,
            tag_count=e.get('tag_count', 0),
            size_mb=e.get('size_mb', 0),
            downloaded=is_local,
            update_available=update,
        ))
    return results


def download_dict_sync(name: str) -> str:
    """Download a dict file from HuggingFace to the local dicts directory."""
    import requests
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail="Invalid dict name")
    os.makedirs(dicts_dir, exist_ok=True)
    url = f"{HF_BASE}/{name}.json"
    log.info(f"Downloading dict: {url}")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download {name}: {e}") from e
    target = os.path.join(dicts_dir, f"{name}.json")
    tmp = target + ".tmp"
    size = 0
    with open(tmp, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
            size += len(chunk)
    os.replace(tmp, target)
    cache.pop(name, None)
    log.info(f"Downloaded dict: {name} ({size / 1024 / 1024:.1f} MB)")
    return target


async def download_dict(name: str):
    """Download a dict from HuggingFace."""
    path = await asyncio.to_thread(download_dict_sync, name)
    entry = await asyncio.to_thread(get_cached, name)
    meta = entry['meta']
    return ItemDict(
        name=meta['name'],
        version=meta['version'],
        tag_count=meta['tag_count'],
        categories=meta['categories'],
        size=entry['size'],
    )


async def delete_dict(name: str):
    """Delete a locally downloaded dict."""
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail="Invalid dict name")
    path = os.path.join(dicts_dir, f"{name}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Dict not found: {name}")
    await asyncio.to_thread(os.remove, path)
    cache.pop(name, None)
    return {"status": "deleted", "name": name}


def register_api(app):
    app.add_api_route("/sdapi/v1/dicts", list_dicts, methods=["GET"], response_model=list[ItemDict], tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/dicts/remote", list_remote, methods=["GET"], response_model=list[ItemDictRemote], tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/dicts/{name}", get_dict, methods=["GET"], response_model=ItemDictContent, tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/dicts/{name}/download", download_dict, methods=["POST"], response_model=ItemDict, tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/dicts/{name}", delete_dict, methods=["DELETE"], tags=["Enumerators"])
