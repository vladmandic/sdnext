"""V1 tag autocomplete endpoints.

Serves pre-built tag files (Danbooru, e621, natural language, artists)
from JSON files in the configured autocomplete directory. Remote files
are hosted on HuggingFace and downloaded on demand.
"""

import asyncio
import collections
import json
import os

from fastapi.exceptions import HTTPException

from modules.api.models import ItemAutocomplete, ItemAutocompleteContent, ItemAutocompleteRemote
from modules.logger import log


autocomplete_dir: str = ""
# LRU cap. Realistic usage enables up to ~16 dictionaries at once; the bound also protects
# against bloat when users disable/re-enable many dicts in one session.
CACHE_MAX_ENTRIES = 16
cache: collections.OrderedDict[str, dict] = collections.OrderedDict()
HF_REPO = "CalamitousFelicitousness/prompt-vocab"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
MANIFEST_CACHE_SEC = 300  # re-fetch manifest every 5 minutes
manifest_cache: dict = {}  # {"data": [...], "fetched_at": float}


def init(path: str) -> None:
    """Set the autocomplete directory path. Called once during API registration."""
    global autocomplete_dir # pylint: disable=global-statement
    autocomplete_dir = path


def get_cached(name: str) -> dict:
    """Load a tag file, returning cached version if file hasn't changed."""
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    path = os.path.join(autocomplete_dir, f"{name}.json")
    if not os.path.isfile(path):
        cache.pop(name, None)
        # Auto-download from HF if available in manifest
        try:
            manifest = fetch_manifest_sync()
            if any(e.get('name') == name for e in manifest):
                log.info(f'Autocomplete: name="{name}" auto-download')
                download_sync(name)
            else:
                raise HTTPException(status_code=404, detail=f"Not found: {name}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Not found: {name} ({e})") from e
    stat = os.stat(path)
    # Translations live in an optional companion file; its mtime is folded into the cache key
    # so edits to either file invalidate a stale entry.
    translations_path = os.path.join(autocomplete_dir, f"{name}.translations.json")
    translations_mtime = os.stat(translations_path).st_mtime if os.path.isfile(translations_path) else 0.0
    entry = cache.get(name)
    if entry and entry['mtime'] == stat.st_mtime and entry.get('translations_mtime', 0.0) == translations_mtime:
        cache.move_to_end(name)
        return entry
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if translations_mtime:
        try:
            with open(translations_path, encoding='utf-8') as tf:
                translations = json.load(tf)
            if isinstance(translations, dict):
                data['translations'] = translations
        except Exception as e:
            log.warning(f'Autocomplete: failed to load translations for "{name}": {e}')
    entry = {
        'mtime': stat.st_mtime,
        'translations_mtime': translations_mtime,
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
    cache.move_to_end(name)
    while len(cache) > CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
    return entry


def list_all_sync() -> list[ItemAutocomplete]:
    """Scan autocomplete directory and return metadata for each tag file."""
    if not autocomplete_dir or not os.path.isdir(autocomplete_dir):
        return []
    items = []
    for filename in sorted(os.listdir(autocomplete_dir)):
        if not filename.endswith('.json') or filename.startswith('.') or filename == 'manifest.json':
            continue
        name = filename.rsplit('.', 1)[0]
        try:
            entry = get_cached(name)
            meta = entry['meta']
            items.append(ItemAutocomplete(
                name=meta['name'],
                version=meta['version'],
                tag_count=meta['tag_count'],
                categories=meta['categories'],
                size=entry['size'],
            ))
        except Exception:
            pass
    return items


async def list_all() -> list[ItemAutocomplete]:
    """List available tag autocomplete files."""
    return await asyncio.to_thread(list_all_sync)


async def get_content(name: str) -> ItemAutocompleteContent:
    """Get full tag file content by name."""
    def _load():
        return get_cached(name)
    try:
        entry = await asyncio.to_thread(_load)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    content = entry['content']
    return ItemAutocompleteContent(
        name=content.get('name', name),
        version=content.get('version', ''),
        categories=content.get('categories', {}),
        tags=content.get('tags', []),
        translations=content.get('translations'),
    )


# -- Remote management --

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
        entries = data.get('entries', data) if isinstance(data, dict) else data
        manifest_cache['data'] = entries
        manifest_cache['fetched_at'] = now
        return entries
    except Exception as e:
        # Zero the timestamp so the next call retries immediately instead of serving the
        # last-known-good payload for the full 5-minute window after a transient failure.
        manifest_cache['fetched_at'] = 0
        log.warning(f"Autocomplete: Failed to fetch manifest: {e}")
        return manifest_cache.get('data', [])


def local_names() -> set[str]:
    """Return set of locally available autocomplete file names."""
    if not autocomplete_dir or not os.path.isdir(autocomplete_dir):
        return set()
    return {
        f.rsplit('.', 1)[0]
        for f in os.listdir(autocomplete_dir)
        if f.endswith('.json') and not f.startswith('.') and f != 'manifest.json'
    }


def local_version(name: str) -> str:
    """Return the version string of a local tag file, or empty string."""
    path = os.path.join(autocomplete_dir, f"{name}.json")
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


async def list_remote() -> list[ItemAutocompleteRemote]:
    """List tag files available for download from HuggingFace."""
    entries = await asyncio.to_thread(fetch_manifest_sync)
    local = await asyncio.to_thread(local_names)
    results = []
    for e in entries:
        name = e['name']
        is_local = name in local
        remote_version = e.get('version', '')
        update = False
        if is_local and remote_version:
            lv = await asyncio.to_thread(local_version, name)
            update = bool(lv and lv != remote_version)
        results.append(ItemAutocompleteRemote(
            name=name,
            description=e.get('description', ''),
            version=remote_version,
            tag_count=e.get('tag_count', 0),
            size_mb=e.get('size_mb', 0),
            downloaded=is_local,
            update_available=update,
        ))
    return results


def download_sync(name: str) -> str:
    """Download a tag file from HuggingFace to the local autocomplete directory.
    If the manifest entry declares `translations: true`, fetches the `{name}.translations.json`
    companion too. Companion failure is logged but does not fail the primary download.
    """
    import requests
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    os.makedirs(autocomplete_dir, exist_ok=True)
    url = f"{HF_BASE}/{name}.json"
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download {name}: {e}") from e
    target = os.path.join(autocomplete_dir, f"{name}.json")
    tmp = target + ".tmp"
    size = 0
    with open(tmp, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
            size += len(chunk)
    os.replace(tmp, target)
    cache.pop(name, None)
    log.info(f'Autocomplete: name="{name}" url={url} ({size / 1024 / 1024:.2f}MB) downloaded')
    # Optional companion translations file. Manifest flag controls whether to attempt the download.
    manifest_entry = next((e for e in manifest_cache.get('data', []) if e.get('name') == name), None)
    if manifest_entry and manifest_entry.get('translations'):
        tr_url = f"{HF_BASE}/{name}.translations.json"
        tr_target = os.path.join(autocomplete_dir, f"{name}.translations.json")
        try:
            tr_resp = requests.get(tr_url, timeout=60)
            tr_resp.raise_for_status()
            tr_tmp = tr_target + ".tmp"
            with open(tr_tmp, 'wb') as f:
                f.write(tr_resp.content)
            os.replace(tr_tmp, tr_target)
            log.info(f'Autocomplete: name="{name}" translations downloaded')
        except Exception as e:
            log.warning(f'Autocomplete: failed to fetch translations for "{name}": {e}')
    return target


async def download(name: str):
    """Download a tag file from HuggingFace."""
    await asyncio.to_thread(download_sync, name)
    entry = await asyncio.to_thread(get_cached, name)
    meta = entry['meta']
    return ItemAutocomplete(
        name=meta['name'],
        version=meta['version'],
        tag_count=meta['tag_count'],
        categories=meta['categories'],
        size=entry['size'],
    )


async def delete(name: str):
    """Delete a locally downloaded tag file."""
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    path = os.path.join(autocomplete_dir, f"{name}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Not found: {name}")
    await asyncio.to_thread(os.remove, path)
    cache.pop(name, None)
    return {"status": "deleted", "name": name}


def register_api(api):
    api.add_api_route("/sdapi/v1/autocomplete", list_all, methods=["GET"], response_model=list[ItemAutocomplete], tags=["Enumerators"])
    api.add_api_route("/sdapi/v1/autocomplete/remote", list_remote, methods=["GET"], response_model=list[ItemAutocompleteRemote], tags=["Enumerators"])
    api.add_api_route("/sdapi/v1/autocomplete/{name}", get_content, methods=["GET"], response_model=ItemAutocompleteContent, tags=["Enumerators"])
    api.add_api_route("/sdapi/v1/autocomplete/{name}/download", download, methods=["POST"], response_model=ItemAutocomplete, tags=["Enumerators"])
    api.add_api_route("/sdapi/v1/autocomplete/{name}", delete, methods=["DELETE"], tags=["Enumerators"])
