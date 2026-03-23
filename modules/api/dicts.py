"""V1 dictionary / tag autocomplete endpoints.

Serves pre-built tag dictionaries (Danbooru, e621, natural language, artists)
from JSON files in the configured dicts directory.
"""

import asyncio
import json
import os
from datetime import datetime

from fastapi.exceptions import HTTPException

from modules.api.models import ItemDict, ItemDictContent

dicts_dir: str = ""
cache: dict[str, dict] = {}


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
        if not filename.endswith('.json') or filename.startswith('.'):
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


def register_api(app):
    app.add_api_route("/sdapi/v1/dicts", list_dicts, methods=["GET"], response_model=list[ItemDict], tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/dicts/{name}", get_dict, methods=["GET"], response_model=ItemDictContent, tags=["Enumerators"])
