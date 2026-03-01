"""CivitAI REST API endpoints.

Provides search, model/version lookup, download queue management,
options discovery, metadata scanning, and user data (bookmarks/bans/history).
"""

import os
from starlette.responses import JSONResponse
from modules.logger import log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_dict(model) -> dict:
    """Convert a Pydantic CivitModel to a JSON-safe dict."""
    return model.dict(by_alias=True)


def _version_to_dict(version) -> dict:
    return version.dict(by_alias=True)


# ---------------------------------------------------------------------------
# Search & Lookup
# ---------------------------------------------------------------------------

def get_search(
    query: str = '',
    tag: str = '',
    types: str = '',
    sort: str = '',
    period: str = '',
    base_models: str = '',
    nsfw: bool = None,
    limit: int = 20,
    cursor: str = None,
    username: str = '',
    favorites: bool = False,
    token: str = None,
):
    """Search CivitAI models with pagination."""
    from modules.civitai.client_civitai import client
    from modules.civitai.userdata_civitai import search_history
    bm_list = [b.strip() for b in base_models.split(',') if b.strip()] if base_models else None
    response = client.search_models(
        query=query, tag=tag, types=types, sort=sort, period=period,
        base_models=bm_list, nsfw=nsfw, limit=limit, cursor=cursor,
        username=username, favorites=favorites, token=token,
    )
    if query:
        search_history.add('query', query)
    elif tag:
        search_history.add('tag', tag)
    return response.dict(by_alias=True)


def get_model(model_id: int, token: str = None):
    """Get a single model by ID (fresh fetch)."""
    from modules.civitai.client_civitai import client
    model = client.get_model(model_id, token=token)
    if model is None:
        return JSONResponse(content={"error": "model not found"}, status_code=404)
    return _model_to_dict(model)


def get_version(version_id: int, token: str = None):
    """Get a single version by ID."""
    from modules.civitai.client_civitai import client
    version = client.get_version(version_id, token=token)
    if version is None:
        return JSONResponse(content={"error": "version not found"}, status_code=404)
    return _version_to_dict(version)


def get_version_by_hash(hash_str: str, token: str = None):
    """Look up a version by file hash."""
    from modules.civitai.client_civitai import client
    version = client.get_version_by_hash(hash_str, token=token)
    if version is None:
        return JSONResponse(content={"error": "version not found"}, status_code=404)
    return _version_to_dict(version)


def get_options():
    """Get valid types, sort, period, base_models from CivitAI API discovery."""
    from modules.civitai.client_civitai import client
    return client.discover_options()


def get_tags(query: str = '', limit: int = 20, page: int = 1):
    """Search CivitAI tags."""
    from modules.civitai.client_civitai import client
    return client.get_tags(query=query, limit=limit, page=page).dict(by_alias=True)


def get_creators(query: str = '', limit: int = 20, page: int = 1):
    """Search CivitAI creators."""
    from modules.civitai.client_civitai import client
    return client.get_creators(query=query, limit=limit, page=page).dict(by_alias=True)


def get_images(model_id: int = None, model_version_id: int = None, limit: int = 20):
    """Get images with generation metadata from CivitAI."""
    from modules.civitai.client_civitai import client
    return {"items": client.get_images_raw(model_id=model_id, model_version_id=model_version_id, limit=limit)}


def get_me(token: str = None):
    """Get authenticated CivitAI user profile."""
    from modules.civitai.client_civitai import client
    profile = client.get_me(token=token)
    if profile is None:
        return JSONResponse(content={"error": "not authenticated"}, status_code=401)
    return profile.dict(by_alias=True)


# ---------------------------------------------------------------------------
# Download Queue
# ---------------------------------------------------------------------------

def post_download(request: dict):
    """Queue a download. Returns the download item with its ID."""
    from modules.civitai.download_civitai import download_manager
    url = request.get('url', '')
    filename = request.get('filename', '')
    folder = request.get('folder', '')
    model_type = request.get('model_type', 'Checkpoint')
    expected_hash = request.get('expected_hash', '')
    token = request.get('token', None)
    model_name = request.get('model_name', '')
    base_model = request.get('base_model', '')
    creator = request.get('creator', '')
    model_id = request.get('model_id', 0)
    version_id = request.get('version_id', 0)
    version_name = request.get('version_name', '')
    nsfw = request.get('nsfw', False)
    if not url:
        return JSONResponse(content={"error": "url is required"}, status_code=400)
    if not folder:
        from modules.civitai.filemanage_civitai import resolve_save_path
        folder = str(resolve_save_path(
            model_type, model_name=model_name, base_model=base_model,
            nsfw=nsfw, creator=creator, model_id=model_id,
            version_id=version_id, version_name=version_name,
        ))
    elif not os.path.isabs(folder):
        from modules import paths
        folder = os.path.join(paths.models_path, folder)
    item = download_manager.enqueue(
        url=url,
        folder=folder,
        filename=filename or "Unknown",
        model_type=model_type,
        expected_hash=expected_hash,
        token=token,
        model_id=int(model_id) if model_id else 0,
        version_id=int(version_id) if version_id else 0,
    )
    return item.to_dict()


def post_download_cancel(download_id: str):
    """Cancel a queued or active download."""
    from modules.civitai.download_civitai import download_manager
    result = download_manager.cancel(download_id)
    if not result:
        return JSONResponse(content={"error": "download not found or already completed"}, status_code=404)
    return {"success": True, "id": download_id}


def get_download_status():
    """Get the full download queue status."""
    from modules.civitai.download_civitai import download_manager
    return download_manager.status()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings():
    """Get CivitAI-related settings."""
    from modules import shared
    return {
        "token_configured": bool(getattr(shared.opts, 'civitai_token', '')),
        "save_subfolder_enabled": getattr(shared.opts, 'civitai_save_subfolder_enabled', False),
        "save_subfolder": getattr(shared.opts, 'civitai_save_subfolder', '{{BASEMODEL}}'),
        "save_type_folders": getattr(shared.opts, 'civitai_save_type_folders', ''),
        "discard_hash_mismatch": getattr(shared.opts, 'civitai_discard_hash_mismatch', True),
        "download_workers": getattr(shared.opts, 'civitai_download_workers', 2),
    }


def post_settings(request: dict):
    """Update CivitAI-related settings and persist to config."""
    from modules import shared
    token = request.get('token')
    save_subfolder = request.get('save_subfolder')
    discard_hash_mismatch = request.get('discard_hash_mismatch')
    if token is not None:
        if token.strip():
            from modules.civitai.client_civitai import client
            user = client.validate_token(token.strip())
            if user is None:
                return JSONResponse(content={"error": "Invalid API token"}, status_code=400)
            log.info(f'CivitAI token validated: user={user.get("username", "?")}')
        from modules import secrets_manager
        secrets_manager.set('civitai_token', token.strip())
    save_subfolder_enabled = request.get('save_subfolder_enabled')
    if save_subfolder_enabled is not None:
        shared.opts.data['civitai_save_subfolder_enabled'] = bool(save_subfolder_enabled)
    if save_subfolder is not None:
        shared.opts.data['civitai_save_subfolder'] = save_subfolder
    if discard_hash_mismatch is not None:
        shared.opts.data['civitai_discard_hash_mismatch'] = discard_hash_mismatch
    shared.opts.save(shared.config_filename)
    return get_settings()


def get_resolve_path(
    model_type: str = 'Checkpoint',
    model_name: str = '',
    base_model: str = '',
    creator: str = '',
    model_id: int = 0,
    version_id: int = 0,
    version_name: str = '',
    nsfw: bool = False,
):
    """Preview where a download would be saved."""
    from modules.civitai.filemanage_civitai import resolve_save_path
    path = resolve_save_path(
        model_type, model_name=model_name, base_model=base_model,
        nsfw=nsfw, creator=creator, model_id=model_id,
        version_id=version_id, version_name=version_name,
    )
    return {"path": str(path)}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def post_metadata_scan(request: dict = None):
    """Scan local models for CivitAI metadata. Optional ``page`` filters by network type (e.g. 'lora', 'model')."""
    from modules.civitai import metadata_civitai
    page = (request or {}).get('page', None)
    results = []
    for batch in metadata_civitai.civit_search_metadata(title=page, raw=True):
        if isinstance(batch, list):
            results = batch
    return {"results": results}


def post_metadata_update():
    """Update local metadata from CivitAI."""
    from modules.civitai import metadata_civitai
    items = []
    for batch in metadata_civitai.civit_update_metadata(raw=True):
        if isinstance(batch, list):
            items = batch
    results = []
    for item in items:
        results.append({
            "file": getattr(item, "file", None),
            "id": getattr(item, "id", None),
            "name": getattr(item, "name", None),
            "sha": getattr(item, "sha", None),
            "versions": getattr(item, "versions", None),
            "latest": getattr(item, "latest_name", None),
            "status": getattr(item, "status", None),
        })
    return {"results": results}


# ---------------------------------------------------------------------------
# User Data — Bookmarks
# ---------------------------------------------------------------------------

def get_bookmarks():
    from modules.civitai.userdata_civitai import bookmarks
    return {"bookmarks": [{"name": n} for n in bookmarks.list()]}


def post_bookmark(request: dict):
    from modules.civitai.userdata_civitai import bookmarks
    name = request.get('name', '')
    if not name:
        return JSONResponse(content={"error": "name is required"}, status_code=400)
    added = bookmarks.add(name)
    return {"success": added, "name": name}


def delete_bookmark(name: str):
    from modules.civitai.userdata_civitai import bookmarks
    removed = bookmarks.remove(name)
    if not removed:
        return JSONResponse(content={"error": "not found"}, status_code=404)
    return {"success": True, "name": name}


# ---------------------------------------------------------------------------
# User Data — Banned
# ---------------------------------------------------------------------------

def get_banned():
    from modules.civitai.userdata_civitai import banned
    return {"banned": [{"name": n} for n in banned.list()]}


def post_banned(request: dict):
    from modules.civitai.userdata_civitai import banned
    name = request.get('name', '')
    if not name:
        return JSONResponse(content={"error": "name is required"}, status_code=400)
    added = banned.add(name)
    return {"success": added, "name": name}


def delete_banned(name: str):
    from modules.civitai.userdata_civitai import banned
    removed = banned.remove(name)
    if not removed:
        return JSONResponse(content={"error": "not found"}, status_code=404)
    return {"success": True, "name": name}


# ---------------------------------------------------------------------------
# User Data — Search History
# ---------------------------------------------------------------------------

def get_history(search_type: str = None):
    from modules.civitai.userdata_civitai import search_history
    return {"history": search_history.list(search_type)}


def delete_history():
    from modules.civitai.userdata_civitai import search_history
    search_history.clear()
    return {"success": True}


# ---------------------------------------------------------------------------
# Sidecar Index (lazy-built from CivitAI .json metadata files)
# ---------------------------------------------------------------------------

_sidecar_index = None


def _build_sidecar_index():
    """Scan CivitAI .json sidecar files to build sha256 -> local file mapping. Lazy, one-time."""
    global _sidecar_index  # pylint: disable=global-statement
    if _sidecar_index is not None:
        return _sidecar_index
    _sidecar_index = {}
    import glob
    from modules import shared, paths
    model_exts = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin')
    dirs = [
        (getattr(shared.cmd_opts, 'lora_dir', None) or os.path.join(paths.models_path, 'Lora'), 'lora'),
        (getattr(shared.opts, 'ckpt_dir', None) or os.path.join(paths.models_path, 'Stable-diffusion'), 'checkpoint'),
    ]
    for model_dir, model_type in dirs:
        if not os.path.isdir(model_dir):
            continue
        for json_path in glob.glob(os.path.join(model_dir, '**', '*.json'), recursive=True):
            try:
                from modules.json_helpers import readfile
                data = readfile(json_path, silent=True, as_type="dict")
                if not isinstance(data, dict) or 'modelVersions' not in data:
                    continue
                # Find the companion model file for this sidecar
                base = os.path.splitext(json_path)[0]
                companion = None
                for ext in model_exts:
                    candidate = base + ext
                    if os.path.isfile(candidate):
                        companion = candidate
                        break
                if not companion:
                    continue
                # Match the companion file to a JSON entry by size (sizeKB)
                companion_size_kb = os.path.getsize(companion) / 1024.0
                companion_name = os.path.basename(base)
                best_sha = None
                best_diff = float('inf')
                for v in data.get('modelVersions', []):
                    for f in v.get('files', []):
                        sha = (f.get('hashes') or {}).get('SHA256')
                        size_kb = f.get('sizeKB', 0)
                        if sha and size_kb and abs(size_kb - companion_size_kb) < 1.0:
                            diff = abs(size_kb - companion_size_kb)
                            if diff < best_diff:
                                best_sha = sha
                                best_diff = diff
                if best_sha:
                    _sidecar_index[best_sha.lower()] = {"filename": companion_name, "type": model_type}
            except Exception:
                continue
    log.debug(f'CivitAI sidecar index: {len(_sidecar_index)} hashes from sidecar files')
    return _sidecar_index


def _invalidate_sidecar_index():
    global _sidecar_index  # pylint: disable=global-statement
    _sidecar_index = None


# ---------------------------------------------------------------------------
# Local Hash Check
# ---------------------------------------------------------------------------

def post_check_local(request: dict):
    """Check which SHA256 hashes correspond to locally downloaded files."""
    from modules import hashes as hash_module
    input_hashes = request.get('hashes', [])
    if not input_hashes:
        return {"found": {}}
    # Build reverse lookup: lowercase sha256 -> {filename, type}
    found = {}
    hash_cache = hash_module.cache("hashes")
    for title, entry in hash_cache.items():
        sha = entry.get("sha256")
        if not sha:
            continue
        parts = title.split("/", 1)
        file_type = parts[0] if len(parts) > 1 else "unknown"
        found[sha.lower()] = {"filename": title, "type": file_type}
    # Supplement from in-memory checkpoint registry
    try:
        from modules.sd_checkpoint import checkpoints_list
        for _title, cp in checkpoints_list.items():
            if cp.sha256:
                key = cp.sha256.lower()
                if key not in found:
                    found[key] = {"filename": cp.filename, "type": "checkpoint"}
    except Exception:
        pass
    # Supplement from in-memory LoRA registry
    try:
        from modules.lora.lora_load import available_networks
        for _name, net in available_networks.items():
            if net.hash:
                key = net.hash.lower()
                if key not in found:
                    found[key] = {"filename": net.filename, "type": "lora"}
    except Exception:
        pass
    # Supplement from sidecar index (covers files never hashed locally)
    sidecar = _build_sidecar_index()
    for h in input_hashes:
        if not h:
            continue
        key = h.lower()
        if key not in found and key in sidecar:
            found[key] = sidecar[key]
    # Match requested hashes
    result = {}
    for h in input_hashes:
        if not h:
            continue
        match = found.get(h.lower())
        if match:
            result[h] = match
    return {"found": result}


# ---------------------------------------------------------------------------
# Legacy Endpoints (backward compatibility)
# ---------------------------------------------------------------------------

def _legacy_get_civitai(
    model_id: int = None,
    query: str = '',
    tag: str = '',
    types: str = '',
    sort: str = '',
    period: str = '',
    nsfw: bool = None,
    limit: int = 0,
    base: str = '',
    token: str = None,
    exact: bool = True,
):
    """Legacy GET /sdapi/v1/civitai — delegates to search or model lookup."""
    from modules.civitai import search_civitai
    if model_id is not None:
        # Legacy model_id lookup used the global cache; now we fetch fresh
        from modules.civitai.client_civitai import client
        model = client.get_model(model_id, token=token)
        if model is None:
            return JSONResponse(content=[], status_code=200)
        return [_model_to_dict(model)]
    if query or tag:
        models = search_civitai.search_civitai(
            query=query, tag=tag, types=types, sort=sort, period=period,
            nsfw=nsfw, limit=limit, base=base, token=token, exact=exact,
        )
        return [_model_to_dict(m) for m in models]
    return JSONResponse(content=[], status_code=200)


def _legacy_post_civitai(page: str = None):
    """Legacy POST /sdapi/v1/civitai — scan metadata."""
    from modules.civitai import metadata_civitai
    result = []
    for r in metadata_civitai.civit_search_metadata(title=page, raw=True):
        result = r
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_api():
    from modules.shared import api

    # New REST API
    api.add_api_route("/sdapi/v2/civitai/search", get_search, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/model/{model_id}", get_model, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/version/{version_id}", get_version, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/version/by-hash/{hash_str}", get_version_by_hash, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/options", get_options, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/tags", get_tags, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/creators", get_creators, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/images", get_images, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/me", get_me, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/download", post_download, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/download/{download_id}/cancel", post_download_cancel, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/download/status", get_download_status, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/settings", get_settings, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/settings", post_settings, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/resolve-path", get_resolve_path, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/metadata/scan", post_metadata_scan, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/metadata/update", post_metadata_update, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/bookmarks", get_bookmarks, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/bookmarks", post_bookmark, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/bookmarks/{name}", delete_bookmark, methods=["DELETE"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/banned", get_banned, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/banned", post_banned, methods=["POST"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/banned/{name}", delete_banned, methods=["DELETE"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/history", get_history, methods=["GET"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/history", delete_history, methods=["DELETE"], tags=["CivitAI"])
    api.add_api_route("/sdapi/v2/civitai/check-local", post_check_local, methods=["POST"], tags=["CivitAI"])

    # Legacy endpoints (backward compatibility)
    api.add_api_route("/sdapi/v1/civitai", _legacy_get_civitai, methods=["GET"], response_model=list, tags=["Models"])
    api.add_api_route("/sdapi/v1/civitai", _legacy_post_civitai, methods=["POST"], response_model=list, tags=["Models"])

    log.debug('CivitAI API: registered endpoints')
