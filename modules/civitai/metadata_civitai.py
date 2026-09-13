import os
import re
import time
import threading
import concurrent.futures
from modules.shared import log, opts, max_workers, state, cmd_opts
from modules.civitai.client_civitai import client
from modules.civitai.filemanage_civitai import hash_cache_title


GIB = 1024 ** 3
sweep_lock = threading.Lock()


class SweepBusy(Exception):
    """A metadata sweep was started while another one was running."""


class CivitModel:
    def __init__(self, name, fn, sha=None, meta=None):
        if meta is None:
            meta = {}
        self.name = name
        self.file = name
        self.id = meta.get('id', 0)
        self.fn = fn
        self.sha = sha
        self.meta = meta
        self.versions = 0
        self.vername = ''
        self.latest = ''
        self.latest_hashes = []
        self.latest_name = ''
        self.url = None
        self.status = 'Not found'


PAGE_KINDS = {'model': 'checkpoint', 'lora': 'lora', 'unet/dit': 'unet', 'vae': 'vae'}


def cache_title(page: str, item: dict) -> str | None:
    """Hash cache key read by the page's own loader, or None when it keeps no cache entry."""
    kind = PAGE_KINDS.get(page)
    return hash_cache_title(kind, item.get('filename') or '', name=item.get('name') if kind in ('checkpoint', 'unet') else None)


def resolve_sha256(entries: list[tuple[str, dict]], size_limit: int | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """File SHA256 per filename plus a note per unresolved file, from the default hash store or by hashing files below size_limit.

    The hashes-addnet store and sshs_model_hash are kohya tensor hashes, which CivitAI never matches.
    """
    from modules import hashes
    resolved, notes, todo, seen = {}, {}, [], set()
    for page, item in entries:
        fn = item.get('filename') or ''
        if fn in seen or not os.path.isfile(fn):
            continue
        seen.add(fn)
        title = cache_title(page, item)
        sha = hashes.sha256_from_cache(fn, title) if title else None
        if sha:
            resolved[fn] = sha.lower()
        elif size_limit is not None and os.path.getsize(fn) >= size_limit:
            notes[fn] = f'not hashed: {size_limit // GIB} GiB or larger'
        else:
            todo.append((fn, title))
    if len(todo) == 0:
        return resolved, notes
    if cmd_opts.no_hashing:
        log.warning(f'CivitAI metadata: unhashed={len(todo)} hashing disabled')
        notes.update({fn: 'not hashed: hashing disabled' for fn, _title in todo})
        return resolved, notes

    def hash_file(fn: str) -> str | None:
        return None if state.interrupted else hashes.calculate_sha256(fn, quiet=True)

    log.info(f'CivitAI metadata: hashing files={len(todo)} size={sum(os.path.getsize(fn) for fn, _title in todo) / GIB:.1f}GB')
    jobid = state.begin('CivitAI hash')
    state.job_count = len(todo)
    cached = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(hash_file, fn): (fn, title) for fn, title in todo}
            for future in concurrent.futures.as_completed(futures):
                fn, title = futures[future]
                state.job_no += 1
                state.textinfo = os.path.basename(fn)
                try:
                    sha = future.result()
                except Exception as e:
                    log.error(f'CivitAI metadata hash: file="{fn}" {e}')
                    notes[fn] = 'not hashed: unreadable'
                    continue
                if sha is None:
                    notes[fn] = 'not hashed: interrupted'
                    continue
                resolved[fn] = sha.lower()
                if title is not None:
                    hashes.cache().add_hash(title, os.path.getmtime(fn), resolved[fn])
                    cached += 1
    finally:
        if cached > 0:
            hashes.save_cache()
        state.end(jobid)
    return resolved, notes


def apply_update_status(model: CivitModel, sha: str, versions: list[dict], local_hashes: set[str]):
    if len(versions) == 0:
        return
    model.latest = versions[0].get('name', '')
    latest_hashes = {str((f.get('hashes') or {}).get('SHA256', '')).upper() for f in versions[0].get('files', [])} - {''}
    model.latest_hashes = sorted(latest_hashes)
    for ver in versions:
        for f in ver.get('files', []):
            if str((f.get('hashes') or {}).get('SHA256', '')).upper() != sha.upper():
                continue
            model.vername = ver.get('name', '')
            model.url = f.get('downloadUrl', None)
            model.latest_name = f.get('name', '')
            if model.vername == model.latest:
                model.status = 'Latest version'
            elif len(local_hashes & latest_hashes) > 0:
                model.status = 'Update downloaded'
            else:
                model.status = 'Update available'
            return


def civit_update_metadata(raw: bool = False):
    def create_update_metadata_table(rows: list[CivitModel]):
        html = """
            <table class="simple-table">
                <thead>
                    <tr><th>File</th><th>ID</th><th>Name</th><th>Hash</th><th>Versions</th><th>Latest</th><th>Status</th></tr>
                </thead>
                <tbody>
                    {tbody}
                </tbody>
            </table>
        """
        tbody = ''
        for row in rows:
            try:
                tbody += f"""
                    <tr>
                        <td>{row.file}</td>
                        <td>{row.id}</td>
                        <td>{row.name}</td>
                        <td>{row.sha}</td>
                        <td>{row.versions}</td>
                        <td>{row.latest}</td>
                        <td>{row.status}</td>
                    </tr>
                """
            except Exception as e:
                log.error(f'Model list: row={row} {e}')
        return html.format(tbody=tbody)

    if not sweep_lock.acquire(blocking=False): # pylint: disable=consider-using-with
        log.warning('CivitAI update metadata: another metadata sweep is running')
        if raw:
            raise SweepBusy('CivitAI metadata sweep already running')
        yield 'CivitAI update metadata: another metadata sweep is running'
        return
    try:
        log.debug('CivitAI update metadata: models')
        from modules import ui_extra_networks
        from modules.civitai.download_civitai import save_civit_meta
        pages = ui_extra_networks.get_pages('Model')
        if len(pages) == 0:
            yield [] if raw else 'CivitAI update metadata: no models found'
            return
        items = [item for item in pages[0].list_items() if item is not None]
        shas, notes = resolve_sha256([('model', item) for item in items])
        rows, failed = client.get_version_ids_by_hash(sorted(set(shas.values())))
        by_hash = {}
        for row in rows: # one hash can match several versions; GET /by-hash/{hash} returns the first
            by_hash.setdefault(str(row.get('hash', '')).lower(), row)
        metas, failed_models = client.get_models_raw(sorted({row['modelId'] for row in by_hash.values() if row.get('modelId')}))
        local_hashes = {sha.upper() for sha in shas.values()}
        results = []
        for item in items:
            fn = item['filename']
            sha = shas.get(fn)
            model = CivitModel(name=item['name'], fn=fn, sha=sha[:10] if sha else item.get('hash', None))
            if sha is None:
                model.status = 'Not hashed' if fn in notes else 'Not found'
            elif sha in failed:
                model.status = 'Lookup failed'
            elif sha in by_hash:
                model.id = by_hash[sha]['modelId']
                meta = metas.get(model.id)
                if meta is None:
                    model.status = 'Lookup failed' if model.id in failed_models else 'Not found'
                else:
                    meta_fn = save_civit_meta(fn, meta)
                    log.info(f'CivitAI download: id={model.id} file="{meta_fn}"')
                    model.meta = meta
                    model.name = meta.get('name', model.name)
                    model.versions = len(meta.get('modelVersions', []))
                    apply_update_status(model, sha, meta.get('modelVersions', []), local_hashes)
            results.append(model)
            yield results if raw else create_update_metadata_table(results)
        yield results if raw else create_update_metadata_table(results)
    finally:
        sweep_lock.release()


def needs_metadata(item: dict) -> bool:
    """True when the item's file exists and it lacks a sidecar, a preview, or preview parameters."""
    from modules.civitai.download_civitai import preview_has_parameters, resolve_preview_file
    filename = item.get('filename') or ''
    if not os.path.isfile(filename):
        return False
    meta = os.path.splitext(filename)[0] + '.json'
    if 'missing.png' in (item.get('preview') or '') or not (os.path.isfile(meta) and os.stat(meta).st_size > 0):
        return True
    actual_preview = resolve_preview_file(item)
    return bool(actual_preview) and not preview_has_parameters(actual_preview)


def download_previews(fn: str, version, result: dict) -> list[dict]:
    """Preview row for the first version image that downloads or gains embedded parameters."""
    from modules.civitai.download_civitai import download_civit_preview, backfill_preview_parameters
    for img in version.images:
        if not img.url:
            continue
        code, size, note = download_civit_preview(fn, img.url, meta=img.meta)
        if code == 200:
            return [{**result, 'code': code, 'size': size, 'note': note, 'type': 'preview'}]
        if code == 304 and backfill_preview_parameters(fn, img.url, img.meta):
            return [{**result, 'code': 200, 'size': '', 'note': 'metadata embedded', 'type': 'preview'}]
    return []


def civit_search_metadata(title: str | None = None, raw: bool = False):
    def create_search_metadata_table(rows):
        html = """
            <table class="simple-table">
                <thead><tr><th>Name</th><th>ID</th><th>Type</th><th>Code</th><th>Hash</th><th>Size</th><th>Note</th></tr></thead>
                <tbody>{tbody}</tbody>
            </table>
        """
        tbody = ''
        for row in rows:
            try:
                tbody += f"""
                    <tr>
                        <td>{row['name']}</td>
                        <td>{row['id']}</td>
                        <td>{row['type']}</td>
                        <td>{row['code']}</td>
                        <td>{row['hash']}</td>
                        <td>{row['size']}</td>
                        <td>{row['note']}</td>
                    </tr>
                """
            except Exception as e:
                log.error(f'Model list: row={row} {e}')
        return html.format(tbody=tbody)

    if not sweep_lock.acquire(blocking=False): # pylint: disable=consider-using-with
        log.warning('CivitAI search metadata: another metadata sweep is running')
        if raw:
            raise SweepBusy('CivitAI metadata sweep already running')
        yield 'CivitAI search metadata: another metadata sweep is running'
        return
    try:
        from modules.ui_extra_networks import get_pages
        from modules.civitai.download_civitai import save_civit_meta
        results = []
        scanned, skipped = 0, 0
        t0 = time.time()
        entries = []
        re_skip = [r.strip() for r in opts.extra_networks_scan_skip.split(',') if len(r.strip()) > 0]
        for page in get_pages():
            if isinstance(title, str) and page.title.lower() != title.lower():
                continue
            if page.name in ('style', 'wildcards'):
                continue
            for item in page.list_items():
                if item is None:
                    continue
                if any(re.search(re_str, item.get('name', '') + item.get('filename', '')) for re_str in re_skip):
                    skipped += 1
                    continue
                scanned += 1
                entries.append((page.name, item))
        log.debug(f'CivitAI search metadata: type={title if isinstance(title, str) else "all"} workers={max_workers} skip={len(re_skip)} items={len(entries)}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            flags = list(executor.map(lambda entry: needs_metadata(entry[1]), entries))
        entries = [entry for entry, flag in zip(entries, flags) if flag]
        shas, notes = resolve_sha256(entries, size_limit=GIB)
        versions, failed = client.get_versions_by_hash(sorted(set(shas.values())))
        by_hash = {}
        for version in versions: # one hash can match several versions; GET /by-hash/{hash} returns the first
            for f in version.files:
                if f.hashes.sha256:
                    by_hash.setdefault(f.hashes.sha256.lower(), version)
        matched = {}
        for _page, item in entries:
            fn = item['filename']
            sha = shas.get(fn)
            result = {'id': '', 'name': item['name'], 'type': '', 'hash': sha[:10] if sha else '', 'code': '', 'size': '', 'note': notes.get(fn, '')}
            if sha is None:
                results.append(result)
            elif sha in failed:
                results.append({**result, 'code': failed[sha], 'note': 'lookup failed'})
            elif sha not in by_hash:
                results.append({**result, 'code': 404})
            else:
                matched[fn] = (by_hash[sha], {**result, 'id': by_hash[sha].model_id})
        metas, failed_models = client.get_models_raw(sorted({version.model_id for version, _result in matched.values()}))
        for fn, (version, result) in matched.items():
            meta = metas.get(version.model_id)
            if meta is None:
                results.append({**result, 'type': 'metadata', 'code': failed_models.get(version.model_id, 404)})
                continue
            meta_fn = save_civit_meta(fn, meta)
            log.info(f'CivitAI download: id={version.model_id} file="{meta_fn}"')
            results.append({**result, 'type': 'metadata', 'code': 200, 'size': len(meta)})
        yield results if raw else create_search_metadata_table(results)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_previews, fn, version, result) for fn, (version, result) in matched.items()]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
                yield results if raw else create_search_metadata_table(results)
        t1 = time.time()
        log.debug(f'CivitAI search metadata: scanned={scanned} skipped={skipped} pending={len(entries)} hashed={len(shas)} matched={len(matched)} time={t1 - t0:.2f}')
        yield results if raw else create_search_metadata_table(results)
    finally:
        sweep_lock.release()
