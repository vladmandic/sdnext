import os
import re
import time
from modules.shared import log, opts, readfile, max_workers
from modules.civitai.client_civitai import client


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

    log.debug('CivitAI update metadata: models')
    from modules import ui_extra_networks
    from modules.civitai.download_civitai import download_civit_meta
    pages = ui_extra_networks.get_pages('Model')
    if len(pages) == 0:
        return 'CivitAI update metadata: no models found'
    page: ui_extra_networks.ExtraNetworksPage = pages[0]
    results = []
    all_hashes = [(item.get('hash', None) or 'XXXXXXXX').upper()[:8] for item in page.list_items()]
    for item in page.list_items():
        model = CivitModel(name=item['name'], fn=item['filename'], sha=item.get('hash', None), meta=item.get('metadata', {}))
        if model.sha is None or len(model.sha) == 0:
            log.debug(f'CivitAI skip search: name="{model.name}" hash=None')
        else:
            version = client.get_version_by_hash(model.sha)
            if version is not None:
                model.id = version.model_id
                download_civit_meta(model.fn, model.id)
                fn = os.path.splitext(item['filename'])[0] + '.json'
                model.meta = readfile(fn, silent=True, as_type="dict")
                model.name = model.meta.get('name', model.name)
                model.versions = len(model.meta.get('modelVersions', []))
            time.sleep(0.25)  # rate limiting
        versions = model.meta.get('modelVersions', [])
        if len(versions) > 0:
            model.latest = versions[0].get('name', '')
            model.latest_hashes.clear()
            for v in versions[0].get('files', []):
                for h in v.get('hashes', {}).values():
                    model.latest_hashes.append(h[:8].upper())
        for ver in versions:
            for f in ver.get('files', []):
                for h in f.get('hashes', {}).values():
                    if h[:8].upper() == model.sha[:8].upper():
                        model.vername = ver.get('name', '')
                        model.url = f.get('downloadUrl', None)
                        model.latest_name = f.get('name', '')
                        if model.vername == model.latest:
                            model.status = 'Latest version'
                        elif any(map(lambda v: v in model.latest_hashes, all_hashes)): # pylint: disable=cell-var-from-loop # noqa: C417
                            model.status = 'Update downloaded'
                        else:
                            model.status = 'Update available'
                        break
        results.append(model)
        yield results if raw else create_update_metadata_table(results)
    yield results if raw else create_update_metadata_table(results)


def atomic_civit_search_metadata(item, results):
    from modules.civitai.download_civitai import download_civit_preview, download_civit_meta
    if item is None:
        return
    try:
        meta = os.path.splitext(item['filename'])[0] + '.json'
    except Exception:
        return
    has_meta = os.path.isfile(meta) and os.stat(meta).st_size > 0
    if ('missing.png' in item['preview'] or not has_meta) and os.path.isfile(item['filename']):
        sha = item.get('hash', None)
        found = False
        result = {
            'id': '',
            'name': item['name'],
            'type': '',
            'hash': '',
            'code': '',
            'size': '',
            'note': '',
        }
        if sha is not None and len(sha) > 0:
            version = client.get_version_by_hash(sha)
            result['hash'] = sha
            if version is not None:
                result['code'] = 200
                result['code'], result['size'], result['note'] = download_civit_meta(item['filename'], version.model_id)
                result['id'] = version.model_id
                result['type'] = 'metadata'
                # Create a new dict for each append to avoid mutation bugs
                results.append(dict(result))
                for img in version.images:
                    if img.url:
                        code, size, note = download_civit_preview(item['filename'], img.url)
                        if code == 200:
                            results.append({**result, 'code': code, 'size': size, 'note': note, 'type': 'preview'})
                            found = True
                            break
            else:
                result['code'] = 404
            time.sleep(0.25)  # rate limiting
        if not found and os.stat(item['filename']).st_size < (1024 * 1024 * 1024):
            from modules import hashes
            sha = hashes.calculate_sha256(item['filename'], quiet=True)[:10]
            version = client.get_version_by_hash(sha)
            result['hash'] = sha
            if version is not None:
                result['code'] = 200
                result['code'], result['size'], result['note'] = download_civit_meta(item['filename'], version.model_id)
                result['id'] = version.model_id
                result['type'] = 'metadata'
                results.append(dict(result))
                for img in version.images:
                    if img.url:
                        code, size, note = download_civit_preview(item['filename'], img.url)
                        if code == 200:
                            results.append({**result, 'code': code, 'size': size, 'note': note, 'type': 'preview'})
                            found = True
                            break
            else:
                result['code'] = 404
            time.sleep(0.25)  # rate limiting
        if not found:
            results.append(dict(result))


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

    from modules.ui_extra_networks import get_pages
    results = []
    scanned, skipped = 0, 0
    t0 = time.time()
    candidates = []
    re_skip = [r.strip() for r in opts.extra_networks_scan_skip.split(',') if len(r.strip()) > 0]
    for page in get_pages():
        if isinstance(title, str):
            if page.title.lower() != title.lower():
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
            candidates.append(item)
    log.debug(f'CivitAI search metadata: type={title if isinstance(title, str) else "all"} workers={max_workers} skip={len(re_skip)} items={len(candidates)}')
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_items = {}
        for candidate in candidates:
            future_items[executor.submit(atomic_civit_search_metadata, candidate, results)] = candidate
        for future in concurrent.futures.as_completed(future_items):
            future.result()
            yield results if raw else create_search_metadata_table(results)

    t1 = time.time()
    log.debug(f'CivitAI search metadata: scanned={scanned} skipped={skipped} time={t1 - t0:.2f}')
    yield results if raw else create_search_metadata_table(results)
