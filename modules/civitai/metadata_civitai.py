import os
import re
import time
import gradio as gr
from modules.shared import log, opts, req, readfile, max_workers


data = []
selected_model = None


class CivitModel:
    def __init__(self, name, fn, sha = None, meta = {}):
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


def civit_update_metadata():
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
            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{model.sha}')
            log.debug(f'CivitAI search: name="{model.name}" hash={model.sha} status={r.status_code}')
            if r.status_code == 200:
                d = r.json()
                model.id = d['modelId']
                download_civit_meta(model.fn, model.id)
                fn = os.path.splitext(item['filename'])[0] + '.json'
                model.meta = readfile(fn, silent=True)
                model.name = model.meta.get('name', model.name)
                model.versions = len(model.meta.get('modelVersions', []))
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
        yield create_update_metadata_table(results)
    return create_update_metadata_table(results)


def civit_search_model(name, tag, model_type):
    # types = 'LORA' if model_type == 'LoRA' else 'Checkpoint'
    url = 'https://civitai.com/api/v1/models?limit=25&Sort=Newest'
    if model_type == 'Model':
        url += '&types=Checkpoint'
    elif model_type == 'LoRA':
        url += '&types=LORA&types=DoRA&types=LoCon'
    elif model_type == 'Embedding':
        url += '&types=TextualInversion'
    elif model_type == 'VAE':
        url += '&types=VAE'
    if name is not None and len(name) > 0:
        url += f'&query={name}'
    if tag is not None and len(tag) > 0:
        url += f'&tag={tag}'
    r = req(url)
    log.debug(f'CivitAI search: type={model_type} name="{name}" tag={tag or "none"} url="{url}" status={r.status_code}')
    if r.status_code != 200:
        log.warning(f'CivitAI search: name="{name}" tag={tag} status={r.status_code}')
        return [], gr.update(visible=False, value=[]), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    try:
        body = r.json()
    except Exception as e:
        log.error(f'CivitAI search: name="{name}" tag={tag} {e}')
        return [], gr.update(visible=False, value=[]), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    global data # pylint: disable=global-statement
    data = body.get('items', [])
    data1 = []
    for model in data:
        found = 0
        if model_type == 'LoRA' and model['type'].lower() in ['lora', 'locon', 'dora', 'lycoris']:
            found += 1
        elif model_type == 'Embedding' and model['type'].lower() in ['textualinversion', 'embedding']:
            found += 1
        elif model_type == 'Model' and model['type'].lower() in ['checkpoint']:
            found += 1
        elif model_type == 'VAE' and model['type'].lower() in ['vae']:
            found += 1
        elif model_type == 'Other':
            found += 1
        if found > 0:
            data1.append([
                model['id'],
                model['name'],
                ', '.join(model['tags']),
                model['stats']['downloadCount'],
                model['stats']['rating']
            ])
    res = f'Search result: name={name} tag={tag or "none"} type={model_type} models={len(data1)}'
    return res, gr.update(visible=len(data1) > 0, value=data1 if len(data1) > 0 else []), gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def atomic_civit_search_metadata(item, results):
    from modules.civitai.download_civitai import download_civit_preview, download_civit_meta
    if item is None:
        return
    try:
        meta = os.path.splitext(item['filename'])[0] + '.json'
    except Exception:
        # log.error(f'CivitAI search metadata: item={item} {e}')
        return
    has_meta = os.path.isfile(meta) and os.stat(meta).st_size > 0
    if ('card-no-preview.png' in item['preview'] or not has_meta) and os.path.isfile(item['filename']):
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
            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{sha}')
            log.debug(f'CivitAI search: name="{item["name"]}" hash={sha} status={r.status_code}')
            result['hash'] = sha
            result['code'] = r.status_code
            if r.status_code == 200:
                d = r.json()
                result['code'], result['size'], result['note'] = download_civit_meta(item['filename'], d['modelId'])
                result['id'] = d['modelId']
                result['type'] = 'metadata'
                results.append(result)
                if d.get('images') is not None:
                    for i in d['images']:
                        result['code'], result['size'], result['note'] = download_civit_preview(item['filename'], i['url'])
                        if result['code'] == 200:
                            result['type'] = 'preview'
                            results.append(result)
                            found = True
                            break
        if not found and os.stat(item['filename']).st_size < (1024 * 1024 * 1024):
            from modules import hashes
            sha = hashes.calculate_sha256(item['filename'], quiet=True)[:10]
            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{sha}')
            log.debug(f'CivitAI search: name="{item["name"]}" hash={sha} status={r.status_code}')
            result['hash'] = sha
            result['code'] = r.status_code
            if r.status_code == 200:
                d = r.json()
                result['code'], result['size'], result['note'] = download_civit_meta(item['filename'], d['modelId'])
                result['id'] = d['modelId']
                result['type'] = 'metadata'
                results.append(result)
                if d.get('images') is not None:
                    for i in d['images']:
                        result['code'], result['size'], result['note'] = download_civit_preview(item['filename'], i['url'])
                        if result['code'] == 200:
                            result['type'] = 'preview'
                            results.append(result)
                            found = True
                            break
        if not found:
            results.append(result)


def civit_search_metadata(title: str = None):
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
        if type(title) == str:
            if page.title != title:
                continue
        if page.name == 'style':
            continue
        for item in page.list_items():
            if item is None:
                continue
            if any(re.search(re_str, item.get('name', '') + item.get('filename', '')) for re_str in re_skip):
                skipped += 1
                continue
            scanned += 1
            candidates.append(item)
    log.debug(f'CivitAI search metadata: type={title if type(title) == str else "all"} workers={max_workers} skip={len(re_skip)} items={len(candidates)}')
    import concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_items = {}
        for fn in candidates:
            future_items[executor.submit(atomic_civit_search_metadata, fn, results)] = fn
        for future in concurrent.futures.as_completed(future_items):
            future.result()
            yield create_search_metadata_table(results)

    t1 = time.time()
    log.debug(f'CivitAI search metadata: scanned={scanned} skipped={skipped} time={t1-t0:.2f}')
    yield create_search_metadata_table(results)
