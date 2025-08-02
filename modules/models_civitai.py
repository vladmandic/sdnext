import os
import re
import time
import json
import gradio as gr
from modules.shared import log, opts, req, readfile, max_workers


data = []
selected_model = None
update_data = []


class CivitModel:
    def __init__(self, name, fn, sha = None, meta = {}):
        self.name = name
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
    def array(self):
        return [self.id, self.fn, self.name, self.versions, self.vername, self.latest, self.status]


def civit_update_metadata():
    log.debug('CivitAI update metadata: models')
    from modules import ui_extra_networks, modelloader
    res = []
    pages = ui_extra_networks.get_pages('Model')
    if len(pages) == 0:
        return 'CivitAI update metadata: no models found'
    page: ui_extra_networks.ExtraNetworksPage = pages[0]
    table_data = []
    update_data.clear()
    all_hashes = [(item.get('hash', None) or 'XXXXXXXX').upper()[:8] for item in page.list_items()]
    for item in page.list_items():
        model = CivitModel(name=item['name'], fn=item['filename'], sha=item.get('hash', None), meta=item.get('metadata', {}))
        if model.sha is None or len(model.sha) == 0:
            res.append(f'CivitAI skip search: name="{model.name}" hash=None')
        else:
            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{model.sha}')
            res.append(f'CivitAI search: name="{model.name}" hash={model.sha} status={r.status_code}')
            if r.status_code == 200:
                d = r.json()
                model.id = d['modelId']
                modelloader.download_civit_meta(model.fn, model.id)
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
                            model.status = 'Latest'
                        elif any(map(lambda v: v in model.latest_hashes, all_hashes)): # pylint: disable=cell-var-from-loop # noqa: C417
                            model.status = 'Downloaded'
                        else:
                            model.status = 'Available'
                        break
        log.debug(res[-1])
        update_data.append(model)
        table_data.append(model.array())
        yield gr.update(value=table_data), '<br>'.join([r for r in res if len(r) > 0])
    return '<br>'.join([r for r in res if len(r) > 0])

def civit_update_select(evt: gr.SelectData, in_data):
    global selected_model # pylint: disable=global-statement
    try:
        selected_model = next([m for m in update_data if m.fn == in_data[evt.index[0]][1]])
    except Exception:
        selected_model = None
    if selected_model is None or selected_model.url is None or selected_model.status != 'Available':
        return [gr.update(value='Model update not available'), gr.update(visible=False)]
    else:
        return [gr.update(), gr.update(visible=True)]

def civit_update_download():
    if selected_model is None or selected_model.url is None or selected_model.status != 'Available':
        return 'Model update not available'
    if selected_model.latest_name is None or len(selected_model.latest_name) == 0:
        model_name = f'{selected_model.name} {selected_model.latest}.safetensors'
    else:
        model_name = selected_model.latest_name
    return civit_download_model(selected_model.url, model_name, model_path='', model_type='Model')


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


def civit_select1(evt: gr.SelectData, in_data):
    model_id = in_data[evt.index[0]][0]
    data2 = []
    preview_img = None
    for model in data:
        if model['id'] == model_id:
            for d in model['modelVersions']:
                try:
                    if d.get('images') is not None and len(d['images']) > 0 and len(d['images'][0]['url']) > 0:
                        preview_img = d['images'][0]['url']
                    data2.append([d.get('id', None), d.get('modelId', None) or model_id, d.get('name', None), d.get('baseModel', None), d.get('createdAt', None) or d.get('publishedAt', None)])
                except Exception as e:
                    log.error(f'CivitAI select: model="{in_data[evt.index[0]]}" {e}')
                    log.error(f'CivitAI version data={type(d)}: {d}')
    log.debug(f'CivitAI select: model="{in_data[evt.index[0]]}" versions={len(data2)}')
    return data2, None, preview_img


def civit_select2(evt: gr.SelectData, in_data):
    variant_id = in_data[evt.index[0]][0]
    model_id = in_data[evt.index[0]][1]
    data3 = []
    for model in data:
        if model['id'] == model_id:
            for variant in model['modelVersions']:
                if variant['id'] == variant_id:
                    for f in variant['files']:
                        try:
                            if os.path.splitext(f['name'])[1].lower() in ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']:
                                data3.append([f['name'], round(f['sizeKB']), json.dumps(f['metadata']), f['downloadUrl']])
                        except Exception:
                            pass
    log.debug(f'CivitAI select: model="{in_data[evt.index[0]]}" files={len(data3)}')
    return data3


def civit_select3(evt: gr.SelectData, in_data):
    log.debug(f'CivitAI select: variant={in_data[evt.index[0]]}')
    return in_data[evt.index[0]][3], in_data[evt.index[0]][0], gr.update(interactive=True)


def civit_download_model(model_url: str, model_name: str, model_path: str, model_type: str, token: str = None):
    if model_url is None or len(model_url) == 0:
        return 'No model selected'
    try:
        from modules.modelloader import download_civit_model
        res = download_civit_model(model_url, model_name, model_path, model_type, token=token)
    except Exception as e:
        res = f"CivitAI model downloaded error: model={model_url} {e}"
        log.error(res)
        return res
    from modules.sd_models import list_models  # pylint: disable=W0621
    list_models()
    return res


def atomic_civit_search_metadata(item, res, rehash):
    from modules.modelloader import download_civit_preview, download_civit_meta
    if item is None:
        return
    meta = os.path.splitext(item['filename'])[0] + '.json'
    has_meta = os.path.isfile(meta) and os.stat(meta).st_size > 0
    if ('card-no-preview.png' in item['preview'] or not has_meta) and os.path.isfile(item['filename']):
        sha = item.get('hash', None)
        found = False
        if sha is not None and len(sha) > 0:
            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{sha}')
            log.debug(f'CivitAI search: name="{item["name"]}" hash={sha} status={r.status_code}')
            if r.status_code == 200:
                d = r.json()
                res.append(download_civit_meta(item['filename'], d['modelId']))
                if d.get('images') is not None:
                    for i in d['images']:
                        preview_url = i['url']
                        img_res = download_civit_preview(item['filename'], preview_url)
                        res.append(img_res)
                        if 'error' not in img_res:
                            found = True
                            break
        if not found and rehash and os.stat(item['filename']).st_size < (1024 * 1024 * 1024):
            from modules import hashes
            sha = hashes.calculate_sha256(item['filename'], quiet=True)[:10]
            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{sha}')
            log.debug(f'CivitAI search: name="{item["name"]}" hash={sha} status={r.status_code}')
            if r.status_code == 200:
                d = r.json()
                res.append(download_civit_meta(item['filename'], d['modelId']))
                if d.get('images') is not None:
                    for i in d['images']:
                        preview_url = i['url']
                        img_res = download_civit_preview(item['filename'], preview_url)
                        res.append(img_res)
                        if 'error' not in img_res:
                            found = True
                            break


def civit_search_metadata(rehash, title):
    log.debug(f'CivitAI search metadata: type={title if type(title) == str else "all"}')
    from modules.ui_extra_networks import get_pages
    res = []
    scanned, skipped = 0, 0
    t0 = time.time()
    candidates = []
    re_skip = [r.strip() for r in opts.extra_networks_scan_skip.split(',') if len(r.strip()) > 0]
    log.debug(f'CivitAI search metadata: skip={re_skip}')
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
            # atomic_civit_search_metadata(item, res, rehash)
    import concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fn in candidates:
            executor.submit(atomic_civit_search_metadata, fn, res, rehash)
    atomic_civit_search_metadata(None, res, rehash)
    t1 = time.time()
    log.debug(f'CivitAI search metadata: scanned={scanned} skipped={skipped} time={t1-t0:.2f}')
    txt = '<br>'.join([r for r in res if len(r) > 0])
    return txt


def civitai_update_token(token):
    log.debug('CivitAI update token')
    opts.civitai_token = token
    opts.save()
