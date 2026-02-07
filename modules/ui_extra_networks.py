import os
import io
import random
import re
import time
import json
import html
import base64
import urllib.parse
import threading
from typing import TYPE_CHECKING
from types import SimpleNamespace
from pathlib import Path
from html.parser import HTMLParser
from collections import OrderedDict
import gradio as gr
from PIL import Image
from starlette.responses import FileResponse, JSONResponse
from modules import paths, shared, files_cache, errors, infotext, ui_symbols, ui_components, modelstats


allowed_dirs = []
refresh_time = 0
extra_pages = shared.extra_networks
debug = shared.log.trace if os.environ.get('SD_EN_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: EN')
card_full = '''
    <div class='card' onclick={card_click} title='{name}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-short='{short}' data-tags='{tags}' data-mtime='{mtime}' data-size='{size}' data-search='{search}' data-version='{version}' style='--data-color: {color}'>
        <div class='overlay'>
            <div class='name {reference}'>{title}</div>
        </div>
        <div class='tags'></div>
        <div class='version'>{version}</div>
        <div class='actions'>
            <span class='details' title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>
            <div class='additional'><ul></ul></div>
        </div>
        <img class='preview' src='{preview}' style='width: {width}; height: {height}; object-fit: {fit}' loading='lazy'></img>
    </div>
'''
card_list = '''
    <div class='card card-list' onclick={card_click} title='{name}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-short='{short}' data-tags='{tags}' data-mtime='{mtime}' data-version='{version}' data-size='{size}' data-search='{search}'>
        <div style='display: flex'>
            <span class='details' title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>&nbsp;
            <div class='name {reference}' style='flex-flow: column'>{title}&nbsp;
                <div class='tags tags-list'></div>
            </div>
        </div>
    </div>
'''
preview_map = None


def init_api():

    def fetch_file(filename: str = ""):
        global allowed_dirs # pylint: disable=global-statement
        if len(allowed_dirs) == 0:
            allowed_dirs = shared.demo.allowed_paths
        if filename is None or len(filename) == 0:
            return JSONResponse({ "error": "no filename" }, status_code=400)
        if not os.path.exists(filename) or not os.path.isfile(filename):
            return JSONResponse({ "error": f"file {filename}: not found" }, status_code=404)
        if filename.startswith('html/') or filename.startswith('models/'):
            return FileResponse(filename, headers={"Accept-Ranges": "bytes"})
        if not any(Path(folder).absolute() in Path(filename).absolute().parents for folder in allowed_dirs):
            return JSONResponse({ "error": f"file {filename}: must be in one of allowed directories" }, status_code=403)
        if os.path.splitext(filename)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            return JSONResponse({"error": f"file {filename}: not an image file"}, status_code=403)
        return FileResponse(filename, headers={"Accept-Ranges": "bytes"})

    def get_metadata(page: str = "", item: str = ""):
        page_dict = next(iter([x for x in shared.extra_networks if x.name.lower() == page.lower()]), None)
        if page_dict is None:
            return JSONResponse({ 'metadata': 'none' })
        metadata = page_dict.metadata.get(item, 'none')
        if metadata is None:
            metadata = ''
        # shared.log.debug(f"Networks metadata: page='{page}' item={item} len={len(metadata)}")
        return JSONResponse({"metadata": metadata})

    def get_info(page: str = "", item: str = ""):
        page_dict = next(iter([x for x in get_pages() if x.name.lower() == page.lower()]), None)
        if page_dict is None:
            return JSONResponse({ 'info': 'none' })
        item_dict = next(iter([x for x in page_dict.items if x['name'].lower() == item.lower()]), None)
        if item_dict is None:
            return JSONResponse({ 'info': 'none' })
        info = page_dict.find_info(item_dict.get('filename', None) or item_dict.get('name', None))
        if info is None:
            info = {}
        # shared.log.debug(f"Networks info: page='{page.name}' item={item['name']} len={len(info)}")
        return JSONResponse({"info": info})

    def get_desc(page: str = "", item: str = ""):
        page_dict = next(iter([x for x in get_pages() if x.name.lower() == page.lower()]), None)
        if page_dict is None:
            return JSONResponse({ 'description': 'none' })
        item_dict = next(iter([x for x in page_dict.items if x['name'].lower() == item.lower()]), None)
        if item_dict is None:
            return JSONResponse({ 'description': 'none' })
        desc = page_dict.find_description(item_dict.get('filename', None) or item_dict.get('name', None))
        if desc is None:
            desc = ''
        # shared.log.debug(f"Networks desc: page='{page.name}' item={item['name']} len={len(desc)}")
        return JSONResponse({"description": desc})

    def get_network(page: str = "", item: str = ""):
        page_dict = next(iter([x for x in get_pages() if x.name.lower() == page.lower()]), None)
        if page_dict is None:
            return JSONResponse({ 'page': 'none' })
        item_dict = next(iter([x for x in page_dict.items if (x['alias'].lower() == item.lower() or x['name'].lower() == item.lower())]), None)
        if item_dict is None:
            return JSONResponse({ 'item': 'none' })
        obj = json.dumps(item_dict, cls=DateTimeEncoder)
        return JSONResponse(obj)

    shared.api.add_api_route("/sdapi/v1/network", get_network, methods=["GET"])
    shared.api.add_api_route("/sdapi/v1/network/thumb", fetch_file, methods=["GET"], auth=False)
    shared.api.add_api_route("/sdapi/v1/network/metadata", get_metadata, methods=["GET"], auth=False)
    shared.api.add_api_route("/sdapi/v1/network/info", get_info, methods=["GET"], auth=False)
    shared.api.add_api_route("/sdapi/v1/network/desc", get_desc, methods=["GET"], auth=False)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        from datetime import datetime
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.allow_negative_prompt = False
        self.metadata = {}
        self.info = {}
        self.html = ''
        self.items = []
        self.missing_thumbs = []
        self.refresh_time = 0
        self.page_time = 0
        self.list_time = 0
        self.info_time = 0
        self.desc_time = 0
        self.preview_time = 0
        self.dirs = {}
        self.view = shared.opts.extra_networks_view
        self.card = card_full if shared.opts.extra_networks_view == 'gallery' else card_list

    def __str__(self):
        return f'Page(title="{self.title}" name="{self.name}" items={len(self.items)})'

    def switch_view(self, tabname: str):
        new_view = 'gallery' if self.view == 'list' else 'list'
        self.view = new_view
        self.card = card_full if new_view == 'gallery' else card_list
        self.html = ''
        self.create_page(tabname)
        if shared.opts.extra_networks_view != new_view:
            shared.opts.extra_networks_view = new_view
            shared.opts.save()

    def refresh(self):
        pass

    def patch(self, text: str, tabname: str):
        return text.replace('~tabname', tabname).replace('txt2img', tabname)

    def create_xyz_grid(self):
        pass

    def find_version(self, item, info):
        all_versions = info.get('modelVersions', [])
        if len(all_versions) == 0:
            return {}
        try:
            if item is None:
                return all_versions[0]
            elif hasattr(item, 'hash') and item.hash is not None:
                current_hash = item.hash[:8].upper()
            elif hasattr(item, 'shorthash') and item.shorthash is not None:
                current_hash = item.shorthash[:8].upper()
            elif hasattr(item, 'sha256') and item.sha256 is not None:
                current_hash = item.sha256[:8].upper()
            else:
                return all_versions[0]
            for v in info.get('modelVersions', []):
                for f in v.get('files', []):
                    if any(h.startswith(current_hash) for h in f.get('hashes', {}).values()):
                        return v
        except Exception as e:
            errors.display(e, 'Network version')
        return all_versions[0]

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        mtime = os.path.getmtime(filename) if os.path.exists(filename) else 0
        preview = f"{shared.opts.subpath}/sdapi/v1/network/thumb?filename={quoted_filename}&mtime={mtime}"
        return preview

    def get_exif(self, image: Image.Image):
        import piexif
        import piexif.helper
        try:
            exifinfo = image.getexif()
            if exifinfo is not None and len(exifinfo) > 0:
                return piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        except Exception:
            pass
        try:
            exifinfo = image.info.get('parameters', None)
            if exifinfo is not None and len(exifinfo) > 0:
                return piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        except Exception:
            pass
        return piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump('', encoding="unicode") } })

    def create_thumb(self):
        debug(f'EN create-thumb: {self.name}')
        created = 0
        for f in self.missing_thumbs:
            if os.path.join('models', 'Reference') in f or not os.path.exists(f):
                continue
            fn = os.path.splitext(f)[0].replace('.preview', '')
            fn = f'{fn}.thumb.jpg'
            if os.path.exists(fn): # thumbnail already exists
                continue
            img = None
            try:
                img = Image.open(f)
                img.load()
            except Exception as e:
                img = None
                shared.log.warning(f'Network removing invalid: image={f} {e}')
            try:
                if img is None:
                    img = None
                    os.remove(f)
                elif (img.width > 1024) or (img.height > 1024) or (os.path.getsize(f) > 65536):
                    exif = self.get_exif(img)
                    img = img.convert('RGB')
                    img.thumbnail((512, 512), Image.Resampling.HAMMING)
                    img.save(fn, quality=50, exif=exif)
                    img.close()
                    created += 1
            except Exception as e:
                shared.log.warning(f'Network create thumbnail={f} {e}')
                errors.display(e, 'thumbnail')
        if created > 0:
            shared.log.info(f'Network thumbnails: type={self.name} created={created}')
            self.missing_thumbs.clear()

    def create_items(self, tabname):
        if self.refresh_time is not None and self.refresh_time > refresh_time: # cached results
            return
        t0 = time.time()
        try:
            self.items = list(self.list_items())
            self.refresh_time = time.time()
        except Exception as e:
            self.items = []
            shared.log.error(f'Networks: listing items class={self.__class__.__name__} tab={tabname} {e}')
            if os.environ.get('SD_EN_DEBUG', None):
                errors.display(e, f'Networks: listing items: class={self.__class__.__name__} tab={tabname}')
        for item in self.items:
            if item is None:
                continue
            self.metadata[item["name"]] = item.get("metadata", {})
        t1 = time.time()
        debug(f'EN create-items: page={self.name} items={len(self.items)} time={t1-t0:.2f}')
        self.list_time += t1-t0

    def create_page(self, tabname, skip = False):
        debug(f'EN create-page: {self.name}')
        if self.page_time > refresh_time and len(self.html) > 0: # cached page
            return self.patch(self.html, tabname)
        self_name_id = self.name.replace(" ", "_")
        if skip:
            return f"<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs'></div><div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>Network page not ready<br>Click refresh to try again</div>"
        subdirs = {}
        allowed_folders = [os.path.abspath(x) for x in self.allowed_directories_for_previews() if os.path.exists(x)]
        diffusers_base = os.path.basename(shared.opts.diffusers_dir)
        for parentdir, dirs in {d: files_cache.walk(d, cached=True, recurse=files_cache.not_hidden) for d in allowed_folders}.items():
            for tgt in dirs:
                tgt = tgt.path
                if os.path.join(paths.models_path, 'Reference') in tgt:
                    continue
                if shared.opts.diffusers_dir in tgt:
                    continue
                if 'models--' in tgt:
                    continue
                subdir = tgt[len(parentdir):].replace("\\", "/")
                while subdir.startswith("/"):
                    subdir = subdir[1:]
                if not subdir:
                    continue
                subdirs[subdir] = 1
        debug(f'Networks: page="{self.name}" subfolders={list(subdirs)}')
        subdirs = OrderedDict(sorted(subdirs.items()))
        if self.name == 'model' and shared.opts.extra_network_reference_enable:
            subdirs['Local'] = 1
            subdirs['Reference'] = 1
            subdirs['Distilled'] = 1
            subdirs['Quantized'] = 1
            subdirs['Nunchaku'] = 1
            subdirs['Community'] = 1
            subdirs['Cloud'] = 1
            subdirs[diffusers_base] = 1
        if self.name == 'style' and shared.opts.extra_networks_styles:
            subdirs['Local'] = 1
            subdirs['Reference'] = 1
        subdirs['All'] = 1
        if 'All' in subdirs:
            subdirs.move_to_end('All', last=False)
        if 'Local' in subdirs:
            subdirs.move_to_end('Local', last=True)
        if os.path.basename(shared.opts.diffusers_dir) in subdirs:
            subdirs.move_to_end(os.path.basename(shared.opts.diffusers_dir), last=True)
        if 'Reference' in subdirs:
            subdirs.move_to_end('Reference', last=True)
        if 'Distilled' in subdirs:
            subdirs.move_to_end('Distilled', last=True)
        if 'Quantized' in subdirs:
            subdirs.move_to_end('Quantized', last=True)
        if 'Nunchaku' in subdirs:
            subdirs.move_to_end('Nunchaku', last=True)
        if 'Community' in subdirs:
            subdirs.move_to_end('Community', last=True)
        if 'Cloud' in subdirs:
            subdirs.move_to_end('Cloud', last=True)
        subdirs_html = ''
        for subdir in subdirs:
            if len(subdir) == 0:
                continue
            if subdir in ['All', 'Local', 'Diffusers', 'Reference', 'Distilled', 'Quantized', 'Nunchaku', 'Community', 'Cloud']:
                style = 'network-reference'
            else:
                style = 'network-folder'
            subdirs_html += f'<button class="lg secondary gradio-button custom-button {style}" onclick="extraNetworksSearchButton(event)">{html.escape(subdir)}</button><br>'

        self.html = ''
        self.create_items(tabname)
        versions = sorted({item.get("version", "") for item in self.items if item.get("version")})
        for v in ['ref', 'reference', 'ready', 'download']:
            if v in versions:
                versions.remove(v)
        versions_html = ''
        for ver in versions:
            versions_html += f'<button class="lg secondary gradio-button custom-button network-model" onclick="extraNetworksFilterVersion(event)">{html.escape(ver)}</button><br>'
        self.create_xyz_grid()
        htmls = []

        if len(self.items) > 0 and self.items[0].get('mtime', None) is not None:
            if shared.opts.extra_networks_sort == 'Default':
                pass
            elif shared.opts.extra_networks_sort == 'Name [A-Z]':
                self.items.sort(key=lambda x: x["name"])
            elif shared.opts.extra_networks_sort == 'Name [Z-A]':
                self.items.sort(key=lambda x: x["name"], reverse=True)
            elif shared.opts.extra_networks_sort == 'Date [Newest]':
                self.items.sort(key=lambda x: x["mtime"], reverse=True)
            elif shared.opts.extra_networks_sort == 'Date [Oldest]':
                self.items.sort(key=lambda x: x["mtime"])
            elif shared.opts.extra_networks_sort == 'Size [Largest]':
                self.items.sort(key=lambda x: x["size"], reverse=True)
            elif shared.opts.extra_networks_sort == 'Size [Smallest]':
                self.items.sort(key=lambda x: x["size"])

        for item in self.items:
            htmls.append(self.create_html(item, tabname))
        self.html += ''.join(htmls)
        self.page_time = time.time()
        self.html = f"""<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs'>{subdirs_html}{versions_html}</div><div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>{self.html}</div>"""
        shared.log.debug(f'Networks: type="{self.name}" items={len(self.items)} subfolders={len(subdirs)} tab={tabname} folders={self.allowed_directories_for_previews()} list={self.list_time:.2f} thumb={self.preview_time:.2f} desc={self.desc_time:.2f} info={self.info_time:.2f} workers={shared.max_workers}')
        if len(self.missing_thumbs) > 0:
            threading.Thread(target=self.create_thumb).start()
        return self.patch(self.html, tabname)

    def list_items(self):
        raise NotImplementedError

    def allowed_directories_for_previews(self):
        return []

    def create_html(self, item, tabname):
        def random_bright_color():
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            return '#{:02x}{:02x}{:02x}'.format(r, g, b) # pylint: disable=consider-using-f-string

        try:
            onclick = f'cardClicked({item.get("prompt", None)})'
            args = {
                # "tabname": tabname,
                "page": self.name,
                "name": html.escape(item.get('name', ''), quote=True),
                "title": os.path.basename(item["name"].replace('_', ' ')),
                "filename": html.escape(item.get('filename', ''), quote=True),
                "short": os.path.splitext(os.path.basename(item.get('filename', '')))[0],
                "tags": '|'.join([item.get('tags')] if isinstance(item.get('tags', {}), str) else list(item.get('tags', {}).keys())),
                "preview": html.escape(item.get('preview', None) or self.link_preview('html/missing.png')),
                "width": 'var(--card-size)',
                "height": 'var(--card-size)' if shared.opts.extra_networks_card_square else 'auto',
                "fit": shared.opts.extra_networks_card_fit,
                "prompt": item.get("prompt", None),
                "search": item.get("search_term", ""),
                "description": item.get("description") or "",
                "card_click": item.get("onclick", '"' + html.escape(onclick) + '"'),
                "mtime": item.get("mtime", 0),
                "size": item.get("size", 0),
                "version": item.get("version", ''),
                "color": random_bright_color(),
                "reference": "reference" if 'Reference' in item.get('name', '') else "",
            }
            # alias = item.get("alias", None)
            # if alias is not None:
            #     args['title'] += f'\nAlias: {alias}'
            return self.card.format(**args)
        except Exception as e:
            shared.log.error(f'Networks: item error: page={tabname} item={item["name"]} {e}')
            if os.environ.get('SD_EN_DEBUG', None) is not None:
                errors.display(e, 'Networks')
            return ""

    def find_preview_file(self, path):
        if path is None:
            return 'html/missing.png'
        if os.path.join('models', 'Reference') in path:
            return path
        exts = ["jpg", "jpeg", "png", "webp", "tiff", "jp2", "jxl"]
        reference_path = os.path.abspath(os.path.join('models', 'Reference'))
        files = list(files_cache.list_files(reference_path, ext_filter=exts, recursive=False))
        if shared.opts.diffusers_dir in path:
            path = os.path.relpath(path, shared.opts.diffusers_dir)
            fn = os.path.join(reference_path, path.replace('models--', '').replace('\\', '/').split('/')[0])
        else:
            fn = os.path.splitext(path)[0]
            files += list(files_cache.list_files(os.path.dirname(path), ext_filter=exts, recursive=False))
        for file in [f'{fn}{mid}{ext}' for ext in exts for mid in ['.thumb.', '.', '.preview.']]:
            if file in files:
                if '.thumb.' not in file:
                    self.missing_thumbs.append(file)
                return file
        return 'html/missing.png'

    def find_preview(self, filename):
        t0 = time.time()
        preview_file = self.find_preview_file(filename)
        self.preview_time += time.time() - t0
        return self.link_preview(preview_file)

    def update_all_previews(self, items):
        global preview_map # pylint: disable=global-statement
        if preview_map is None:
            preview_file = os.path.join('data', 'previews.json')
            preview_map = shared.readfile(preview_file, silent=True, as_type="dict")
        t0 = time.time()
        reference_path = os.path.abspath(os.path.join('models', 'Reference'))
        possible_paths = list(set([os.path.dirname(item['filename']) for item in items] + [reference_path]))
        exts = ["jpg", "jpeg", "png", "webp", "tiff", "jp2", "jxl"]
        all_previews = list(files_cache.list_files(*possible_paths, ext_filter=exts, recursive=False))
        all_previews_fn = [os.path.basename(x) for x in all_previews]
        for item in items:
            if item.get('preview', None) is not None:
                continue
            base = os.path.splitext(item['filename'])[0]
            if item.get('local_preview', None) is None:
                item['local_preview'] = f'{base}.{shared.opts.samples_format}'
            if shared.opts.diffusers_dir in base:
                if 'models--' in base:
                    match = re.search(r"models--([^/^\\]+)[/\\]", base)
                    if match is None:
                        match = re.search(r"models--(.*)", base)
                    base = os.path.join(reference_path, match[1])
                    model_path = os.path.join(shared.opts.diffusers_dir, match[0])
                    item['local_preview'] = f'{os.path.join(model_path, match[1])}.{shared.opts.samples_format}'
                    all_previews += list(files_cache.list_files(model_path, ext_filter=exts, recursive=False))
                else:
                    if os.path.isdir(base):
                        item['local_preview'] = os.path.join(base, f'{os.path.basename(base)}.{shared.opts.samples_format}')
            base = os.path.basename(base)
            for file in [f'{base}{mid}{ext}' for ext in exts for mid in ['.thumb.', '.', '.preview.']]:
                if file in all_previews_fn:
                    file_idx = all_previews_fn.index(os.path.basename(file))
                    if '.thumb.' not in file:
                        self.missing_thumbs.append(all_previews[file_idx])
                    item['preview'] = self.link_preview(all_previews[file_idx])
                    break
            if item.get('preview', None) is None:
                found = preview_map.get(base, None)
                if found is not None:
                    item['preview'] = self.link_preview(found)
                    debug(f'EN mapped-preview: {item["name"]}={found}')
            if item.get('preview', None) is None:
                item['preview'] = self.link_preview('html/missing.png')
                debug(f'EN missing-preview: {item["name"]}')
        self.preview_time += time.time() - t0

    def find_description(self, path, info=None):
        t0 = time.time()
        class HTMLFilter(HTMLParser):
            text = ""
            def handle_data(self, data):
                self.text += data
            def handle_endtag(self, tag):
                if tag == 'p':
                    self.text += '\n'

        if path is not None:
            fn = os.path.splitext(path)[0] + '.txt'
            if os.path.exists(fn):
                try:
                    with open(fn, "r", encoding="utf-8", errors="replace") as f:
                        txt = f.read()
                        txt = re.sub('[<>]', '', txt)
                        return txt
                except OSError:
                    pass
            if info is None:
                info = self.find_info(path)
        if not isinstance(info, dict):
            self.desc_time += time.time() - t0
            return ''
        desc = info.get('description', '') or ''
        f = HTMLFilter()
        f.feed(desc)
        t1 = time.time()
        self.desc_time += t1-t0
        return f.text

    def find_info(self, path):
        data = {}
        if shared.cmd_opts.no_metadata:
            return data
        if path is not None:
            t0 = time.time()
            fn = os.path.splitext(path)[0] + '.json'
            if not data and os.path.exists(fn):
                data = shared.readfile(fn, silent=True, as_type="dict")
            fn = os.path.join(path, 'model_index.json')
            if not data and os.path.exists(fn):
                data = shared.readfile(fn, silent=True, as_type="dict")
            t1 = time.time()
            self.info_time += t1-t0
        return data


def initialize():
    shared.extra_networks.clear()


def register_page(page: ExtraNetworksPage):
    # registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions
    debug(f'EN register-page: {page}')
    if page in shared.extra_networks:
        debug(f'EN register-page: {page} already registered')
        return
    shared.extra_networks.append(page)
    # allowed_dirs.clear()
    # for pg in shared.extra_networks:
    for folder in page.allowed_directories_for_previews():
        if folder not in allowed_dirs:
            allowed_dirs.append(os.path.abspath(folder))


def register_pages():
    debug('EN register-pages')
    shared.extra_networks.clear()
    allowed_dirs.clear()
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    register_page(ExtraNetworksPageCheckpoints())
    from modules.ui_extra_networks_vae import ExtraNetworksPageVAEs
    register_page(ExtraNetworksPageVAEs())
    from modules.ui_extra_networks_styles import ExtraNetworksPageStyles
    register_page(ExtraNetworksPageStyles())
    from modules.ui_extra_networks_lora import ExtraNetworksPageLora
    register_page(ExtraNetworksPageLora())
    from modules.ui_extra_networks_wildcards import ExtraNetworksPageWildcards
    register_page(ExtraNetworksPageWildcards())
    if shared.opts.latent_history > 0:
        from modules.ui_extra_networks_history import ExtraNetworksPageHistory
        register_page(ExtraNetworksPageHistory())
    if shared.opts.diffusers_enable_embed:
        from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
        register_page(ExtraNetworksPageTextualInversion())
    from modules.video_models.models_def import models # pylint: disable=unused-import


def get_pages(title=None):
    visible = shared.opts.extra_networks
    pages: list[ExtraNetworksPage] = []
    if 'All' in visible or visible == []: # default en sort order
        visible = ['Model', 'Lora', 'Style', 'Wildcards', 'Embedding', 'VAE', 'History', 'Hypernetwork']

    titles = [page.title for page in shared.extra_networks]
    if title is None:
        for page in visible:
            try:
                idx = titles.index(page)
                pages.append(shared.extra_networks[idx])
            except ValueError:
                continue
    else:
        try:
            idx = titles.index(title)
            pages.append(shared.extra_networks[idx])
        except ValueError:
            pass
    return pages


class ExtraNetworksUi:
    def __init__(self):
        self.tabname: str = None
        self.pages: list[str] = None
        self.visible: gr.State = None
        self.state: gr.Textbox = None
        self.details: gr.Group = None
        self.details_tabs: gr.Group = None
        self.details_text: gr.Group = None
        self.tabs: gr.Tabs = None
        self.gallery: gr.Gallery = None
        self.description: gr.Textbox = None
        self.search: gr.Textbox = None
        self.button_details: gr.Button = None
        self.button_refresh: gr.Button = None
        self.button_scan: gr.Button = None
        self.button_view: gr.Button = None
        self.button_quicksave: gr.Button = None
        self.button_save: gr.Button = None
        self.button_sort: gr.Button = None
        self.button_apply: gr.Button = None
        self.button_close: gr.Button = None
        self.button_model: gr.Checkbox = None
        self.details_components: list = []
        self.last_item: dict = None
        self.last_page: ExtraNetworksPage = None
        self.state: gr.State = None


def create_ui(container, button_parent, tabname, skip_indexing = False):
    if 'networks' in shared.opts.ui_disabled:
        return None
    debug(f'EN create-ui: {tabname}')
    ui = ExtraNetworksUi()
    ui.tabname = tabname
    ui.pages = []
    ui.state = gr.Textbox('{}', elem_id=f"{tabname}_extra_state", visible=False)
    ui.visible = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    ui.details = gr.Group(elem_id=f"{tabname}_extra_details", elem_classes=["extra-details"], visible=False)
    ui.tabs = gr.Tabs(elem_id=f"{tabname}_extra_tabs")
    ui.button_details = gr.Button('Details', elem_id=f"{tabname}_extra_details_btn", visible=False)
    state = {}

    def get_item(state, params = None):
        if params is not None and type(params) == dict:
            page = next(iter([x for x in get_pages() if x.title == 'Style']), None)
            item = page.create_style(params)
        else:
            if state is None or not hasattr(state, 'page') or not hasattr(state, 'item'):
                return None, None
            page = next(iter([x for x in get_pages() if x.title == state.page]), None)
            if page is None:
                return None, None
            item = next(iter([x for x in page.items if x["name"] == state.item]), None)
            if item is None:
                return page, None
        item = SimpleNamespace(**item)
        ui.last_item = item
        ui.last_page = page
        return page, item

    # main event that is triggered when js updates state text field with json values, used to communicate js -> python
    def state_change(state_text):
        try:
            nonlocal state
            state = SimpleNamespace(**json.loads(state_text))
        except Exception as e:
            shared.log.error(f'Networks: state error: {e}')
            return
        _page, _item = get_item(state)
        # shared.log.debug(f'Extra network: op={state.op} page={page.title if page is not None else None} item={item.filename if item is not None else None}')

    def toggle_visibility(is_visible):
        is_visible = not is_visible
        return is_visible, gr.update(visible=is_visible), gr.update(variant=("secondary-down" if is_visible else "secondary"))

    with ui.details:
        details_close = ui_components.ToolButton(ui_symbols.close, elem_id=f"{tabname}_extra_details_close", elem_classes=['extra-details-close'])
        details_close.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[ui.details])
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.HTML('<div>title</div>')
                ui.details_components.append(text)
            with gr.Column(scale=1):
                img = gr.Image(value=None, show_label=False, interactive=False, container=False, show_download_button=False, elem_id=f"{tabname}_extra_details_img", elem_classes=['extra-details-img'])
                ui.details_components.append(img)
                with gr.Row():
                    btn_save_img = gr.Button('Replace', elem_classes=['small-button'])
                    btn_delete_img = gr.Button('Delete', elem_classes=['small-button'])
        with gr.Group(elem_id=f"{tabname}_extra_details_tabs", visible=False) as ui.details_tabs:
            with gr.Tabs():
                with gr.Tab('Description', elem_classes=['extra-details-tabs']):
                    desc = gr.Textbox('', show_label=False, lines=8, placeholder="Network description...")
                    ui.details_components.append(desc)
                    with gr.Row():
                        btn_save_desc = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_desc')
                        btn_delete_desc = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_desc')
                        btn_close_desc = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_desc')
                        btn_close_desc.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])
                with gr.Tab('Model metadata', elem_classes=['extra-details-tabs']):
                    info = gr.JSON({}, show_label=False)
                    ui.details_components.append(info)
                    with gr.Row():
                        btn_save_info = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_info')
                        btn_delete_info = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_info')
                        btn_close_info = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_info')
                        btn_close_info.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])
                with gr.Tab('Embedded metadata', elem_classes=['extra-details-tabs']):
                    meta = gr.JSON({}, show_label=False)
                    ui.details_components.append(meta)
                with gr.Tab('Preview metadata', elem_classes=['extra-details-tabs']):
                    thumb = gr.JSON({}, show_label=False)
                    ui.details_components.append(thumb)
        with gr.Group(elem_id=f"{tabname}_extra_details_text", elem_classes=["extra-details-text"], visible=False) as ui.details_text:
            description = gr.Textbox(label='Description', lines=1, placeholder="Style description...")
            prompt = gr.Textbox(label='Network prompt', lines=2, placeholder="Prompt...")
            negative = gr.Textbox(label='Network negative prompt', lines=2, placeholder="Negative prompt...")
            extra = gr.Textbox(label='Network parameters', lines=2, placeholder="Generation parameters overrides...")
            wildcards = gr.Textbox(label='Wildcards', lines=2, placeholder="Wildcard prompt replacements...")
            ui.details_components += [description, prompt, negative, extra, wildcards]
            with gr.Row():
                btn_save_style = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_style')
                btn_delete_style = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_style')
                btn_close_style = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_style')
                btn_close_style.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])

    with ui.tabs:
        def ui_tab_change(page):
            scan_visible = page in ['Model', 'Lora', 'VAE', 'Hypernetwork', 'Embedding']
            save_visible = page in ['Style']
            model_visible = page in ['Model']
            return [gr.update(visible=scan_visible), gr.update(visible=save_visible), gr.update(visible=model_visible)]

        ui.button_refresh = ui_components.ToolButton(ui_symbols.refresh, elem_id=f"{tabname}_extra_refresh")
        ui.button_scan = ui_components.ToolButton(ui_symbols.scan, elem_id=f"{tabname}_extra_scan", visible=True)
        ui.button_quicksave = ui_components.ToolButton(ui_symbols.book, elem_id=f"{tabname}_extra_quicksave", visible=False)
        ui.button_save = ui_components.ToolButton(ui_symbols.book, elem_id=f"{tabname}_extra_save", visible=False)
        ui.button_sort = ui_components.ToolButton(ui_symbols.sort, elem_id=f"{tabname}_extra_sort", visible=True)
        ui.button_view = ui_components.ToolButton(ui_symbols.view, elem_id=f"{tabname}_extra_view", visible=True)
        ui.button_close = ui_components.ToolButton(ui_symbols.close, elem_id=f"{tabname}_extra_close", visible=True)
        ui.button_model = ui_components.ToolButton(ui_symbols.refine, elem_id=f"{tabname}_extra_model", visible=True)
        ui.search = gr.Textbox('', show_label=False, elem_id=f"{tabname}_extra_search", placeholder="Search...", elem_classes="textbox", lines=2, container=False)
        ui.description = gr.Textbox('', show_label=False, elem_id=f"{tabname}_description", elem_classes=["textbox", "extra-description"], lines=2, interactive=False, container=False)

        if ui.tabname == 'txt2img': # refresh only once
            global refresh_time # pylint: disable=global-statement
            refresh_time = time.time()
        if not skip_indexing:
            import concurrent
            with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
                for page in get_pages():
                    executor.submit(page.create_items, ui.tabname)
        for page in get_pages():
            page.create_page(ui.tabname, skip_indexing)
            with gr.Tab(page.title, id=page.title.lower().replace(" ", "_"), elem_classes="extra-networks-tab") as tab:
                page_html = gr.HTML(page.patch(page.html, tabname), elem_id=f'{tabname}{page.name}_extra_page', elem_classes="extra-networks-page")
                ui.pages.append(page_html)
                tab.select(ui_tab_change, _js="getENActivePage", inputs=[ui.button_details], outputs=[ui.button_scan, ui.button_save, ui.button_model])

    def fn_save_img(image):
        if ui.last_item is None or ui.last_item.local_preview is None:
            return 'html/missing.png'
        images = []
        if ui.gallery is not None:
            images = list(ui.gallery.temp_files) # gallery cannot be used as input component so looking at most recently registered temp files
        if len(images) < 1:
            shared.log.warning(f'Network no image: item="{ui.last_item.name}"')
            return 'html/missing.png'
        try:
            images.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            image = Image.open(images[0])
        except Exception as e:
            shared.log.error(f'Network error opening image: item="{ui.last_item.name}" {e}')
            return 'html/missing.png'
        fn_delete_img(image)
        if image.width > 512 or image.height > 512:
            image = image.convert('RGB')
            image.thumbnail((512, 512), Image.Resampling.HAMMING)
        try:
            image.save(ui.last_item.local_preview, quality=50)
            shared.log.debug(f'Networks save image: item="{ui.last_item.name}" filename="{ui.last_item.local_preview}"')
        except Exception as e:
            shared.log.error(f'Network save image: item="{ui.last_item.name}" filename="{ui.last_item.local_preview}" {e}')
        return image

    def fn_delete_img(_image):
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2", "jxl"]
        fn = os.path.splitext(ui.last_item.filename)[0]
        for file in [f'{fn}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if os.path.exists(file):
                os.remove(file)
                shared.log.debug(f'Network delete image: item="{ui.last_item.name}" filename="{file}"')
        return 'html/missing.png'

    def fn_save_desc(desc):
        if hasattr(ui.last_item, 'type') and ui.last_item.type == 'Style':
            params = ui.last_page.parse_desc(desc)
            if params is not None:
                fn_save_info(params)
        else:
            fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(desc)
            shared.log.debug(f'Network save desc: item="{ui.last_item.name}" filename="{fn}"')
        return desc

    def fn_delete_network(desc):
        if ui.last_item is None:
            return desc
        basename = os.path.splitext(ui.last_item.filename)[0]
        extensions = ['.safetensors', '.ckpt', '.txt', '.json', '.thumb.jpg', '.jpg', '.jpeg', '.png', '.webp', '.tiff', '.jp2', '.jxl']
        candidates = []
        for ext in extensions:
            fn = basename + ext
            if os.path.exists(fn) and os.path.isfile(fn):
                candidates.append(fn)
        msg = f'Network delete: item="{ui.last_item.name}" files={candidates}'
        shared.log.debug(msg)
        for fn in candidates:
            os.remove(fn)
        return msg

    def fn_save_info(info):
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        shared.writefile(info, fn, silent=True)
        shared.log.debug(f'Network save info: item="{ui.last_item.name}" filename="{fn}"')
        return info

    def fn_save_style(info, description, prompt, negative, extra, wildcards):
        if not isinstance(info, dict) or isinstance(info, list):
            shared.log.warning(f'Network save style skip: item="{ui.last_item.name}" not a dict: {type(info)}')
            return info
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if hasattr(ui.last_item, 'type') and ui.last_item.type == 'Style':
            info.update(**{ 'description': description, 'prompt': prompt, 'negative': negative, 'extra': extra, 'wildcards': wildcards })
            shared.writefile(info, fn, silent=True)
            shared.log.debug(f'Network save style: item="{ui.last_item.name}" filename="{fn}"')
        return info

    def fn_delete_style(info):
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if os.path.exists(fn):
            shared.log.debug(f'Network delete style: item="{ui.last_item.name}" filename="{fn}"')
            os.remove(fn)
            return {}
        return info

    btn_save_img.click(fn=fn_save_img, _js='closeDetailsEN', inputs=[img], outputs=[img])
    btn_delete_img.click(fn=fn_delete_img, _js='closeDetailsEN', inputs=[img], outputs=[img])
    btn_save_desc.click(fn=fn_save_desc, _js='closeDetailsEN', inputs=[desc], outputs=[desc])
    btn_delete_desc.click(fn=fn_delete_network, _js='closeDetailsEN', inputs=[desc], outputs=[desc])
    btn_save_info.click(fn=fn_save_info, _js='closeDetailsEN', inputs=[info], outputs=[info])
    btn_delete_info.click(fn=fn_delete_network, _js='closeDetailsEN', inputs=[info], outputs=[desc])
    btn_save_style.click(fn=fn_save_style, _js='closeDetailsEN', inputs=[info, description, prompt, negative, extra, wildcards], outputs=[info])
    btn_delete_style.click(fn=fn_delete_style, _js='closeDetailsEN', inputs=[info], outputs=[info])

    def show_details(text, img, desc, info, meta, thumb, description, prompt, negative, parameters, wildcards, params, _dummy1=None, _dummy2=None):
        from modules import images
        page, item = get_item(state, params)
        is_style = (page is not None) and (page.title == 'Style')
        is_valid = (item is not None) and hasattr(item, 'name') and hasattr(item, 'filename')

        if is_valid:
            if TYPE_CHECKING:
                assert item is not None # Part of the definition of "is_valid"
            stat_size, stat_mtime = modelstats.stat(item.filename)
            if hasattr(item, 'size') and item.size > 0:
                stat_size = item.size
            if hasattr(item, 'mtime') and item.mtime is not None:
                stat_mtime = item.mtime
            desc = item.description
            fullinfo = shared.readfile(os.path.splitext(item.filename)[0] + '.json', silent=True, as_type="dict")
            if 'modelVersions' in fullinfo: # sanitize massive objects
                fullinfo['modelVersions'] = []
            info = fullinfo
            if isinstance(info, list):
                item.filename = None
                shared.log.warning('Network: show details not supported for compound item')
                info = None
            if prompt is not None and len(prompt) > 0:
                item.prompt = prompt
            if negative is not None and len(negative) > 0:
                item.negative = negative
            if description is not None and len(description) > 0:
                item.description = description
            if wildcards is not None and len(wildcards) > 0:
                item.wildcards = wildcards

            meta = page.metadata.get(item.name, {}) or {}
            if type(meta) is str:
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            if ui.last_item.preview.startswith('data:'):
                b64str = ui.last_item.preview.split(',',1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64str)))
            elif hasattr(item, 'local_preview') and os.path.exists(item.local_preview):
                img = item.local_preview
            else:
                img = page.find_preview_file(item.filename)

            _geninfo, thumb = images.read_info_from_image(img)

            lora = ''
            model = ''
            style = ''
            note = ''
            if item.filename is not None and not os.path.exists(item.filename):
                note = f'<br>Target filename: {item.filename}'
            if page.title == 'Model':
                merge = len(list(meta.get('sd_merge_models', {})))
                if merge > 0:
                    model += f'<tr><td>Merge models</td><td>{merge} recipes</td></tr>'
                if meta.get('modelspec.architecture', None) is not None:
                    model += f'''
                        <tr><td>Architecture</td><td>{meta.get('modelspec.architecture', 'N/A')}</td></tr>
                        <tr><td>Title</td><td>{meta.get('modelspec.title', 'N/A')}</td></tr>
                        <tr><td>Resolution</td><td>{meta.get('modelspec.resolution', 'N/A')}</td></tr>
                    '''
            if page.title == 'Lora':
                try:
                    tags = getattr(item, 'tags', {})
                    tags = [f'{name}:{tags[name]}' for i, name in enumerate(tags)]
                    tags = ' '.join(tags)
                except Exception:
                    tags = ''
                try:
                    triggers = ' '.join(info.get('tags', []))
                except Exception:
                    triggers = ''
                lora = f'''
                    <tr><td>Model tags</td><td>{tags}</td></tr>
                    <tr><td>User tags</td><td>{triggers}</td></tr>
                    <tr><td>Base model</td><td>{meta.get('ss_sd_model_name', 'N/A')}</td></tr>
                    <tr><td>Resolution</td><td>{meta.get('ss_resolution', 'N/A')}</td></tr>
                    <tr><td>Training images</td><td>{meta.get('ss_num_train_images', 'N/A')}</td></tr>
                    <tr><td>Comment</td><td>{meta.get('ss_training_comment', 'N/A')}</td></tr>
                '''
            if page.title == 'Style':
                description = item.description
                prompt = item.prompt
                negative = item.negative
                parameters = item.extra
                wildcards = item.wildcards
                style = f'''
                    <tr><td>Name</td><td>{item.name}</td></tr>
                    <tr><td>Description</td><td>{item.description}</td></tr>
                    <tr><td>Preview Embedded</td><td>{item.preview.startswith('data:')}</td></tr>
                '''
            if item.name.startswith('Diffusers'):
                url = item.name.replace('Diffusers/', '')
                url = f'<a href="https://huggingface.co/{url}" target="_blank">https://huggingface.co/models/{url}</a>' if url is not None else 'N/A'
            else:
                url = info.get('id', None) if info is not None else None
                url = f'<a href="https://civitai.com/models/{url}" target="_blank">civitai.com/models/{url}</a>' if url is not None else 'N/A'
            text = f'''
                <h2 style="border-bottom: 1px solid var(--button-primary-border-color); margin: 0em 0px 1em 0 !important">{item.name}</h2>
                <table style="width: 100%; line-height: 1.5em;"><tbody>
                    <tr><td>Type</td><td>{page.title}</td></tr>
                    <tr><td>Alias</td><td>{getattr(item, 'alias', 'N/A')}</td></tr>
                    <tr><td>Filename</td><td>{item.filename}</td></tr>
                    <tr><td>Hash</td><td>{getattr(item, 'hash', 'N/A')}</td></tr>
                    <tr><td>Size</td><td>{round(stat_size/1024/1024, 2)} MB</td></tr>
                    <tr><td>Last modified</td><td>{stat_mtime}</td></tr>
                    <tr><td>Source URL</td><td>{url}</td></tr>
                    <tr><td style="border-top: 1px solid var(--button-primary-border-color);"></td><td></td></tr>
                    {lora}
                    {model}
                    {style}
                </tbody></table>
                {note}
            '''
        return [
            text, # gr.html
            img, # gr.image
            desc, # gr.textbox
            info, # gr.json
            meta, # gr.json
            thumb, # gr.json
            description, # gr.textbox
            gr.update(value=prompt, visible=is_style), # gr.textbox
            gr.update(value=negative, visible=is_style), # gr.textbox
            gr.update(value=parameters, visible=is_style), # gr.textbox
            gr.update(value=wildcards, visible=is_style), # gr.textbox
            gr.update(visible=is_valid), # details ui visible
            gr.update(visible=not is_style), # details ui tabs visible
            gr.update(visible=is_style), # details ui text visible
        ]

    def ui_refresh_click(title):
        pages = []
        for page in get_pages():
            if page.title != title:
                pages.append(page.html)
                continue
            page.page_time = 0
            page.refresh_time = 0
            page.refresh()
            page.create_page(ui.tabname)
            shared.log.debug(f'Networks: refresh page="{page.title}" items={len(page.items)} tab={ui.tabname}')
            pages.append(page.html)
        ui.search.update(title)
        return pages

    def ui_view_cards(title):
        pages = []
        for page in get_pages():
            page.switch_view(ui.tabname)
            shared.log.debug(f'Networks: refresh page="{page.title}" items={len(page.items)} tab={ui.tabname} view={page.view}')
            pages.append(page.html)
        ui.search.update(title)
        return pages

    def ui_scan_click(title):
        from modules.civitai.metadata_civitai import civit_search_metadata
        for _generator in civit_search_metadata(title): # need to read generator output so python does not optimize function away
            pass
        return ui_refresh_click(title)

    def ui_save_click():
        from modules.processing_info import get_last_args
        params, text = get_last_args()
        if (not params) or (not text) or (len(text) == 0):
            if os.path.exists(paths.params_path):
                with open(paths.params_path, "r", encoding="utf8") as file:
                    text = file.read()
            else:
                text = ''
            params = infotext.parse(text)
        prompt = params.get('Original prompt', None) or params.get('Prompt', '')
        negative = params.get('Original negative', None) or params.get('Negative prompt', '')
        res = show_details(text=None, img=None, desc=None, info=None, meta=None, thumb=None, parameters=None, description=None, prompt=prompt, negative=negative, wildcards=None, params=params)
        return res

    def ui_quicksave_click(name):
        if name is None or len(name) < 1:
            shared.log.warning("Network quick save style: no name provided")
            return
        from modules.processing_info import get_last_args
        params, text = get_last_args()
        if (not params) or (not text) or (len(text) == 0):
            if os.path.exists(paths.params_path):
                with open(paths.params_path, "r", encoding="utf8") as file:
                    text = file.read()
            else:
                text = ''
            params = infotext.parse(text)
        fn = os.path.join(shared.opts.styles_dir, os.path.splitext(name)[0] + '.json')
        prompt = params.get('Original prompt', None) or params.get('Prompt', '')
        negative = params.get('Original negative', None) or params.get('Negative prompt', '')
        item = {
            "name": name,
            "description": '',
            "prompt": prompt,
            "negative": negative,
            "extra": '',
        }
        shared.writefile(item, fn, silent=True)
        if len(prompt) > 0:
            shared.log.debug(f'Networks type=style quicksave style: item="{name}" filename="{fn}" prompt="{prompt}"')
        else:
            shared.log.warning(f'Networks type=style quicksave model: item="{name}" filename="{fn}" prompt is empty')

    def ui_sort_cards(sort_order):
        if shared.opts.extra_networks_sort != sort_order:
            shared.opts.extra_networks_sort = sort_order
            shared.opts.save()
        return f'Networks: sort={sort_order}'

    dummy = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    button_parent.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container, button_parent])
    ui.button_close.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container])
    ui.button_sort.click(fn=ui_sort_cards, _js='sortExtraNetworks', inputs=[ui.search], outputs=[ui.description])
    ui.button_view.click(fn=ui_view_cards, inputs=[ui.search], outputs=ui.pages)
    ui.button_refresh.click(fn=ui_refresh_click, _js='getENActivePage', inputs=[ui.search], outputs=ui.pages)
    ui.button_scan.click(fn=ui_scan_click, _js='getENActivePage', inputs=[ui.search], outputs=ui.pages)
    ui.button_save.click(fn=ui_save_click, inputs=[], outputs=ui.details_components + [ui.details])
    ui.button_quicksave.click(fn=ui_quicksave_click, _js="() => prompt('Prompt name', '')", inputs=[ui.search], outputs=[])
    ui.button_details.click(show_details, _js="getCardDetails", inputs=ui.details_components + [dummy, dummy, dummy], outputs=ui.details_components + [ui.details, ui.details_tabs, ui.details_text])
    ui.state.change(state_change, inputs=[ui.state], outputs=[])
    return ui


def setup_ui(ui, gallery: gr.Gallery = None):
    if ui is None:
        return
    ui.gallery = gallery
