#!/usr/bin/env python
from dataclasses import dataclass
import os
import sys
import json
import time
import logging


full_dct = False
full_html = False
debug = False
logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


@dataclass
class ModelImage():
    def __init__(self, dct: dict):
        if isinstance(dct, str):
            dct = json.loads(dct)
        self.id: int = dct.get('id', 0)
        self.url: str = dct.get('url', '')
        self.width: int = dct.get('width', 0)
        self.height: int = dct.get('height', 0)
        self.type: str = dct.get('type', 'Unknown')
        self.dct: dict = dct if full_dct else {}

    def __str__(self):
        return f'ModelImage(id={self.id} url="{self.url}" width={self.width} height={self.height} type="{self.type}")'

@dataclass
class ModelFile():
    def __init__(self, dct: dict):
        if isinstance(dct, str):
            dct = json.loads(dct)
        self.id: int = dct.get('id', 0)
        self.size: int = int(1024 * dct.get('sizeKB', 0))
        self.name: str = dct.get('name', 'Unknown')
        self.type: str = dct.get('type', 'Unknown')
        self.hashes: list[str] = dct.get('hashes', {}).values()
        self.url: str = dct.get('downloadUrl', '')
        self.dct: dict = dct if full_dct else {}

    def __str__(self):
        return f'ModelFile(id={self.id} name="{self.name}" size={self.size} type="{self.type}" url="{self.url}")'


@dataclass
class ModelVersion():
    def __init__(self, dct: dict):
        import bs4
        if isinstance(dct, str):
            dct = json.loads(dct)
        self.id: int = dct.get('id', 0)
        self.name: str = dct.get('name', 'Unknown')
        self.base: str = dct.get('baseModel', 'Unknown')
        self.mtime: str = dct.get('publishedAt', '')
        self.downloads: int = dct.get('stats', {}).get('downloadCount', 0)
        self.availability: str = dct.get('availability', 'Unknown')
        self.html: str = dct.get('description', '') or '' if full_html else ''
        self.desc: str = bs4.BeautifulSoup(dct.get('description', '') or '', features="html.parser").get_text()
        self.files = [ModelFile(f) for f in dct.get('files', [])]
        self.images = [ModelImage(i) for i in dct.get('images', [])]
        self.dct: dict = dct if full_dct else {}

    def __str__(self):
        return f'ModelVersion(id={self.id} name="{self.name}" base="{self.base}" mtime="{self.mtime}" downloads={self.downloads} availability={self.availability} desc="{self.desc[:30]}...")'


@dataclass
class Model():
    def __init__(self, dct: dict):
        import bs4
        if isinstance(dct, str):
            dct = json.loads(dct)
        self.id: int = dct.get('id', 0)
        self.url: str = f'https://civitai.com/models/{self.id}'
        self.type: str = dct.get('type', 'Unknown')
        self.name: str = dct.get('name', 'Unknown')
        self.html: str = dct.get('description', '') or '' if full_html else ''
        self.desc: str = bs4.BeautifulSoup(dct.get('description', '') or '', features="html.parser").get_text()
        self.tags: list[str] = dct.get('tags', [])
        self.nsfw: bool = dct.get('nsfw', False)
        self.level: str = dct.get('nsfwLevel', 0)
        self.availability: str = dct.get('availability', 'Unknown')
        self.downloads: int = dct.get('stats', {}).get('downloadCount', 0)
        self.creator: str = dct.get('creator', {}).get('username', 'Unknown')
        self.versions: list[ModelVersion] = [ModelVersion(v) for v in dct.get('modelVersions', [])]
        self.dct: dict = dct if full_dct else {}

    def __str__(self):
        return f'Model(id={self.id} type={self.type} name="{self.name}" versions={len(self.versions)} nsfw={self.nsfw}/{self.level} downloads={self.downloads} author="{self.creator}" tags={self.tags} desc="{self.desc[:30]}...")'


def search_civitai(
        query:str,
        tag:str = '', # optional:tag name
        types:str = '', # (Checkpoint, TextualInversion, Hypernetwork, AestheticGradient, LORA, Controlnet, Poses)
        sort:str = '', # (Highest Rated, Most Downloaded, Newest)
        period:str = '', # (AllTime, Year, Month, Week, Day)
        nsfw:bool = None, # optional:bool
        limit:int = 0,
        base:list[str] = [], # list
        token:str = None,
        exact:bool = True,
):
    import requests
    from urllib.parse import urlencode

    if len(query) == 0:
        log.error('CivitAI: empty query')
        return []

    t0 = time.time()
    dct = { 'query': query }
    if len(tag) > 0:
        dct['tag'] = tag
    if nsfw is not None:
        dct['nsfw'] = 'true' if nsfw else 'false'
    if limit > 0:
        dct['limit'] = limit
    if len(types) > 0:
        dct['types'] = types
    if len(sort) > 0:
        dct['sort'] = sort
    if len(period) > 0:
        dct['period'] = period
    if len(base) > 0:
        dct['baseModels'] = ','.join(base)
    encoded = urlencode(dct)

    headers = {}
    if token is None:
        token = os.environ.get('CIVITAI_TOKEN', None)
    if token is not None and len(token) > 0:
        headers['Authorization'] = f'Bearer {token}'

    url = 'https://civitai.com/api/v1/models'
    uri = f'{url}?{encoded}'
    log.info(f'CivitAI request: uri="{uri}" dct={dct} token={token is not None}')
    result = requests.get(uri, headers=headers, timeout=60)

    if result.status_code != 200:
        log.error(f'CivitAI: code={result.status_code} reason={result.reason} uri={result.url}')
        return []

    models: list[Model] = []
    exact_models: list[Model] = []
    items = result.json().get('items', [])
    for item in items:
        models.append(Model(item))

    if exact:
        for model in models:
            model_names = [model.name.lower()]
            version_names = [v.name.lower() for v in model.versions]
            file_names = [f.name.lower() for v in model.versions for f in v.files]
            if any([query.lower() in name for name in model_names + version_names + file_names]): # noqa: C419
                exact_models.append(model)

    t1 = time.time()
    log.info(f'CivitAI result: code={result.status_code} exact={len(exact_models)} total={len(models)} time={t1-t0:.2f}')
    return exact_models if len(exact_models) > 0 else models


def models_to_dct(all_models:list, model_id:int=None):
    dct = []
    for model in all_models:
        if model_id is not None and model.id != model_id:
            continue
        model_dct = model.__dict__.copy()
        versions_dct = []
        for version in model.versions:
            version_dct = version.__dict__.copy()
            version_dct['files'] = [f.__dict__.copy() for f in version.files]
            version_dct['images'] = [i.__dict__.copy() for i in version.images]
            versions_dct.append(version_dct)
        model_dct['versions'] = versions_dct
        dct.append(model_dct)
    return dct


def print_models(models: list[Model]):
    if debug:
        from rich import print as dbg
    else:
        dbg = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment
    for model in models:
        log.info(f' {model}')
        dbg('Model', model.dct)
        for version in model.versions:
            log.info(f'  {version}')
            dbg('ModelVersion', version.dct)
            for file in version.files:
                log.info(f'   {file}')
                dbg('ModelFile', file.dct)
            for image in version.images:
                log.info(f'   {image}')
                dbg('ModelImage', image.dct)


if __name__ == "__main__":
    sys.argv.pop(0)
    txt = ' '.join(sys.argv)
    res = search_civitai(
        query=txt,
        # tag = '',
        # types = '',
        # sort = 'Most Downloaded',
        # period = 'Year',
        # nsfw = True,
        # base = [],
        # exact= True,
        # limit=100,
    )
    print_models(res)
