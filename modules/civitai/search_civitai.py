from dataclasses import dataclass
import os
import json
import time
from installer import install, log


full_dct = False
full_html = False
base_models = ['', 'ODOR', 'SD 1.4', 'SD 1.5', 'SD 1.5 LCM', 'SD 1.5 Hyper', 'SD 2.0', 'SD 2.0 768', 'SD 2.1', 'SD 2.1 768', 'SD 2.1 Unclip', 'SDXL 0.9', 'SDXL 1.0', 'SD 3', 'SD 3.5', 'SD 3.5 Medium', 'SD 3.5 Large', 'SD 3.5 Large Turbo', 'Pony', 'Flux.1 S', 'Flux.1 D', 'Flux.1 Kontext', 'AuraFlow', 'SDXL 1.0 LCM', 'SDXL Distilled', 'SDXL Turbo', 'SDXL Lightning', 'SDXL Hyper', 'Stable Cascade', 'SVD', 'SVD XT', 'Playground v2', 'PixArt a', 'PixArt E', 'Hunyuan 1', 'Hunyuan Video', 'Lumina', 'Kolors', 'Illustrious', 'Mochi', 'LTXV', 'CogVideoX', 'NoobAI', 'Wan Video', 'Wan Video 1.3B t2v', 'Wan Video 14B t2v', 'Wan Video 14B i2v 480p', 'Wan Video 14B i2v 720p', 'HiDream', 'OpenAI', 'Imagen4', 'Other']


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
        self.hashes: list[str] = [str(h) for h in dct.get('hashes', {}).values()]
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


models: list[Model] = []  # global cache for civitai search results


def search_civitai(
        query:str,
        tag:str = '', # optional:tag name
        types:str = '', # (Checkpoint, TextualInversion, Hypernetwork, AestheticGradient, LORA, Controlnet, Poses)
        sort:str = '', # (Highest Rated, Most Downloaded, Newest)
        period:str = '', # (AllTime, Year, Month, Week, Day)
        nsfw:bool = None, # optional:bool
        limit:int = 0,
        base:str = '', # list
        token:str = None,
        exact:bool = True,
):
    global models # pylint: disable=global-statement
    import requests
    from urllib.parse import urlencode
    install('bs4')  # Ensure BeautifulSoup is installed

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
        dct['baseModels'] = base
    encoded = urlencode(dct)

    headers = {}
    if token is None:
        token = os.environ.get('CIVITAI_TOKEN', None)
    if token is not None and len(token) > 0:
        headers['Authorization'] = f'Bearer {token}'

    url = 'https://civitai.com/api/v1/models'
    if query.isnumeric():
        uri = f'{url}/{query}'
    else:
        uri = f'{url}?{encoded}'

    log.info(f'CivitAI request: uri="{uri}" dct={dct} token={token is not None}')
    result = requests.get(uri, headers=headers, timeout=60)

    if result.status_code != 200:
        log.error(f'CivitAI: code={result.status_code} reason={result.reason} uri={result.url}')
        return []

    all_models: list[Model] = []
    exact_models: list[Model] = []
    dct = result.json()
    if 'items' not in dct:
        items = [dct] # single model
    else:
        items = dct.get('items', [])
    for item in items:
        all_models.append(Model(item))

    if exact:
        for model in all_models:
            model_names = [model.name.lower()]
            version_names = [v.name.lower() for v in model.versions]
            file_names = [f.name.lower() for v in model.versions for f in v.files]
            if any([query.lower() in name for name in model_names + version_names + file_names]): # noqa: C419 # pylint: disable=use-a-generator
                exact_models.append(model)

    t1 = time.time()
    log.info(f'CivitAI result: code={result.status_code} exact={len(exact_models)} total={len(models)} time={t1-t0:.2f}')
    models = exact_models if len(exact_models) > 0 else all_models
    return models


def create_model_cards(all_models: list[Model]) -> str:
    details = """
        <div id="model-details">
        </div>
    """
    cards = """
        <div id="model-cards" class="extra-network-cards">
            {cards}
        </div>
    """
    card = """
        <div class="card" data-id="{id}" onclick="modelCardClick({id})">
            <div class="overlay"><div class="name">{name}</div></div>
            <div class="version">{type}</div>
            <img class="preview" src="{preview}" alt="{name}" loading="lazy" />
        </div>
    """
    all_cards = ''
    for model in all_models:
        previews = []
        for version in model.versions:
            for image in version.images:
                if image.url and len(image.url) > 0 and not image.url.lower().endswith('.mp4'):
                    previews.append(image.url)
        if len(previews) == 0:
            previews = ['/sdapi/v1/network/thumb?filename=html/card-no-preview.png']
        all_cards += card.format(id=model.id, name=model.name, type=model.type, preview=previews[0])
    html = details + cards.format(cards=all_cards)
    return html


def print_models(all_models: list[Model]):
    for model in all_models:
        log.info(f' {model}')
        log.trace('Model', model.dct)
        for version in model.versions:
            log.info(f'  {version}')
            log.trace('ModelVersion', version.dct)
            for file in version.files:
                log.info(f'   {file}')
                log.trace('ModelFile', file.dct)
            for image in version.images:
                log.info(f'   {image}')
                log.trace('ModelImage', image.dct)
