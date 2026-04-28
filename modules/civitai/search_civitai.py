import re
import time
from installer import log
from modules.civitai.client_civitai import client
from modules.civitai.models_civitai import CivitModel, CivitSearchResponse


# Hardcoded fallback list — used by Gradio UI if discover_options() fails
base_models = ['', 'AuraFlow', 'Chroma', 'CogVideoX', 'Flux.1 S', 'Flux.1 D', 'Flux.1 Krea', 'Flux.1 Kontext', 'Flux.2 D', 'HiDream', 'Hunyuan 1', 'Hunyuan Video', 'Illustrious', 'Kolors', 'LTXV', 'Lumina', 'Mochi', 'NoobAI', 'PixArt a', 'PixArt E', 'Pony', 'Pony V7', 'Qwen', 'SD 1.4', 'SD 1.5', 'SD 1.5 LCM', 'SD 1.5 Hyper', 'SD 2.0', 'SD 2.1', 'SDXL 1.0', 'SDXL Lightning', 'SDXL Hyper', 'Wan Video 1.3B t2v', 'Wan Video 14B t2v', 'Wan Video 14B i2v 480p', 'Wan Video 14B i2v 720p', 'Wan Video 2.2 TI2V-5B', 'Wan Video 2.2 I2V-A14B', 'Wan Video 2.2 T2V-A14B', 'Wan Video 2.5 T2V', 'Wan Video 2.5 I2V', 'ZImageTurbo', 'Other']


def search_civitai(
        query: str,
        tag: str = '',
        types: str = '',
        sort: str = '',
        period: str = '',
        nsfw: bool | None = None,
        limit: int = 0,
        base: str = '',
        token: str | None = None,
        exact: bool = True,
) -> list[CivitModel]:
    if not query and not tag and not sort:
        log.error('CivitAI: no search criteria provided')
        return []

    t0 = time.time()

    # URL query → extract model ID (e.g. https://civitai.com/models/967405/nova-orange-xl)
    url_match = re.match(r'https?://civitai\.(?:com|red)/models/(\d+)', query.strip())
    if url_match:
        query = url_match.group(1)
        log.info(f'CivitAI: extracted model id={query} from URL')

    # Numeric query → single model fetch
    if query.isnumeric():
        model = client.get_model(int(query), token=token)
        if model:
            t1 = time.time()
            log.info(f'CivitAI result: id={query} time={t1 - t0:.2f}')
            return [model]
        return []

    response: CivitSearchResponse = client.search_models(
        query=query,
        tag=tag,
        types=types,
        sort=sort,
        period=period,
        base_models=[base] if base else None,
        nsfw=nsfw,
        limit=limit if limit > 0 else 20,
        token=token,
    )

    all_models = response.items
    exact_models: list[CivitModel] = []
    if exact and query:
        q_lower = query.lower()
        for model in all_models:
            names = [model.name.lower()]
            names.extend(v.name.lower() for v in model.versions)
            names.extend(f.name.lower() for v in model.versions for f in v.files)
            if any(q_lower in name for name in names):
                exact_models.append(model)

    result = exact_models if exact_models else all_models
    t1 = time.time()
    log.info(f'CivitAI result: exact={len(exact_models)} total={len(all_models)} time={t1 - t0:.2f}')
    return result


def create_model_cards(all_models: list[CivitModel]) -> str:
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
                if image.url and not image.url.lower().endswith('.mp4'):
                    previews.append(image.url)
        if not previews:
            previews = ['/sdapi/v1/network/thumb?filename=html/missing.png']
        all_cards += card.format(id=model.id, name=model.name, type=model.type, preview=previews[0])
    html = details + cards.format(cards=all_cards)
    return html
