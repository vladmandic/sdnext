import re
import time
from html import escape
from installer import log
from modules.civitai.client_civitai import client
from modules.civitai.models_civitai import CivitModel, CivitSearchResponse


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
) -> CivitSearchResponse:
    if not query and not tag and not sort:
        log.error('CivitAI: no search criteria provided')
        return CivitSearchResponse(error='no search criteria provided')

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
            return CivitSearchResponse(items=[model])
        return CivitSearchResponse(error=f'model {query} not found')

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

    response.items = exact_models if exact_models else all_models
    t1 = time.time()
    log.info(f'CivitAI result: exact={len(exact_models)} total={len(all_models)} time={t1 - t0:.2f}')
    return response


def create_model_cards(response: CivitSearchResponse) -> str:
    if response.error:
        notice = f'CivitAI: {escape(response.error)}'
    elif not response.items:
        notice = 'No models found'
    else:
        notice = ''
    details = f'<div id="model-details">{notice}</div>'
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
    for model in response.items:
        previews = []
        for version in model.versions:
            for image in version.images:
                if image.url and not image.url.lower().endswith('.mp4'):
                    previews.append(image.url)
        if not previews:
            previews = ['/sdapi/v1/network/thumb?filename=ui/assets/missing.png']
        all_cards += card.format(id=model.id, name=model.name, type=model.type, preview=previews[0])
    html = details + cards.format(cards=all_cards)
    return html
